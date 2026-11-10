"""Free-tier LLM providers behind one OpenAI-compatible client.

NVIDIA build.nvidia.com, Groq, Gemini, OpenRouter and a local Ollama server all expose the
OpenAI chat-completions API, so one client covers them; switching providers is configuration.

Each pipeline role picks its model from an environment variable holding "provider" or
"provider:model" (the model part may itself contain "/" or ":"):

    MEDX_PLANNER_LLM  query planning between retrieval rounds   (default: first of nvidia, groq, gemini, openrouter)
    MEDX_ANSWER_LLM   final grounded answer                     (default: first of nvidia, groq, gemini, openrouter)
    MEDX_JUDGE_LLM    grading answers during evaluation         (default: first of nvidia, groq, gemini, openrouter)

On NVIDIA the judge defaults to a different model family (GLM) than the answer model
(Nemotron), so evaluation is not a model grading its own output. If a call fails outright,
the next provider with a key is tried (set MEDX_FALLBACK=0 to disable).

Every response is cached on disk under .llm_cache/, keyed by provider, model, messages and
parameters, so rerunning an evaluation costs no quota and returns identical results.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - only used when optional dependency is missing.
    load_dotenv = None


ENV_PATH = Path(__file__).with_name(".env")
CACHE_DIR = Path(__file__).with_name(".llm_cache")

PLACEHOLDER_KEYS = {"", "your_key_here", "paste_your_gemini_api_key_here"}


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: Optional[str]
    default_model: str
    # Requests per minute we allow ourselves; kept under each free tier's published limit.
    rpm: int
    # Whether the endpoint accepts response_format={"type": "json_object"}.
    json_mode: bool


PROVIDERS: Dict[str, Provider] = {
    "nvidia": Provider(
        name="nvidia",
        base_url="https://integrate.api.nvidia.com/v1",
        key_env="NVIDIA_API_KEY",
        default_model=os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3-super-120b-a12b"),
        rpm=int(os.environ.get("NVIDIA_RPM", "20")),
        json_mode=False,
    ),
    "groq": Provider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        key_env="GROQ_API_KEY",
        default_model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        rpm=int(os.environ.get("GROQ_RPM", "25")),
        json_mode=True,
    ),
    "gemini": Provider(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        key_env="GEMINI_API_KEY",
        default_model=os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"),
        rpm=int(os.environ.get("GEMINI_RPM", "8")),
        json_mode=True,
    ),
    "openrouter": Provider(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        key_env="OPENROUTER_API_KEY",
        default_model=os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
        rpm=int(os.environ.get("OPENROUTER_RPM", "15")),
        json_mode=True,
    ),
    "ollama": Provider(
        name="ollama",
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        key_env=None,
        default_model=os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"),
        rpm=0,
        json_mode=True,
    ),
}

ROLE_ENV = {"planner": "MEDX_PLANNER_LLM", "answer": "MEDX_ANSWER_LLM", "judge": "MEDX_JUDGE_LLM"}
ROLE_PREFERENCE = {
    "planner": ["nvidia", "groq", "gemini", "openrouter"],
    "answer": ["nvidia", "groq", "gemini", "openrouter"],
    "judge": ["nvidia", "groq", "gemini", "openrouter"],
}
# The judge must not be the model that wrote the answers. NVIDIA hosts other model families,
# so its judge default is GLM rather than Nemotron.
ROLE_MODEL_DEFAULTS = {
    ("judge", "nvidia"): os.environ.get("NVIDIA_JUDGE_MODEL", "z-ai/glm-5.3-flash"),
}

THINK_RE = re.compile(r"<think>.*?</think>", re.S)


class LLMUnavailable(RuntimeError):
    """No provider is configured for a role, or the provider's quota is exhausted for now."""


def load_project_env() -> None:
    if load_dotenv is not None:
        load_dotenv(ENV_PATH, override=True)


def get_api_key(env_name: Optional[str]) -> Optional[str]:
    if env_name is None:
        return None
    load_project_env()
    value = (os.environ.get(env_name) or "").strip().strip('"')
    if env_name == "GEMINI_API_KEY" and not value:
        value = (os.environ.get("GOOGLE_API_KEY") or "").strip().strip('"')
    return None if value in PLACEHOLDER_KEYS else value


def provider_available(name: str) -> bool:
    provider = PROVIDERS[name]
    if provider.key_env is None:
        # Ollama needs no key; count it only once the user points us at a running server.
        load_project_env()
        return bool(os.environ.get("OLLAMA_BASE_URL"))
    return get_api_key(provider.key_env) is not None


def resolve_spec(role: str, spec: Optional[str] = None) -> Optional[str]:
    """Return "provider:model" for a role, or None when no provider is configured."""
    load_project_env()
    spec = spec or os.environ.get(ROLE_ENV[role], "").strip()
    if not spec:
        for name in ROLE_PREFERENCE[role]:
            if provider_available(name):
                return f"{name}:{_default_model(role, name)}"
        return None
    name, _, model = spec.partition(":")
    if name not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider {name!r} in {spec!r}; choose from {', '.join(PROVIDERS)}")
    return f"{name}:{model or _default_model(role, name)}"


def _default_model(role: str, provider: str) -> str:
    return ROLE_MODEL_DEFAULTS.get((role, provider)) or PROVIDERS[provider].default_model


def fallback_specs(role: str, primary: str) -> List[str]:
    """Backup "provider:model" specs for a role: other providers with keys, in preference order.

    Set MEDX_FALLBACK=0 to disable, e.g. for an evaluation where every answer must come from one model.
    """
    load_project_env()
    if os.environ.get("MEDX_FALLBACK", "1").strip().lower() in ("0", "false", "no"):
        return []
    primary_provider = primary.partition(":")[0]
    return [
        f"{name}:{_default_model(role, name)}"
        for name in ROLE_PREFERENCE[role]
        if name != primary_provider and provider_available(name)
    ]


class _RateLimiter:
    """Spaces requests to each provider so we stay under its per-minute free-tier limit."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_allowed: Dict[str, float] = {}

    def wait(self, provider: Provider) -> None:
        if provider.rpm <= 0:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_allowed.get(provider.name, now))
            self._next_allowed[provider.name] = start + 60.0 / provider.rpm
        if start > now:
            time.sleep(start - now)


_RATE_LIMITER = _RateLimiter()


class LLM:
    def __init__(self, spec: str):
        name, _, model = spec.partition(":")
        self.provider = PROVIDERS[name]
        self.model = model or self.provider.default_model
        api_key = get_api_key(self.provider.key_env)
        if self.provider.key_env and not api_key:
            raise LLMUnavailable(f"{self.provider.key_env} is not set in .env")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The OpenAI client is not installed. Install it with: pip install openai") from exc

        # max_retries=0: retries are handled below so daily-quota errors can fail fast.
        self.client = OpenAI(base_url=self.provider.base_url, api_key=api_key or "ollama", max_retries=0, timeout=120)

    @property
    def label(self) -> str:
        return f"{self.provider.name}:{self.model}"

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        json_output: bool = False,
    ) -> str:
        messages = [{"role": "system", "content": self._system_prompt(system)}, {"role": "user", "content": user}]
        params: Dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens}
        if json_output and self.provider.json_mode:
            params["response_format"] = {"type": "json_object"}

        cache_path = self._cache_path(messages, params)
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))["text"]

        text = self._call_with_retries(messages, params)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"model": self.label, "messages": messages, "params": params, "text": text}, ensure_ascii=False),
            encoding="utf-8",
        )
        return text

    def chat_json(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 4096) -> Dict[str, Any]:
        return extract_json(self.chat(system, user, temperature=temperature, max_tokens=max_tokens, json_output=True))

    def _system_prompt(self, system: str) -> str:
        # Nemotron reasoning models think out loud unless told not to; the pipeline only needs answers.
        if "nemotron" in self.model.lower():
            return f"/no_think\n{system}"
        return system

    def _cache_path(self, messages: List[Dict[str, str]], params: Dict[str, Any]) -> Path:
        key = json.dumps({"model": self.label, "messages": messages, "params": params}, sort_keys=True)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return CACHE_DIR / self.provider.name / f"{digest}.json"

    def _call_with_retries(self, messages: List[Dict[str, str]], params: Dict[str, Any], attempts: int = 8) -> str:
        import openai

        for attempt in range(attempts):
            _RATE_LIMITER.wait(self.provider)
            try:
                response = self.client.chat.completions.create(model=self.model, messages=messages, **params)
            except (openai.APIConnectionError, openai.APITimeoutError) as exc:
                error: Exception = exc
                retry_after = None
            except openai.APIStatusError as exc:
                if exc.status_code not in (408, 429, 500, 502, 503, 504):
                    raise
                error, retry_after = exc, _retry_after_seconds(exc)
                if exc.status_code == 429 and _is_daily_quota(exc, retry_after):
                    raise LLMUnavailable(f"{self.label} daily quota exhausted: {exc.message}") from exc
            else:
                choice = response.choices[0]
                text = _strip_reasoning(choice.message.content or "")
                if not text:
                    raise RuntimeError(f"{self.label} returned no text (finish_reason={choice.finish_reason}).")
                if choice.finish_reason == "length":
                    # Hidden reasoning tokens count toward max_tokens on some models (e.g. Gemini 3).
                    print(f"{self.label}: reply cut off at max_tokens={params['max_tokens']}", file=sys.stderr)
                    text += "\n\n[Answer truncated: the model hit its output limit.]"
                return text

            if attempt == attempts - 1:
                raise error
            delay = retry_after if retry_after is not None else min(4.0 * (2**attempt), 60.0)
            print(f"{self.label}: {error.__class__.__name__}; retrying in {delay:.0f}s", file=sys.stderr)
            time.sleep(delay)
        raise AssertionError("unreachable")


def _retry_after_seconds(exc: Any) -> Optional[float]:
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    value = headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _is_daily_quota(exc: Any, retry_after: Optional[float]) -> bool:
    # Waiting hours for a daily quota to reset would only hang the pipeline.
    if retry_after is not None and retry_after > 120:
        return True
    message = str(getattr(exc, "message", "") or exc).lower()
    return any(marker in message for marker in ("per day", "perday", "daily", "tokens per day", "rpd"))


def _strip_reasoning(text: str) -> str:
    text = THINK_RE.sub("", text)
    # Some reasoning models emit only a closing tag when the opening one is in the prompt template.
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def extract_json(text: str) -> Dict[str, Any]:
    """Parse the first JSON object in a model reply, tolerating code fences and surrounding prose."""
    text = _strip_reasoning(text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"No JSON object in model reply: {text[:200]!r}")
    return json.loads(text[start : end + 1])


class FallbackLLM:
    """Tries the primary model, then each backup provider, when a call fails outright.

    label reports the model that produced the most recent reply, so outputs record which model
    actually answered even after a fallback.
    """

    def __init__(self, primary: str, backups: List[str]):
        self.specs = [primary] + backups
        self._clients: Dict[str, LLM] = {}
        self.label = primary

    def _client(self, spec: str) -> LLM:
        if spec not in self._clients:
            self._clients[spec] = LLM(spec)
        return self._clients[spec]

    def chat(self, *args: Any, **kwargs: Any) -> str:
        return self._first_success("chat", *args, **kwargs)

    def chat_json(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self._first_success("chat_json", *args, **kwargs)

    def _first_success(self, method: str, *args: Any, **kwargs: Any) -> Any:
        errors: List[str] = []
        for spec in self.specs:
            try:
                result = getattr(self._client(spec), method)(*args, **kwargs)
            except Exception as exc:
                errors.append(f"{spec}: {exc.__class__.__name__}: {exc}")
                if spec != self.specs[-1]:
                    print(f"{spec} failed ({exc.__class__.__name__}); falling back", file=sys.stderr)
                continue
            self.label = spec
            return result
        raise LLMUnavailable("All configured models failed:\n  " + "\n  ".join(errors))


_CLIENTS: Dict[str, FallbackLLM] = {}


def get_llm(role: str, spec: Optional[str] = None) -> Optional[FallbackLLM]:
    """The client for a pipeline role ("planner", "answer" or "judge"), or None if no provider is set up."""
    resolved = resolve_spec(role, spec)
    if resolved is None:
        return None
    backups = fallback_specs(role, resolved)
    key = "|".join([resolved] + backups)
    if key not in _CLIENTS:
        _CLIENTS[key] = FallbackLLM(resolved, backups)
    return _CLIENTS[key]


def describe_roles() -> Dict[str, Optional[str]]:
    descriptions: Dict[str, Optional[str]] = {}
    for role in ROLE_ENV:
        resolved = resolve_spec(role)
        backups = fallback_specs(role, resolved) if resolved else []
        descriptions[role] = resolved and (resolved + (f"  (fallback: {', '.join(backups)})" if backups else ""))
    return descriptions


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("Configured providers:", ", ".join(name for name in PROVIDERS if provider_available(name)) or "none")
    for role, spec in describe_roles().items():
        print(f"  {role:8s} -> {spec or 'not configured'}")
    if len(sys.argv) > 1 and sys.argv[1] == "ping":
        for role in ROLE_ENV:
            llm = get_llm(role)
            if llm is None:
                continue
            try:
                reply = llm.chat("Reply with exactly one word.", f"Say OK. ({role} check)", max_tokens=200)
                print(f"  {role:8s} ok: {reply[:60]!r}")
            except Exception as exc:
                print(f"  {role:8s} FAILED: {exc.__class__.__name__}: {exc}")
