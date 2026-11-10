"""LLM-only baseline with an Ollama-compatible generation client."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .data import MedQARecord
from .scope import ScopeAnnotation


class LLMClient(Protocol):
    """Minimal interface required by the LLM-only runner."""

    def generate(self, prompt: str) -> str:
        """Generate one response for a prompt."""


class OllamaClient:
    """Client for Ollama's local ``/api/generate`` endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        temperature: float = 0.0,
        timeout_seconds: float = 120.0,
    ) -> None:
        if not model.strip():
            raise ValueError("model must be non-empty")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.model = model.strip()
        self.endpoint = base_url.rstrip("/") + "/api/generate"
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    def generate(
        self, prompt: str, response_format: dict[str, Any] | str | None = None
    ) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if response_format is not None:
            payload["format"] = response_format
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.endpoint}: {exc.reason}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned invalid JSON") from exc

        generated = body.get("response")
        if not isinstance(generated, str):
            raise RuntimeError("Ollama response did not contain a text 'response' field")
        return generated


def build_prompt(record: MedQARecord) -> str:
    """Build the fixed no-retrieval prompt used by every baseline question."""
    options = "\n".join(f"{label}. {text}" for label, text in record.options.items())
    labels = ", ".join(record.options)
    return (
        "You are answering a medical multiple-choice benchmark question. "
        "You have no external documents or retrieval context. "
        "Choose the single best option using only your general knowledge.\n\n"
        f"Question:\n{record.question}\n\n"
        f"Options:\n{options}\n\n"
        f"Return only valid JSON in this exact shape: {{\"answer\": \"{labels.split(', ')[0]}\"}}. "
        f"The answer value must be exactly one of: {labels}."
    )


def parse_answer(response: str, valid_labels: tuple[str, ...]) -> str | None:
    """Extract one option label without accepting arbitrary generated text."""
    if not response.strip():
        return None
    labels = {label.upper() for label in valid_labels}
    candidate = response.strip()
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE)
    decoded_objects: list[dict[str, Any]] = []
    try:
        decoded = json.loads(candidate)
        if isinstance(decoded, dict):
            decoded_objects.append(decoded)
    except json.JSONDecodeError:
        pass
    for match in re.finditer(r"\{.*?\}", response, flags=re.DOTALL):
        try:
            decoded = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            decoded_objects.append(decoded)
    for decoded in reversed(decoded_objects):
        for field in ("answer", "predicted_answer", "option", "choice"):
            value = decoded.get(field)
            if isinstance(value, str):
                match = re.fullmatch(r"\s*([A-Za-z])\s*", value)
                if match and match.group(1).upper() in labels:
                    return match.group(1).upper()

    patterns = (
        r"^\s*([A-Za-z])\s*$",
        r"^\s*(?:answer|choice|option)\s*[:=-]\s*([A-Za-z])(?:\b|[.)])",
        r"^\s*(?:the\s+)?(?:answer|choice|option)\s+is\s+([A-Za-z])(?:\b|[.)])",
        r"^\s*.*?\bcorrect\s+answer\s+is\s*[:=-]?\s*([A-Za-z])(?:\b|[.)])",
    )
    for pattern in patterns:
        match = re.search(pattern, response, flags=re.IGNORECASE | re.MULTILINE)
        if match and match.group(1).upper() in labels:
            return match.group(1).upper()
    return None


@dataclass(frozen=True)
class LLMOnlyResult:
    """One serializable result row for the common evaluation schema."""

    question_id: str
    system_name: str
    question: str
    predicted_answer: str | None
    expected_answer: str | None
    is_correct: bool | None
    retrieved_chunk_ids: list[str]
    citations: list[str]
    latency_ms: float
    model: str
    temperature: float
    prompt: str
    raw_response: str | None
    error: str | None
    abstention_reason: str | None
    scope: str | None
    evidence_expected: bool | None
    annotation_version: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_record(
    record: MedQARecord,
    client: LLMClient,
    model: str,
    temperature: float,
    annotation: ScopeAnnotation | None = None,
) -> LLMOnlyResult:
    """Run one benchmark record and preserve failures as explicit result rows."""
    prompt = build_prompt(record)
    started = time.perf_counter()
    raw_response: str | None = None
    error: str | None = None
    predicted: str | None = None
    try:
        raw_response = client.generate(prompt)
        predicted = parse_answer(raw_response, tuple(record.options))
        if predicted is None:
            error = "model response did not contain one valid option label"
    except (OSError, RuntimeError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    expected = record.answer_letter
    return LLMOnlyResult(
        question_id=record.id,
        system_name="llm_only",
        question=record.question,
        predicted_answer=predicted,
        expected_answer=expected,
        is_correct=predicted == expected if predicted is not None and expected is not None else None,
        retrieved_chunk_ids=[],
        citations=[],
        latency_ms=latency_ms,
        model=model,
        temperature=temperature,
        prompt=prompt,
        raw_response=raw_response,
        error=error,
        abstention_reason=error if predicted is None else None,
        scope=annotation.scope if annotation else None,
        evidence_expected=annotation.evidence_expected if annotation else None,
        annotation_version=annotation.annotation_version if annotation else None,
    )


def write_results(path: str | Path, results: list[LLMOnlyResult]) -> None:
    """Write result rows as deterministic UTF-8 JSONL."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(json.dumps(result.as_dict(), ensure_ascii=False) + "\n")
