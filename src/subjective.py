"""Subjective medical QA response and insufficient-evidence policy."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .bm25 import RetrievedChunk


EVIDENCE_STATUSES = (
    "supported",
    "partial",
    "unsupported",
    "abstained",
)


class SubjectiveLLMClient(Protocol):
    def generate(self, prompt: str, response_format: dict[str, Any] | None = None) -> str:
        """Generate a structured subjective answer."""


@dataclass(frozen=True)
class SubjectiveResponse:
    """One source-cited free-form medical answer."""

    question: str
    method: str
    answer: str
    evidence_status: str
    citations: list[str]
    limitations: str
    abstention_reason: str | None
    retrieved_chunk_ids: list[str]
    retrieval_results: list[dict[str, Any]]
    latency_ms: float
    model: str
    raw_response: str | None
    error: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_subjective_prompt(
    question: str,
    retrieved: list[RetrievedChunk],
) -> str:
    if not question.strip():
        raise ValueError("question must be non-empty")
    contexts = "\n\n".join(
        f"[{chunk.chunk_id}] {chunk.textbook_name} - {chunk.title}\n{chunk.text}"
        for chunk in retrieved
    )
    allowed = ", ".join(chunk.chunk_id for chunk in retrieved)
    return (
        "You are a medical evidence-grounding assistant. Answer the user's "
        "subjective medical question using only the retrieved textbook passages "
        "when they provide adequate support. Do not claim clinical certainty "
        "and do not invent facts or citations.\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Retrieved evidence:\n{contexts or 'No adequate evidence retrieved.'}\n\n"
        f"Allowed citation chunk IDs: {allowed or 'none'}\n"
        "Return only valid JSON with this exact shape:\n"
        "{"
        "\"answer\":\"...\","
        "\"evidence_status\":\"supported|partial|unsupported|abstained\","
        "\"citations\":[\"chunk_id\"],"
        "\"limitations\":\"...\","
        "\"abstention_reason\":\"... or null\""
        "}\n"
        "Use evidence_status=abstained if the passages do not support a reliable "
        "answer. Citations must contain only allowed chunk IDs."
    )


def subjective_response_schema(retrieved: list[RetrievedChunk]) -> dict[str, Any]:
    allowed_ids = [chunk.chunk_id for chunk in retrieved]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "answer": {"type": "string", "minLength": 1},
            "evidence_status": {
                "type": "string",
                "enum": list(EVIDENCE_STATUSES),
            },
            "citations": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": allowed_ids,
                },
            },
            "limitations": {"type": "string"},
            "abstention_reason": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "null"},
                ]
            },
        },
        "required": [
            "answer",
            "evidence_status",
            "citations",
            "limitations",
            "abstention_reason",
        ],
    }


def evaluate_abstention(
    retrieved: list[RetrievedChunk],
    *,
    method: str,
    dense_threshold: float = 0.25,
    hybrid_threshold: float = 0.015,
    min_chunk_chars: int = 100,
) -> tuple[bool, str | None]:
    """Apply a documented, bounded insufficient-evidence policy."""
    if dense_threshold < -1 or dense_threshold > 1:
        raise ValueError("dense_threshold must be between -1 and 1")
    if hybrid_threshold < 0:
        raise ValueError("hybrid_threshold must be non-negative")
    if min_chunk_chars <= 0:
        raise ValueError("min_chunk_chars must be positive")
    if not retrieved:
        return True, "No textbook chunks were retrieved"

    adequate = [
        chunk
        for chunk in retrieved
        if len(chunk.text.strip()) >= min_chunk_chars
    ]
    if not adequate:
        return True, "Retrieved chunks do not contain sufficient passage text"

    if method == "dense":
        if max(chunk.score for chunk in adequate) < dense_threshold:
            return True, (
                f"Dense similarity did not reach the configured threshold "
                f"{dense_threshold:.3f}"
            )
    elif method == "hybrid":
        if max(chunk.score for chunk in adequate) < hybrid_threshold:
            return True, (
                f"Hybrid RRF score did not reach the configured threshold "
                f"{hybrid_threshold:.3f}"
            )
    elif method == "bm25":
        if max(chunk.score for chunk in adequate) <= 0:
            return True, "BM25 did not retrieve a positive relevance score"
    else:
        raise ValueError(f"Unknown retrieval method: {method}")
    return False, None


def parse_subjective_response(
    response: str,
    allowed_ids: set[str],
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    candidate = response.strip().strip("`")
    try:
        decoded = json.loads(candidate)
        if isinstance(decoded, dict):
            candidates.append(decoded)
    except json.JSONDecodeError:
        pass
    for match in re.finditer(r"\{.*?\}", response, flags=re.DOTALL):
        try:
            decoded = json.loads(match.group(0))
            if isinstance(decoded, dict):
                candidates.append(decoded)
        except json.JSONDecodeError:
            continue

    for decoded in reversed(candidates):
        answer = decoded.get("answer")
        status = decoded.get("evidence_status")
        citations = decoded.get("citations")
        limitations = decoded.get("limitations")
        abstention_reason = decoded.get("abstention_reason")
        if not isinstance(answer, str) or not answer.strip():
            continue
        if status not in EVIDENCE_STATUSES:
            continue
        if not isinstance(citations, list):
            continue
        if any(
            not isinstance(chunk_id, str) or chunk_id not in allowed_ids
            for chunk_id in citations
        ):
            continue
        if not isinstance(limitations, str):
            continue
        if abstention_reason is not None and not isinstance(abstention_reason, str):
            continue
        if status == "abstained" and not abstention_reason:
            continue
        return {
            "answer": answer.strip(),
            "evidence_status": status,
            "citations": list(dict.fromkeys(citations)),
            "limitations": limitations,
            "abstention_reason": abstention_reason,
        }
    return None


def run_subjective_question(
    question: str,
    *,
    method: str,
    retrieved: list[RetrievedChunk],
    client: SubjectiveLLMClient | None,
    model: str,
    dense_threshold: float = 0.25,
    hybrid_threshold: float = 0.015,
) -> SubjectiveResponse:
    """Run one subjective question with explicit insufficient-evidence behavior."""
    started = time.perf_counter()
    should_abstain, reason = evaluate_abstention(
        retrieved,
        method=method,
        dense_threshold=dense_threshold,
        hybrid_threshold=hybrid_threshold,
    )
    raw_response: str | None = None
    error: str | None = None
    if should_abstain:
        answer = "I cannot provide a reliable answer from the retrieved evidence."
        status = "abstained"
        citations: list[str] = []
        limitations = (
            "The configured retrieval-evidence rule did not find adequate "
            "textbook support."
        )
        abstention_reason = reason
    else:
        if client is None:
            raise ValueError("A generation client is required for adequate evidence")
        prompt = build_subjective_prompt(question, retrieved)
        try:
            raw_response = client.generate(
                prompt,
                subjective_response_schema(retrieved),
            )
            parsed = parse_subjective_response(
                raw_response,
                {chunk.chunk_id for chunk in retrieved},
            )
            if parsed is None:
                error = "model response did not satisfy the subjective response schema"
                answer = "I cannot provide a reliable answer from the retrieved evidence."
                status = "abstained"
                citations = []
                limitations = "The model response could not be structurally validated."
                abstention_reason = error
            else:
                answer = parsed["answer"]
                status = parsed["evidence_status"]
                citations = parsed["citations"]
                limitations = parsed["limitations"]
                abstention_reason = parsed["abstention_reason"]
        except (OSError, RuntimeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            answer = "I cannot provide a reliable answer from the retrieved evidence."
            status = "abstained"
            citations = []
            limitations = "The answer-generation process failed."
            abstention_reason = error

    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    return SubjectiveResponse(
        question=question,
        method=method,
        answer=answer,
        evidence_status=status,
        citations=citations,
        limitations=limitations,
        abstention_reason=abstention_reason,
        retrieved_chunk_ids=[chunk.chunk_id for chunk in retrieved],
        retrieval_results=[chunk.as_dict() for chunk in retrieved],
        latency_ms=latency_ms,
        model=model,
        raw_response=raw_response,
        error=error,
    )
