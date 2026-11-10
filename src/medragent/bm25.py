"""BM25 retrieval and vanilla RAG generation for the fixed Psychiatry corpus."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from .data import MedQARecord, TextChunk
from .llm_only import LLMClient, OllamaClient, parse_answer
from .scope import ScopeAnnotation


def tokenize(text: str) -> list[str]:
    """Use a deterministic, lowercase alphanumeric tokenizer for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass(frozen=True)
class RetrievedChunk:
    """A ranked BM25 result with its stable corpus position and score."""

    chunk_id: str
    title: str
    text: str
    rank: int
    score: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class BM25Retriever:
    """In-memory BM25 index over a fixed, ordered list of text chunks."""

    def __init__(self, chunks: list[TextChunk]) -> None:
        if not chunks:
            raise ValueError("BM25 corpus must contain at least one chunk")
        self.chunks = list(chunks)
        self.index = BM25Okapi([tokenize(chunk.text) for chunk in self.chunks])

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("query must be non-empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scores = self.index.get_scores(tokenize(query))
        ranked_indices = sorted(
            range(len(self.chunks)),
            key=lambda index: (-float(scores[index]), index),
        )[:top_k]
        return [
            RetrievedChunk(
                chunk_id=self.chunks[index].chunk_id,
                title=self.chunks[index].title,
                text=self.chunks[index].text,
                rank=rank,
                score=float(scores[index]),
            )
            for rank, index in enumerate(ranked_indices, start=1)
        ]


def build_rag_prompt(record: MedQARecord, retrieved: list[RetrievedChunk]) -> str:
    """Build the fixed BM25 grounded prompt with explicit citation requirements."""
    options = "\n".join(f"{label}. {text}" for label, text in record.options.items())
    contexts = "\n\n".join(
        f"[{chunk.chunk_id}] {chunk.title}\n{chunk.text}" for chunk in retrieved
    )
    labels = ", ".join(record.options)
    chunk_ids = ", ".join(chunk.chunk_id for chunk in retrieved)
    return (
        "You are answering a medical multiple-choice benchmark question. "
        "Use the retrieved evidence below when it is relevant. Do not invent citations. "
        "Choose the single best option.\n\n"
        f"Question:\n{record.question}\n\n"
        f"Options:\n{options}\n\n"
        f"Retrieved evidence:\n{contexts}\n\n"
        f"Allowed citation chunk IDs: {chunk_ids}\n"
        "Return only valid JSON in this exact shape: "
        '{"answer": "A", "citations": ["chunk_id"]}. '
        f"The answer must be exactly one of: {labels}. "
        "The citations array must contain only IDs from the allowed citation chunk IDs."
    )


def response_schema(record: MedQARecord, retrieved: list[RetrievedChunk]) -> dict[str, Any]:
    """Return the Ollama JSON schema for one grounded response."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "answer": {"type": "string", "enum": list(record.options)},
            "citations": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [chunk.chunk_id for chunk in retrieved],
                },
                "minItems": 1,
            },
        },
        "required": ["answer", "citations"],
    }
def parse_citations(response: str, allowed_ids: set[str]) -> list[str] | None:
    """Parse and validate citations from the model's JSON response."""
    candidates: list[dict[str, Any]] = []
    try:
        decoded = json.loads(response.strip().strip("`"))
        if isinstance(decoded, dict):
            candidates.append(decoded)
    except json.JSONDecodeError:
        pass
    for match in re.finditer(r"\{.*?\}", response, flags=re.DOTALL):
        try:
            decoded = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            candidates.append(decoded)
    for decoded in reversed(candidates):
        citations = decoded.get("citations")
        if not isinstance(citations, list):
            continue
        if all(isinstance(value, str) and value in allowed_ids for value in citations):
            return list(dict.fromkeys(citations))
    return None


@dataclass(frozen=True)
class BM25RAGResult:
    """One BM25 vanilla-RAG result row."""

    question_id: str
    system_name: str
    question: str
    predicted_answer: str | None
    expected_answer: str | None
    is_correct: bool | None
    retrieved_chunk_ids: list[str]
    retrieval_results: list[dict[str, Any]]
    citations: list[str]
    latency_ms: float
    model: str
    temperature: float
    top_k: int
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
    retriever: BM25Retriever,
    client: LLMClient,
    model: str,
    temperature: float,
    top_k: int,
    annotation: ScopeAnnotation | None = None,
) -> BM25RAGResult:
    """Retrieve, generate, validate answer/citations, and preserve failures."""
    retrieved = retriever.retrieve(record.question, top_k=top_k)
    prompt = build_rag_prompt(record, retrieved)
    started = time.perf_counter()
    raw_response: str | None = None
    predicted: str | None = None
    citations: list[str] = []
    error: str | None = None
    try:
        if isinstance(client, OllamaClient):
            raw_response = client.generate(
                prompt, response_format=response_schema(record, retrieved)
            )
        else:
            raw_response = client.generate(prompt)
        predicted = parse_answer(raw_response, tuple(record.options))
        citations_value = parse_citations(
            raw_response, {chunk.chunk_id for chunk in retrieved}
        )
        if predicted is None:
            error = "model response did not contain one valid option label"
        elif citations_value is None:
            error = "model response did not contain valid citations"
        else:
            citations = citations_value
    except (OSError, RuntimeError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    expected = record.answer_letter
    return BM25RAGResult(
        question_id=record.id,
        system_name="bm25_rag",
        question=record.question,
        predicted_answer=predicted,
        expected_answer=expected,
        is_correct=(
            predicted == expected
            if predicted is not None and expected is not None and error is None
            else None
        ),
        retrieved_chunk_ids=[chunk.chunk_id for chunk in retrieved],
        retrieval_results=[chunk.as_dict() for chunk in retrieved],
        citations=citations,
        latency_ms=latency_ms,
        model=model,
        temperature=temperature,
        top_k=top_k,
        prompt=prompt,
        raw_response=raw_response,
        error=error,
        abstention_reason=error if error else None,
        scope=annotation.scope if annotation else None,
        evidence_expected=annotation.evidence_expected if annotation else None,
        annotation_version=annotation.annotation_version if annotation else None,
    )


def write_results(path: str | Path, results: list[BM25RAGResult]) -> None:
    """Write BM25 result rows as UTF-8 JSONL."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(json.dumps(result.as_dict(), ensure_ascii=False) + "\n")
