"""Dense vanilla RAG using Ollama embeddings and a FAISS inner-product index."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import faiss
import numpy as np

from .bm25 import RetrievedChunk, build_rag_prompt, parse_citations, response_schema
from .data import MedQARecord, TextChunk
from .llm_only import LLMClient, OllamaClient, parse_answer
from .scope import ScopeAnnotation


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""


class DenseRetriever:
    """Normalized dense index with deterministic corpus-order tie-breaking."""

    def __init__(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> None:
        if not chunks or len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be non-empty and have equal length")
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise ValueError("embeddings must be a two-dimensional non-empty matrix")
        self.chunks = list(chunks)
        self.dimension = int(matrix.shape[1])
        faiss.normalize_L2(matrix)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(matrix)

    @classmethod
    def from_client(
        cls,
        chunks: list[TextChunk],
        client: EmbeddingClient,
        batch_size: int = 32,
    ) -> "DenseRetriever":
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        embeddings: list[list[float]] = []
        for start in range(0, len(chunks), batch_size):
            embeddings.extend(client.embed([chunk.text for chunk in chunks[start : start + batch_size]]))
        return cls(chunks, embeddings)

    def retrieve(self, query: str, client: EmbeddingClient, top_k: int = 3) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("query must be non-empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        vector = np.asarray(client.embed([query]), dtype=np.float32)
        if vector.shape != (1, self.dimension):
            raise ValueError("query embedding dimension does not match the index")
        faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, min(top_k, len(self.chunks)))
        ranked = sorted(
            zip(indices[0], scores[0]),
            key=lambda pair: (-float(pair[1]), int(pair[0])),
        )
        return [
            RetrievedChunk(
                chunk_id=self.chunks[int(index)].chunk_id,
                title=self.chunks[int(index)].title,
                text=self.chunks[int(index)].text,
                rank=rank,
                score=float(score),
            )
            for rank, (index, score) in enumerate(ranked, start=1)
        ]


@dataclass(frozen=True)
class DenseRAGResult:
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
    embedding_model: str
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
    retriever: DenseRetriever,
    embedding_client: EmbeddingClient,
    generation_client: LLMClient,
    model: str,
    embedding_model: str,
    temperature: float,
    top_k: int,
    annotation: ScopeAnnotation | None = None,
) -> DenseRAGResult:
    retrieved = retriever.retrieve(record.question, embedding_client, top_k)
    prompt = build_rag_prompt(record, retrieved)
    started = time.perf_counter()
    raw_response: str | None = None
    predicted: str | None = None
    citations: list[str] = []
    error: str | None = None
    try:
        if isinstance(generation_client, OllamaClient):
            raw_response = generation_client.generate(
                prompt, response_format=response_schema(record, retrieved)
            )
        else:
            raw_response = generation_client.generate(prompt)
        predicted = parse_answer(raw_response, tuple(record.options))
        citations_value = parse_citations(raw_response, {chunk.chunk_id for chunk in retrieved})
        if predicted is None:
            error = "model response did not contain one valid option label"
        elif citations_value is None:
            error = "model response did not contain valid citations"
        else:
            citations = citations_value
    except (OSError, RuntimeError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return DenseRAGResult(
        question_id=record.id,
        system_name="dense_rag",
        question=record.question,
        predicted_answer=predicted,
        expected_answer=record.answer_letter,
        is_correct=(
            predicted == record.answer_letter
            if predicted is not None and record.answer_letter is not None and error is None
            else None
        ),
        retrieved_chunk_ids=[chunk.chunk_id for chunk in retrieved],
        retrieval_results=[chunk.as_dict() for chunk in retrieved],
        citations=citations,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        model=model,
        embedding_model=embedding_model,
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


def write_results(path: str | Path, results: list[DenseRAGResult]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            import json
            handle.write(json.dumps(result.as_dict(), ensure_ascii=False) + "\n")
