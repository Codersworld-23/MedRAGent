"""Hybrid vanilla RAG using BM25 and dense Reciprocal Rank Fusion."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .bm25 import (
    BM25Retriever,
    RetrievedChunk,
    build_rag_prompt,
    parse_citations,
    response_schema,
)
from .data import MedQARecord
from .dense import DenseRetriever, EmbeddingClient
from .llm_only import LLMClient, OllamaClient, parse_answer
from .scope import ScopeAnnotation


class HybridRetriever:
    """Fuse BM25 and dense rankings with deterministic Reciprocal Rank Fusion."""

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        embedding_client: EmbeddingClient,
        rrf_constant: int = 60,
    ) -> None:
        if rrf_constant <= 0:
            raise ValueError("rrf_constant must be positive")
        self.bm25 = bm25
        self.dense = dense
        self.embedding_client = embedding_client
        self.rrf_constant = rrf_constant

    def retrieve(
        self, query: str, top_k: int = 3
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk], list[RetrievedChunk]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        bm25_results = self.bm25.retrieve(query, top_k=top_k)
        dense_results = self.dense.retrieve(query, self.embedding_client, top_k=top_k)
        by_id = {chunk.chunk_id: chunk for chunk in bm25_results + dense_results}
        bm25_ranks = {chunk.chunk_id: chunk.rank for chunk in bm25_results}
        dense_ranks = {chunk.chunk_id: chunk.rank for chunk in dense_results}

        def rrf_score(chunk_id: str) -> float:
            score = 0.0
            if chunk_id in bm25_ranks:
                score += 1 / (self.rrf_constant + bm25_ranks[chunk_id])
            if chunk_id in dense_ranks:
                score += 1 / (self.rrf_constant + dense_ranks[chunk_id])
            return score

        fused = sorted(
            by_id.values(),
            key=lambda chunk: (-rrf_score(chunk.chunk_id), chunk.chunk_id),
        )[:top_k]
        fused = [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                text=chunk.text,
                rank=rank,
                score=rrf_score(chunk.chunk_id),
            )
            for rank, chunk in enumerate(fused, start=1)
        ]
        return bm25_results, dense_results, fused


@dataclass(frozen=True)
class HybridRAGResult:
    question_id: str
    system_name: str
    question: str
    predicted_answer: str | None
    expected_answer: str | None
    is_correct: bool | None
    retrieved_chunk_ids: list[str]
    retrieval_results: list[dict[str, Any]]
    bm25_retrieval_results: list[dict[str, Any]]
    dense_retrieval_results: list[dict[str, Any]]
    citations: list[str]
    latency_ms: float
    model: str
    embedding_model: str
    temperature: float
    top_k: int
    rrf_constant: int
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
    retriever: HybridRetriever,
    generation_client: LLMClient,
    model: str,
    embedding_model: str,
    temperature: float,
    top_k: int,
    annotation: ScopeAnnotation | None = None,
) -> HybridRAGResult:
    bm25_results, dense_results, retrieved = retriever.retrieve(record.question, top_k)
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
    return HybridRAGResult(
        question_id=record.id,
        system_name="hybrid_rag",
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
        bm25_retrieval_results=[chunk.as_dict() for chunk in bm25_results],
        dense_retrieval_results=[chunk.as_dict() for chunk in dense_results],
        citations=citations,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        model=model,
        embedding_model=embedding_model,
        temperature=temperature,
        top_k=top_k,
        rrf_constant=retriever.rrf_constant,
        prompt=prompt,
        raw_response=raw_response,
        error=error,
        abstention_reason=error if error else None,
        scope=annotation.scope if annotation else None,
        evidence_expected=annotation.evidence_expected if annotation else None,
        annotation_version=annotation.annotation_version if annotation else None,
    )


def write_results(path: str | Path, results: list[HybridRAGResult]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(json.dumps(result.as_dict(), ensure_ascii=False) + "\n")
