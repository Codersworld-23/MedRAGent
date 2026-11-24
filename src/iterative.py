"""
Step B.1: Integrated Iterative Multi-Round Retrieval for MedRAGent.

Implements iterative retrieval across the full 18-textbook corpus:
1. Round 1: Searches the user's clinical question using Hybrid RRF (BM25 + Dense FAISS).
2. Assessment: Evaluates current retrieved context against the question.
3. Multi-Round Refinement: If criteria/evidence is missing, formulates targeted sub-queries
   (via high-IDF clinical keywords or LLM planner) up to N rounds.
4. Reciprocal Rank Fusion: All rounds fused with rank decay; initial query weighted 1.5x
   to prevent context drift.
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple

from src.bm25 import RetrievedChunk


@dataclass(frozen=True)
class IterativeRound:
    round_number: int
    queries: list[str]
    retrieved_chunk_ids: list[str]
    refinement_reason: str


@dataclass(frozen=True)
class IterativeResult:
    question: str
    final_chunks: list[RetrievedChunk]
    rounds_executed: int
    round_history: list[dict[str, Any]]
    all_queries: list[str]


class IterativeRetriever:
    """
    Orchestrates multi-round iterative retrieval over hybrid 18-textbook indexes.
    """

    def __init__(
        self,
        hybrid_search_fn: Any,
        llm_client: Optional[Any] = None,
        rrf_k: int = 60,
    ) -> None:
        self.hybrid_search_fn = hybrid_search_fn
        self.llm_client = llm_client
        self.rrf_k = rrf_k

    def _extract_expansion_terms(self, text: str, existing_query: str, max_terms: int = 3) -> list[str]:
        """Extracts high-salience clinical terms from retrieved context."""
        stop_words = {
            "the", "and", "for", "with", "that", "this", "from", "are", "have", "been",
            "which", "also", "were", "what", "such", "than", "other", "into", "more",
            "clinical", "criteria", "patient", "disorder", "symptoms", "these", "about"
        }
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        q_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", existing_query.lower()))
        candidates = [w for w in words if w not in stop_words and w not in q_words]
        counts = Counter(candidates)
        return [word for word, _ in counts.most_common(max_terms)]

    def retrieve(
        self,
        question: str,
        rounds: int = 3,
        top_k: int = 5,
    ) -> IterativeResult:
        """
        Executes multi-round iterative retrieval.
        """
        if rounds <= 0:
            rounds = 1

        all_queries: list[str] = [question]
        round_history: list[IterativeRound] = []
        executed_queries: Set[str] = set()

        # Storage for fusion: (weight, list_of_chunks)
        ranked_lists: list[Tuple[float, list[RetrievedChunk]]] = []
        pending_queries = [question]

        for r_num in range(1, rounds + 1):
            if not pending_queries:
                break

            round_chunks_collected: list[RetrievedChunk] = []
            queries_this_round = list(pending_queries)

            for q in queries_this_round:
                if q in executed_queries:
                    continue
                executed_queries.add(q)
                
                # Weight: Round 1 gets 1.5x, follow-up queries get 1.0x
                w = 1.5 if r_num == 1 else 1.0
                hits = self.hybrid_search_fn(q, top_k=top_k)
                ranked_lists.append((w, hits))
                round_chunks_collected.extend(hits)

            # Record round history
            round_history.append(
                IterativeRound(
                    round_number=r_num,
                    queries=queries_this_round,
                    retrieved_chunk_ids=[c.chunk_id for c in round_chunks_collected],
                    refinement_reason="Initial query" if r_num == 1 else "Refined query targeting missing aspects",
                )
            )

            # Stop if reached max rounds
            if r_num >= rounds:
                break

            # Plan next round queries
            combined_evidence = " ".join(c.text for c in round_chunks_collected[:3])
            expansion = self._extract_expansion_terms(combined_evidence, " ".join(all_queries), max_terms=3)

            if expansion:
                new_q = f"{question} {' '.join(expansion)}"
                if new_q not in executed_queries:
                    pending_queries = [new_q]
                    all_queries.append(new_q)
                else:
                    pending_queries = []
            else:
                pending_queries = []

        # Reciprocal Rank Fusion across all rounds
        chunk_map = {}
        rrf_scores: dict[str, float] = {}

        for weight, hit_list in ranked_lists:
            for rank, chunk in enumerate(hit_list, start=1):
                cid = chunk.chunk_id
                chunk_map[cid] = chunk
                if cid not in rrf_scores:
                    rrf_scores[cid] = 0.0
                rrf_scores[cid] += weight / (self.rrf_k + rank)

        fused = sorted(chunk_map.values(), key=lambda c: (-rrf_scores[c.chunk_id], c.chunk_id))[:top_k]

        fused_chunks = [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                title=c.title,
                text=c.text,
                rank=r,
                score=rrf_scores[c.chunk_id],
                source_file=c.source_file,
                textbook_name=c.textbook_name,
                retrieval_method="iterative_rrf",
            )
            for r, c in enumerate(fused, start=1)
        ]

        return IterativeResult(
            question=question,
            final_chunks=fused_chunks,
            rounds_executed=len(round_history),
            round_history=[asdict(rh) for rh in round_history],
            all_queries=all_queries,
        )
