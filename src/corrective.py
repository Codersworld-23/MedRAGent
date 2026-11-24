"""
Step D.4: Corrective Retrieval-Augmented Generation (CRAG) for MedRAGent.

Implements bounded self-correction over the full 18-textbook corpus:
1. Initial Retrieval: Hybrid RRF (BM25 + Dense FAISS).
2. Evidence Quality Evaluator: Grades retrieved chunks as CORRECT, AMBIGUOUS, or INCORRECT.
3. Corrective Action:
   - CORRECT (> 0.70 confidence): Proceed to direct grounded generation.
   - AMBIGUOUS (0.35 - 0.70 confidence): Extract missing clinical entities, reformulate query,
     and execute max 1 bounded re-retrieval round. Merge and deduplicate evidence.
   - INCORRECT (< 0.35 confidence): Safe clinical abstention to prevent hallucination.
4. Final Grounded Generation: Produces SubjectiveResponse according to the project schema.
5. Complete Audit Trail: Records trigger reasons, query evolution, and quality shifts.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.bm25 import RetrievedChunk
from src.subjective import SubjectiveResponse, build_subjective_prompt


@dataclass(frozen=True)
class EvidenceGrade:
    """Assessment of evidence adequacy for a clinical question."""
    status: str  # "CORRECT", "AMBIGUOUS", or "INCORRECT"
    confidence: float  # 0.0 to 1.0
    reason: str
    missing_entities: list[str]


@dataclass(frozen=True)
class CorrectiveAudit:
    """Comprehensive audit trail of the Corrective RAG pipeline."""
    original_query: str
    original_retrieved_ids: list[str]
    original_grade: dict[str, Any]
    correction_triggered: bool
    correction_reason: Optional[str]
    corrected_query: Optional[str]
    corrected_retrieved_ids: Optional[list[str]]
    final_retrieved_ids: list[str]
    final_evidence_status: str
    latency_ms: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class CorrectiveRAGPipeline:
    """
    End-to-end Corrective RAG pipeline over the full 18-textbook corpus.
    """

    def __init__(
        self,
        bm25_path: str | Path,
        faiss_path: str | Path,
        metadata_path: str | Path,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        llm_client: Optional[Any] = None,
        rrf_k: int = 60,
    ) -> None:
        self.bm25_path = Path(bm25_path)
        self.faiss_path = Path(faiss_path)
        self.metadata_path = Path(metadata_path)
        self.rrf_k = rrf_k
        self.llm_client = llm_client

        # 1. Load BM25 Index
        if not self.bm25_path.exists():
            raise FileNotFoundError(f"Missing BM25 index: {self.bm25_path}")
        with self.bm25_path.open("rb") as f:
            self.bm25 = pickle.load(f)

        # 2. Load FAISS Dense Index
        if not self.faiss_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {self.faiss_path}")
        self.faiss_index = faiss.read_index(str(self.faiss_path))

        # 3. Load Metadata & Chunk Mapping
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Missing dense metadata: {self.metadata_path}")
        with self.metadata_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
            self.chunk_manifest = meta["chunks"]

        # 4. Load Embedding Model
        self.embedding_model = SentenceTransformer(embedding_model_name, device="cpu")

    def hybrid_search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """
        Executes Reciprocal Rank Fusion (RRF) across BM25 and Dense FAISS.
        """
        # A. BM25 Search
        bm25_hits = self.bm25.retrieve(query, top_k=top_k * 2)

        # B. Dense FAISS Search
        q_vec = self.embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=False)
        q_vec = np.asarray(q_vec, dtype=np.float32)
        faiss.normalize_L2(q_vec)
        scores, indices = self.faiss_index.search(q_vec, top_k * 2)

        dense_hits: list[RetrievedChunk] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0 or idx >= len(self.chunk_manifest):
                continue
            item = self.chunk_manifest[idx]
            # retrieve original text from bm25 corpus chunks
            orig_chunk = self.bm25.chunks[idx]
            dense_hits.append(
                RetrievedChunk(
                    chunk_id=item["chunk_id"],
                    title=item["title"],
                    text=orig_chunk.text,
                    rank=rank,
                    score=float(score),
                    source_file=item["source_file"],
                    textbook_name=item["textbook_name"],
                    retrieval_method="dense",
                )
            )

        # C. Reciprocal Rank Fusion (RRF)
        bm25_ranks = {hit.chunk_id: hit.rank for hit in bm25_hits}
        dense_ranks = {hit.chunk_id: hit.rank for hit in dense_hits}
        all_hits = {hit.chunk_id: hit for hit in bm25_hits + dense_hits}

        def rrf_score(cid: str) -> float:
            score = 0.0
            if cid in bm25_ranks:
                score += 1.0 / (self.rrf_k + bm25_ranks[cid])
            if cid in dense_ranks:
                score += 1.0 / (self.rrf_k + dense_ranks[cid])
            return score

        fused = sorted(all_hits.values(), key=lambda c: (-rrf_score(c.chunk_id), c.chunk_id))[:top_k]

        return [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                title=c.title,
                text=c.text,
                rank=r,
                score=rrf_score(c.chunk_id),
                source_file=c.source_file,
                textbook_name=c.textbook_name,
                retrieval_method="hybrid_rrf",
            )
            for r, c in enumerate(fused, start=1)
        ]

    def grade_evidence(self, query: str, chunks: list[RetrievedChunk]) -> EvidenceGrade:
        """
        Retrieval Evaluator: Assesses whether the retrieved passages adequately
        cover the clinical entities and criteria required to answer the query.
        """
        if not chunks:
            return EvidenceGrade(
                status="INCORRECT",
                confidence=0.0,
                reason="No passages retrieved.",
                missing_entities=["all"],
            )

        # If an LLM client is available, use clinical evaluator prompt
        if self.llm_client:
            evidence_sample = "\n\n".join(f"[{c.chunk_id}] {c.textbook_name}: {c.text[:250]}" for c in chunks[:3])
            eval_prompt = (
                "You are an objective clinical retrieval evaluator. Evaluate whether the provided textbook "
                "passages contain adequate clinical evidence to answer the user query.\n\n"
                f"Query: {query}\n\n"
                f"Retrieved Passages:\n{evidence_sample}\n\n"
                "Respond ONLY with valid JSON in this exact format:\n"
                "{\n"
                '  "status": "CORRECT|AMBIGUOUS|INCORRECT",\n'
                '  "confidence": 0.85,\n'
                '  "missing_entities": ["entity1", "entity2"],\n'
                '  "reason": "explanation of coverage"\n'
                "}"
            )
            try:
                raw = self.llm_client.generate(eval_prompt)
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    d = json.loads(m.group(0))
                    status = d.get("status", "AMBIGUOUS").upper()
                    if status not in {"CORRECT", "AMBIGUOUS", "INCORRECT"}:
                        status = "AMBIGUOUS"
                    return EvidenceGrade(
                        status=status,
                        confidence=float(d.get("confidence", 0.5)),
                        reason=str(d.get("reason", "LLM evaluation")),
                        missing_entities=list(d.get("missing_entities", [])),
                    )
            except Exception:
                pass  # Fallback to heuristic evaluation below

        # Robust Content-Entity Overlap Grader
        STOP_WORDS = {
            "how", "what", "why", "when", "where", "which", "who", "whom", "this", "that",
            "these", "those", "with", "from", "for", "and", "the", "are", "can", "could",
            "should", "would", "does", "did", "have", "has", "had", "about", "into", "some",
            "such", "than", "then", "very", "also", "most", "other", "their", "there", "they"
        }
        all_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", query.lower()))
        content_words = {w for w in all_words if w not in STOP_WORDS}
        if not content_words:
            content_words = all_words

        # Calculate per-chunk and collective coverage of clinical content words
        single_chunk_coverages = [
            len({w for w in content_words if w in c.text.lower()}) / max(1, len(content_words))
            for c in chunks
        ]
        max_single_coverage = max(single_chunk_coverages) if single_chunk_coverages else 0.0

        combined_text = " ".join(c.text.lower() for c in chunks)
        covered_words = {w for w in content_words if w in combined_text}
        collective_coverage = len(covered_words) / max(1, len(content_words))
        missing = list(content_words - covered_words)

        # 1. INCORRECT: Collective coverage is negligible (< 35%) -> Out of corpus
        if collective_coverage < 0.35:
            return EvidenceGrade(
                status="INCORRECT",
                confidence=round(collective_coverage, 3),
                reason=f"Negligible clinical concept coverage ({collective_coverage:.1%}). Question appears out-of-corpus.",
                missing_entities=missing,
            )

        # 2. CORRECT: At least one single chunk has comprehensive direct coverage (>= 60%)
        if max_single_coverage >= 0.60:
            return EvidenceGrade(
                status="CORRECT",
                confidence=round(max_single_coverage, 3),
                reason=f"Strong direct single-passage evidence coverage ({max_single_coverage:.1%}).",
                missing_entities=[],
            )

        # 3. AMBIGUOUS: Evidence is scattered across multiple chunks without a single comprehensive passage
        return EvidenceGrade(
            status="AMBIGUOUS",
            confidence=round(max_single_coverage, 3),
            reason=f"Scattered / partial evidence (max single passage coverage: {max_single_coverage:.1%}, collective: {collective_coverage:.1%}). Key criteria need targeted re-search.",
            missing_entities=missing[:3] if missing else [w for w in content_words if w not in chunks[0].text.lower()][:3],
        )

    def reformulate_query(self, query: str, missing_entities: list[str]) -> str:
        """
        Query Reformulator: Synthesizes a focused follow-up query incorporating missing entities.
        """
        if not missing_entities:
            return query

        if self.llm_client:
            prompt = (
                f"Original medical query: {query}\n"
                f"Missing diagnostic entities: {', '.join(missing_entities)}\n"
                "Rewrite the medical search query to specifically target these missing clinical details. "
                "Output ONLY the concise search query without preamble."
            )
            try:
                refined = self.llm_client.generate(prompt).strip()
                if refined and len(refined) < 200:
                    return refined
            except Exception:
                pass

        # Deterministic refinement fallback
        return f"{query} {' '.join(missing_entities)}"

    def run(
        self,
        query: str,
        top_k: int = 5,
        model_name: str = "medragent-crag-v1",
    ) -> tuple[SubjectiveResponse, CorrectiveAudit]:
        """
        Executes Corrective RAG:
        1. Hybrid retrieve
        2. Grade evidence
        3. If AMBIGUOUS: reformulate query and execute 1 retry
        4. If INCORRECT: abstain
        5. Generate subjective cited response
        """
        t0 = time.perf_counter()

        # Step 1: Initial Retrieval
        initial_chunks = self.hybrid_search(query, top_k=top_k)
        grade = self.grade_evidence(query, initial_chunks)

        correction_triggered = False
        correction_reason = None
        corrected_query = None
        corrected_chunks: Optional[list[RetrievedChunk]] = None
        final_chunks = initial_chunks

        # Step 2: Handle Triggers
        if grade.status == "AMBIGUOUS":
            correction_triggered = True
            correction_reason = f"Evidence grade was AMBIGUOUS: {grade.reason}"
            corrected_query = self.reformulate_query(query, grade.missing_entities)
            
            # 1-round bounded re-retrieval
            retry_chunks = self.hybrid_search(corrected_query, top_k=top_k)
            corrected_chunks = retry_chunks

            # Merge and deduplicate evidence (original + corrected)
            seen_ids = set()
            merged = []
            for c in initial_chunks + retry_chunks:
                if c.chunk_id not in seen_ids:
                    seen_ids.add(c.chunk_id)
                    merged.append(c)
            final_chunks = merged[:top_k]

        elif grade.status == "INCORRECT":
            # Clinical safety abstention policy
            latency = (time.perf_counter() - t0) * 1000
            audit = CorrectiveAudit(
                original_query=query,
                original_retrieved_ids=[c.chunk_id for c in initial_chunks],
                original_grade=asdict(grade),
                correction_triggered=False,
                correction_reason="Retrieval Evaluator graded evidence as INCORRECT. Abstaining.",
                corrected_query=None,
                corrected_retrieved_ids=None,
                final_retrieved_ids=[c.chunk_id for c in initial_chunks],
                final_evidence_status="abstained",
                latency_ms=round(latency, 2),
            )
            response = SubjectiveResponse(
                question=query,
                method="corrective_rag",
                answer="The 18 medical textbooks do not contain sufficient verified evidence to answer this clinical question safely.",
                evidence_status="abstained",
                citations=[],
                limitations="Insufficient evidence found in the corpus. Model strictly abstained to prevent clinical hallucination.",
                abstention_reason=grade.reason,
                retrieved_chunk_ids=[c.chunk_id for c in initial_chunks],
                retrieval_results=[c.as_dict() for c in initial_chunks],
                latency_ms=round(latency, 2),
                model=model_name,
                raw_response=None,
                error=None,
            )
            return response, audit

        # Step 3: Grounded Subjective Generation
        latency = (time.perf_counter() - t0) * 1000
        citations = [c.chunk_id for c in final_chunks[:3]]
        evidence_status = "supported" if grade.status == "CORRECT" else "partial"

        # Generate answer with LLM if client is provided
        answer_text = ""
        limitations_text = ""
        if self.llm_client:
            prompt = build_subjective_prompt(query, final_chunks)
            try:
                raw_resp = self.llm_client.generate(prompt)
                m = re.search(r"\{.*\}", raw_resp, re.DOTALL)
                if m:
                    parsed = json.loads(m.group(0))
                    answer_text = parsed.get("answer", "")
                    evidence_status = parsed.get("evidence_status", evidence_status)
                    citations = [cid for cid in parsed.get("citations", []) if cid in {c.chunk_id for c in final_chunks}]
                    limitations_text = parsed.get("limitations", "")
            except Exception:
                pass

        if not answer_text:
            # Concise synthesis of top evidence
            answer_text = f"Based on evidence from {final_chunks[0].textbook_name} ([{final_chunks[0].chunk_id}]): {final_chunks[0].text[:300]}..."
            limitations_text = "Automatic heuristic synthesis; LLM client omitted."

        response = SubjectiveResponse(
            question=query,
            method="corrective_rag",
            answer=answer_text,
            evidence_status=evidence_status,
            citations=citations,
            limitations=limitations_text,
            abstention_reason=None,
            retrieved_chunk_ids=[c.chunk_id for c in final_chunks],
            retrieval_results=[c.as_dict() for c in final_chunks],
            latency_ms=round(latency, 2),
            model=model_name,
            raw_response=None,
            error=None,
        )

        audit = CorrectiveAudit(
            original_query=query,
            original_retrieved_ids=[c.chunk_id for c in initial_chunks],
            original_grade=asdict(grade),
            correction_triggered=correction_triggered,
            correction_reason=correction_reason,
            corrected_query=corrected_query,
            corrected_retrieved_ids=[c.chunk_id for c in corrected_chunks] if corrected_chunks else None,
            final_retrieved_ids=[c.chunk_id for c in final_chunks],
            final_evidence_status=evidence_status,
            latency_ms=round(latency, 2),
        )

        return response, audit
