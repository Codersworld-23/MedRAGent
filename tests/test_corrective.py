"""
Tests for Step D.4: Corrective RAG Pipeline (CRAG) over the 18-textbook corpus.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.corrective import CorrectiveRAGPipeline


def test_corrective_rag_pipeline():
    bm25_path = PROJECT_ROOT / "indexes" / "full_textbooks" / "bm25_full_textbooks.pkl"
    faiss_path = PROJECT_ROOT / "indexes" / "full_textbooks_dense" / "dense_full_textbooks.faiss"
    metadata_path = PROJECT_ROOT / "indexes" / "full_textbooks_dense" / "dense_index_metadata.json"

    print("\n--- 1. Initializing Corrective RAG Pipeline ---")
    pipeline = CorrectiveRAGPipeline(
        bm25_path=bm25_path,
        faiss_path=faiss_path,
        metadata_path=metadata_path,
        embedding_model_name="all-MiniLM-L6-v2",
    )
    print("Pipeline successfully initialized.")

    # Test Case 1: In-Scope Medical Question (Should be CORRECT or AMBIGUOUS with citations)
    print("\n--- 2. Testing In-Scope Medical Query ---")
    in_scope_query = "What are the clinical diagnostic criteria for major depressive disorder in DSM-5?"
    resp_med, audit_med = pipeline.run(in_scope_query, top_k=3)

    print(f"Query: {in_scope_query}")
    print(f"Evidence Status: {resp_med.evidence_status}")
    print(f"Citations: {resp_med.citations}")
    print(f"Correction Triggered: {audit_med.correction_triggered}")
    print(f"Latency: {resp_med.latency_ms:.2f} ms")
    print(f"Answer snippet: {resp_med.answer[:150]}...")

    assert resp_med.evidence_status in {"supported", "partial"}, "Medical query should have supported or partial evidence"
    assert len(resp_med.citations) > 0, "Medical query must have supporting citations"
    assert audit_med.final_evidence_status in {"supported", "partial"}

    # Test Case 2: Out-of-Corpus / Non-Medical Question (Should trigger SAFE ABSTENTION)
    print("\n--- 3. Testing Out-of-Scope Query (Safe Abstention Verification) ---")
    out_of_scope_query = "How to configure a Kubernetes cluster with Docker on Ubuntu?"
    resp_ood, audit_ood = pipeline.run(out_of_scope_query, top_k=3)

    print(f"Query: {out_of_scope_query}")
    print(f"Evidence Status: {resp_ood.evidence_status}")
    print(f"Abstention Reason: {resp_ood.abstention_reason}")
    print(f"Correction Triggered: {audit_ood.correction_triggered}")
    print(f"Latency: {resp_ood.latency_ms:.2f} ms")

    assert resp_ood.evidence_status == "abstained", "Out-of-scope query must trigger safe abstention"
    assert resp_ood.abstention_reason is not None, "Abstention reason must be provided"
    assert len(resp_ood.citations) == 0, "Abstained response must not have false citations"

    # Test Case 3: Ambiguous / Partial Query (Should trigger Corrective Re-retrieval)
    print("\n--- 4. Testing Ambiguous Query (Correction Trigger Verification) ---")
    ambiguous_query = "What are the ocular manifestations and optic nerve involvement in persistent depressive disorder?"
    resp_amb, audit_amb = pipeline.run(ambiguous_query, top_k=3)

    print(f"Query: {ambiguous_query}")
    print(f"Evidence Status: {resp_amb.evidence_status}")
    print(f"Correction Triggered: {audit_amb.correction_triggered}")
    print(f"Correction Reason: {audit_amb.correction_reason}")
    print(f"Corrected Query: {audit_amb.corrected_query}")
    print(f"Final Retrieved Chunks: {audit_amb.final_retrieved_ids}")
    print(f"Latency: {resp_amb.latency_ms:.2f} ms")

    assert audit_amb.correction_triggered is True, "Ambiguous query should trigger corrective reformulation"
    assert audit_amb.corrected_query is not None, "Corrected query must be generated"
    assert len(resp_amb.citations) > 0, "Must contain supporting citations after re-retrieval"

    print("\nALL CORRECTIVE RAG TESTS PASSED SUCCESSFULLY! ✅")


if __name__ == "__main__":
    test_corrective_rag_pipeline()
