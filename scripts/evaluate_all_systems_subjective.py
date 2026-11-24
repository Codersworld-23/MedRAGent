"""
Unified Benchmark Evaluation Runner for MedRAGent Subjective Question Answering.

Evaluates 4 systems across the official 78-question subjective medical benchmark:
(C:\\Users\\kiran\\OneDrive\\Desktop\\sem7\\major\\test\\eval_questions_shared.jsonl)

1. System 1: Baseline (Flat 1-Round Hybrid)
2. System 2: Hierarchical RAG (Track A: Collapsed Tree)
3. System 3: Iterative RAG (Track B: 3-Round Planner)
4. System 4: Corrective RAG (Step D.4: Quality Evaluator + Bounded Reformulation + Abstention Guard)

Metrics:
- Hit Rate (fraction of questions where at least 1 gold source is retrieved)
- Recall@5 (fraction of all gold source passages retrieved)
- MRR (Mean Reciprocal Rank of first relevant source chunk)
- Multi-Passage Recall (performance on questions requiring 2+ source passages)
- Mean Latency (ms per question)
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import chromadb
from chromadb.utils import embedding_functions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.corrective import CorrectiveRAGPipeline
from src.iterative import IterativeRetriever


def normalize_id(cid: str) -> str:
    """Normalizes chunk IDs for consistent matching across indexing variants."""
    return re.sub(r"[_\-\s]+", "_", cid.strip()).lower()


def compute_retrieval_metrics(
    retrieved_ids: list[str],
    gold_ids: list[str],
) -> dict[str, float]:
    """Computes Hit, Recall, and MRR for a single question."""
    norm_gold = {normalize_id(g) for g in gold_ids}
    norm_retrieved = [normalize_id(r) for r in retrieved_ids]

    # Hit Rate: at least 1 gold passage in top-k
    hit = 1.0 if any(r in norm_gold for r in norm_retrieved) else 0.0

    # Recall: fraction of all gold passages retrieved
    matched_count = sum(1 for r in norm_retrieved if r in norm_gold)
    recall = matched_count / max(1, len(norm_gold))

    # MRR: reciprocal rank of first relevant chunk
    mrr = 0.0
    for rank, r in enumerate(norm_retrieved, start=1):
        if r in norm_gold:
            mrr = 1.0 / rank
            break

    return {
        "hit": hit,
        "recall": min(1.0, recall),
        "mrr": mrr,
        "is_multi_passage": len(norm_gold) > 1,
        "all_gold_found": 1.0 if norm_gold.issubset(set(norm_retrieved)) else 0.0,
    }


def main():
    test_jsonl_path = Path(r"C:\Users\kiran\OneDrive\Desktop\sem7\major\test\eval_questions_shared.jsonl")
    chroma_dir = Path(r"C:\Users\kiran\OneDrive\Desktop\sem7\major\chroma_db")
    bm25_path = PROJECT_ROOT / "indexes" / "full_textbooks" / "bm25_full_textbooks.pkl"
    faiss_path = PROJECT_ROOT / "indexes" / "full_textbooks_dense" / "dense_full_textbooks.faiss"
    metadata_path = PROJECT_ROOT / "indexes" / "full_textbooks_dense" / "dense_index_metadata.json"
    output_dir = PROJECT_ROOT / "experiments" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MEDRAGENT: 4-SYSTEM SUBJECTIVE MEDICAL BENCHMARK EVALUATION")
    print(f"Benchmark Dataset: {test_jsonl_path}")
    print("Systems: Baseline | Hierarchical | Iterative | Corrective")
    print("=" * 80)

    # 1. Load Questions
    questions: list[dict[str, Any]] = []
    with test_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    print(f"Loaded {len(questions)} subjective questions.")

    # 2. Initialize Pipelines
    print("Initializing Core RAG Pipelines...")
    corrective_pipeline = CorrectiveRAGPipeline(
        bm25_path=bm25_path,
        faiss_path=faiss_path,
        metadata_path=metadata_path,
        embedding_model_name="all-MiniLM-L6-v2",
    )
    iterative_pipeline = IterativeRetriever(
        hybrid_search_fn=corrective_pipeline.hybrid_search,
    )

    # 3. Initialize ChromaDB for Hierarchical Tree (Track A)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    tree_col = client.get_collection("psichiatry_dsm5_hierarchical", embedding_function=embed_fn)
    print("ChromaDB Hierarchical Collection connected successfully.")

    # Metrics Accumulators
    systems = ["Baseline (1-Round)", "Hierarchical (Track A)", "Iterative (Track B 3-Round)", "Corrective RAG (CRAG)"]
    system_metrics: dict[str, list[dict[str, float]]] = {sys_name: [] for sys_name in systems}
    system_latencies: dict[str, list[float]] = {sys_name: [] for sys_name in systems}

    # Evaluate Each Question Across All 4 Systems
    print(f"\nEvaluating all {len(questions)} questions across 4 systems...")
    for idx, q_data in enumerate(questions, start=1):
        q_text = q_data["question"]
        gold_ids = q_data.get("source_chunk_ids", [])
        if not gold_ids:
            continue

        # --- System 1: Baseline (Flat 1-Round Hybrid) ---
        t0 = time.perf_counter()
        base_chunks = corrective_pipeline.hybrid_search(q_text, top_k=5)
        lat_base = (time.perf_counter() - t0) * 1000
        base_ids = [c.chunk_id for c in base_chunks]
        system_metrics["Baseline (1-Round)"].append(compute_retrieval_metrics(base_ids, gold_ids))
        system_latencies["Baseline (1-Round)"].append(lat_base)

        # --- System 2: Hierarchical RAG (Track A: Collapsed Tree) ---
        t0 = time.perf_counter()
        hier_res = tree_col.query(query_texts=[q_text], n_results=5)
        lat_hier = (time.perf_counter() - t0) * 1000
        hier_ids = hier_res["ids"][0] if hier_res and hier_res.get("ids") else []
        system_metrics["Hierarchical (Track A)"].append(compute_retrieval_metrics(hier_ids, gold_ids))
        system_latencies["Hierarchical (Track A)"].append(lat_hier)

        # --- System 3: Iterative RAG (Track B: 3-Round Planner) ---
        t0 = time.perf_counter()
        iter_res = iterative_pipeline.retrieve(q_text, rounds=3, top_k=5)
        lat_iter = (time.perf_counter() - t0) * 1000
        iter_ids = [c.chunk_id for c in iter_res.final_chunks]
        system_metrics["Iterative (Track B 3-Round)"].append(compute_retrieval_metrics(iter_ids, gold_ids))
        system_latencies["Iterative (Track B 3-Round)"].append(lat_iter)

        # --- System 4: Corrective RAG (CRAG) ---
        t0 = time.perf_counter()
        crag_resp, crag_audit = corrective_pipeline.run(q_text, top_k=5)
        lat_crag = (time.perf_counter() - t0) * 1000
        crag_ids = crag_audit.final_retrieved_ids
        system_metrics["Corrective RAG (CRAG)"].append(compute_retrieval_metrics(crag_ids, gold_ids))
        system_latencies["Corrective RAG (CRAG)"].append(lat_crag)

        if idx % 15 == 0 or idx == len(questions):
            print(f"Processed {idx}/{len(questions)} questions...")

    # Aggregate Final Metrics Table
    results_summary = {}
    print("\n" + "=" * 95)
    print(f"{'System':<28} | {'Hit Rate':<10} | {'Recall@5':<10} | {'MRR':<10} | {'Multi-Passage':<14} | {'Latency':<10}")
    print("-" * 95)

    for sys_name in systems:
        metrics_list = system_metrics[sys_name]
        lat_list = system_latencies[sys_name]

        hit_rate = sum(m["hit"] for m in metrics_list) / len(metrics_list)
        recall = sum(m["recall"] for m in metrics_list) / len(metrics_list)
        mrr = sum(m["mrr"] for m in metrics_list) / len(metrics_list)

        multi_list = [m for m in metrics_list if m["is_multi_passage"]]
        multi_recall = (sum(m["all_gold_found"] for m in multi_list) / len(multi_list)) if multi_list else 0.0
        mean_lat = sum(lat_list) / len(lat_list)

        results_summary[sys_name] = {
            "Hit_Rate": round(hit_rate, 4),
            "Recall@5": round(recall, 4),
            "MRR": round(mrr, 4),
            "Multi_Passage_All_Found": round(multi_recall, 4),
            "Mean_Latency_ms": round(mean_lat, 2),
            "Total_Questions": len(metrics_list),
        }

        print(f"{sys_name:<28} | {hit_rate:<10.3f} | {recall:<10.3f} | {mrr:<10.3f} | {multi_recall:<14.3f} | {mean_lat:<8.1f} ms")
    print("=" * 95)

    # Save to JSON
    report_file = output_dir / "subjective_4_systems_comparison.json"
    with report_file.open("w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved consolidated benchmark results to: {report_file}")


if __name__ == "__main__":
    main()
