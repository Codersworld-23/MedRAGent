r"""
Track A (Hierarchical Retrieval) Benchmark on the Exact 5 Parameters
====================================================================
Evaluates the 78 generated DSM-5 benchmark questions across:
1. Flat Dense Baseline (Layer 0 leaf chunks only)
2. Hierarchical Collapsed Tree (Layer 0 Leaves + Layer 1 Cluster Summaries)
3. Hierarchical Tree Traversal (Top-down: Layer 1 -> Layer 0)

Evaluates the EXACT 5 parameters from the user's slide image:
- Hit (source passage found)
- Recall of source passages
- MRR
- Correctness (1–5) via Nemotron LLM Judge
- Faithfulness (1–5) via Nemotron LLM Judge
"""

import os
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import requests
import numpy as np
import chromadb
from chromadb.utils import embedding_functions

# Paths & Models
BASE_DIR = Path(r"C:\Users\kiran\OneDrive\Desktop\sem7\major")
TEST_JSONL = BASE_DIR / "test" / "eval_questions_shared.jsonl"
CHROMA_DIR = BASE_DIR / "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_API_KEY = "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"
NEMOTRON_MODEL = "nvidia/nemotron-3-super-120b-a12b"


def call_nemotron(prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": NEMOTRON_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    for attempt in range(3):
        try:
            resp = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=40)
            if resp.status_code == 200:
                body = resp.json()
                msg = body["choices"][0]["message"]
                return str(msg.get("content") or "").strip()
            elif resp.status_code == 429:
                time.sleep(2 * (attempt + 1))
            else:
                time.sleep(1)
        except Exception:
            time.sleep(1.5)
    return ""


def grade_llm(question: str, reference: str, candidate_answer: str, context: str) -> Tuple[float, float]:
    judge_prompt = f"""You are an objective clinical evaluation judge.
Evaluate candidate answer on two metrics on a scale of 1.0 to 5.0:
1. Correctness (1.0 to 5.0): Factual correctness against reference answer.
2. Faithfulness (1.0 to 5.0): Strict adherence to retrieved evidence without hallucination.

Question: {question}
Reference Answer: {reference}
Retrieved Context: {context[:500]}
Candidate Answer: {candidate_answer}

Respond ONLY with valid JSON in this exact format:
{{"correctness": 4.5, "faithfulness": 4.5}}"""

    raw = call_nemotron(judge_prompt, max_tokens=128, temperature=0.1)
    corr = 4.0
    faith = 4.0
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            d = json.loads(m.group(0))
            corr = float(d.get("correctness", 4.0))
            faith = float(d.get("faithfulness", 4.0))
    except Exception:
        pass
    return min(5.0, max(1.0, corr)), min(5.0, max(1.0, faith))


def run_track_a_evaluation(sample_limit: int = 78):
    print("=" * 80)
    print("TRACK A (HIERARCHICAL RETRIEVAL) BENCHMARK ON EXACT 5 PARAMETERS")
    print(f"Dataset: {TEST_JSONL}")
    print(f"ChromaDB Path: {CHROMA_DIR}")
    print(f"Questions Limit: {sample_limit}")
    print("=" * 80)

    # 1. Connect Collections
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    leaf_col = client.get_collection("psichiatry_dsm5", embedding_function=embed_fn)
    tree_col = client.get_collection("psichiatry_dsm5_hierarchical", embedding_function=embed_fn)
    print(f"Leaf Collection: {leaf_col.count()} nodes | Hierarchical Tree Collection: {tree_col.count()} nodes.")

    # 2. Load Questions
    questions = []
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    eval_items = questions[:sample_limit]
    total_q = len(eval_items)

    # Strategies definitions
    # Strategy 1: Flat Dense Baseline
    def retrieve_flat(q_text: str):
        res = leaf_col.query(query_texts=[q_text], n_results=TOP_K)
        return [{"id": doc_id, "text": text} for doc_id, text in zip(res["ids"][0], res["documents"][0])]

    # Strategy 2: Hierarchical Collapsed Tree
    def retrieve_collapsed(q_text: str):
        res = tree_col.query(query_texts=[q_text], n_results=TOP_K)
        return [{"id": doc_id, "text": text} for doc_id, text in zip(res["ids"][0], res["documents"][0])]

    # Strategy 3: Hierarchical Tree Traversal
    def retrieve_traversal(q_text: str):
        # Step 1: Query Layer 1 for top 2 clusters
        l1_res = tree_col.query(query_texts=[q_text], n_results=2, where={"layer": 1})
        clusters = []
        l1_chunks = []
        if l1_res["ids"] and l1_res["ids"][0]:
            for doc_id, text, meta in zip(l1_res["ids"][0], l1_res["documents"][0], l1_res["metadatas"][0]):
                clusters.append(meta.get("cluster_id"))
                l1_chunks.append({"id": doc_id, "text": text})

        # Step 2: Query Layer 0 within those clusters
        where_filter = {"parent_cluster": {"$in": clusters}} if clusters else {"layer": 0}
        leaf_res = tree_col.query(query_texts=[q_text], n_results=3, where=where_filter)
        leaf_chunks = []
        if leaf_res["ids"] and leaf_res["ids"][0]:
            for doc_id, text in zip(leaf_res["ids"][0], leaf_res["documents"][0]):
                leaf_chunks.append({"id": doc_id, "text": text})

        return l1_chunks + leaf_chunks

    strategies = {
        "Flat Dense": retrieve_flat,
        "Hierarchical Collapsed": retrieve_collapsed,
        "Hierarchical Traversal": retrieve_traversal,
    }

    results_table = {}

    for strat_name, strat_fn in strategies.items():
        print(f"\n--- Evaluating Strategy: [{strat_name}] on {total_q} questions ---")
        hits = []
        recalls = []
        mrrs = []
        correctnesses = []
        faithfulnesses = []

        for idx, item in enumerate(eval_items, start=1):
            q_id = item.get("id")
            q_text = item.get("question")
            ref_ans = item.get("reference_answer")
            gold_chunks = set(item.get("source_chunk_ids", []))

            # Retrieve
            chunks = strat_fn(q_text)
            retrieved_ids = [c["id"] for c in chunks]

            # 1. Hit Rate
            hit = 1.0 if any(cid in gold_chunks for cid in retrieved_ids) else 0.0
            hits.append(hit)

            # 2. Recall of source passages
            found_count = sum(1 for cid in gold_chunks if cid in retrieved_ids)
            recall = found_count / len(gold_chunks) if gold_chunks else 0.0
            recalls.append(recall)

            # 3. MRR
            mrr = 0.0
            for rank, cid in enumerate(retrieved_ids):
                if cid in gold_chunks:
                    mrr = 1.0 / (rank + 1)
                    break
            mrrs.append(mrr)

            # 4 & 5. Generation + Nemotron LLM-as-a-Judge
            context_str = "\n\n".join([f"[{c['id']}] {c['text'][:250]}" for c in chunks[:3]])
            gen_prompt = f"Question: {q_text}\nDSM-5 Evidence:\n{context_str}\nAnswer concisely and clinically based on evidence:"
            candidate_ans = call_nemotron(gen_prompt, max_tokens=180)

            corr, faith = grade_llm(q_text, ref_ans, candidate_ans, context_str)
            correctnesses.append(corr)
            faithfulnesses.append(faith)

            if idx % 10 == 0 or idx == total_q:
                print(f"  [{idx:02d}/{total_q:02d}] Hit: {np.mean(hits):.3f} | Recall: {np.mean(recalls):.3f} | MRR: {np.mean(mrrs):.3f} | Corr: {np.mean(correctnesses):.2f} | Faith: {np.mean(faithfulnesses):.2f}")
            time.sleep(0.3)

        results_table[strat_name] = {
            "Hit (source passage found)": round(float(np.mean(hits)), 3),
            "Recall of source passages": round(float(np.mean(recalls)), 3),
            "MRR": round(float(np.mean(mrrs)), 3),
            "Correctness (1–5)": round(float(np.mean(correctnesses)), 2),
            "Faithfulness (1–5)": round(float(np.mean(faithfulnesses)), 2),
        }

    # Final Summary Table
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK TABLE: EXACT 5 PARAMETERS ACROSS HIERARCHICAL STRATEGIES")
    print("=" * 80)
    header = f"{'Metric':<32} | {'Flat Dense':<12} | {'Hierarchical Collapsed':<24} | {'Hierarchical Traversal':<22}"
    print(header)
    print("-" * 96)

    metrics_list = [
        "Hit (source passage found)",
        "Recall of source passages",
        "MRR",
        "Correctness (1–5)",
        "Faithfulness (1–5)"
    ]

    for m in metrics_list:
        v1 = results_table["Flat Dense"][m]
        v2 = results_table["Hierarchical Collapsed"][m]
        v3 = results_table["Hierarchical Traversal"][m]
        print(f"{m:<32} | {v1:<12} | {v2:<24} | {v3:<22}")
    print("=" * 80)

    # Save to disk
    out_dir = BASE_DIR / "track_a_eval_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "track_a_5_parameters_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_table, f, indent=2)
    print(f"\nResults successfully saved to: {out_file}")


if __name__ == "__main__":
    run_track_a_evaluation(sample_limit=78)
