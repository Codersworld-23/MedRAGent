"""Run reproducible Phase 3 baseline retrieval experiments.

Examples:
    python scripts/run_phase3.py --retriever bm25 --limit 100
    python scripts/run_phase3.py --retriever all --limit 20
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.retrieval_metrics import (
    _relevant_ids,
    aggregate_metrics,
    evaluate_ranked_results,
)
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever


TOP_K_VALUES = (1, 3, 5, 10, 15)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_retrievers(chunks, method, model_name):
    bm25 = BM25Retriever()
    bm25.load_chunks(chunks)
    if method == "bm25":
        return {"bm25": bm25}

    dense = DenseRetriever(model_name=model_name)
    dense.load_chunks(chunks)
    if method == "dense":
        return {"dense": dense}
    return {"bm25": bm25, "dense": dense, "hybrid": HybridRetriever(bm25, dense)}


def run_method(retriever, questions, top_k_values):
    max_k = max(top_k_values)
    query_results = []
    metric_results = {k: [] for k in top_k_values}

    for question in questions:
        ranked = retriever.search(question.get("question", ""), top_k=max_k)
        query_results.append({
            "question_id": question.get("id"),
            "question": question.get("question", ""),
            "results": [
                {"score": score, "chunk": chunk}
                for score, chunk in ranked
            ],
        })
        for k in top_k_values:
            metric_results[k].append(
                evaluate_ranked_results(ranked, _relevant_ids(question), k)
            )

    return query_results, {str(k): aggregate_metrics(values) for k, values in metric_results.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Run MedRAGent Phase 3 retrieval experiments")
    parser.add_argument("--retriever", choices=("bm25", "dense", "hybrid", "all"), default="bm25")
    parser.add_argument("--chunk-file", default="baseline_chunks_500.jsonl")
    parser.add_argument("--question-file", default="medqa_us.jsonl")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0, help="Limit questions; 0 means all")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Embedding models to compare; overrides --model for dense/hybrid",
    )
    parser.add_argument("--output-dir", default="experiments/results/phase3")
    return parser.parse_args()


def main():
    args = parse_args()
    chunks_path = ROOT / "data" / "processed" / "chunks" / args.chunk_file
    questions_path = ROOT / "data" / "processed" / "questions" / args.question_file
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {chunks_path}")
    if not questions_path.exists():
        raise FileNotFoundError(f"Question file not found: {questions_path}")

    questions = load_jsonl(questions_path)
    if args.split:
        questions = [question for question in questions if question.get("split") == args.split]
    if args.limit > 0:
        questions = questions[:args.limit]
    if not questions:
        raise ValueError("No questions matched the requested split/limit")

    chunks = load_jsonl(chunks_path)
    methods = ("bm25", "dense", "hybrid") if args.retriever == "all" else (args.retriever,)
    models = args.models or [args.model]
    if methods == ("bm25",):
        models = [None]
    summary = {
        "chunk_file": args.chunk_file,
        "question_file": args.question_file,
        "split": args.split,
        "question_count": len(questions),
        "chunk_count": len(chunks),
        "top_k_values": list(TOP_K_VALUES),
        "methods": {},
    }

    for model in models:
        retrievers = build_retrievers(
            chunks,
            args.retriever if args.retriever != "all" else "hybrid",
            model or args.model,
        )
        model_key = (model or "none").replace("/", "_")
        for method in methods:
            if method not in retrievers:
                continue
            results, metrics = run_method(retrievers[method], questions, TOP_K_VALUES)
            result_key = method if len(models) == 1 else f"{method}:{model}"
            summary["methods"][result_key] = metrics
            with open(output_dir / f"{method}_{model_key}_results.jsonl", "w", encoding="utf-8") as handle:
                for result in results:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    with open(output_dir / "phase3_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()