"""Run the Phase 4 vanilla RAG generation and evaluation pipeline.

Examples:
    python scripts/run_phase4.py --backend extractive --limit 10
    python scripts/run_phase4.py --backend ollama --model llama3.1:8b
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.vanilla_rag import VanillaRAG
from src.evaluation.accuracy import aggregate_accuracy, evaluate_answer
from src.evaluation.faithfulness import evaluate_faithfulness
from src.evaluation.hallucination import evaluate_hallucination
from src.evaluation.relevance import evaluate_relevance
from src.generation.llm import ExtractiveGenerator, OllamaGenerator
from src.retrieval.bm25 import BM25Retriever


def load_jsonl(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Run MedRAGent Phase 4 vanilla RAG")
    parser.add_argument("--backend", choices=("extractive", "ollama"), default="extractive")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--chunk-file", default="baseline_chunks_500.jsonl")
    parser.add_argument("--question-file", default="medqa_us.jsonl")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="Limit questions; 0 means all")
    parser.add_argument("--output-dir", default="experiments/results/phase4")
    return parser.parse_args()


def main():
    args = parse_args()
    chunks_path = ROOT / "data" / "processed" / "chunks" / args.chunk_file
    questions_path = ROOT / "data" / "processed" / "questions" / args.question_file
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_jsonl(chunks_path)
    questions = [q for q in load_jsonl(questions_path) if not args.split or q.get("split") == args.split]
    if args.limit > 0:
        questions = questions[:args.limit]
    if not questions:
        raise ValueError("No questions matched the requested split/limit")

    retriever = BM25Retriever()
    retriever.load_chunks(chunks)
    generator = ExtractiveGenerator() if args.backend == "extractive" else OllamaGenerator(model=args.model)
    pipeline = VanillaRAG(retriever, generator, top_k=args.top_k)

    records = []
    accuracy = []
    faithfulness = []
    hallucination = []
    relevance = []
    for question in questions:
        prediction = pipeline.answer(question)
        prediction.pop("prompt", None)
        accuracy.append(evaluate_answer(prediction, question))
        faith = evaluate_faithfulness(prediction)
        faithfulness.append(faith)
        hallucination.append(evaluate_hallucination(faith))
        relevance.append(evaluate_relevance(prediction))
        prediction["evaluation"] = {
            "accuracy": accuracy[-1],
            "faithfulness": faithfulness[-1],
            "hallucination": hallucination[-1],
            "relevance": relevance[-1],
        }
        records.append(prediction)

    with open(output_dir / "vanilla_rag_results.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "backend": args.backend,
        "model": args.model if args.backend == "ollama" else None,
        "chunk_file": args.chunk_file,
        "question_file": args.question_file,
        "split": args.split,
        "top_k": args.top_k,
        "question_count": len(questions),
        "accuracy": aggregate_accuracy(accuracy),
        "faithfulness": {
            "mean": sum(item["faithfulness"] for item in faithfulness) / len(faithfulness),
            "mean_valid_citations": sum(item["valid_citations"] for item in faithfulness) / len(faithfulness),
        },
        "hallucination": {
            "mean_rate": sum(item["hallucination_rate"] for item in hallucination) / len(hallucination),
            "uncited_responses": sum(item["uncited_response"] for item in hallucination),
        },
        "relevance": {
            "mean": sum(item["mean_relevance"] for item in relevance) / len(relevance),
            "mean_top": sum(item["top_relevance"] for item in relevance) / len(relevance),
        },
    }
    with open(output_dir / "phase4_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()