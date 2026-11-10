"""Run the BM25 vanilla RAG baseline against MedQA-US."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medragent.bm25 import BM25Retriever, run_record, write_results
from medragent.data import load_medqa_jsonl, load_text_chunks_jsonl
from medragent.llm_only import OllamaClient
from medragent.scope import load_annotations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "MedQA" / "questions" / "US" / "dev.jsonl",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=PROJECT_ROOT
        / "datasets"
        / "MedRAG"
        / "chunk"
        / "Psichiatry_DSM-5.jsonl",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "annotations" / "medqa_scope_dev.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "bm25_rag_dev.jsonl",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    records = load_medqa_jsonl(args.input)
    annotations = load_annotations(args.annotations)
    missing = [record.id for record in records if record.id not in annotations]
    if missing:
        parser.error(f"{len(missing)} input records have no scope annotation")
    if args.limit is not None:
        records = records[: args.limit]
    chunks = load_text_chunks_jsonl(args.corpus)
    retriever = BM25Retriever(chunks)
    client = OllamaClient(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        timeout_seconds=args.timeout,
    )
    results = [
        run_record(
            record,
            retriever,
            client,
            model=args.model,
            temperature=args.temperature,
            top_k=args.top_k,
            annotation=annotations[record.id],
        )
        for record in records
    ]
    write_results(args.output, results)
    failures = sum(result.error is not None for result in results)
    correct = sum(result.is_correct is True for result in results)
    print(f"Wrote {len(results)} results to {args.output}")
    print(f"Correct: {correct}/{len(results)}; failures/abstentions: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
