"""Run dense vanilla RAG against the fixed MedQA and Psychiatry corpus."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medragent.data import load_medqa_jsonl, load_text_chunks_jsonl
from medragent.dense import DenseRetriever, run_record, write_results
from medragent.embeddings import OllamaEmbeddingClient
from medragent.llm_only import OllamaClient
from medragent.scope import load_annotations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "datasets" / "MedQA" / "questions" / "US" / "dev.jsonl")
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "datasets" / "MedRAG" / "chunk" / "Psichiatry_DSM-5.jsonl")
    parser.add_argument("--annotations", type=Path, default=PROJECT_ROOT / "experiments" / "annotations" / "medqa_scope_dev.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "experiments" / "results" / "dense_rag_dev.jsonl")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.top_k <= 0 or args.batch_size <= 0:
        parser.error("--top-k and --batch-size must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    records = load_medqa_jsonl(args.input)
    annotations = load_annotations(args.annotations)
    if args.limit is not None:
        records = records[: args.limit]
    missing = [record.id for record in records if record.id not in annotations]
    if missing:
        parser.error(f"{len(missing)} input records have no scope annotation")
    chunks = load_text_chunks_jsonl(args.corpus)
    embedding_client = OllamaEmbeddingClient(args.embedding_model, args.base_url, args.timeout)
    started = time.perf_counter()
    retriever = DenseRetriever.from_client(chunks, embedding_client, args.batch_size)
    index_build_ms = round((time.perf_counter() - started) * 1000, 3)
    generation_client = OllamaClient(args.model, args.base_url, args.temperature, args.timeout)
    results = [
        run_record(
            record,
            retriever,
            embedding_client,
            generation_client,
            args.model,
            args.embedding_model,
            args.temperature,
            args.top_k,
            annotations[record.id],
        )
        for record in records
    ]
    write_results(args.output, results)
    metadata_path = args.output.with_suffix(".index.json")
    metadata_path.write_text(
        json.dumps(
            {
                "embedding_model": args.embedding_model,
                "corpus": str(args.corpus),
                "chunk_count": len(chunks),
                "embedding_dimension": retriever.dimension,
                "normalization": "L2; inner-product search equals cosine similarity",
                "index_type": "faiss.IndexFlatIP",
                "batch_size": args.batch_size,
                "index_build_ms": index_build_ms,
                "top_k": args.top_k,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    failures = sum(result.error is not None for result in results)
    correct = sum(result.is_correct is True for result in results)
    print(f"Index build: {index_build_ms} ms")
    print(f"Wrote {len(results)} results to {args.output}")
    print(f"Wrote index metadata to {metadata_path}")
    print(f"Correct: {correct}/{len(results)}; failures/abstentions: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
