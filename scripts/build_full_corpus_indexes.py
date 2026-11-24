"""Build persisted BM25 and dense indexes for the full 18-textbook corpus."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import faiss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medragent.data import load_full_textbook_chunks
from medragent.bm25 import BM25Retriever
from medragent.dense import DenseRetriever
from medragent.embeddings import OllamaEmbeddingClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["bm25", "dense", "both"], default="bm25")
    parser.add_argument("--chunk-dir", type=Path, default=PROJECT_ROOT / "datasets" / "MedRAG" / "chunk")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "indexes" / "full_textbooks")
    parser.add_argument("--embedding-model", default="nomic-embed-text:latest")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    started = time.perf_counter()
    chunks = load_full_textbook_chunks(args.chunk_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "corpus_chunk_count": len(chunks),
        "embedding_model": args.embedding_model if args.method != "bm25" else None,
        "indexes": {},
    }

    if args.method in {"bm25", "both"}:
        bm25_started = time.perf_counter()
        bm25 = BM25Retriever(chunks)
        bm25_path = args.output_dir / "bm25_full_textbooks.pkl"
        with bm25_path.open("wb") as handle:
            pickle.dump(bm25, handle)
        report["indexes"]["bm25"] = {
            "path": str(bm25_path),
            "build_time_ms": round((time.perf_counter() - bm25_started) * 1000, 3),
            "chunk_count": len(chunks),
        }

    if args.method in {"dense", "both"}:
        dense_started = time.perf_counter()
        embedding_client = OllamaEmbeddingClient(
            args.embedding_model,
            args.base_url,
            args.timeout,
        )
        dense = DenseRetriever.from_client(chunks, embedding_client, args.batch_size)
        dense_path = args.output_dir / "dense_full_textbooks.faiss"
        faiss.write_index(dense.index, str(dense_path))
        report["indexes"]["dense"] = {
            "path": str(dense_path),
            "build_time_ms": round((time.perf_counter() - dense_started) * 1000, 3),
            "chunk_count": len(chunks),
            "embedding_dimension": dense.dimension,
            "normalization": "L2; inner-product search equals cosine similarity",
        }

    report["total_build_time_ms"] = round((time.perf_counter() - started) * 1000, 3)
    report_path = args.output_dir / "index_metadata.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
