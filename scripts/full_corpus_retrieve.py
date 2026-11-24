"""Source-aware retrieval over all 18 MedRAG textbook corpora."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medragent.data import load_full_textbook_chunks
from medragent.bm25 import BM25Retriever, RetrievedChunk
from medragent.dense import DenseRetriever
from medragent.embeddings import OllamaEmbeddingClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--method", choices=["bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-dir", type=Path, default=PROJECT_ROOT / "datasets" / "MedRAG" / "chunk")
    parser.add_argument("--embedding-model", default="nomic-embed-text:latest")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rrf-constant", type=int, default=60)
    parser.add_argument("--bm25-index", type=Path)
    parser.add_argument("--dense-index", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    started = time.perf_counter()
    chunks = load_full_textbook_chunks(args.chunk_dir)
    if args.bm25_index:
        with args.bm25_index.open("rb") as handle:
            bm25 = pickle.load(handle)
        if len(bm25.chunks) != len(chunks):
            raise ValueError("Persisted BM25 corpus size does not match the full corpus")
    else:
        bm25 = BM25Retriever(chunks)
    bm25_results = bm25.retrieve(args.query, args.top_k)

    if args.method == "bm25":
        results = bm25_results
        dense_results = []
        retrieval_time_ms = round((time.perf_counter() - started) * 1000, 3)
    else:
        embedding_client = OllamaEmbeddingClient(
            args.embedding_model,
            args.base_url,
            args.timeout,
        )
        if args.dense_index:
            index = faiss.read_index(str(args.dense_index))
            vector = np.asarray(
                embedding_client.embed([args.query]),
                dtype=np.float32,
            )
            faiss.normalize_L2(vector)
            scores, indices = index.search(
                vector,
                min(args.top_k, len(chunks)),
            )
            ranked = sorted(
                zip(indices[0], scores[0]),
                key=lambda pair: (-float(pair[1]), int(pair[0])),
            )
            dense_results = [
                RetrievedChunk(
                    chunk_id=chunks[int(index)].chunk_id,
                    title=chunks[int(index)].title,
                    text=chunks[int(index)].text,
                    rank=rank,
                    score=float(score),
                    source_file=chunks[int(index)].source_file,
                    textbook_name=chunks[int(index)].textbook_name,
                    retrieval_method="dense",
                )
                for rank, (index, score) in enumerate(ranked, start=1)
            ]
        else:
            dense = DenseRetriever.from_client(
                chunks,
                embedding_client,
                args.batch_size,
            )
            dense_results = dense.retrieve(args.query, embedding_client, args.top_k)
        if args.method == "dense":
            results = dense_results
        else:
            bm25_ranks = {item.chunk_id: item.rank for item in bm25_results}
            dense_ranks = {item.chunk_id: item.rank for item in dense_results}
            by_id = {
                item.chunk_id: item
                for item in bm25_results + dense_results
            }
            fused = sorted(
                by_id.values(),
                key=lambda item: (
                    -(
                        (1 / (args.rrf_constant + bm25_ranks[item.chunk_id])
                         if item.chunk_id in bm25_ranks else 0.0)
                        + (1 / (args.rrf_constant + dense_ranks[item.chunk_id])
                           if item.chunk_id in dense_ranks else 0.0)
                    ),
                    item.chunk_id,
                ),
            )[:args.top_k]
            results = [
                type(item)(
                    chunk_id=item.chunk_id,
                    title=item.title,
                    text=item.text,
                    rank=rank,
                    score=(
                        (1 / (args.rrf_constant + bm25_ranks[item.chunk_id])
                         if item.chunk_id in bm25_ranks else 0.0)
                        + (1 / (args.rrf_constant + dense_ranks[item.chunk_id])
                           if item.chunk_id in dense_ranks else 0.0)
                    ),
                    source_file=item.source_file,
                    textbook_name=item.textbook_name,
                    retrieval_method="hybrid_rrf",
                )
                for rank, item in enumerate(fused, start=1)
            ]
        retrieval_time_ms = round((time.perf_counter() - started) * 1000, 3)

    report = {
        "query": args.query,
        "method": args.method,
        "top_k": args.top_k,
        "corpus_chunk_count": len(chunks),
        "retrieval_time_ms": retrieval_time_ms,
        "rrf_constant": args.rrf_constant if args.method == "hybrid" else None,
        "embedding_model": args.embedding_model if args.method != "bm25" else None,
        "results": [item.as_dict() for item in results],
        "bm25_results": [item.as_dict() for item in bm25_results] if args.method == "hybrid" else [],
        "dense_results": [item.as_dict() for item in dense_results] if args.method == "hybrid" else [],
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
