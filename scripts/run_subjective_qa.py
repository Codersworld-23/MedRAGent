"""Run one subjective evidence-grounded question across the 18 textbooks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medragent.bm25 import BM25Retriever
from medragent.data import load_full_textbook_chunks
from medragent.dense import DenseRetriever
from medragent.embeddings import OllamaEmbeddingClient
from medragent.hybrid import HybridRetriever
from medragent.llm_only import OllamaClient
from medragent.subjective import run_subjective_question


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--method", choices=["bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("--model", default="llama3.2:latest")
    parser.add_argument("--embedding-model", default="nomic-embed-text:latest")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rrf-constant", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--dense-threshold", type=float, default=0.25)
    parser.add_argument("--hybrid-threshold", type=float, default=0.015)
    parser.add_argument("--chunk-dir", type=Path, default=PROJECT_ROOT / "datasets" / "MedRAG" / "chunk")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    chunks = load_full_textbook_chunks(args.chunk_dir)
    bm25 = BM25Retriever(chunks)
    if args.method == "bm25":
        retrieved = bm25.retrieve(args.question, args.top_k)
    else:
        embedding = OllamaEmbeddingClient(
            args.embedding_model,
            args.base_url,
            args.timeout,
        )
        dense = DenseRetriever.from_client(chunks, embedding, args.batch_size)
        if args.method == "dense":
            retrieved = dense.retrieve(args.question, embedding, args.top_k)
        else:
            hybrid = HybridRetriever(
                bm25,
                dense,
                embedding,
                args.rrf_constant,
            )
            _, _, retrieved = hybrid.retrieve(args.question, args.top_k)

    generation = OllamaClient(
        args.model,
        args.base_url,
        0.0,
        args.timeout,
    )
    result = run_subjective_question(
        args.question,
        method=args.method,
        retrieved=retrieved,
        client=generation,
        model=args.model,
        dense_threshold=args.dense_threshold,
        hybrid_threshold=args.hybrid_threshold,
    )
    text = json.dumps(result.as_dict(), indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
