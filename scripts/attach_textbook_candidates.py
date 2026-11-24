"""Attach reproducible BM25 textbook-evidence candidates to subjective questions.

Candidates are review aids only.  This script never writes them to
supporting_chunk_ids and therefore never turns automatic retrieval into gold
evidence.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from medragent.bm25 import BM25Retriever
from medragent.data import load_full_textbook_chunks
from src.subjective_benchmark import validate_records


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, help="Map only the first N records")
    parser.add_argument("--chunk-dir", type=Path, default=PROJECT_ROOT / "datasets" / "MedRAG" / "chunk")
    parser.add_argument("--bm25-index", type=Path, default=PROJECT_ROOT / "indexes" / "full_textbooks" / "bm25_full_textbooks.pkl")
    args = parser.parse_args()
    if args.top_k <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--top-k and --limit must be positive")

    records = load_jsonl(args.input)
    validate_records(records)
    if args.bm25_index.exists():
        with args.bm25_index.open("rb") as handle:
            retriever = pickle.load(handle)
        if not isinstance(retriever, BM25Retriever) or len(retriever.chunks) != 125847:
            raise ValueError("Persisted BM25 index is not the expected full textbook corpus")
    else:
        retriever = BM25Retriever(load_full_textbook_chunks(args.chunk_dir))

    target_count = args.limit or len(records)
    for record in records[:target_count]:
        results = retriever.retrieve(record["question"], args.top_k)
        record["candidate_textbook_evidence"] = [
            {
                "chunk_id": item.chunk_id,
                "textbook_name": item.textbook_name,
                "rank": item.rank,
                "score": item.score,
                "text_excerpt": item.text[:500],
            }
            for item in results
        ]
        record["candidate_retrieval"] = {
            "method": "bm25",
            "top_k": args.top_k,
            "corpus": "MedRAG_18_textbooks_125847_chunks",
            "gold_evidence": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "mapped_records": target_count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
