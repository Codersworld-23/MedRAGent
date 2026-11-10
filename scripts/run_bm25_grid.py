"""Run BM25 top-k candidates on the MedQA development split."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 3, 5, 8])
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "datasets" / "MedQA" / "questions" / "US" / "dev.jsonl")
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "datasets" / "MedRAG" / "chunk" / "Psichiatry_DSM-5.jsonl")
    parser.add_argument("--annotations", type=Path, default=PROJECT_ROOT / "experiments" / "annotations" / "medqa_scope_dev.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments" / "results" / "bm25_grid")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for top_k in args.top_k:
        if top_k <= 0:
            parser.error("--top-k values must be positive")
        output = args.output_dir / f"bm25_rag_dev_topk{top_k}.jsonl"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_bm25_rag.py"),
            "--model", args.model,
            "--top-k", str(top_k),
            "--input", str(args.input),
            "--corpus", str(args.corpus),
            "--annotations", str(args.annotations),
            "--output", str(output),
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        subprocess.run(command, check=True)
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate_bm25.py"), str(output)],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
