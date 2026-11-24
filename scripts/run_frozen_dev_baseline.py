"""Run or verify the frozen MedQA-US development baseline gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-annotations", type=Path, required=True)
    parser.add_argument("--grounding-review", nargs="+", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        raise FileNotFoundError("Project virtual environment was not found")

    manual = args.manual_annotations
    if not manual.exists():
        raise FileNotFoundError(f"Manual annotation sidecar not found: {manual}")

    run([
        str(python),
        "scripts/review_scope_annotations.py",
        "validate",
        "--input",
        str(manual),
        "--require-complete",
    ])

    if args.grounding_review:
        run([
            str(python),
            "scripts/validate_grounding_review.py",
            *(str(path) for path in args.grounding_review),
            "--output",
            "experiments/analysis/grounding_review_summary.json",
        ])

    freeze_outputs = [
        PROJECT_ROOT / "experiments/results/llm_only_dev.jsonl",
        PROJECT_ROOT / "experiments/results/bm25_rag_dev_topk3.jsonl",
        PROJECT_ROOT / "experiments/results/dense_rag_dev_topk3.jsonl",
        PROJECT_ROOT / "experiments/results/hybrid_rag_dev_topk3.jsonl",
    ]
    missing = [path for path in freeze_outputs if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing development outputs: " + ", ".join(str(path) for path in missing)
        )
    if args.verify_only:
        run([
            str(python),
            "scripts/freeze_baseline.py",
            "--split",
            "dev",
            "--manual-annotations",
            str(manual),
            "--check-outputs",
            *(str(path.relative_to(PROJECT_ROOT)) for path in freeze_outputs),
            "--output",
            "experiments/analysis/baseline_freeze_dev_reviewed.json",
        ])
        run([
            str(python),
            "evaluations/statistical_tests.py",
            "--baseline",
            "experiments/results/llm_only_dev.jsonl",
            "--systems",
            "experiments/results/bm25_rag_dev_topk3.jsonl",
            "experiments/results/dense_rag_dev_topk3.jsonl",
            "experiments/results/hybrid_rag_dev_topk3.jsonl",
            "--output",
            "experiments/analysis/baseline_paired_statistics_reviewed.json",
        ])
        print("Reviewed frozen development verification complete.")
    else:
        print(
            "Manual sidecar and grounding gate passed. "
            "Add the exact rerun commands for the four systems after review freeze."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
