"""Export BioASQ summary questions into a textbook-review-ready JSONL benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.subjective_benchmark import build_unreviewed_records, load_bioasq_summary_questions, validate_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bioasq",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "BioASQ" / "BioASQ-training13b" / "training13b.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Export the first N deterministic records")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    questions = load_bioasq_summary_questions(args.bioasq)
    if args.limit is not None:
        questions = questions[: args.limit]
    records = build_unreviewed_records(questions, source_name="BioASQ-13b")
    summary = validate_records(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
