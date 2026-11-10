"""Create a versioned MedQA-US scope annotation file."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medragent.data import load_medqa_jsonl
from medragent.scope import annotate_records, write_annotations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "MedQA" / "questions" / "US" / "dev.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "annotations" / "medqa_scope_dev.jsonl",
    )
    args = parser.parse_args()
    annotations = annotate_records(load_medqa_jsonl(args.input))
    write_annotations(args.output, annotations)
    counts = Counter(annotation.scope for annotation in annotations)
    print(f"Wrote {len(annotations)} annotations to {args.output}")
    for scope, count in sorted(counts.items()):
        print(f"{scope}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
