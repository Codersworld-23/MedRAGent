"""Validate subjective benchmark records and optionally verify textbook chunk IDs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from medragent.data import load_full_textbook_chunks
from src.subjective_benchmark import validate_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--verify-chunk-ids", action="store_true")
    parser.add_argument("--chunk-dir", type=Path, default=PROJECT_ROOT / "datasets" / "MedRAG" / "chunk")
    args = parser.parse_args()
    with args.input.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    chunk_ids = None
    if args.verify_chunk_ids:
        chunk_ids = {chunk.chunk_id for chunk in load_full_textbook_chunks(args.chunk_dir)}
    print(json.dumps(validate_records(records, known_chunk_ids=chunk_ids), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
