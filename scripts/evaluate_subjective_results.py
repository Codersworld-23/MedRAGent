"""Summarize coverage and abstention behavior for subjective QA outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("subjective result file contains no rows")

    statuses = Counter(row["evidence_status"] for row in rows)
    citations = sum(bool(row.get("citations")) for row in rows)
    valid_structural_citations = sum(
        bool(row.get("citations"))
        and set(row["citations"]).issubset(set(row["retrieved_chunk_ids"]))
        for row in rows
    )
    report = {
        "questions": len(rows),
        "evidence_status_counts": dict(sorted(statuses.items())),
        "coverage": round(len(rows) / len(rows), 6),
        "answer_coverage": round(
            sum(row["evidence_status"] != "abstained" for row in rows) / len(rows),
            6,
        ),
        "abstention_rate": round(
            sum(row["evidence_status"] == "abstained" for row in rows) / len(rows),
            6,
        ),
        "citation_coverage": round(citations / len(rows), 6),
        "structural_citation_validity": round(
            valid_structural_citations / len(rows),
            6,
        ),
        "mean_latency_ms": round(
            sum(float(row["latency_ms"]) for row in rows) / len(rows),
            3,
        ),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
