"""Summarize a BM25 JSONL result file with overall and scope metrics."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("result file contains no rows")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("scope") or "unannotated"].append(row)

    def group_metrics(group: list[dict[str, Any]]) -> dict[str, Any]:
        successful = [row for row in group if row.get("error") is None]
        latencies = [float(row["latency_ms"]) for row in group]
        citation_valid = [
            row for row in successful
            if row.get("citations")
            and set(row["citations"]).issubset(set(row.get("retrieved_chunk_ids", [])))
        ]
        return {
            "questions": len(group),
            "correct": sum(row.get("is_correct") is True for row in group),
            "accuracy_on_all": round(
                sum(row.get("is_correct") is True for row in group) / len(group), 6
            ),
            "failures_or_abstentions": sum(row.get("error") is not None for row in group),
            "failure_rate": round(
                sum(row.get("error") is not None for row in group) / len(group), 6
            ),
            "citation_valid_successes": len(citation_valid),
            "citation_valid_rate_on_successes": round(
                len(citation_valid) / len(successful), 6
            ) if successful else 0.0,
            "mean_latency_ms": round(statistics.mean(latencies), 3),
            "median_latency_ms": round(statistics.median(latencies), 3),
        }

    return {
        "questions": len(rows),
        "correct": sum(row.get("is_correct") is True for row in rows),
        "accuracy_on_all": round(
            sum(row.get("is_correct") is True for row in rows) / len(rows), 6
        ),
        "failures_or_abstentions": sum(row.get("error") is not None for row in rows),
        "failure_rate": round(
            sum(row.get("error") is not None for row in rows) / len(rows), 6
        ),
        "mean_latency_ms": round(
            statistics.mean(float(row["latency_ms"]) for row in rows), 3
        ),
        "median_latency_ms": round(
            statistics.median(float(row["latency_ms"]) for row in rows), 3
        ),
        "mean_retrieved_chunks": round(
            statistics.mean(len(row.get("retrieved_chunk_ids", [])) for row in rows), 3
        ),
        "scopes": {scope: group_metrics(group) for scope, group in sorted(grouped.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(load_rows(args.input))
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
