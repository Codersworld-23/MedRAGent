"""Common evaluation for MedRAGent experiment JSONL outputs.

The evaluator is deliberately schema-aware:
- answer metrics use the normalized MedQA option letters;
- citation metrics are structural and do not claim medical factuality;
- retrieval metrics are computed only when a gold-evidence sidecar is supplied;
- scope metrics preserve the project’s psychiatry/out-of-corpus stratification.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


REQUIRED_RESULT_FIELDS = {
    "question_id",
    "system_name",
    "predicted_answer",
    "expected_answer",
    "is_correct",
    "retrieved_chunk_ids",
    "citations",
    "latency_ms",
    "error",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_gold_evidence(path: Path | None) -> dict[str, set[str]]:
    if path is None:
        return {}
    labels: dict[str, set[str]] = {}
    for row in load_jsonl(path):
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("gold evidence rows require a non-empty question_id")
        if question_id in labels:
            raise ValueError(f"{question_id}: duplicate gold evidence row")
        values = row.get("supporting_chunk_ids", row.get("gold_chunk_ids", []))
        if not isinstance(values, list):
            raise ValueError(f"{question_id}: supporting_chunk_ids must be a list")
        labels[question_id] = {str(value) for value in values}
    return labels


def load_manual_evidence(path: Path | None) -> tuple[dict[str, set[str]], dict[str, str]]:
    """Load reviewed evidence and labels without inferring them from text."""
    if path is None:
        return {}, {}
    labels: dict[str, set[str]] = {}
    scope_labels: dict[str, str] = {}
    for row in load_jsonl(path):
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("manual evidence rows require a non-empty question_id")
        if question_id in labels:
            raise ValueError(f"{question_id}: duplicate manual evidence row")
        values = row.get("supporting_chunk_ids")
        scope = row.get("scope_label")
        if not isinstance(values, list):
            raise ValueError(f"{question_id}: supporting_chunk_ids must be a list")
        if scope not in {
            "psychiatry_relevant_supported",
            "psychiatry_relevant_unsupported",
            "out_of_corpus",
        }:
            raise ValueError(f"{question_id}: invalid scope_label")
        labels[question_id] = {str(value) for value in values}
        scope_labels[question_id] = scope
    return labels, scope_labels


def _validate_rows(rows: list[dict[str, Any]]) -> None:
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        missing = REQUIRED_RESULT_FIELDS - row.keys()
        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(f"result row {index} is missing required fields: {fields}")
        question_id = row["question_id"]
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"result row {index} has an invalid question_id")
        if question_id in seen_ids:
            raise ValueError(f"duplicate result question_id: {question_id}")
        seen_ids.add(question_id)
        for field in ("retrieved_chunk_ids", "citations"):
            if not isinstance(row[field], list):
                raise ValueError(f"result row {index} field '{field}' must be a list")
        if row["latency_ms"] is not None:
            try:
                latency = float(row["latency_ms"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"result row {index} has invalid latency_ms") from exc
            if not math.isfinite(latency) or latency < 0:
                raise ValueError(f"result row {index} has invalid latency_ms")


def _retrieval_metrics(
    rows: list[dict[str, Any]], gold: dict[str, set[str]], k: int
) -> dict[str, Any]:
    labeled = []
    for row in rows:
        relevant = gold.get(str(row["question_id"]))
        if not relevant:
            continue
        retrieved = [str(value) for value in row.get("retrieved_chunk_ids", [])[:k]]
        hits = [index for index, value in enumerate(retrieved) if value in relevant]
        hit_count = len(set(retrieved) & relevant)
        dcg = sum(1.0 / math.log2(index + 2) for index in hits)
        ideal = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal))
        labeled.append(
            {
                "recall": hit_count / len(relevant),
                "precision": hit_count / len(retrieved) if retrieved else 0.0,
                "mrr": 1.0 / (hits[0] + 1) if hits else 0.0,
                "ndcg": dcg / idcg if idcg else 0.0,
            }
        )
    if not labeled:
        return {"labeled_queries": 0, "k": k}
    return {
        "labeled_queries": len(labeled),
        "k": k,
        "recall_at_k": round(statistics.mean(item["recall"] for item in labeled), 6),
        "precision_at_k": round(statistics.mean(item["precision"] for item in labeled), 6),
        "mrr": round(statistics.mean(item["mrr"] for item in labeled), 6),
        "ndcg_at_k": round(statistics.mean(item["ndcg"] for item in labeled), 6),
    }


def _retrieval_metrics_at_k(
    rows: list[dict[str, Any]],
    gold: dict[str, set[str]],
) -> dict[str, Any]:
    at_one = _retrieval_metrics(rows, gold, 1)
    at_three = _retrieval_metrics(rows, gold, 3)
    return {
        "labeled_queries": at_three.get("labeled_queries", 0),
        "recall_at_1": at_one.get("recall_at_k"),
        "recall_at_3": at_three.get("recall_at_k"),
        "precision_at_1": at_one.get("precision_at_k"),
        "precision_at_3": at_three.get("precision_at_k"),
        "mrr": at_three.get("mrr"),
        "ndcg_at_3": at_three.get("ndcg_at_k"),
        "k": 3,
    }


def _metrics(
    rows: list[dict[str, Any]],
    gold: dict[str, set[str]],
    k: int,
    scope_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("result file contains no rows")
    successful = [row for row in rows if row.get("error") is None]
    valid_citations = [
        row for row in successful
        if row.get("citations")
        and set(row["citations"]).issubset(set(row.get("retrieved_chunk_ids", [])))
    ]
    latencies = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
    correct = sum(row.get("is_correct") is True for row in rows)
    proportion = correct / len(rows)
    z = 1.96
    denominator = 1 + z * z / len(rows)
    centre = (proportion + z * z / (2 * len(rows))) / denominator
    margin = (
        z
        * ((proportion * (1 - proportion) / len(rows) + z * z / (4 * len(rows) ** 2)) ** 0.5)
        / denominator
    )
    return {
        "questions": len(rows),
        "evaluated_answers": len(successful),
        "correct": correct,
        "accuracy_on_all_questions": round(correct / len(rows), 6),
        "accuracy_95_ci": [round(max(0.0, centre - margin), 6), round(min(1.0, centre + margin), 6)],
        "accuracy_on_successful_answers": round(
            sum(row.get("is_correct") is True for row in successful) / len(successful), 6
        ) if successful else None,
        "failures_or_abstentions": len(rows) - len(successful),
        "failure_rate": round((len(rows) - len(successful)) / len(rows), 6),
        "citation_valid_successes": len(valid_citations),
        "citation_valid_rate_on_successes": round(
            len(valid_citations) / len(successful), 6
        ) if successful else None,
        "citation_coverage_on_all": round(
            sum(bool(row.get("citations")) for row in rows) / len(rows), 6
        ),
        "mean_latency_ms": round(statistics.mean(latencies), 3) if latencies else None,
        "median_latency_ms": round(statistics.median(latencies), 3) if latencies else None,
        "mean_retrieved_chunks": round(
            statistics.mean(len(row.get("retrieved_chunk_ids", [])) for row in rows), 3
        ),
        "retrieval": {
            **_retrieval_metrics(rows, gold, k),
            **_retrieval_metrics_at_k(rows, gold),
        },
        "unsupported": {
            "count": sum(
                scope_labels.get(str(row["question_id"])) in {
                    "psychiatry_relevant_unsupported",
                    "out_of_corpus",
                }
                for row in rows
            ) if scope_labels else 0,
            "percentage": round(
                100.0 * sum(
                    scope_labels.get(str(row["question_id"])) in {
                        "psychiatry_relevant_unsupported",
                        "out_of_corpus",
                    }
                    for row in rows
                ) / len(rows),
                6,
            ) if scope_labels else 0.0,
        },
    }


def evaluate(
    rows: list[dict[str, Any]],
    gold: dict[str, set[str]] | None = None,
    k: int = 3,
    scope_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    if k <= 0:
        raise ValueError("k must be positive")
    if not rows:
        raise ValueError("result file contains no rows")
    _validate_rows(rows)
    gold = gold or {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("scope") or "unannotated"].append(row)
    return {
        "system_name": rows[0].get("system_name"),
        "result_rows": len(rows),
        "protocol": {
            "answer_metric": "exact MedQA option-letter accuracy",
            "failure_policy": "failures and abstentions count incorrect in accuracy_on_all_questions",
            "citation_metric": "citation IDs must be non-empty and belong to retrieved_chunk_ids",
            "retrieval_metric_policy": "Recall, Precision, MRR, and nDCG require gold evidence sidecar",
            "grounding_metric_policy": "lexical overlap is diagnostic only; no factuality claim",
            "k": k,
        },
        "overall": _metrics(rows, gold, k, scope_labels),
        "by_scope": {
            scope: _metrics(scope_rows, gold, k, scope_labels)
            for scope, scope_rows in sorted(grouped.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--gold-evidence", type=Path)
    parser.add_argument("--manual-annotations", type=Path)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("--k must be positive")
    gold = load_gold_evidence(args.gold_evidence)
    scope_labels = None
    if args.manual_annotations:
        gold, scope_labels = load_manual_evidence(args.manual_annotations)
    report = evaluate(
        load_jsonl(args.input),
        gold,
        args.k,
        scope_labels,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
