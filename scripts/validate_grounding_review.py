"""Validate claim-level grounding reviews and report reviewer agreement."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

REVIEW_FIELDS = (
    "question_id",
    "system_name",
    "option_correct",
    "evidence_supports_option",
    "unsupported_claim_present",
    "should_abstain",
    "citation_points_to_relevant_passage",
    "reviewer",
    "review_notes",
)
VALID_BOOLEAN = {"yes", "no", "uncertain"}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            rows.append(row)
    return rows


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    if len(labels_a) != len(labels_b):
        raise ValueError("review lists must have equal lengths")
    if not labels_a:
        raise ValueError("review lists cannot be empty")
    categories = sorted(set(labels_a) | set(labels_b))
    n = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = sum(
        (counts_a[category] / n) * (counts_b[category] / n)
        for category in categories
    )
    return (observed - expected) / (1 - expected) if 1 - expected else 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="+", type=Path)
    parser.add_argument("--agreement-field", default="evidence_supports_option")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    all_rows = []
    for path in args.input:
        rows = load_jsonl(path)
        for line_number, row in enumerate(rows, start=1):
            missing = [field for field in REVIEW_FIELDS if field not in row]
            if missing:
                raise ValueError(
                    f"{path}:{line_number}: missing fields: {', '.join(missing)}"
                )
            for field in (
                "option_correct",
                "evidence_supports_option",
                "unsupported_claim_present",
                "should_abstain",
                "citation_points_to_relevant_passage",
            ):
                if row[field] not in VALID_BOOLEAN:
                    raise ValueError(
                        f"{path}:{line_number}: invalid {field} '{row[field]}'"
                    )
            if not str(row["reviewer"]).strip():
                raise ValueError(f"{path}:{line_number}: reviewer is required")
        all_rows.extend(rows)

    key_pairs = defaultdict(list)
    for row in all_rows:
        key = (row["question_id"], row["system_name"])
        key_pairs[key].append(row)

    duplicate_claims = [
        key for key, rows in key_pairs.items()
        if len({row["reviewer"] for row in rows}) < 2
    ]
    kappas = []
    paired_labels: dict[str, tuple[list[str], list[str]]] = {}
    for key, rows in key_pairs.items():
        by_reviewer = {
            row["reviewer"]: str(row[args.agreement_field]).strip().lower()
            for row in rows
        }
        if len(by_reviewer) < 2:
            continue
        reviewers = sorted(by_reviewer)
        for reviewer_a, reviewer_b in combinations(reviewers, 2):
            pair_name = f"{reviewer_a}|{reviewer_b}"
            labels_a, labels_b = paired_labels.setdefault(pair_name, ([], []))
            labels_a.append(by_reviewer[reviewer_a])
            labels_b.append(by_reviewer[reviewer_b])
    for labels_a, labels_b in paired_labels.values():
        kappas.append(cohen_kappa(labels_a, labels_b))

    exact = Counter(
        (row["question_id"], row["system_name"], row["reviewer"])
        for row in all_rows
    )
    duplicate_rows = [key for key, count in exact.items() if count > 1]
    if duplicate_rows:
        raise ValueError("duplicate reviewer rows for question/system pair")

    counts = Counter()
    for row in all_rows:
        counts["option_correct_" + row["option_correct"]] += 1
        counts["evidence_supports_option_" + row["evidence_supports_option"]] += 1
        counts["unsupported_claim_present_" + row["unsupported_claim_present"]] += 1
        counts["should_abstain_" + row["should_abstain"]] += 1
        counts["citation_relevant_" + row["citation_points_to_relevant_passage"]] += 1

    report = {
        "review_row_count": len(all_rows),
        "unique_question_system_pairs": len(key_pairs),
        "pairs_with_only_one_reviewer": len(duplicate_claims),
        "agreement_field": args.agreement_field,
        "cohen_kappa_for_complete_paired_items": (
            sum(kappas) / len(kappas) if kappas else None
        ),
        "counts": dict(sorted(counts.items())),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
