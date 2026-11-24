"""Export a bounded claim-level grounding review sample."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medragent.scope import load_annotations, load_manual_annotations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", nargs="+", type=Path)
    parser.add_argument("--annotations", type=Path, default=PROJECT_ROOT / "experiments" / "annotations" / "medqa_scope_dev.jsonl")
    parser.add_argument("--manual-annotations", type=Path)
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "datasets" / "MedRAG" / "chunk" / "Psichiatry_DSM-5.jsonl")
    parser.add_argument("--questions", type=Path, default=PROJECT_ROOT / "datasets" / "MedQA" / "questions" / "US" / "dev.jsonl")
    parser.add_argument("--sample-per-group", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    annotations = load_annotations(args.annotations)
    manual = None
    if args.manual_annotations:
        manual = load_manual_annotations(
            args.manual_annotations,
            args.questions,
            args.corpus,
        )
    rng = random.Random(args.seed)
    for result_path in args.result:
        rows = [
            json.loads(line)
            for line in result_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        groups = {
            "supported": [],
            "unsupported": [],
            "out_of_corpus": [],
        }
        for row in rows:
            if manual:
                label = manual[row["question_id"]].scope_label
                if label == "psychiatry_relevant_supported":
                    scope = "supported"
                elif label == "psychiatry_relevant_unsupported":
                    scope = "unsupported"
                else:
                    scope = "out_of_corpus"
            else:
                annotation = annotations.get(row["question_id"])
                scope = "supported" if (
                    annotation and annotation.scope == "psychiatry_relevant"
                ) else "out_of_corpus"
            if scope in groups:
                groups[scope].append(row)
        selected = []
        for scope, rows_in_group in groups.items():
            selected.extend(
                rng.sample(
                    rows_in_group,
                    min(args.sample_per_group, len(rows_in_group)),
                )
            )
        output = args.output.with_name(
            args.output.stem + "_" + result_path.stem + args.output.suffix
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="\n") as handle:
            for row in selected:
                handle.write(json.dumps({
                    "question_id": row["question_id"],
                    "system_name": row["system_name"],
                    "question": row["question"],
                    "predicted_answer": row["predicted_answer"],
                    "expected_answer": row["expected_answer"],
                    "retrieved_chunk_ids": row["retrieved_chunk_ids"],
                    "citations": row["citations"],
                    "scope": (
                        manual[row["question_id"]].scope_label
                        if manual
                        else annotations[row["question_id"]].scope
                    ),
                    "option_correct": "",
                    "evidence_supports_option": "",
                    "unsupported_claim_present": "",
                    "should_abstain": "",
                    "citation_points_to_relevant_passage": "",
                    "reviewer": "",
                    "review_notes": "",
                }, ensure_ascii=False) + "\n")
        print(f"Wrote {len(selected)} grounding-review rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
