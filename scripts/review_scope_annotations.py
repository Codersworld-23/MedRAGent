"""Export or validate manual MedQA-US scope/evidence annotations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medragent.data import load_medqa_jsonl
from medragent.scope import (
    ManualScopeAnnotation,
    load_annotations,
    load_manual_annotations,
    validate_manual_annotation_coverage,
    write_manual_annotations,
)


def export_template(
    questions_path: Path,
    lexical_path: Path,
    output_path: Path,
    *,
    psychiatrists_only: bool,
    reviewers: str,
    version: str,
) -> list[ManualScopeAnnotation]:
    records = load_medqa_jsonl(questions_path)
    lexical = load_annotations(lexical_path)
    annotations = []
    for record in records:
        candidate = lexical.get(record.id)
        if psychiatrists_only and (
            candidate is None or candidate.scope != "psychiatry_relevant"
        ):
            continue
        annotations.append(
            ManualScopeAnnotation(
                question_id=record.id,
                split="dev",
                scope_label="",
                supporting_chunk_ids=[],
                annotation_confidence="",
                annotation_notes="",
                reviewer=reviewers,
                annotation_version=version,
            )
        )
    write_manual_annotations(output_path, annotations)
    return annotations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="Create a review-friendly template")
    export.add_argument("--questions", type=Path, default=PROJECT_ROOT / "datasets" / "MedQA" / "questions" / "US" / "dev.jsonl")
    export.add_argument("--lexical", type=Path, default=PROJECT_ROOT / "experiments" / "annotations" / "medqa_scope_dev.jsonl")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--version", default="manual_scope_v1")
    export.add_argument("--reviewers", default="pending_reviewer")
    export.add_argument("--psychiatry-candidates-only", action="store_true")

    validate = subparsers.add_parser("validate", help="Validate and summarize a manual sidecar")
    validate.add_argument("--input", type=Path, required=True)
    validate.add_argument("--questions", type=Path, default=PROJECT_ROOT / "datasets" / "MedQA" / "questions" / "US" / "dev.jsonl")
    validate.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "datasets" / "MedRAG" / "chunk" / "Psichiatry_DSM-5.jsonl")
    validate.add_argument("--require-complete", action="store_true")
    validate.add_argument("--output", type=Path)

    args = parser.parse_args()
    if args.command == "export":
        rows = export_template(
            args.questions,
            args.lexical,
            args.output,
            psychiatrists_only=args.psychiatry_candidates_only,
            reviewers=args.reviewers,
            version=args.version,
        )
        print(f"Wrote {len(rows)} review rows to {args.output}")
        return 0

    annotations = load_manual_annotations(args.input, args.questions, args.corpus)
    summary = validate_manual_annotation_coverage(annotations, args.questions)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.require_complete and summary["missing_count"]:
        print(
            f"ERROR: {summary['missing_count']} questions are not manually annotated",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
