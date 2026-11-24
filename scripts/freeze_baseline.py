"""Freeze and preflight the MedQA-US baseline configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medragent.data import load_medqa_jsonl, load_text_chunks_jsonl
from medragent.scope import load_annotations


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_result_ids(path: Path) -> list[str]:
    ids = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                ids.append(row["question_id"])
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate question IDs")
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--model", default="llama3.2:latest")
    parser.add_argument("--embedding-model", default="nomic-embed-text:latest")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rrf-constant", type=int, default=60)
    parser.add_argument("--prompt-version", default="baseline_v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-outputs", nargs="*", type=Path)
    parser.add_argument("--require-frozen-test", action="store_true")
    parser.add_argument("--manual-annotations", type=Path)
    args = parser.parse_args()
    if args.temperature < 0 or args.top_k <= 0 or args.rrf_constant <= 0:
        parser.error("temperature, top-k, and rrf-constant are invalid")

    questions = PROJECT_ROOT / "datasets" / "MedQA" / "questions" / "US" / f"{args.split}.jsonl"
    corpus = PROJECT_ROOT / "datasets" / "MedRAG" / "chunk" / "Psichiatry_DSM-5.jsonl"
    annotations = PROJECT_ROOT / "experiments" / "annotations" / "medqa_scope_dev.jsonl"
    records = load_medqa_jsonl(questions)
    chunks = load_text_chunks_jsonl(corpus)
    if args.split == "dev":
        scope = load_annotations(annotations)
        missing = [record.id for record in records if record.id not in scope]
        if missing:
            raise ValueError(f"{len(missing)} questions lack scope annotations")
    if args.manual_annotations:
        if args.split != "dev":
            raise ValueError("--manual-annotations is supported for dev only")
        from medragent.scope import load_manual_annotations

        manual = load_manual_annotations(
            args.manual_annotations,
            questions,
            corpus,
        )
        if len(manual) != len(records):
            raise ValueError(
                f"manual annotations cover {len(manual)} of {len(records)} dev questions"
            )

    expected_ids = [record.id for record in records]
    output_checks = []
    for path in args.check_outputs or []:
        ids = load_result_ids(path)
        output_checks.append({
            "path": str(path),
            "rows": len(ids),
            "ordered_ids_match": ids == expected_ids,
            "missing_count": len(set(expected_ids) - set(ids)),
            "unexpected_count": len(set(ids) - set(expected_ids)),
        })
        if ids != expected_ids:
            raise ValueError(f"{path}: question IDs differ from frozen {args.split} split")

    frozen = {
        "freeze_status": "ready" if args.split == "dev" else "test_config_frozen",
        "split": args.split,
        "question_count": len(records),
        "question_file": str(questions),
        "question_sha256": sha256(questions),
        "corpus_file": str(corpus),
        "corpus_sha256": sha256(corpus),
        "corpus_chunk_count": len(chunks),
        "scope_annotation_file": str(annotations),
        "scope_annotation_sha256": sha256(annotations),
        "manual_annotation_file": (
            str(args.manual_annotations) if args.manual_annotations else None
        ),
        "manual_annotation_sha256": (
            sha256(args.manual_annotations) if args.manual_annotations else None
        ),
        "generator_model": args.model,
        "embedding_model": args.embedding_model,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "rrf_constant": args.rrf_constant,
        "prompt_version": args.prompt_version,
        "output_checks": output_checks,
    }
    if args.require_frozen_test and args.split != "test":
        raise ValueError("--require-frozen-test requires --split test")
    text = json.dumps(frozen, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
