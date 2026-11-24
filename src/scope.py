"""Reproducible scope annotations for the Psychiatry evidence corpus."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .data import MedQARecord


_PSYCHIATRY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("psychiatry", r"\bpsychiatr\w*\b"),
    ("mental health", r"\bmental\s+health\b"),
    ("depression", r"\bdepress\w*\b"),
    ("anxiety", r"\banx\w*\b"),
    ("schizophrenia", r"\bschizophren\w*\b"),
    ("psychosis", r"\bpsychos\w*\b"),
    ("bipolar", r"\bbipolar\b"),
    ("mania", r"\bmanic\b|\bmania\b"),
    ("obsessive compulsive", r"\bobsessive[-\s]+compulsive\b|\bOCD\b"),
    ("panic", r"\bpanic\s+attack\w*\b|\bpanic\s+disorder\b"),
    ("phobia", r"\bphobi\w*\b"),
    ("post-traumatic stress", r"\bpost[-\s]?traumatic\s+stress\b|\bPTSD\b"),
    ("personality disorder", r"\bpersonality\s+disorder\w*\b"),
    ("autism", r"\bautis\w*\b"),
    ("attention deficit", r"\bADHD\b|\battention[-\s]+deficit\b"),
    ("delirium", r"\bdeliri\w*\b"),
    ("suicide", r"\bsuicid\w*\b"),
    ("hallucination", r"\bhallucination\w*\b"),
    ("delusion", r"\bdelusion\w*\b"),
    ("psychotherapy", r"\bpsychotherap\w*\b|\bcognitive[-\s]+behavioral\s+therapy\b"),
    ("psychiatric medication", r"\bantipsychotic\w*\b|\bantidepressant\w*\b"),
    ("mood stabilizer", r"\bmood\s+stabilizer\w*\b"),
)


@dataclass(frozen=True)
class ScopeAnnotation:
    """One versioned, auditable annotation for a benchmark question."""

    question_id: str
    question: str
    options: dict[str, str]
    expected_answer: str | None
    scope: str
    evidence_expected: bool
    topic: str
    reason: str
    annotator: str
    annotation_version: str
    matched_terms: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


MANUAL_SCOPE_LABELS = (
    "psychiatry_relevant_supported",
    "psychiatry_relevant_unsupported",
    "out_of_corpus",
)
MANUAL_ANNOTATION_CONFIDENCE = ("high", "medium", "low")


@dataclass(frozen=True)
class ManualScopeAnnotation:
    """Human-reviewed scope and supporting-evidence annotation."""

    question_id: str
    split: str
    scope_label: str
    supporting_chunk_ids: list[str]
    annotation_confidence: str
    annotation_notes: str
    reviewer: str
    annotation_version: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def annotate_record(
    record: MedQARecord,
    *,
    annotator: str = "lexical_candidate",
    annotation_version: str = "scope_v1",
) -> ScopeAnnotation:
    """Create a conservative, reproducible candidate scope label.

    This is a triage label, not a substitute for expert manual review. Questions
    whose wording explicitly contains a psychiatry-related term are marked
    relevant; all other questions are marked out_of_corpus for the initial
    stratification and can be revised through the versioned annotation file.
    """
    text = f"{record.question} {' '.join(record.options.values())}"
    matched_terms = [
        term for term, pattern in _PSYCHIATRY_PATTERNS
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]
    if matched_terms:
        return ScopeAnnotation(
            question_id=record.id,
            question=record.question,
            options=dict(record.options),
            expected_answer=record.answer_letter,
            scope="psychiatry_relevant",
            evidence_expected=True,
            topic="psychiatry",
            reason="Explicit psychiatry-related term matched in question or options",
            annotator=annotator,
            annotation_version=annotation_version,
            matched_terms=matched_terms,
        )
    return ScopeAnnotation(
        question_id=record.id,
        question=record.question,
        options=dict(record.options),
        expected_answer=record.answer_letter,
        scope="out_of_corpus",
        evidence_expected=False,
        topic="general_medical",
        reason="No explicit psychiatry-related term matched in question or options",
        annotator=annotator,
        annotation_version=annotation_version,
        matched_terms=[],
    )


def annotate_records(records: Iterable[MedQARecord]) -> list[ScopeAnnotation]:
    """Annotate records in their existing deterministic order."""
    return [annotate_record(record) for record in records]


def write_annotations(path: str | Path, annotations: Iterable[ScopeAnnotation]) -> None:
    """Write annotations as UTF-8 JSONL."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for annotation in annotations:
            handle.write(json.dumps(annotation.as_dict(), ensure_ascii=False) + "\n")


def load_annotations(path: str | Path) -> dict[str, ScopeAnnotation]:
    """Load annotations keyed by question ID and reject duplicate or malformed rows."""
    annotations: dict[str, ScopeAnnotation] = {}
    annotation_path = Path(path)
    try:
        lines = annotation_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{annotation_path}:{line_number}: invalid JSON") from exc
        required = (
            "question_id",
            "question",
            "options",
            "expected_answer",
            "scope",
            "evidence_expected",
            "topic",
            "reason",
            "annotator",
            "annotation_version",
            "matched_terms",
        )
        if any(field not in raw for field in required):
            raise ValueError(f"{annotation_path}:{line_number}: missing annotation field")
        question_id = raw["question_id"]
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"{annotation_path}:{line_number}: invalid question_id")
        if question_id in annotations:
            raise ValueError(f"{annotation_path}:{line_number}: duplicate question_id '{question_id}'")
        annotations[question_id] = ScopeAnnotation(
            question_id=question_id,
            question=str(raw["question"]),
            options=dict(raw["options"]),
            expected_answer=(
                str(raw["expected_answer"])
                if raw["expected_answer"] is not None
                else None
            ),
            scope=str(raw["scope"]),
            evidence_expected=bool(raw["evidence_expected"]),
            topic=str(raw["topic"]),
            reason=str(raw["reason"]),
            annotator=str(raw["annotator"]),
            annotation_version=str(raw["annotation_version"]),
            matched_terms=list(raw["matched_terms"]),
        )
    return annotations


def load_manual_annotations(
    path: str | Path,
    question_path: str | Path,
    corpus_path: str | Path,
) -> dict[str, ManualScopeAnnotation]:
    """Load and validate a manual scope/evidence annotation sidecar."""
    from .data import load_medqa_jsonl, load_text_chunks_jsonl

    annotation_path = Path(path)
    question_records = load_medqa_jsonl(question_path)
    corpus_records = load_text_chunks_jsonl(corpus_path)
    question_ids = {record.id for record in question_records}
    chunk_ids = {record.chunk_id for record in corpus_records}

    annotations: dict[str, ManualScopeAnnotation] = {}
    try:
        lines = annotation_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Manual annotation file not found: {annotation_path}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{annotation_path}:{line_number}: invalid JSON"
            ) from exc

        required = (
            "question_id",
            "split",
            "scope_label",
            "supporting_chunk_ids",
            "annotation_confidence",
            "annotation_notes",
            "reviewer",
            "annotation_version",
        )
        missing = [field for field in required if field not in raw]
        if missing:
            raise ValueError(
                f"{annotation_path}:{line_number}: missing fields: {', '.join(missing)}"
            )

        question_id = raw["question_id"]
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"{annotation_path}:{line_number}: invalid question_id")
        if question_id in annotations:
            raise ValueError(
                f"{annotation_path}:{line_number}: duplicate question_id '{question_id}'"
            )
        if question_id not in question_ids:
            raise ValueError(
                f"{annotation_path}:{line_number}: unknown question_id '{question_id}'"
            )
        if raw["split"] != "dev":
            raise ValueError(
                f"{annotation_path}:{line_number}: split must be 'dev'"
            )

        scope_label = raw["scope_label"]
        if not isinstance(scope_label, str) or not scope_label:
            raise ValueError(
                f"{annotation_path}:{line_number}: scope_label is required"
            )
        if scope_label not in MANUAL_SCOPE_LABELS:
            raise ValueError(
                f"{annotation_path}:{line_number}: invalid scope_label '{scope_label}'"
            )
        confidence = raw["annotation_confidence"]
        if confidence not in MANUAL_ANNOTATION_CONFIDENCE:
            raise ValueError(
                f"{annotation_path}:{line_number}: invalid annotation_confidence "
                f"'{confidence}'"
            )

        supporting = raw["supporting_chunk_ids"]
        if not isinstance(supporting, list):
            raise ValueError(
                f"{annotation_path}:{line_number}: supporting_chunk_ids must be a list"
            )
        supporting_ids = [str(value) for value in supporting]
        if len(supporting_ids) != len(set(supporting_ids)):
            raise ValueError(
                f"{annotation_path}:{line_number}: duplicate supporting_chunk_id"
            )
        unknown_chunks = sorted(set(supporting_ids) - chunk_ids)
        if unknown_chunks:
            raise ValueError(
                f"{annotation_path}:{line_number}: unknown supporting_chunk_id(s): "
                f"{', '.join(unknown_chunks)}"
            )
        if scope_label == "psychiatry_relevant_supported" and not supporting_ids:
            raise ValueError(
                f"{annotation_path}:{line_number}: supported labels require at least "
                "one supporting_chunk_id"
            )
        if scope_label != "psychiatry_relevant_supported" and supporting_ids:
            raise ValueError(
                f"{annotation_path}:{line_number}: unsupported and out-of-corpus "
                "labels must use an empty supporting_chunk_ids list"
            )

        reviewer = raw["reviewer"]
        version = raw["annotation_version"]
        notes = raw["annotation_notes"]
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError(f"{annotation_path}:{line_number}: reviewer is required")
        if not isinstance(version, str) or not version.strip():
            raise ValueError(
                f"{annotation_path}:{line_number}: annotation_version is required"
            )
        if not isinstance(notes, str):
            raise ValueError(
                f"{annotation_path}:{line_number}: annotation_notes must be text"
            )

        annotations[question_id] = ManualScopeAnnotation(
            question_id=question_id,
            split="dev",
            scope_label=scope_label,
            supporting_chunk_ids=supporting_ids,
            annotation_confidence=confidence,
            annotation_notes=notes,
            reviewer=reviewer.strip(),
            annotation_version=version.strip(),
        )
    return annotations


def write_manual_annotations(
    path: str | Path,
    annotations: Iterable[ManualScopeAnnotation],
) -> None:
    """Write a review-ready manual annotation JSONL file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for annotation in annotations:
            handle.write(
                json.dumps(annotation.as_dict(), ensure_ascii=False) + "\n"
            )


def validate_manual_annotation_coverage(
    annotations: dict[str, ManualScopeAnnotation],
    question_path: str | Path,
) -> dict[str, Any]:
    """Check annotation coverage and return summary metadata."""
    from .data import load_medqa_jsonl

    records = load_medqa_jsonl(question_path)
    question_ids = [record.id for record in records]
    missing = [question_id for question_id in question_ids if question_id not in annotations]
    unexpected = sorted(set(annotations) - set(question_ids))
    if unexpected:
        raise ValueError(
            f"annotations contain question IDs outside the input split: "
            f"{', '.join(unexpected)}"
        )
    counts = Counter(annotation.scope_label for annotation in annotations.values())
    reviewers = sorted({annotation.reviewer for annotation in annotations.values()})
    versions = sorted({annotation.annotation_version for annotation in annotations.values()})
    return {
        "question_count": len(question_ids),
        "annotation_count": len(annotations),
        "missing_count": len(missing),
        "missing_question_ids": missing,
        "coverage": (
            len(annotations) / len(question_ids) if question_ids else 0.0
        ),
        "scope_label_counts": dict(sorted(counts.items())),
        "reviewers": reviewers,
        "annotation_versions": versions,
    }
