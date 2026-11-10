"""Reproducible scope annotations for the Psychiatry evidence corpus."""

from __future__ import annotations

import json
import re
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
