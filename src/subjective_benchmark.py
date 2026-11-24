"""BioASQ-to-textbook subjective benchmark construction and validation.

The exported records deliberately separate BioASQ's PubMed evidence from
candidate evidence retrieved from the MedRAG textbook corpus.  A candidate
chunk is *not* gold evidence until a qualified reviewer approves it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


BENCHMARK_VERSION = "subjective_textbook_v1"
REVIEW_STATUSES = {"unreviewed", "model_reviewed", "expert_reviewed"}
EVIDENCE_STATUSES = {"supported", "partial", "unsupported", "expected_abstention"}


class SubjectiveBenchmarkError(ValueError):
    """Raised when a subjective benchmark record is malformed."""


def load_bioasq_summary_questions(path: str | Path) -> list[dict[str, Any]]:
    """Load BioASQ summary questions with non-empty ideal answers."""
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"BioASQ file not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise SubjectiveBenchmarkError(f"{source}: invalid JSON: {exc.msg}") from exc
    questions = raw.get("questions") if isinstance(raw, dict) else None
    if not isinstance(questions, list):
        raise SubjectiveBenchmarkError(f"{source}: expected a 'questions' list")

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for position, item in enumerate(questions, start=1):
        if not isinstance(item, dict) or item.get("type") != "summary":
            continue
        question_id = item.get("id")
        question = item.get("body")
        ideal_answers = item.get("ideal_answer")
        if not isinstance(question_id, str) or not question_id.strip():
            raise SubjectiveBenchmarkError(f"{source}: summary item {position} has no id")
        if question_id in seen_ids:
            raise SubjectiveBenchmarkError(f"{source}: duplicate BioASQ id '{question_id}'")
        if not isinstance(question, str) or not question.strip():
            raise SubjectiveBenchmarkError(f"{source}: {question_id} has no question body")
        if not isinstance(ideal_answers, list) or not all(
            isinstance(answer, str) and answer.strip() for answer in ideal_answers
        ):
            raise SubjectiveBenchmarkError(f"{source}: {question_id} has no ideal answer")
        seen_ids.add(question_id)
        records.append(item)
    if not records:
        raise SubjectiveBenchmarkError(f"{source}: no valid BioASQ summary questions found")
    return records


def _pubmed_evidence(question: dict[str, Any]) -> list[dict[str, str]]:
    """Keep external gold evidence for provenance, never as textbook gold."""
    snippets = question.get("snippets", [])
    if not isinstance(snippets, list):
        return []
    evidence: list[dict[str, str]] = []
    for snippet in snippets:
        if not isinstance(snippet, dict):
            continue
        text = snippet.get("text")
        document = snippet.get("document")
        if isinstance(text, str) and text.strip():
            evidence.append(
                {
                    "document": document.strip() if isinstance(document, str) else "",
                    "text": text.strip(),
                }
            )
    return evidence


def build_unreviewed_records(
    questions: Iterable[dict[str, Any]], *, source_name: str
) -> list[dict[str, Any]]:
    """Convert BioASQ summaries into review-ready textbook benchmark records."""
    output: list[dict[str, Any]] = []
    for question in questions:
        ideal_answers = [answer.strip() for answer in question["ideal_answer"]]
        output.append(
            {
                "benchmark_version": BENCHMARK_VERSION,
                "question_id": f"bioasq13b_summary_{question['id']}",
                "question": question["body"].strip(),
                "question_type": "summary",
                "domain": None,
                "reference_answers": ideal_answers,
                "reference_answer": ideal_answers[0],
                "source_dataset": source_name,
                "source_question_id": question["id"],
                "external_gold_evidence": _pubmed_evidence(question),
                "candidate_textbook_evidence": [],
                "supporting_chunk_ids": [],
                "expected_evidence_status": None,
                "review_status": "unreviewed",
                "reviewer": None,
                "reviewer_confidence": None,
                "review_notes": (
                    "BioASQ answer and PubMed evidence are external references. "
                    "Review textbook candidates before assigning support labels."
                ),
            }
        )
    return output


def validate_records(
    records: Iterable[dict[str, Any]], *, known_chunk_ids: set[str] | None = None
) -> dict[str, int]:
    """Validate benchmark records and prohibit unreviewed gold evidence."""
    count = 0
    reviewed = 0
    supporting = 0
    seen_ids: set[str] = set()
    for record in records:
        count += 1
        if not isinstance(record, dict):
            raise SubjectiveBenchmarkError(f"record {count}: expected JSON object")
        question_id = record.get("question_id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise SubjectiveBenchmarkError(f"record {count}: missing question_id")
        if question_id in seen_ids:
            raise SubjectiveBenchmarkError(f"record {count}: duplicate question_id '{question_id}'")
        seen_ids.add(question_id)
        if not isinstance(record.get("question"), str) or not record["question"].strip():
            raise SubjectiveBenchmarkError(f"{question_id}: missing question")
        answers = record.get("reference_answers")
        if not isinstance(answers, list) or not answers or not all(
            isinstance(answer, str) and answer.strip() for answer in answers
        ):
            raise SubjectiveBenchmarkError(f"{question_id}: reference_answers must be non-empty text")
        status = record.get("review_status")
        if status not in REVIEW_STATUSES:
            raise SubjectiveBenchmarkError(f"{question_id}: invalid review_status")
        evidence_status = record.get("expected_evidence_status")
        chunk_ids = record.get("supporting_chunk_ids")
        if not isinstance(chunk_ids, list) or not all(isinstance(item, str) for item in chunk_ids):
            raise SubjectiveBenchmarkError(f"{question_id}: supporting_chunk_ids must be a list of strings")
        if len(chunk_ids) != len(set(chunk_ids)):
            raise SubjectiveBenchmarkError(f"{question_id}: duplicate supporting chunk IDs")
        if known_chunk_ids is not None and any(item not in known_chunk_ids for item in chunk_ids):
            raise SubjectiveBenchmarkError(f"{question_id}: contains unknown textbook chunk ID")
        if status == "unreviewed" and (chunk_ids or evidence_status is not None):
            raise SubjectiveBenchmarkError(
                f"{question_id}: unreviewed records cannot contain gold textbook evidence"
            )
        if status in {"model_reviewed", "expert_reviewed"}:
            reviewed += 1
            if evidence_status not in EVIDENCE_STATUSES:
                raise SubjectiveBenchmarkError(f"{question_id}: reviewed record needs evidence status")
            if evidence_status in {"supported", "partial"} and not chunk_ids:
                raise SubjectiveBenchmarkError(f"{question_id}: supported record needs chunk IDs")
            if evidence_status in {"unsupported", "expected_abstention"} and chunk_ids:
                raise SubjectiveBenchmarkError(f"{question_id}: unsupported record cannot have chunk IDs")
        supporting += len(chunk_ids)
    if not count:
        raise SubjectiveBenchmarkError("benchmark contains no records")
    return {"records": count, "reviewed": reviewed, "supporting_chunk_ids": supporting}
