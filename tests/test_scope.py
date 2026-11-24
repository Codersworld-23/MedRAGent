"""Tests for reproducible MedQA scope annotation."""

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medragent.data import MedQARecord
from medragent.scope import (
    ManualScopeAnnotation,
    annotate_record,
    load_manual_annotations,
    validate_manual_annotation_coverage,
)


def record(question: str) -> MedQARecord:
    return MedQARecord(
        id="q1",
        question=question,
        options={"A": "one", "B": "two"},
        answer_index=0,
        split="dev",
        source_file="test.jsonl",
        metadata={},
    )


class ScopeAnnotationTests(unittest.TestCase):
    def test_explicit_psychiatry_question_is_relevant(self) -> None:
        annotation = annotate_record(record("Which finding is typical of schizophrenia?"))
        self.assertEqual(annotation.scope, "psychiatry_relevant")
        self.assertTrue(annotation.evidence_expected)
        self.assertIn("schizophrenia", annotation.matched_terms)
        self.assertEqual(annotation.expected_answer, "A")
        self.assertEqual(annotation.options["A"], "one")

    def test_unmatched_question_is_out_of_corpus(self) -> None:
        annotation = annotate_record(record("Which artery supplies the kidney?"))
        self.assertEqual(annotation.scope, "out_of_corpus")
        self.assertFalse(annotation.evidence_expected)
        self.assertEqual(annotation.matched_terms, [])

    def test_annotations_are_versioned(self) -> None:
        annotation = annotate_record(record("What is bipolar disorder?"))
        self.assertEqual(annotation.annotation_version, "scope_v1")
        self.assertEqual(annotation.annotator, "lexical_candidate")

    def test_manual_annotation_requires_supported_evidence(self) -> None:
        from medragent.scope import MANUAL_SCOPE_LABELS

        self.assertIn("psychiatry_relevant_supported", MANUAL_SCOPE_LABELS)
        annotation = ManualScopeAnnotation(
            question_id="dev-000001",
            split="dev",
            scope_label="psychiatry_relevant_supported",
            supporting_chunk_ids=["chunk-1"],
            annotation_confidence="high",
            annotation_notes="Reviewed evidence",
            reviewer="reviewer@example",
            annotation_version="manual_scope_v1",
        )
        self.assertEqual(annotation.supporting_chunk_ids, ["chunk-1"])

    def test_manual_annotation_loader_rejects_unknown_question(self) -> None:
        import tempfile
        import json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question_path = root / "dev.jsonl"
            corpus_path = root / "corpus.jsonl"
            annotation_path = root / "manual.jsonl"
            question_path.write_text(
                json.dumps({
                    "id": "dev-000001",
                    "question": "Kidney question",
                    "options": {"A": "one", "B": "two"},
                    "answer_idx": 0,
                }) + "\n",
                encoding="utf-8",
            )
            corpus_path.write_text(
                json.dumps({
                    "id": "chunk-1",
                    "title": "DSM-5",
                    "content": "Evidence",
                }) + "\n",
                encoding="utf-8",
            )
            annotation_path.write_text(
                json.dumps({
                    "question_id": "unknown",
                    "split": "dev",
                    "scope_label": "out_of_corpus",
                    "supporting_chunk_ids": [],
                    "annotation_confidence": "high",
                    "annotation_notes": "Outside scope",
                    "reviewer": "reviewer@example",
                    "annotation_version": "manual_scope_v1",
                }) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unknown question_id"):
                load_manual_annotations(annotation_path, question_path, corpus_path)

    def test_manual_annotation_loader_rejects_unknown_chunk(self) -> None:
        import tempfile
        import json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question_path = root / "dev.jsonl"
            corpus_path = root / "corpus.jsonl"
            annotation_path = root / "manual.jsonl"
            question_path.write_text(
                json.dumps({
                    "id": "dev-000001",
                    "question": "Depression question",
                    "options": {"A": "one", "B": "two"},
                    "answer_idx": 0,
                }) + "\n",
                encoding="utf-8",
            )
            corpus_path.write_text(
                json.dumps({
                    "id": "chunk-1",
                    "title": "DSM-5",
                    "content": "Evidence",
                }) + "\n",
                encoding="utf-8",
            )
            annotation_path.write_text(
                json.dumps({
                    "question_id": "dev-000001",
                    "split": "dev",
                    "scope_label": "psychiatry_relevant_supported",
                    "supporting_chunk_ids": ["chunk-404"],
                    "annotation_confidence": "high",
                    "annotation_notes": "Reviewed evidence",
                    "reviewer": "reviewer@example",
                    "annotation_version": "manual_scope_v1",
                }) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unknown supporting_chunk_id"):
                load_manual_annotations(annotation_path, question_path, corpus_path)

    def test_empty_support_is_allowed_for_out_of_corpus(self) -> None:
        import tempfile
        import json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question_path = root / "dev.jsonl"
            corpus_path = root / "corpus.jsonl"
            annotation_path = root / "manual.jsonl"
            question_path.write_text(
                json.dumps({
                    "id": "dev-000001",
                    "question": "Kidney question",
                    "options": {"A": "one", "B": "two"},
                    "answer_idx": 0,
                }) + "\n",
                encoding="utf-8",
            )
            corpus_path.write_text(
                json.dumps({
                    "id": "chunk-1",
                    "title": "DSM-5",
                    "content": "Evidence",
                }) + "\n",
                encoding="utf-8",
            )
            annotation_path.write_text(
                json.dumps({
                    "question_id": "dev-000001",
                    "split": "dev",
                    "scope_label": "out_of_corpus",
                    "supporting_chunk_ids": [],
                    "annotation_confidence": "high",
                    "annotation_notes": "Outside corpus",
                    "reviewer": "reviewer@example",
                    "annotation_version": "manual_scope_v1",
                }) + "\n",
                encoding="utf-8",
            )
            annotations = load_manual_annotations(
                annotation_path, question_path, corpus_path
            )
            summary = validate_manual_annotation_coverage(
                annotations, question_path
            )
            self.assertEqual(summary["coverage"], 1.0)
            self.assertEqual(summary["missing_count"], 0)


if __name__ == "__main__":
    unittest.main()
