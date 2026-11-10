"""Tests for reproducible MedQA scope annotation."""

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medragent.data import MedQARecord
from medragent.scope import annotate_record


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


if __name__ == "__main__":
    unittest.main()
