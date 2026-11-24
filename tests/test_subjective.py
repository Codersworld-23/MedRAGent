"""Tests for subjective evidence-grounded answers and abstention."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medragent.bm25 import RetrievedChunk
from medragent.subjective import (
    evaluate_abstention,
    parse_subjective_response,
    run_subjective_question,
)


class FakeSubjectiveClient:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str, response_format=None) -> str:
        return self.response


def chunk(score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="InternalMed_Harrison_1",
        title="Internal Medicine",
        text="Diabetic ketoacidosis is a serious metabolic emergency. " * 3,
        rank=1,
        score=score,
        source_file="InternalMed_Harrison.jsonl",
        textbook_name="InternalMed_Harrison",
        retrieval_method="bm25",
    )


class SubjectiveTests(unittest.TestCase):
    def test_no_evidence_abstains(self) -> None:
        abstain, reason = evaluate_abstention([], method="bm25")
        self.assertTrue(abstain)
        self.assertIn("No textbook chunks", reason)

    def test_bm25_positive_score_is_adequate(self) -> None:
        abstain, reason = evaluate_abstention([chunk(1.0)], method="bm25")
        self.assertFalse(abstain)
        self.assertIsNone(reason)

    def test_dense_below_threshold_abstains(self) -> None:
        abstain, reason = evaluate_abstention(
            [chunk(0.10)],
            method="dense",
            dense_threshold=0.25,
        )
        self.assertTrue(abstain)
        self.assertIn("threshold", reason)

    def test_valid_subjective_response_is_parsed(self) -> None:
        parsed = parse_subjective_response(
            '{"answer":"It is a metabolic emergency.",'
            '"evidence_status":"supported",'
            '"citations":["InternalMed_Harrison_1"],'
            '"limitations":"General educational information.",'
            '"abstention_reason":null}',
            {"InternalMed_Harrison_1"},
        )
        self.assertEqual(parsed["evidence_status"], "supported")
        self.assertEqual(parsed["citations"], ["InternalMed_Harrison_1"])

    def test_unknown_citation_is_rejected(self) -> None:
        parsed = parse_subjective_response(
            '{"answer":"Answer","evidence_status":"supported",'
            '"citations":["unknown"],"limitations":"","abstention_reason":null}',
            {"InternalMed_Harrison_1"},
        )
        self.assertIsNone(parsed)

    def test_abstention_preserves_result_fields(self) -> None:
        result = run_subjective_question(
            "Explain diabetic ketoacidosis",
            method="bm25",
            retrieved=[],
            client=FakeSubjectiveClient("{}"),
            model="fake:test",
        )
        self.assertEqual(result.evidence_status, "abstained")
        self.assertEqual(result.citations, [])
        self.assertIn("No textbook chunks", result.abstention_reason)


if __name__ == "__main__":
    unittest.main()
