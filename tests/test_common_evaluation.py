"""Tests for the shared experiment evaluation protocol."""

from pathlib import Path
import json
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluations.common_evaluation import (
    evaluate,
    load_gold_evidence,
    load_manual_evidence,
)


class CommonEvaluationTests(unittest.TestCase):
    def rows(self):
        return [
            {
                "question_id": "q1",
                "system_name": "bm25_rag",
                "predicted_answer": "A",
                "expected_answer": "A",
                "is_correct": True,
                "retrieved_chunk_ids": ["c1", "c2"],
                "citations": ["c1"],
                "latency_ms": 10,
                "error": None,
                "scope": "psychiatry_relevant",
            },
            {
                "question_id": "q2",
                "system_name": "bm25_rag",
                "predicted_answer": None,
                "expected_answer": "B",
                "is_correct": None,
                "retrieved_chunk_ids": ["c3"],
                "citations": [],
                "latency_ms": 20,
                "error": "invalid response",
                "scope": "out_of_corpus",
            },
        ]

    def test_failures_count_in_all_question_accuracy(self):
        report = evaluate(self.rows())
        self.assertEqual(report["overall"]["accuracy_on_all_questions"], 0.5)
        self.assertEqual(len(report["overall"]["accuracy_95_ci"]), 2)
        self.assertEqual(report["overall"]["failure_rate"], 0.5)
        self.assertEqual(report["overall"]["citation_coverage_on_all"], 0.5)

    def test_retrieval_metrics_require_explicit_gold_labels(self):
        report = evaluate(self.rows(), {"q1": {"c1"}}, k=2)
        retrieval = report["overall"]["retrieval"]
        self.assertEqual(retrieval["labeled_queries"], 1)
        self.assertEqual(retrieval["recall_at_k"], 1.0)
        self.assertEqual(report["by_scope"]["out_of_corpus"]["retrieval"]["labeled_queries"], 0)

    def test_empty_support_labels_are_excluded_from_retrieval_metrics(self):
        labels = [{"question_id": "q1", "supporting_chunk_ids": []}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gold.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in labels) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_gold_evidence(path), {"q1": set()})

    def test_duplicate_result_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate result question_id"):
            evaluate(self.rows() + [self.rows()[0]])

    def test_manual_evidence_preserves_supported_and_unsupported_labels(self):
        labels = [
            {
                "question_id": "q1",
                "scope_label": "psychiatry_relevant_supported",
                "supporting_chunk_ids": ["c1"],
            },
            {
                "question_id": "q2",
                "scope_label": "out_of_corpus",
                "supporting_chunk_ids": [],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manual.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in labels) + "\n",
                encoding="utf-8",
            )
            gold, scopes = load_manual_evidence(path)
            self.assertEqual(gold["q1"], {"c1"})
            self.assertEqual(gold["q2"], set())
            self.assertEqual(scopes["q2"], "out_of_corpus")

    def test_manual_evidence_requires_valid_scope_label(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manual.jsonl"
            path.write_text(
                json.dumps({
                    "question_id": "q1",
                    "scope_label": "unknown",
                    "supporting_chunk_ids": [],
                }) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "invalid scope_label"):
                load_manual_evidence(path)


if __name__ == "__main__":
    unittest.main()
