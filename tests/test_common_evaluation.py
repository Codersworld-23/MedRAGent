"""Tests for the shared experiment evaluation protocol."""

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluations.common_evaluation import evaluate, load_gold_evidence


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


if __name__ == "__main__":
    unittest.main()
