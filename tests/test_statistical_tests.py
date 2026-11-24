"""Tests for paired baseline significance utilities."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluations.statistical_tests import (
    bootstrap_accuracy_difference,
    holm_bonferroni,
    mcnemar_exact,
)


class StatisticalTests(unittest.TestCase):
    def test_mcnemar_contingency_counts(self) -> None:
        result = mcnemar_exact(
            both_correct=8,
            both_wrong=2,
            rag_only=5,
            llm_only=1,
        )
        self.assertEqual(result["total_questions"], 16)
        self.assertEqual(result["discordant"], 6)
        self.assertEqual(result["rag_only"], 5)
        self.assertEqual(result["llm_only"], 1)

    def test_mcnemar_no_discordance_is_not_significant(self) -> None:
        result = mcnemar_exact(10, 10, 0, 0)
        self.assertEqual(result["exact_mcnemar_p"], 1.0)

    def test_holm_bonferroni_is_monotonic_and_two_sided(self) -> None:
        adjusted = holm_bonferroni([
            ("bm25", 0.01),
            ("dense", 0.04),
            ("hybrid", 0.005),
        ])
        self.assertEqual(adjusted["hybrid"], 0.015)
        self.assertEqual(adjusted["bm25"], 0.02)
        self.assertEqual(adjusted["dense"], 0.04)

    def test_bootstrap_difference_is_paired_and_deterministic(self) -> None:
        baseline = [True, True, False, False]
        system = [True, False, True, False]
        first = bootstrap_accuracy_difference(baseline, system)
        second = bootstrap_accuracy_difference(baseline, system)
        self.assertEqual(first, second)
        self.assertEqual(first["accuracy_difference"], 0.0)
        self.assertEqual(len(first["bootstrap_95_ci"]), 2)


if __name__ == "__main__":
    unittest.main()
