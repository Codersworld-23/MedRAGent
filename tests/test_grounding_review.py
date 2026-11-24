"""Tests for claim-level grounding review validation."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_grounding_review import cohen_kappa


class GroundingReviewTests(unittest.TestCase):
    def test_complete_agreement_is_one(self) -> None:
        labels = ["yes", "no", "yes", "uncertain"]
        self.assertEqual(cohen_kappa(labels, labels), 1.0)

    def test_opposed_labels_have_negative_or_low_kappa(self) -> None:
        kappa = cohen_kappa(
            ["yes", "yes", "no", "no"],
            ["no", "no", "yes", "yes"],
        )
        self.assertLess(kappa, 0.0)

    def test_empty_reviews_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            cohen_kappa([], [])


if __name__ == "__main__":
    unittest.main()
