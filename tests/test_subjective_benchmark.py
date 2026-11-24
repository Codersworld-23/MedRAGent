"""Tests for the BioASQ-to-textbook subjective benchmark boundary."""

import unittest

from src.subjective_benchmark import SubjectiveBenchmarkError, build_unreviewed_records, validate_records


def bioasq_question():
    return {
        "id": "bio-1",
        "body": "Explain a medical condition.",
        "ideal_answer": ["A reference paragraph."],
        "snippets": [{"document": "pubmed:1", "text": "External evidence."}],
    }


class SubjectiveBenchmarkTests(unittest.TestCase):
    def test_export_is_unreviewed_and_has_no_textbook_gold(self):
        record = build_unreviewed_records([bioasq_question()], source_name="BioASQ-13b")[0]
        self.assertEqual(record["review_status"], "unreviewed")
        self.assertEqual(record["supporting_chunk_ids"], [])
        self.assertEqual(validate_records([record])["records"], 1)

    def test_unreviewed_cannot_claim_gold_textbook_evidence(self):
        record = build_unreviewed_records([bioasq_question()], source_name="BioASQ-13b")[0]
        record["supporting_chunk_ids"] = ["InternalMed_Harrison_1"]
        with self.assertRaises(SubjectiveBenchmarkError):
            validate_records([record])

    def test_reviewed_supported_record_requires_real_chunk_id(self):
        record = build_unreviewed_records([bioasq_question()], source_name="BioASQ-13b")[0]
        record.update(
            review_status="model_reviewed",
            expected_evidence_status="supported",
            supporting_chunk_ids=["InternalMed_Harrison_1"],
        )
        self.assertEqual(
            validate_records([record], known_chunk_ids={"InternalMed_Harrison_1"})["reviewed"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
