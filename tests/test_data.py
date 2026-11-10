"""Focused tests for the Step 1 dataset loaders."""

from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medragent.data import (  # noqa: E402
    DatasetValidationError,
    load_medqa_jsonl,
    load_text_chunks_jsonl,
)


class DatasetLoaderTests(unittest.TestCase):
    def test_loads_real_medqa_dev_schema(self) -> None:
        path = PROJECT_ROOT / "datasets" / "MedQA" / "questions" / "US" / "dev.jsonl"
        records = load_medqa_jsonl(path)
        self.assertEqual(len(records), 1272)
        self.assertEqual(records[0].split, "dev")
        self.assertTrue(records[0].id.startswith("dev-"))
        self.assertEqual(set(records[0].options), {"A", "B", "C", "D", "E"})
        self.assertIn(records[0].answer_index, range(5))
        self.assertEqual(records[0].answer_letter, "D")

    def test_loads_real_psychiatry_chunks(self) -> None:
        path = (
            PROJECT_ROOT
            / "datasets"
            / "MedRAG"
            / "chunk"
            / "Psichiatry_DSM-5.jsonl"
        )
        records = load_text_chunks_jsonl(path)
        self.assertEqual(len(records), 4057)
        self.assertTrue(records[0].chunk_id)
        self.assertTrue(records[0].text)
        self.assertEqual(records[0].source_file, str(path))

    def test_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.jsonl"
            path.write_text(
                '{"id":"q1","question":"q","options":{"A":"a"},"answer_idx":"A"}\n'
                '{"id":"q1","question":"q2","options":{"A":"a"},"answer_idx":"A"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DatasetValidationError, "duplicate id"):
                load_medqa_jsonl(path)

    def test_rejects_invalid_answer_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.jsonl"
            path.write_text(
                '{"id":"q1","question":"q","options":{"A":"a"},"answer_idx":1}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DatasetValidationError, "outside"):
                load_medqa_jsonl(path)


if __name__ == "__main__":
    unittest.main()
