"""Tests for frozen test baseline safeguards."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_publication_package import package_results, sha256


class FrozenTestRunnerTests(unittest.TestCase):
    def test_package_hashes_test_result_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bm25_test.jsonl"
            path.write_text(
                json.dumps({
                    "question_id": "test-000001",
                    "system_name": "bm25_rag",
                }) + "\n",
                encoding="utf-8",
            )
            packages = package_results(Path(directory))
            self.assertEqual(packages[0]["row_count"], 1)
            self.assertEqual(packages[0]["system_name"], "bm25_rag")
            self.assertEqual(len(packages[0]["sha256"]), 64)

    def test_sha256_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "file.txt"
            path.write_text("test", encoding="utf-8")
            self.assertEqual(sha256(path), sha256(path))
            self.assertEqual(len(sha256(path)), 64)


if __name__ == "__main__":
    unittest.main()
