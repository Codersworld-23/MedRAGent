"""Offline tests for the Step 2 LLM-only baseline."""

from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medragent.data import load_medqa_jsonl
from medragent.llm_only import build_prompt, parse_answer, run_record, write_results
from medragent.scope import annotate_record


class FakeClient:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.response


class LLMOnlyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.record = load_medqa_jsonl(
            PROJECT_ROOT / "datasets" / "MedQA" / "questions" / "US" / "dev.jsonl"
        )[0]

    def test_prompt_contains_question_and_no_retrieval_context(self) -> None:
        prompt = build_prompt(self.record)
        self.assertIn(self.record.question, prompt)
        self.assertNotIn("context:", prompt.lower())
        self.assertIn('"answer"', prompt)

    def test_parser_accepts_json_and_label_forms(self) -> None:
        labels = ("A", "B", "C", "D", "E")
        self.assertEqual(parse_answer('{"answer":"D"}', labels), "D")
        self.assertEqual(parse_answer('Explanation first. {"answer":"D"}', labels), "D")
        self.assertEqual(parse_answer("Answer: d", labels), "D")
        self.assertEqual(parse_answer("Answer: C. Obtain a urine culture", labels), "C")
        self.assertEqual(parse_answer("The correct answer is C.", labels), "C")
        self.assertEqual(parse_answer("The correct answer is:\n\nC.", labels), "C")
        self.assertEqual(
            parse_answer("Explanation first.\n\nAnswer: C. Obtain a urine culture.", labels),
            "C",
        )
        self.assertEqual(parse_answer("C", labels), "C")
        self.assertIsNone(parse_answer("I do not know", labels))
        self.assertIsNone(parse_answer('{"answer":"Z"}', labels))

    def test_run_record_produces_common_result_shape(self) -> None:
        client = FakeClient('{"answer":"D"}')
        result = run_record(self.record, client, model="fake:test", temperature=0.0)
        self.assertEqual(result.system_name, "llm_only")
        self.assertEqual(result.predicted_answer, "D")
        self.assertTrue(result.is_correct)
        self.assertEqual(result.retrieved_chunk_ids, [])
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.latency_ms, 0)
        self.assertIsNone(result.scope)

    def test_run_record_preserves_scope_annotation(self) -> None:
        result = run_record(
            self.record,
            FakeClient('{"answer":"D"}'),
            model="fake:test",
            temperature=0.0,
            annotation=annotate_record(self.record),
        )
        self.assertEqual(result.scope, "out_of_corpus")
        self.assertFalse(result.evidence_expected)
        self.assertEqual(result.annotation_version, "scope_v1")

    def test_invalid_response_is_explicit_failure(self) -> None:
        result = run_record(
            self.record, FakeClient("unsupported response"), model="fake:test", temperature=0.0
        )
        self.assertIsNone(result.predicted_answer)
        self.assertIsNone(result.is_correct)
        self.assertIsNotNone(result.error)
        self.assertEqual(result.abstention_reason, result.error)

    def test_results_are_written_as_jsonl(self) -> None:
        result = run_record(self.record, FakeClient('{"answer":"D"}'), "fake:test", 0.0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            write_results(output, [result])
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
