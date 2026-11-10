"""Offline tests for BM25 retrieval and vanilla RAG output validation."""

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medragent.bm25 import (
    BM25Retriever,
    build_rag_prompt,
    parse_citations,
    response_schema,
    run_record,
)
from medragent.data import MedQARecord, TextChunk
from medragent.scope import annotate_record


class FakeClient:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return self.response


def record() -> MedQARecord:
    return MedQARecord(
        id="q1",
        question="Which symptom is associated with anxiety?",
        options={"A": "Fever", "B": "Worry"},
        answer_index=1,
        split="dev",
        source_file="test.jsonl",
        metadata={},
    )


class BM25Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            TextChunk("c1", "General", "fever cough infection", "test.jsonl", {}),
            TextChunk("c2", "Anxiety", "anxiety worry and panic symptoms", "test.jsonl", {}),
            TextChunk("c3", "Cardiology", "chest pain and myocardial infarction", "test.jsonl", {}),
        ]
        self.retriever = BM25Retriever(self.chunks)

    def test_retrieval_is_ranked_and_bounded(self) -> None:
        results = self.retriever.retrieve("anxiety worry", top_k=2)
        self.assertEqual([result.chunk_id for result in results], ["c2", "c1"])
        self.assertEqual([result.rank for result in results], [1, 2])

    def test_prompt_contains_only_retrieved_context(self) -> None:
        results = self.retriever.retrieve("anxiety", top_k=1)
        prompt = build_rag_prompt(record(), results)
        self.assertIn("[c2]", prompt)
        self.assertIn("Allowed citation chunk IDs: c2", prompt)
        self.assertNotIn("fever cough infection", prompt)

    def test_citations_reject_unknown_ids(self) -> None:
        self.assertEqual(parse_citations('{"citations":["c2"]}', {"c2"}), ["c2"])
        self.assertEqual(
            parse_citations('Explanation. {"answer":"B","citations":["c2"]}', {"c2"}),
            ["c2"],
        )
        self.assertIsNone(parse_citations('{"citations":["unknown"]}', {"c2"}))

    def test_response_schema_restricts_answers_and_citations(self) -> None:
        schema = response_schema(record(), self.retriever.retrieve("anxiety", top_k=1))
        self.assertEqual(schema["properties"]["answer"]["enum"], ["A", "B"])
        self.assertEqual(schema["properties"]["citations"]["items"]["enum"], ["c2"])
        self.assertEqual(schema["required"], ["answer", "citations"])

    def test_run_record_requires_valid_citations(self) -> None:
        result = run_record(
            record(),
            self.retriever,
            FakeClient('{"answer":"B","citations":["c2"]}'),
            "fake:test",
            0.0,
            1,
            annotate_record(record()),
        )
        self.assertEqual(result.predicted_answer, "B")
        self.assertTrue(result.is_correct)
        self.assertEqual(result.citations, ["c2"])
        self.assertEqual(result.scope, "psychiatry_relevant")

    def test_missing_citations_are_explicit_failure(self) -> None:
        result = run_record(
            record(), self.retriever, FakeClient('{"answer":"B"}'), "fake:test", 0.0, 1
        )
        self.assertIsNone(result.is_correct)
        self.assertIn("citations", result.error or "")


if __name__ == "__main__":
    unittest.main()
