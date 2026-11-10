"""Offline tests for dense retrieval and dense vanilla RAG."""

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medragent.data import MedQARecord, TextChunk
from medragent.dense import DenseRetriever, run_record


class FakeEmbeddingClient:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]


class FakeGenerationClient:
    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return '{"answer":"B","citations":["c2"]}'


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


class DenseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            TextChunk("c1", "General", "fever", "test.jsonl", {}),
            TextChunk("c2", "Anxiety", "anxiety worry", "test.jsonl", {}),
            TextChunk("c3", "Cardiology", "chest pain", "test.jsonl", {}),
        ]
        self.embeddings = FakeEmbeddingClient(
            {
                "fever": [1.0, 0.0],
                "anxiety worry": [0.0, 1.0],
                "chest pain": [-1.0, 0.0],
                "Which symptom is associated with anxiety?": [0.0, 1.0],
            }
        )
        self.retriever = DenseRetriever.from_client(self.chunks, self.embeddings)

    def test_retrieval_uses_normalized_inner_product(self) -> None:
        results = self.retriever.retrieve("anxiety worry", self.embeddings, top_k=2)
        self.assertEqual([result.chunk_id for result in results], ["c2", "c1"])
        self.assertEqual([result.rank for result in results], [1, 2])

    def test_dense_rag_requires_valid_citations(self) -> None:
        result = run_record(
            record(),
            self.retriever,
            self.embeddings,
            FakeGenerationClient(),
            "fake:test",
            "fake-embeddings",
            0.0,
            1,
        )
        self.assertEqual(result.predicted_answer, "B")
        self.assertTrue(result.is_correct)
        self.assertEqual(result.retrieved_chunk_ids, ["c2"])


if __name__ == "__main__":
    unittest.main()
