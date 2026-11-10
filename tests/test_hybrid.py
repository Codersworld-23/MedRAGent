"""Offline tests for hybrid Reciprocal Rank Fusion retrieval."""

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medragent.bm25 import BM25Retriever
from medragent.data import TextChunk
from medragent.dense import DenseRetriever
from medragent.hybrid import HybridRetriever


class FakeEmbeddingClient:
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = {
            "alpha": [1.0, 0.0],
            "beta": [0.0, 1.0],
            "alpha beta": [0.7, 0.7],
        }
        return [vectors[text] for text in texts]


class HybridTests(unittest.TestCase):
    def test_rrf_is_deterministic_and_bounded(self) -> None:
        chunks = [
            TextChunk("c1", "Alpha", "alpha", "test.jsonl", {}),
            TextChunk("c2", "Beta", "beta", "test.jsonl", {}),
        ]
        embeddings = FakeEmbeddingClient()
        retriever = HybridRetriever(
            BM25Retriever(chunks),
            DenseRetriever.from_client(chunks, embeddings),
            embeddings,
            rrf_constant=60,
        )
        bm25, dense, fused = retriever.retrieve("alpha beta", top_k=2)
        self.assertEqual(len(bm25), 2)
        self.assertEqual(len(dense), 2)
        self.assertEqual(len(fused), 2)
        self.assertEqual([chunk.rank for chunk in fused], [1, 2])
        self.assertEqual([chunk.chunk_id for chunk in fused], ["c1", "c2"])


if __name__ == "__main__":
    unittest.main()
