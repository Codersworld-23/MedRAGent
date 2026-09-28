"""
Phase 3 -- BM25 Sparse Lexical Retrieval
"""
from rank_bm25 import BM25Okapi
import numpy as np
import re

def _tokenize(text):
    """Simple tokenizer for BM25."""
    if not text:
        return []
    # Lowercase and keep alphanumeric
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


class BM25Retriever:
    def __init__(self):
        self.chunks = []
        self.bm25 = None
        self.tokenized_corpus = []

    def load_chunks(self, chunks):
        """
        Load a list of chunk dictionaries and index them.
        Each chunk must have 'text' and 'chunk_id'.
        """
        if not chunks:
            raise ValueError("chunks must contain at least one record")
        if any(not chunk.get("chunk_id") for chunk in chunks):
            raise ValueError("every chunk must contain a non-empty chunk_id")

        self.chunks = list(chunks)
        print("  Tokenizing corpus for BM25...")
        self.tokenized_corpus = [_tokenize(chunk.get("text", "")) for chunk in self.chunks]
        print("  Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"  BM25 index built with {len(self.chunks)} chunks.")

    def search(self, query, top_k=5):
        """
        Search the BM25 index for the top-k chunks.
        Returns a list of tuples: (score, chunk_dict)
        """
        if self.bm25 is None:
            raise ValueError("BM25 index is empty. Call load_chunks() first.")

        if top_k <= 0:
            return []

        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            results.append((score, self.chunks[idx]))
                
        return results
