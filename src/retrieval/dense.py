"""
Phase 3 -- Dense Vector Retrieval using Sentence Transformers and FAISS
"""
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class DenseRetriever:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5", device="cpu"):
        """
        Initialize the dense retriever.
        We default to bge-small-en-v1.5 for faster local embedding on CPU, 
        but it can easily be swapped to bge-base.
        """
        self.model_name = model_name
        self.device = device
        self.chunks = []
        self.index = None
        self.embeddings = None
        
        print(f"  Loading embedding model: {self.model_name} on {self.device}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def _get_embeddings(self, texts, batch_size=32):
        """Helper to get embeddings with progress reporting."""
        # For BGE models, passages do not require a special instruction
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)

    def load_chunks(self, chunks):
        """
        Load chunks, compute embeddings, and build FAISS index.
        """
        if not chunks:
            raise ValueError("chunks must contain at least one record")
        if any(not chunk.get("chunk_id") for chunk in chunks):
            raise ValueError("every chunk must contain a non-empty chunk_id")

        self.chunks = list(chunks)
        texts = [c.get("text", "") for c in chunks]
        
        print("  Computing dense embeddings for chunks...")
        embeddings = np.asarray(self._get_embeddings(texts), dtype="float32")
        self.embeddings = embeddings
        
        d = embeddings.shape[1]
        print(f"  Building FAISS IndexFlatIP (dim={d})...")
        # Use Inner Product (IP) since embeddings are normalized (equivalent to cosine similarity)
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)
        print(f"  FAISS index built with {len(self.chunks)} chunks.")

    def search(self, query, top_k=5):
        """
        Search for top-k chunks.
        For BGE models, queries should be prefixed with an instruction if specified by the model.
        bge-small/base requires: "Represent this sentence for searching relevant passages: "
        """
        if self.index is None:
            raise ValueError("FAISS index is empty. Call load_chunks() first.")

        if top_k <= 0:
            return []

        # BGE query prefix
        instruction = "Represent this sentence for searching relevant passages: "
        query_text = instruction + query

        # Encode query
        query_embedding = np.asarray(
            self.model.encode([query_text], normalize_embeddings=True),
            dtype="float32",
        )
        
        # Search index
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
        
        results = []
        for j, idx in enumerate(indices[0]):
            if idx != -1:
                score = float(scores[0][j])
                results.append((score, self.chunks[idx]))
                
        return results
