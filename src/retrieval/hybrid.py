"""
Phase 3 -- Hybrid Retrieval

Combines BM25 and Dense retrieval scores using Reciprocal Rank Fusion (RRF).
"""

def reciprocal_rank_fusion(bm25_results, dense_results, k=60):
    """
    Combine two ranked lists using Reciprocal Rank Fusion (RRF).
    
    RRF Score = sum(1 / (k + rank)) for each rank list.
    
    Args:
        bm25_results: list of (score, chunk_dict) from BM25
        dense_results: list of (score, chunk_dict) from Dense search
        k: smoothing constant, typically 60
        
    Returns:
        List of (rrf_score, chunk_dict) sorted by score.
    """
    
    if k <= 0:
        raise ValueError("k must be positive")

    rrf_scores = {}
    chunk_map = {}
    
    # Process BM25 rankings
    for rank, (score, chunk) in enumerate(bm25_results):
        cid = chunk["chunk_id"]
        if cid not in rrf_scores:
            rrf_scores[cid] = 0.0
            chunk_map[cid] = chunk
        rrf_scores[cid] += 1.0 / (k + rank + 1)  # rank is 0-indexed
        
    # Process Dense rankings
    for rank, (score, chunk) in enumerate(dense_results):
        cid = chunk["chunk_id"]
        if cid not in rrf_scores:
            rrf_scores[cid] = 0.0
            chunk_map[cid] = chunk
        rrf_scores[cid] += 1.0 / (k + rank + 1)
        
    # Sort by combined score
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    final_results = []
    for cid, r_score in sorted_results:
        final_results.append((r_score, chunk_map[cid]))
        
    return final_results


class HybridRetriever:
    def __init__(self, bm25_retriever, dense_retriever):
        """
        Takes instantiated and loaded BM25 and Dense retrievers.
        """
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        
    def search(self, query, top_k=5, fetch_k=20):
        """
        Search both indexes, combine with RRF, and return top_k.
        fetch_k dictates how many results to fetch from each sub-retriever 
        before combining.
        """
        if top_k <= 0:
            return []
        if fetch_k < top_k:
            fetch_k = top_k

        bm25_res = self.bm25.search(query, top_k=fetch_k)
        dense_res = self.dense.search(query, top_k=fetch_k)
        
        hybrid_res = reciprocal_rank_fusion(bm25_res, dense_res, k=60)
        
        return hybrid_res[:top_k]
