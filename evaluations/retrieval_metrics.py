"""Metrics and evaluation helpers for Phase 3 retrieval experiments."""

import math


def _relevant_ids(record):
	"""Read optional chunk labels without imposing them on existing datasets."""
	values = record.get("relevant_chunk_ids", record.get("gold_chunk_ids", []))
	if isinstance(values, str):
		return {values}
	return {str(value) for value in values or []}


def evaluate_ranked_results(results, relevant_ids, k):
	"""Calculate Recall@k, Precision@k, MRR, and nDCG for one query."""
	relevant_ids = {str(value) for value in relevant_ids}
	ranked_ids = [str(chunk["chunk_id"]) for _, chunk in results[:k]]
	hits = [index for index, chunk_id in enumerate(ranked_ids) if chunk_id in relevant_ids]
	retrieved_relevant = len(hits)

	recall = retrieved_relevant / len(relevant_ids) if relevant_ids else None
	precision = retrieved_relevant / k if k else 0.0
	reciprocal_rank = 1.0 / (hits[0] + 1) if hits else 0.0
	dcg = sum(1.0 / math.log2(index + 2) for index in hits)
	ideal_hits = min(len(relevant_ids), k)
	idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))

	return {
		"recall_at_k": recall,
		"precision_at_k": precision,
		"mrr": reciprocal_rank,
		"ndcg_at_k": dcg / idcg if idcg else None,
		"retrieved_relevant": retrieved_relevant,
		"labeled": bool(relevant_ids),
	}


def aggregate_metrics(metrics):
	"""Average labeled metrics while retaining the number of evaluated queries."""
	labeled = [metric for metric in metrics if metric["labeled"]]
	if not labeled:
		return {"labeled_queries": 0, "total_queries": len(metrics)}

	names = ("recall_at_k", "precision_at_k", "mrr", "ndcg_at_k")
	return {
		"labeled_queries": len(labeled),
		"total_queries": len(metrics),
		**{name: sum(metric[name] or 0.0 for metric in labeled) / len(labeled) for name in names},
	}
