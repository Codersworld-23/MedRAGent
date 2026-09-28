"""Lightweight citation and claim-support faithfulness checks."""

import re


def evaluate_faithfulness(prediction, min_overlap=0.15):
	answer = prediction.get("answer", {})
	text = answer.get("text", "")
	contexts = prediction.get("contexts", [])
	citations = answer.get("citations", [])
	valid_citations = [index for index in citations if 1 <= index <= len(contexts)]
	claims = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
	supported = 0
	for claim in claims:
		claim_terms = set(re.findall(r"[a-z0-9]+", claim.lower()))
		evidence_terms = set()
		for index in valid_citations:
			evidence_terms.update(re.findall(r"[a-z0-9]+", contexts[index - 1].get("text", "").lower()))
		if claim_terms and len(claim_terms & evidence_terms) / len(claim_terms) >= min_overlap:
			supported += 1
	return {
		"claims": len(claims),
		"supported_claims": supported,
		"citation_count": len(citations),
		"valid_citations": len(valid_citations),
		"faithfulness": supported / len(claims) if claims else 0.0,
	}
