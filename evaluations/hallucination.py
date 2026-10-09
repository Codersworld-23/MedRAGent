"""Hallucination indicators derived from citation and support checks."""


def evaluate_hallucination(faithfulness):
	claims = faithfulness.get("claims", 0)
	unsupported = claims - faithfulness.get("supported_claims", 0)
	return {
		"unsupported_claims": unsupported,
		"claims": claims,
		"hallucination_rate": unsupported / claims if claims else 0.0,
		"uncited_response": faithfulness.get("citation_count", 0) == 0,
	}
