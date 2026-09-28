"""Evidence relevance metrics based on question/evidence term overlap."""

import re


def evaluate_relevance(prediction):
    question_terms = set(re.findall(r"[a-z0-9]+", prediction.get("question", "").lower()))
    contexts = prediction.get("contexts", [])
    scores = []
    for context in contexts:
        evidence_terms = set(re.findall(r"[a-z0-9]+", context.get("text", "").lower()))
        scores.append(len(question_terms & evidence_terms) / len(question_terms) if question_terms else 0.0)
    return {
        "contexts": len(contexts),
        "mean_relevance": sum(scores) / len(scores) if scores else 0.0,
        "top_relevance": max(scores) if scores else 0.0,
    }