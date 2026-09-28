"""Answer accuracy metrics for normalized multiple-choice questions."""


def evaluate_answer(prediction, record):
	expected = record.get("answer")
	predicted = prediction.get("answer", {}).get("answer_letter")
	comparable = expected is not None and predicted is not None
	return {
		"evaluated": comparable,
		"correct": bool(comparable and str(predicted).upper() == str(expected).upper()),
		"expected": expected,
		"predicted": predicted,
	}


def aggregate_accuracy(results):
	evaluated = [result for result in results if result["evaluated"]]
	return {
		"evaluated": len(evaluated),
		"total": len(results),
		"accuracy": sum(result["correct"] for result in evaluated) / len(evaluated) if evaluated else None,
	}
