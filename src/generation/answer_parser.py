"""Parse answer letters and evidence citations from model output."""

import re


def parse_answer(text, options=None):
	text = (text or "").strip()
	citations = [int(value) for value in re.findall(r"\[C(\d+)\]", text, re.IGNORECASE)]
	answer_letter = None
	if options:
		valid = "|".join(re.escape(str(key)) for key in options)
		match = re.search(rf"(?:^|\b)(?:answer\s*[:\-]?\s*)?({valid})(?:\b|\.)", text, re.IGNORECASE)
		if match:
			answer_letter = match.group(1).upper()
	return {"text": text, "answer_letter": answer_letter, "citations": citations}
