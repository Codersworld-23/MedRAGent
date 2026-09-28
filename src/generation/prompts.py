"""Prompt construction for grounded vanilla RAG generation."""


def build_vanilla_rag_prompt(question, contexts, options=None):
	"""Build a citation-constrained prompt from ranked chunk contexts."""
	evidence = []
	for index, context in enumerate(contexts, 1):
		metadata = context.get("metadata", {})
		location = " > ".join(
			value for value in (metadata.get("chapter"), metadata.get("section")) if value
		)
		label = f"[C{index}] {context.get('chunk_id', 'unknown')}"
		if location:
			label += f" ({location})"
		evidence.append(f"{label}\n{context.get('text', '')}")

	option_text = ""
	if options:
		option_text = "\n\nOptions:\n" + "\n".join(
			f"{key}. {value}" for key, value in options.items()
		)

	return (
		"You are answering a medical question using only the supplied textbook evidence.\n"
		"Do not use outside knowledge. If the evidence is insufficient, say so.\n"
		"For multiple-choice questions, begin with the answer letter and explain briefly.\n"
		"Cite every factual claim with one or more evidence labels such as [C1].\n\n"
		f"Question:\n{question}{option_text}\n\n"
		"Textbook evidence:\n" + "\n\n".join(evidence) + "\n\n"
		"Answer:"
	)
