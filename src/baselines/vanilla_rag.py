"""Vanilla retrieve-then-generate baseline for Phase 4."""

from src.generation.answer_parser import parse_answer
from src.generation.prompts import build_vanilla_rag_prompt


class VanillaRAG:
	def __init__(self, retriever, generator, top_k=5):
		self.retriever = retriever
		self.generator = generator
		self.top_k = top_k

	def answer(self, question_record):
		question = question_record.get("question", "")
		contexts = [chunk for _, chunk in self.retriever.search(question, self.top_k)]
		options = question_record.get("options")
		prompt = build_vanilla_rag_prompt(question, contexts, options)
		try:
			text = self.generator.generate(
				prompt, contexts=contexts, question=question, options=options
			)
		except TypeError:
			text = self.generator.generate(prompt)
		return {
			"question_id": question_record.get("id"),
			"question": question,
			"answer": parse_answer(text, question_record.get("options")),
			"contexts": contexts,
			"prompt": prompt,
		}
