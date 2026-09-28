"""Generation backends used by the Phase 4 vanilla RAG baseline."""

import json
import re
import urllib.error
import urllib.request


class OllamaGenerator:
	"""Small dependency-free client for a local Ollama chat endpoint."""

	def __init__(self, model="llama3.1:8b", host="http://localhost:11434", timeout=120):
		self.model = model
		self.url = host.rstrip("/") + "/api/generate"
		self.timeout = timeout

	def generate(self, prompt):
		payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode()
		request = urllib.request.Request(
			self.url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
		)
		try:
			with urllib.request.urlopen(request, timeout=self.timeout) as response:
				body = json.loads(response.read().decode("utf-8"))
		except (urllib.error.URLError, TimeoutError) as exc:
			raise RuntimeError(f"Ollama request failed at {self.url}: {exc}") from exc
		return body.get("response", "").strip()


class ExtractiveGenerator:
	"""Deterministic fallback that returns the most question-overlapping sentence."""

	_token_pattern = re.compile(r"[a-z0-9]+")

	def generate(self, prompt, contexts=None, question="", options=None):
		if not contexts:
			return "Insufficient evidence."
		query_terms = set(self._token_pattern.findall(question.lower()))
		if options:
			evidence_terms = set()
			for context in contexts:
				evidence_terms.update(self._token_pattern.findall(context.get("text", "").lower()))
			ranked_options = sorted(
				options.items(),
				key=lambda item: len(set(self._token_pattern.findall(str(item[1]).lower())) & evidence_terms),
				reverse=True,
			)
			if ranked_options:
				return f"{ranked_options[0][0]}. {ranked_options[0][1]} [C1]"
		best_sentence = contexts[0].get("text", "").strip()
		best_index = 1
		best_score = -1
		for index, context in enumerate(contexts, 1):
			for sentence in re.split(r"(?<=[.!?])\s+", context.get("text", "")):
				terms = set(self._token_pattern.findall(sentence.lower()))
				score = len(query_terms & terms)
				if score > best_score:
					best_score = score
					best_sentence = sentence.strip()
					best_index = index
		return f"{best_sentence} [C{best_index}]"
