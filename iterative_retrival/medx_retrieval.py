from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from medx_llm import FallbackLLM, LLMUnavailable, get_api_key, get_llm, load_project_env

load_project_env()


DEFAULT_EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001")
DEFAULT_EMBED_DIMENSIONS = 768

# Reciprocal rank fusion constant; 60 is the standard value from Cormack et al. (2009).
RRF_K = 60
# The user's own question anchors the fused ranking so refined queries cannot drift it away.
ORIGINAL_QUERY_WEIGHT = 1.5
REFINED_QUERY_WEIGHT = 1.0

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]+")
# The PDF extraction left line-break hyphens inside words ("anx- iety", "dis- orders").
BROKEN_HYPHEN_RE = re.compile(r"([a-z])- +([a-z])")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

STOPWORDS = {
    "about",
    "after",
    "all",
    "also",
    "among",
    "and",
    "any",
    "are",
    "because",
    "been",
    "being",
    "between",
    "both",
    "but",
    "can",
    "chapter",
    "could",
    "did",
    "does",
    "during",
    "each",
    "for",
    "four",
    "from",
    "had",
    "has",
    "have",
    "her",
    "his",
    "how",
    "into",
    "its",
    "may",
    "more",
    "most",
    "much",
    "must",
    "not",
    "one",
    "only",
    "other",
    "our",
    "over",
    "page",
    "same",
    "should",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "three",
    "through",
    "two",
    "under",
    "used",
    "very",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "within",
    "without",
    "would",
    "you",
    "your",
}


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    round_number: int
    rank: int
    score: float
    document: Document
    query: str = ""


@dataclass(frozen=True)
class RetrievalRound:
    round_number: int
    queries: List[str]
    refiner: str
    sufficient: bool = False
    missing_information: str = ""
    note: str = ""


@dataclass(frozen=True)
class RetrievalOutcome:
    sources: List[SearchResult]
    trace: List[SearchResult]
    rounds: List[RetrievalRound]
    stop_reason: str

    @property
    def queries(self) -> List[str]:
        return [query for round_ in self.rounds for query in round_.queries]


@dataclass(frozen=True)
class RAGResponse:
    question: str
    answer: str
    sources: List[SearchResult]
    queries: List[str]
    rounds: List[RetrievalRound]
    stop_reason: str
    retrieval_trace: List[SearchResult]
    generator: str
    retriever: str


def tokenize(text: str) -> List[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 2 and token.lower() not in STOPWORDS
    ]


def load_jsonl(path: Union[str, Path]) -> List[Document]:
    documents: List[Document] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            title = str(row.get("title", ""))
            # Prefer "content": "contents" repeats the title in front of every chunk, which
            # pollutes both lexical scores and embeddings.
            text = str(row.get("content") or row.get("contents") or "")
            if title and text.startswith(f"{title}. "):
                text = text[len(title) + 2 :]
            text = BROKEN_HYPHEN_RE.sub(r"\1\2", text)
            documents.append(Document(id=str(row.get("id", line_number)), title=title, text=text))
    if not documents:
        raise ValueError(f"No documents found in {path}")
    return documents


def corpus_fingerprint(documents: Sequence[Document]) -> str:
    digest = hashlib.sha256()
    for document in documents:
        digest.update(document.id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(document.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def normalize_vector(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else list(vector)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Tuple[float, Sequence[Document]]],
    k: int = RRF_K,
) -> List[Tuple[float, Document]]:
    """Fuse ranked lists by rank, not raw score, so lists with different score scales are comparable."""
    scores: DefaultDict[str, float] = defaultdict(float)
    documents: Dict[str, Document] = {}
    for weight, documents_in_list in ranked_lists:
        for rank, document in enumerate(documents_in_list, start=1):
            scores[document.id] += weight / (k + rank)
            documents.setdefault(document.id, document)
    return sorted(
        ((score, documents[document_id]) for document_id, score in scores.items()),
        key=lambda item: item[0],
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Lexical retrieval
# ---------------------------------------------------------------------------


class BM25Index:
    def __init__(self, documents: Iterable[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self.term_frequencies: List[Counter[str]] = []
        self.document_frequency: Counter[str] = Counter()
        self.document_lengths: List[int] = []

        for document in self.documents:
            counts = Counter(tokenize(document.text))
            self.term_frequencies.append(counts)
            self.document_frequency.update(counts.keys())
            self.document_lengths.append(sum(counts.values()))

        self.average_document_length = sum(self.document_lengths) / max(len(self.document_lengths), 1)
        self.idf = {
            term: math.log(1 + (len(self.documents) - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in self.document_frequency.items()
        }

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, Document]]:
        query_terms = tokenize(query)
        scores: List[Tuple[float, Document]] = []
        for index, document in enumerate(self.documents):
            score = self._score_document(index, query_terms)
            if score > 0:
                scores.append((score, document))
        return sorted(scores, key=lambda item: item[0], reverse=True)[:top_k]

    def _score_document(self, document_index: int, query_terms: List[str]) -> float:
        counts = self.term_frequencies[document_index]
        length = self.document_lengths[document_index] or 1
        score = 0.0

        for term in query_terms:
            frequency = counts.get(term, 0)
            if frequency == 0:
                continue
            numerator = frequency * (self.k1 + 1)
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * length / self.average_document_length
            )
            score += self.idf.get(term, 0.0) * numerator / denominator
        return score


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def _with_retries(call: Callable[[], Any], attempts: int = 8, base_delay: float = 4.0) -> Any:
    """Retry Gemini calls on rate limits and transient server errors with exponential backoff."""
    from google.genai import errors

    for attempt in range(attempts):
        try:
            return call()
        except errors.APIError as exc:
            retryable = exc.code in (429, 500, 502, 503, 504)
            if not retryable or attempt == attempts - 1:
                raise
            suggested = _suggested_retry_seconds(exc)
            if suggested is not None and suggested > 120:
                # Daily quotas reset hours later; waiting here would only hang the pipeline.
                raise RuntimeError(
                    f"Gemini quota exhausted; it resets in about {suggested / 3600:.1f}h. "
                    f"Details: {exc.message}"
                ) from exc
            delay = suggested if suggested is not None else min(base_delay * (2**attempt), 90.0)
            print(f"Gemini API error {exc.code}; retrying in {delay:.0f}s", file=sys.stderr)
            time.sleep(delay)
    raise AssertionError("unreachable")


def _suggested_retry_seconds(exc: Any) -> Optional[float]:
    """Read google.rpc.RetryInfo (e.g. "57077s") from a Gemini error, if present."""
    details = ((exc.details or {}).get("error") or {}).get("details") or []
    for detail in details:
        delay = str(detail.get("retryDelay", ""))
        if delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                return None
    return None


class GeminiEmbeddingFunction:
    """Semantic embeddings from the Gemini API, with separate document and query task types."""

    batch_size = 100

    def __init__(
        self,
        model: str = DEFAULT_EMBED_MODEL,
        dimensions: int = DEFAULT_EMBED_DIMENSIONS,
        api_key: Union[str, None] = None,
    ):
        api_key = get_gemini_api_key(api_key)
        if not api_key:
            raise RuntimeError(
                "Gemini API key not found. Set GEMINI_API_KEY in .env, or use --embedder local."
            )
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.dimensions = dimensions
        self.name = f"gemini-{model}-{dimensions}"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed(texts[start : start + self.batch_size], "RETRIEVAL_DOCUMENT"))
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]

    def _embed(self, texts: List[str], task_type: str) -> List[List[float]]:
        from google.genai import types

        response = _with_retries(
            lambda: self.client.models.embed_content(
                model=self.model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.dimensions,
                ),
            )
        )
        # Truncated Gemini embeddings are not unit length, so normalize before cosine search.
        return [normalize_vector(embedding.values) for embedding in response.embeddings]


class LocalEmbeddingFunction:
    """Free semantic embeddings: Chroma's bundled ONNX all-MiniLM-L6-v2 model, run on the CPU.

    The model (~80 MB) downloads once to ~/.cache/chroma; no API key or quota is involved.
    """

    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self.model = DefaultEmbeddingFunction()
        self.name = "local-minilm-l6-384"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [normalize_vector([float(value) for value in vector]) for vector in self.model(texts)]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class HashingEmbeddingFunction:
    """Offline fallback: hashed word n-grams. Lexical only, so it cannot match synonyms."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions
        self.name = f"hash-{dimensions}"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_text(text)

    def embed_text(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        tokens = tokenize(text)
        features: List[Tuple[str, float]] = []
        features.extend((token, 1.0) for token in tokens)
        features.extend((f"{left}_{right}", 1.8) for left, right in zip(tokens, tokens[1:]))
        features.extend(
            (f"{first}_{second}_{third}", 2.4)
            for first, second, third in zip(tokens, tokens[1:], tokens[2:])
        )

        for feature, weight in features:
            bucket = self._stable_hash(feature) % self.dimensions
            vector[bucket] += weight
        return normalize_vector(vector)

    @staticmethod
    def _stable_hash(value: str) -> int:
        result = 2166136261
        for char in value:
            result ^= ord(char)
            result = (result * 16777619) & 0xFFFFFFFF
        return result


def build_embedder(embedder: str = "auto") -> Any:
    # "auto" picks the local model: the Gemini free tier caps embeddings at 1,000 chunks per day,
    # so a full Gemini index of this corpus takes several days unless billing is enabled.
    if embedder in ("auto", "local"):
        return LocalEmbeddingFunction()
    if embedder == "gemini":
        return GeminiEmbeddingFunction()
    if embedder == "hash":
        return HashingEmbeddingFunction()
    raise ValueError("embedder must be one of: auto, local, gemini, hash")


# ---------------------------------------------------------------------------
# Dense and hybrid retrieval
# ---------------------------------------------------------------------------


class ChromaRetriever:
    def __init__(
        self,
        db_path: Union[str, Path] = "chroma_db",
        collection_name: str = "medx_dsm5",
        embedder: Any = None,
    ):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "ChromaDB is not installed. Install it with: pip install chromadb"
            ) from exc

        self.embedder = embedder or build_embedder("auto")
        self.db_path = str(db_path)
        # One collection per embedder, so switching embedders never mixes vector spaces.
        self.collection_name = f"{collection_name}_{self.embedder.name}".replace(".", "-")
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self._open_collection()

    def _open_collection(self, fingerprint: str = "") -> Any:
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "MedX DSM-5 chunks",
                "embedder": self.embedder.name,
                "corpus_fingerprint": fingerprint,
                "hnsw:space": "cosine",
            },
        )

    def count(self) -> int:
        return int(self.collection.count())

    def ingest(self, documents: List[Document], batch_size: int = 100, reset: bool = False) -> int:
        fingerprint = corpus_fingerprint(documents)
        stored_fingerprint = (self.collection.metadata or {}).get("corpus_fingerprint")
        # Rebuild when asked to, or when the corpus text changed since the vectors were stored.
        if reset or stored_fingerprint != fingerprint:
            if self.count():
                print(f"Rebuilding Chroma collection {self.collection_name}", file=sys.stderr)
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = self._open_collection(fingerprint)

        # Resumable: only embed chunks that are not stored yet, so an ingest interrupted by
        # rate limits picks up where it stopped.
        existing_ids = set(self.collection.get(include=[])["ids"])
        missing = [document for document in documents if document.id not in existing_ids]
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            self.collection.upsert(
                ids=[document.id for document in batch],
                documents=[document.text for document in batch],
                metadatas=[
                    {"title": document.title, "source": "Psichiatry_DSM-5.jsonl"}
                    for document in batch
                ],
                embeddings=self.embedder.embed_documents([document.text for document in batch]),
            )
            print(
                f"Ingested {len(existing_ids) + start + len(batch)}/{len(documents)} chunks",
                file=sys.stderr,
            )
        return self.count()

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, Document]]:
        results = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits: List[Tuple[float, Document]] = []
        for document_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            document = Document(
                id=str(document_id),
                title=str((metadata or {}).get("title", "")),
                text=str(text or ""),
            )
            hits.append((1.0 - float(distance), document))
        return hits


class HybridRetriever:
    """Dense Chroma search plus BM25, fused with reciprocal rank fusion."""

    def __init__(self, dense: ChromaRetriever, lexical: BM25Index, candidate_k: int = 30):
        self.dense = dense
        self.lexical = lexical
        self.candidate_k = candidate_k

    @property
    def idf(self) -> Dict[str, float]:
        return self.lexical.idf

    def count(self) -> int:
        return self.dense.count()

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, Document]]:
        candidate_k = max(self.candidate_k, top_k)
        dense_hits = [document for _, document in self.dense.search(query, top_k=candidate_k)]
        lexical_hits = [document for _, document in self.lexical.search(query, top_k=candidate_k)]
        return reciprocal_rank_fusion([(1.0, dense_hits), (1.0, lexical_hits)])[:top_k]


def ensure_chroma_ingested(
    data_path: Union[str, Path] = "Psichiatry_DSM-5.jsonl",
    db_path: Union[str, Path] = "chroma_db",
    collection_name: str = "medx_dsm5",
    reset: bool = False,
    embedder: Union[str, Any] = "auto",
    documents: Optional[List[Document]] = None,
) -> ChromaRetriever:
    documents = documents or load_jsonl(data_path)
    if isinstance(embedder, str):
        embedder = build_embedder(embedder)
    retriever = ChromaRetriever(db_path=db_path, collection_name=collection_name, embedder=embedder)
    retriever.ingest(documents, reset=reset)
    return retriever


def build_retriever(
    data_path: Union[str, Path] = "Psichiatry_DSM-5.jsonl",
    retriever_name: str = "hybrid",
    chroma_path: Union[str, Path] = "chroma_db",
    collection_name: str = "medx_dsm5",
    reset_chroma: bool = False,
    embedder: str = "auto",
) -> Any:
    documents = load_jsonl(data_path)
    if retriever_name == "bm25":
        return BM25Index(documents)

    dense = ensure_chroma_ingested(
        data_path=data_path,
        db_path=chroma_path,
        collection_name=collection_name,
        reset=reset_chroma,
        embedder=embedder,
        documents=documents,
    )
    if retriever_name == "chroma":
        return dense
    if retriever_name == "hybrid":
        return HybridRetriever(dense, BM25Index(documents))
    raise ValueError("retriever_name must be one of: hybrid, chroma, bm25")


# ---------------------------------------------------------------------------
# Query refinement
# ---------------------------------------------------------------------------


REFINE_SYSTEM = """You are the retrieval planner for MedX, a RAG assistant over the DSM-5 psychiatry manual.

Decide whether the evidence retrieved so far is enough to answer the user's question completely.
If it is not, write up to {max_queries} new search queries that target the missing information.

Rules for new queries:
- Each query must look for something the current evidence lacks; do not repeat queries already tried.
- Use the clinical vocabulary the DSM-5 itself uses (disorder names, specifiers, criteria wording).
- Keep each query short and specific, like a search-engine query, not a full sentence.
- If the question has several parts, give each missing part its own query.
- Do not add the words "DSM-5" to queries; every chunk is from the DSM-5 already.
- The DSM-5 covers diagnostic criteria, symptoms, features, specifiers, prevalence, development and course,
  risk factors, differential diagnosis and comorbidity. It does not cover treatment, drug choice,
  pharmacology, lab tests or clinical management, so never search for those.
- Set sufficient to true, with no new queries, when the evidence answers the question, or when everything
  still missing is information the DSM-5 does not contain.

Respond with only a JSON object, no other text:
{{"sufficient": true or false, "missing_information": "what is still missing, empty if sufficient", "next_queries": ["query", ...]}}"""

REFINE_USER = """User question:
{question}

Queries already tried:
{tried}

Evidence retrieved so far:
{evidence}"""


def _llm_query_refiner(llm: FallbackLLM, max_queries: int) -> Callable[..., Tuple[bool, str, List[str]]]:
    def refine(question: str, tried: List[str], evidence: List[Tuple[float, Document]]) -> Tuple[bool, str, List[str]]:
        reply = llm.chat_json(
            REFINE_SYSTEM.format(max_queries=max_queries),
            REFINE_USER.format(
                question=question,
                tried="\n".join(f"- {query}" for query in tried),
                evidence=build_context([document for _, document in evidence], max_chars=6000, per_chunk_chars=700),
            ),
        )
        sufficient = reply.get("sufficient") in (True, "true", "True")
        queries = [str(query).strip() for query in reply.get("next_queries") or [] if str(query).strip()]
        return sufficient, str(reply.get("missing_information") or "").strip(), queries[:max_queries]

    return refine


def keyword_expansion_query(
    question: str,
    evidence: List[Tuple[float, Document]],
    idf: Optional[Dict[str, float]],
    limit: int = 4,
    feedback_docs: int = 3,
) -> str:
    """Pseudo-relevance feedback: add a few rare, repeated terms from the top fused evidence."""
    existing = set(tokenize(question))
    weighted_terms: DefaultDict[str, float] = defaultdict(float)
    for rank, (_, document) in enumerate(evidence[:feedback_docs], start=1):
        for term, frequency in Counter(tokenize(document.text)).items():
            if term in existing:
                continue
            # IDF keeps generic words like "disorder" or "criteria" from dominating the expansion.
            weighted_terms[term] += frequency * (idf or {}).get(term, 1.0) / rank
    new_terms = [
        term
        for term, _ in sorted(weighted_terms.items(), key=lambda item: item[1], reverse=True)[:limit]
    ]
    return f"{question} {' '.join(new_terms)}" if new_terms else ""


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


def iterative_retrieve(
    retriever: Any,
    question: str,
    rounds: int = 3,
    top_k: int = 5,
    source_limit: int = 8,
    refiner: str = "auto",
    planner_llm: Optional[str] = None,
    max_queries_per_round: int = 3,
    expansion_terms: int = 4,
) -> RetrievalOutcome:
    """Retrieve, ask the refiner what is missing, retrieve again with refined queries, and fuse everything.

    Round 1 searches the user's question. After each round the refiner sees the fused evidence: the
    planner LLM either declares it sufficient or writes targeted follow-up queries; the keyword refiner
    adds high-IDF terms from the top evidence. Every query's ranked list is fused with reciprocal rank
    fusion, with the original question weighted highest, so later rounds add evidence without drowning
    it out. planner_llm is "provider" or "provider:model" (see medx_llm); None uses MEDX_PLANNER_LLM.
    """
    if refiner not in ("auto", "llm", "keywords"):
        raise ValueError("refiner must be one of: auto, llm, keywords")
    planner = get_llm("planner", planner_llm) if refiner != "keywords" else None
    if refiner == "llm" and planner is None:
        raise LLMUnavailable("No LLM provider key found in .env for the planner; use --refiner keywords.")
    llm_refine = _llm_query_refiner(planner, max_queries_per_round) if planner is not None else None

    ranked_lists: List[Tuple[float, List[Document]]] = []
    trace: List[SearchResult] = []
    round_log: List[RetrievalRound] = []
    tried: Set[str] = set()
    pending: List[str] = [question]
    pending_refiner = "question"
    pending_sufficient = False
    pending_missing = ""
    pending_note = ""
    fused: List[Tuple[float, Document]] = []
    stop_reason = f"reached the {rounds}-round limit"

    for round_number in range(1, rounds + 1):
        executed: List[str] = []
        for query in pending:
            key = _normalize_query(query)
            if key in tried:
                continue
            tried.add(key)
            executed.append(query)
            hits = retriever.search(query, top_k=top_k)
            for rank, (score, document) in enumerate(hits, start=1):
                trace.append(SearchResult(round_number, rank, score, document, query))
            weight = ORIGINAL_QUERY_WEIGHT if query == question else REFINED_QUERY_WEIGHT
            ranked_lists.append((weight, [document for _, document in hits]))

        round_log.append(
            RetrievalRound(
                round_number=round_number,
                queries=executed,
                refiner=pending_refiner,
                sufficient=pending_sufficient,
                missing_information=pending_missing,
                note=pending_note,
            )
        )
        fused = reciprocal_rank_fusion(ranked_lists)
        if round_number == rounds:
            break

        # Plan the next round from everything retrieved so far.
        evidence = fused[: max(source_limit, top_k) + 2]
        pending_note = ""
        if llm_refine is not None:
            try:
                sufficient, missing, next_queries = llm_refine(
                    question, [query for round_ in round_log for query in round_.queries], evidence
                )
                pending_refiner, pending_sufficient, pending_missing = planner.label, sufficient, missing
                if sufficient:
                    stop_reason = f"{planner.label} judged the evidence sufficient after round {round_number}"
                    break
                pending = next_queries
            except Exception as exc:
                # Keep the pipeline usable when the planner is unavailable; record why we fell back.
                pending_refiner, pending_sufficient, pending_missing = "keywords", False, ""
                pending_note = f"{planner.label} refinement failed ({exc.__class__.__name__}: {exc}); used keyword expansion"
                pending = [keyword_expansion_query(question, evidence, getattr(retriever, "idf", None), expansion_terms)]
        else:
            pending_refiner = "keywords"
            pending = [keyword_expansion_query(question, evidence, getattr(retriever, "idf", None), expansion_terms)]

        pending = [query for query in pending if query and _normalize_query(query) not in tried]
        if not pending:
            stop_reason = f"no new queries after round {round_number}"
            break

    sources = [
        SearchResult(_first_round(trace, document.id), rank, score, document)
        for rank, (score, document) in enumerate(_select_sources(fused, ranked_lists, source_limit), start=1)
    ]
    return RetrievalOutcome(sources=sources, trace=trace, rounds=round_log, stop_reason=stop_reason)


def _select_sources(
    fused: List[Tuple[float, Document]],
    ranked_lists: List[Tuple[float, List[Document]]],
    limit: int,
) -> List[Tuple[float, Document]]:
    """Give each query's best hit a slot, then fill by fused rank.

    Fusion alone lets the original question's results crowd out chunks that refined queries found
    for exactly the information the planner said was missing.
    """
    fused_score = {document.id: score for score, document in fused}
    selected: Dict[str, Document] = {}
    for _, documents in ranked_lists:
        if documents and len(selected) < limit:
            selected.setdefault(documents[0].id, documents[0])
    for _, document in fused:
        if len(selected) >= limit:
            break
        selected.setdefault(document.id, document)
    return sorted(
        ((fused_score[document_id], document) for document_id, document in selected.items()),
        key=lambda item: item[0],
        reverse=True,
    )


def _first_round(trace: List[SearchResult], document_id: str) -> int:
    return min(result.round_number for result in trace if result.document.id == document_id)


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------


def format_result(result: SearchResult, preview_chars: int = 450) -> str:
    preview = " ".join(result.document.text.split())[:preview_chars]
    query = f" | query: {result.query}" if result.query else ""
    return (
        f"Round {result.round_number} | #{result.rank} | "
        f"score={result.score:.4f} | {result.document.id}{query}\n{preview}"
    )


def build_context(documents: List[Document], max_chars: int = 9000, per_chunk_chars: int = 2000) -> str:
    context_parts: List[str] = []
    current_length = 0
    for document in documents:
        chunk = f"[{document.id}] " + " ".join(document.text.split())[:per_chunk_chars]
        if current_length + len(chunk) > max_chars:
            break
        context_parts.append(chunk)
        current_length += len(chunk)
    return "\n\n".join(context_parts)


def grounded_answer(question: str, sources: List[SearchResult], max_sentences: int = 5) -> str:
    question_terms = set(tokenize(question))
    ranked_sentences: List[Tuple[float, str, str]] = []

    for source in sources:
        sentences = SENTENCE_RE.split(" ".join(source.document.text.split()))
        for sentence in sentences:
            overlap = question_terms.intersection(tokenize(sentence))
            if not overlap:
                continue
            # Rank by overlap first, then by the source's fused rank.
            score = len(overlap) - source.rank / 100.0
            ranked_sentences.append((score, source.document.id, sentence.strip()))

    if not ranked_sentences:
        return (
            "I could not find enough grounded evidence in the retrieved corpus to answer this. "
            "Try rephrasing the question or adding more source documents."
        )

    selected: List[str] = []
    seen: Set[str] = set()
    for _, document_id, sentence in sorted(ranked_sentences, key=lambda item: item[0], reverse=True):
        if sentence in seen:
            continue
        selected.append(f"- {sentence} [{document_id}]")
        seen.add(sentence)
        if len(selected) >= max_sentences:
            break

    return (
        "This is a source-grounded draft for review, not medical advice or a diagnosis.\n"
        + "\n".join(selected)
    )


OUT_OF_SCOPE_MESSAGE = "This question is outside MedX's scope: it covers psychiatry and mental health, based on the DSM-5."

SCOPE_SYSTEM = """You are the scope check for MedX, an assistant for psychiatry and mental health based on the DSM-5.

In scope: mental disorders (diagnosis, criteria, symptoms, course, prevalence, risk factors, differential
diagnosis), psychiatric treatment and psychopharmacology, substance use, intoxication and withdrawal (including
acute management of overdose, intoxication and withdrawal, e.g. naloxone for opioid overdose), sleep,
eating, neurocognitive and neurodevelopmental disorders, personality, psychotherapy, suicide and self-harm,
and behavioral science.

Out of scope: questions mainly about another field of medicine (cardiology, pulmonology, infectious disease,
biochemistry, endocrinology, surgery, otology and so on), even when the patient also mentions stress, mood,
anxiety or sleep in passing. Judge what the question is asking, not incidental details of the case.

Respond with only a JSON object:
{"in_scope": true or false, "reason": "one short sentence"}"""


def check_scope(question: str, llm: FallbackLLM) -> Tuple[bool, str]:
    """Ask the planner model whether a question is about psychiatry; refusing here skips retrieval entirely."""
    reply = llm.chat_json(SCOPE_SYSTEM, f"Question:\n{question}", temperature=0.0, max_tokens=2048)
    return reply.get("in_scope") not in (False, "false", "False"), str(reply.get("reason", "")).strip()


ANSWER_SYSTEM = """You are MedX, a careful medical RAG assistant for psychiatry and mental health.

If the question is mainly about another field of medicine, do not answer it. Reply only:
\"""" + OUT_OF_SCOPE_MESSAGE + """\"

Use only the provided retrieved source chunks to answer the user's question.
Do not diagnose the user. Do not invent facts that are not in the sources.
If the sources are insufficient, say what is missing.
Use concise language and cite source ids inline like [Psichiatry_DSM-5_123].
End with: "This is educational information, not medical advice.\""""


def build_answer_prompt(question: str, sources: List[SearchResult]) -> str:
    context = build_context([source.document for source in sources])
    return f"""Question:
{question}

Retrieved sources:
{context}

Answer:"""


def get_gemini_api_key(api_key: Union[str, None] = None) -> Union[str, None]:
    return api_key or get_api_key("GEMINI_API_KEY")


def generate_with_llm(question: str, sources: List[SearchResult], llm: FallbackLLM) -> str:
    return llm.chat(ANSWER_SYSTEM, build_answer_prompt(question, sources), temperature=0.2, max_tokens=4096)


def answer_question(
    question: str,
    retriever: Any,
    rounds: int = 3,
    top_k: int = 5,
    source_limit: int = 8,
    generator: str = "auto",
    refiner: str = "auto",
    planner_llm: Optional[str] = None,
    answer_llm: Optional[str] = None,
    expansion_terms: int = 4,
    scope_check: bool = True,
) -> RAGResponse:
    """Check scope, run iterative retrieval, then answer from the selected sources.

    planner_llm / answer_llm are "provider" or "provider:model" (e.g. "groq", "nvidia:meta/llama-3.3-70b-instruct");
    None uses MEDX_PLANNER_LLM / MEDX_ANSWER_LLM from .env, else the first provider with a key.
    With scope_check, the planner model first decides whether the question is about psychiatry; if not,
    MedX refuses without retrieving or answering. A failed check lets the question through.
    """
    if generator not in ("auto", "llm", "extractive"):
        raise ValueError("generator must be one of: auto, llm, extractive")

    gate = get_llm("planner", planner_llm) if scope_check and refiner != "keywords" else None
    if gate is not None:
        try:
            in_scope, reason = check_scope(question, gate)
        except Exception as exc:
            print(f"Scope check failed ({exc.__class__.__name__}: {exc}); answering anyway", file=sys.stderr)
            in_scope = True
        if not in_scope:
            return RAGResponse(
                question=question,
                answer=OUT_OF_SCOPE_MESSAGE,
                sources=[],
                queries=[],
                rounds=[],
                stop_reason=f"out of scope ({gate.label}): {reason}",
                retrieval_trace=[],
                generator=gate.label,
                retriever=retriever.__class__.__name__,
            )

    outcome = iterative_retrieve(
        retriever,
        question,
        rounds=rounds,
        top_k=top_k,
        source_limit=source_limit,
        refiner=refiner,
        planner_llm=planner_llm,
        expansion_terms=expansion_terms,
    )

    answerer = get_llm("answer", answer_llm) if generator != "extractive" else None
    if generator == "llm" and answerer is None:
        raise LLMUnavailable("No LLM provider key found in .env for answering; use --generator extractive.")
    if answerer is not None:
        answer = generate_with_llm(question, outcome.sources, answerer)
        used_generator = answerer.label
    else:
        answer = grounded_answer(question, outcome.sources)
        used_generator = "extractive"

    return RAGResponse(
        question=question,
        answer=answer,
        sources=outcome.sources,
        queries=outcome.queries,
        rounds=outcome.rounds,
        stop_reason=outcome.stop_reason,
        retrieval_trace=outcome.trace,
        generator=used_generator,
        retriever=retriever.__class__.__name__,
    )


def answer_from_jsonl(
    question: str,
    data_path: Union[str, Path] = "Psichiatry_DSM-5.jsonl",
    rounds: int = 3,
    top_k: int = 5,
    source_limit: int = 8,
    generator: str = "auto",
    refiner: str = "auto",
    planner_llm: Optional[str] = None,
    answer_llm: Optional[str] = None,
    retriever: str = "hybrid",
    embedder: str = "auto",
    chroma_path: Union[str, Path] = "chroma_db",
    collection_name: str = "medx_dsm5",
) -> RAGResponse:
    searcher = build_retriever(
        data_path=data_path,
        retriever_name=retriever,
        chroma_path=chroma_path,
        collection_name=collection_name,
        embedder=embedder,
    )
    return answer_question(
        question,
        searcher,
        rounds=rounds,
        top_k=top_k,
        source_limit=source_limit,
        generator=generator,
        refiner=refiner,
        planner_llm=planner_llm,
        answer_llm=answer_llm,
    )


def response_to_dict(response: RAGResponse) -> dict:
    return {
        "question": response.question,
        "answer": response.answer,
        "generator": response.generator,
        "retriever": response.retriever,
        "stop_reason": response.stop_reason,
        "rounds": [
            {
                "round": round_.round_number,
                "refiner": round_.refiner,
                "queries": round_.queries,
                "missing_information": round_.missing_information,
                "note": round_.note,
            }
            for round_ in response.rounds
        ],
        "sources": [
            {
                "id": source.document.id,
                "title": source.document.title,
                "score": round(source.score, 5),
                "first_round": source.round_number,
                "rank": source.rank,
                "preview": " ".join(source.document.text.split())[:450],
            }
            for source in response.sources
        ],
        "retrieval_trace": [
            {
                "id": result.document.id,
                "score": round(result.score, 4),
                "round": result.round_number,
                "rank": result.rank,
                "query": result.query,
            }
            for result in response.retrieval_trace
        ],
    }


def load_questions(path: Union[str, Path]) -> List[dict]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))

    questions: List[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            questions.append(item)
    return questions


def evaluate_questions(
    questions_path: Union[str, Path],
    data_path: Union[str, Path] = "Psichiatry_DSM-5.jsonl",
    output_path: Union[str, Path] = "rag_test_results.jsonl",
    rounds: int = 3,
    top_k: int = 5,
    source_limit: int = 8,
    generator: str = "auto",
    refiner: str = "auto",
    planner_llm: Optional[str] = None,
    answer_llm: Optional[str] = None,
    retriever: str = "hybrid",
    embedder: str = "auto",
    chroma_path: Union[str, Path] = "chroma_db",
    collection_name: str = "medx_dsm5",
) -> dict:
    searcher = build_retriever(
        data_path=data_path,
        retriever_name=retriever,
        chroma_path=chroma_path,
        collection_name=collection_name,
        embedder=embedder,
    )
    questions = load_questions(questions_path)
    evaluated = 0
    hits = 0

    with Path(output_path).open("w", encoding="utf-8") as file:
        for item in questions:
            question = str(item.get("question", "")).strip()
            if not question:
                continue
            expected_ids = item.get("expected_doc_ids") or []
            if isinstance(expected_ids, str):
                expected_ids = [value.strip() for value in expected_ids.split(",") if value.strip()]

            response = answer_question(
                question,
                searcher,
                rounds=rounds,
                top_k=top_k,
                source_limit=source_limit,
                generator=generator,
                refiner=refiner,
                planner_llm=planner_llm,
                answer_llm=answer_llm,
            )
            payload = response_to_dict(response)
            payload["expected_doc_ids"] = expected_ids
            payload["hit_expected_doc"] = bool(
                expected_ids and {source["id"] for source in payload["sources"]}.intersection(expected_ids)
            )
            if expected_ids:
                evaluated += 1
                hits += int(payload["hit_expected_doc"])
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return {
        "questions": len(questions),
        "evaluated_with_expected_ids": evaluated,
        "hit_at_sources": hits / evaluated if evaluated else None,
        "output_path": str(output_path),
    }


def retrieve_from_jsonl(
    query: str,
    data_path: Union[str, Path] = "Psichiatry_DSM-5.jsonl",
    rounds: int = 3,
    top_k: int = 5,
    refiner: str = "keywords",
) -> RetrievalOutcome:
    """Notebook-friendly helper that does not use argparse. Uses BM25 so it needs no API calls."""
    index = BM25Index(load_jsonl(data_path))
    return iterative_retrieve(index, query, rounds=rounds, top_k=top_k, refiner=refiner)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Iterative retrieval and RAG testing for the MedX corpus.")
    parser.add_argument("mode_or_query", nargs="?", default="criteria for major depressive disorder")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--data", default="Psichiatry_DSM-5.jsonl")
    parser.add_argument("--questions")
    parser.add_argument("--output", default="rag_test_results.jsonl")
    parser.add_argument("--chroma-path", default="chroma_db")
    parser.add_argument("--collection", default="medx_dsm5")
    parser.add_argument("--reset-chroma", action="store_true")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5, help="Chunks retrieved per query.")
    parser.add_argument("--sources", type=int, default=8, help="Fused chunks passed to the generator.")
    parser.add_argument("--expansion-terms", type=int, default=4)
    parser.add_argument("--generator", choices=["auto", "llm", "extractive"], default="auto")
    parser.add_argument("--refiner", choices=["auto", "llm", "keywords"], default="auto")
    parser.add_argument(
        "--planner-llm",
        help='Query-planning model as "provider" or "provider:model" (nvidia, groq, gemini, openrouter, ollama). '
        "Defaults to MEDX_PLANNER_LLM in .env.",
    )
    parser.add_argument("--answer-llm", help="Answering model, same format. Defaults to MEDX_ANSWER_LLM in .env.")
    parser.add_argument("--retriever", choices=["hybrid", "chroma", "bm25"], default="hybrid")
    parser.add_argument("--embedder", choices=["auto", "local", "gemini", "hash"], default="auto")
    args = parser.parse_args()

    if args.mode_or_query == "ingest":
        try:
            searcher = ensure_chroma_ingested(
                data_path=args.data,
                db_path=args.chroma_path,
                collection_name=args.collection,
                reset=args.reset_chroma,
                embedder=args.embedder,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        print(
            json.dumps(
                {
                    "chroma_path": args.chroma_path,
                    "collection": searcher.collection_name,
                    "embedder": searcher.embedder.name,
                    "chunks": searcher.count(),
                },
                indent=2,
            )
        )
        return

    if args.mode_or_query == "test":
        if not args.questions:
            raise SystemExit("Use: python medx_retrieval.py test --questions questions.jsonl")
        try:
            summary = evaluate_questions(
                args.questions,
                data_path=args.data,
                output_path=args.output,
                rounds=args.rounds,
                top_k=args.top_k,
                source_limit=args.sources,
                generator=args.generator,
                refiner=args.refiner,
                planner_llm=args.planner_llm,
                answer_llm=args.answer_llm,
                retriever=args.retriever,
                embedder=args.embedder,
                chroma_path=args.chroma_path,
                collection_name=args.collection,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        print(json.dumps(summary, indent=2))
        return

    if args.mode_or_query == "ask":
        query = " ".join(args.question) or "criteria for major depressive disorder"
    else:
        query = " ".join([args.mode_or_query] + args.question)

    try:
        searcher = build_retriever(
            data_path=args.data,
            retriever_name=args.retriever,
            chroma_path=args.chroma_path,
            collection_name=args.collection,
            reset_chroma=args.reset_chroma,
            embedder=args.embedder,
        )
        response = answer_question(
            query,
            searcher,
            rounds=args.rounds,
            top_k=args.top_k,
            source_limit=args.sources,
            generator=args.generator,
            refiner=args.refiner,
            planner_llm=args.planner_llm,
            answer_llm=args.answer_llm,
            expansion_terms=args.expansion_terms,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Retriever: {response.retriever}")
    print(f"Generator: {response.generator}")
    print(f"Stopped: {response.stop_reason}")
    for round_ in response.rounds:
        print(f"\nRound {round_.round_number} ({round_.refiner})")
        if round_.missing_information:
            print(f"  missing: {round_.missing_information}")
        if round_.note:
            print(f"  note: {round_.note}")
        for query in round_.queries:
            print(f"  query: {query}")
    print()
    print(response.answer)
    print()
    print("Selected sources")
    for result in response.sources:
        print(format_result(result))
        print("-" * 80)


if __name__ == "__main__":
    main()
