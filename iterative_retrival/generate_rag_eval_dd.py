"""
Generate a RAG evaluation set with NVIDIA NeMo Data Designer + Nemotron.

Install:
  pip install data-designer pandas scikit-learn pydantic

Environment:
  NVIDIA_API_KEY in .env (loaded automatically)
  optionally NEMOTRON_MODEL and JUDGE_MODEL; the judge (gpt-oss-20b) is a different model family
  from the generator (Nemotron), so questions are not graded by the model that wrote them; GLM was
  accurate but took ~2 hours for 180 grades

Examples:
  python generate_rag_eval_dd.py --chunks Psichiatry_DSM-5.jsonl --n 100 --preview
  python generate_rag_eval_dd.py --chunks Psichiatry_DSM-5.jsonl --n 100 --out eval_questions.jsonl

Output JSONL rows:
  {"question", "answer", "expected_doc_ids", "type", "difficulty"}
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

import data_designer.config as dd

from medx_llm import load_project_env

load_project_env()

# nemotron-3-super answered a planning benchmark in ~6-10s; nemotron-3.5-lightning took 80-95s.
GEN_MODEL = os.environ.get("NEMOTRON_MODEL", "nvidia/nemotron-3-super-120b-a12b")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-oss-20b")
PROVIDER = "nvidia"

MIN_WORDS = 70
MAX_CONTEXT_CHARS = 5000
OVERGEN = 1.8
MULTI_FRACTION = 0.25
DISTANT_SHARE = 0.60

QUESTION_TYPES = {
    "diagnostic_criteria": (
        0.25,
        "asks about features, duration, thresholds, exclusions, or diagnosis-defining criteria",
    ),
    "diagnostic_distinction": (
        0.20,
        "asks how one condition is distinguished from, overlaps with, or interacts with a similar condition",
    ),
    "course_and_risk": (
        0.15,
        "asks about prevalence, onset, course, risk factors, prognosis, or comorbidity",
    ),
    "definition": (
        0.15,
        "asks what a term, specifier, concept, or category means",
    ),
    "clinical_application": (
        0.15,
        "uses a short clinical vignette and asks what conclusion follows from the provided material",
    ),
    "safety_or_limitation": (
        0.10,
        "asks about cautions, limitations, differential diagnosis, or when the information is insufficient",
    ),
}

DIFFICULTY_WEIGHTS = {"easy": 0.30, "medium": 0.50, "hard": 0.20}

SKIP_PATTERNS = [
    "all rights reserved",
    "library of congress",
    "board of trustees",
    "manufactured in the united states",
    "correct citation",
    "index of",
    "copyright",
]


class QA(BaseModel):
    question: str = Field(description="Standalone user-style question; no passage or id references.")
    answer: str = Field(description="One to three concise sentences, fully supported by the passages.")
    used_ids: list[str] = Field(description="Only the passage ids required to answer.")


@dataclass(frozen=True)
class Chunk:
    id: str
    content: str


def normalize_text(text: str) -> str:
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_jsonl_or_json(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"No data found in {path}")
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of chunk objects.")
        return data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_chunks(path: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for line_number, row in enumerate(load_jsonl_or_json(path), start=1):
        # Prefer "content": "contents" repeats the title before every chunk. Matches medx_retrieval.load_jsonl,
        # so questions are written from the same text the retriever indexes.
        content = str(row.get("content") or row.get("contents") or row.get("text") or "")
        title = str(row.get("title") or "")
        if title and content.startswith(f"{title}. "):
            content = content[len(title) + 2 :]
        chunk_id = row.get("id") or f"chunk_{line_number}"
        content = normalize_text(content)
        if not content:
            continue
        chunks.append(Chunk(id=str(chunk_id), content=content))
    if not chunks:
        raise ValueError(f"No usable chunks found in {path}")
    return chunks


def is_useful_chunk(chunk: Chunk) -> bool:
    content_lower = chunk.content.lower()
    if len(chunk.content.split()) < MIN_WORDS:
        return False
    return not any(pattern in content_lower for pattern in SKIP_PATTERNS)


def doc_key(chunk_id: str) -> tuple[str, int]:
    match = re.match(r"(.+)_(\d+)$", chunk_id)
    if not match:
        return chunk_id, -1
    return match.group(1), int(match.group(2))


def build_related_index(chunks: list[Chunk], k: int = 15) -> dict[int, list[int]]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
    )
    matrix = vectorizer.fit_transform([chunk.content for chunk in chunks])
    neighbors: dict[int, list[int]] = {}
    for start in range(0, len(chunks), 500):
        sims = (matrix[start : start + 500] @ matrix.T).toarray()
        for offset, row in enumerate(sims):
            index = start + offset
            neighbors[index] = [int(i) for i in row.argsort()[::-1] if int(i) != index][:k]
    return neighbors


def far_partners(
    chunks: list[Chunk],
    neighbors: dict[int, list[int]],
    index: int,
    count: int,
    min_gap: int = 5,
) -> list[Chunk]:
    doc, number = doc_key(chunks[index].id)
    candidates: list[Chunk] = []
    for candidate_index in neighbors.get(index, []):
        candidate_doc, candidate_number = doc_key(chunks[candidate_index].id)
        if candidate_doc != doc or abs(candidate_number - number) >= min_gap:
            candidates.append(chunks[candidate_index])
    random.shuffle(candidates)
    return candidates[:count]


def adjacent_partners(by_id: dict[str, Chunk], chunk: Chunk, count: int) -> list[Chunk]:
    doc, number = doc_key(chunk.id)
    if number < 0:
        return []
    partners = []
    for step in range(1, count + 1):
        partner = by_id.get(f"{doc}_{number + step}")
        if partner is not None and is_useful_chunk(partner):
            partners.append(partner)
    return partners


def truncate_group(group: list[Chunk], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    parts: list[str] = []
    used = 0
    per_chunk_budget = max(800, max_chars // max(len(group), 1))
    for chunk in group:
        content = chunk.content[:per_chunk_budget]
        part = f"[{chunk.id}]\n{content}"
        if used + len(part) > max_chars:
            break
        parts.append(part)
        used += len(part)
    return "\n\n".join(parts)


def build_seed(chunks: list[Chunk], row_count: int) -> pd.DataFrame:
    usable = [chunk for chunk in chunks if is_useful_chunk(chunk)]
    if len(usable) < 10:
        raise ValueError("Too few useful chunks after filtering. Lower MIN_WORDS or inspect the input data.")

    by_id = {chunk.id: chunk for chunk in usable}
    positions = {chunk.id: index for index, chunk in enumerate(usable)}
    neighbors = build_related_index(usable)
    type_names = list(QUESTION_TYPES)
    type_weights = [QUESTION_TYPES[name][0] for name in type_names]

    rows: list[dict[str, str]] = []
    pool: list[Chunk] = []
    while len(rows) < row_count:
        if not pool:
            pool = usable[:]
            random.shuffle(pool)

        seed = pool.pop()
        group = [seed]
        qtype = random.choices(type_names, weights=type_weights)[0]

        if random.random() < MULTI_FRACTION:
            if random.random() < DISTANT_SHARE:
                group.extend(far_partners(usable, neighbors, positions[seed.id], random.choice([1, 2])))
                if len(group) > 1:
                    qtype = "multi_chunk_distant"
            else:
                group.extend(adjacent_partners(by_id, seed, random.choice([1, 2])))
                if len(group) > 1:
                    qtype = "multi_chunk_adjacent"

        multi = len(group) > 1
        random.shuffle(group)
        rows.append(
            {
                "context": truncate_group(group),
                "chunk_ids": json.dumps([chunk.id for chunk in group]),
                "question_type": qtype,
                "type_instr": (
                    "requires combining information from at least two passages, where each passage contributes "
                    "a necessary part of the answer"
                    if multi
                    else QUESTION_TYPES[qtype][1]
                ),
            }
        )
    return pd.DataFrame(rows)


GEN_PROMPT = """You are building a retrieval benchmark for a psychiatry textbook RAG system.

PASSAGES, each tagged with an id:
{{ context }}

Write exactly one high-quality question that {{ type_instr }}.

Difficulty: {{ difficulty }}
- easy: direct factual lookup
- medium: requires understanding and paraphrase
- hard: requires synthesis, comparison, or careful exclusion

Rules:
- The question must be fully standalone.
- Never mention passages, excerpts, context, source ids, or the phrase "according to".
- Paraphrase heavily; do not copy distinctive wording from the passages.
- The answer must be fully supported by the passages and must not use outside knowledge.
- Do not ask about headings, page numbers, OCR artifacts, formatting, or fragments cut off at passage edges.
- For used_ids, list only ids that are required to answer correctly."""


JUDGE_PROMPT = """PASSAGES:
{{ context }}

QUESTION:
{{ qa.question }}

ANSWER:
{{ qa.answer }}

Evaluate strictly against the passages only. Penalize outside knowledge, vague support, copied wording,
questions that depend on seeing the passage, and answers that omit important caveats."""


SCORES = [
    dd.Score(
        name="Groundedness",
        description="Is every answer claim fully supported by the passages?",
        options={
            1: "Unsupported",
            2: "Mostly unsupported",
            3: "Partly supported",
            4: "Mostly supported",
            5: "Fully supported",
        },
    ),
    dd.Score(
        name="Standalone",
        description="Does the question make sense without seeing the passages and avoid source references?",
        options={
            1: "Depends on the passages",
            3: "Somewhat standalone",
            5: "Fully standalone",
        },
    ),
    dd.Score(
        name="RetrievalValue",
        description="Would this question be useful for testing retrieval rather than being trivial or malformed?",
        options={
            1: "Not useful",
            3: "Somewhat useful",
            5: "Very useful",
        },
    ),
]


def build_config() -> dd.DataDesignerConfigBuilder:
    def model_config(alias: str, model: str) -> dd.ModelConfig:
        return dd.ModelConfig(
            alias=alias,
            model=model,
            provider=PROVIDER,
            inference_parameters=dd.ChatCompletionInferenceParams(
                temperature=0.6 if alias == "gen" else 0.1,
                max_tokens=4096,
                max_parallel_requests=4,
            ),
        )

    return dd.DataDesignerConfigBuilder(
        model_configs=[
            model_config("gen", GEN_MODEL),
            model_config("judge", JUDGE_MODEL),
        ]
    )


def parse_jsonish(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    raise TypeError(f"Expected dict or JSON string, got {type(value).__name__}")


def score_value(value: Any) -> float:
    if isinstance(value, dict):
        value = value.get("score", value.get("value"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def judge_scores(value: Any) -> dict[str, float]:
    parsed = parse_jsonish(value)
    return {name: score_value(score) for name, score in parsed.items()}


def normalize_question_key(question: str) -> str:
    return re.sub(r"\W+", " ", question.lower()).strip()


def create_dataset(args: argparse.Namespace) -> None:
    from data_designer.interface import DataDesigner

    random.seed(args.seed)
    chunks = load_chunks(args.chunks)
    seed_count = 5 if args.preview else int(args.n * args.overgen)
    seed_df = build_seed(chunks, seed_count)

    config = build_config()
    config.with_seed_dataset(dd.DataFrameSeedSource(df=seed_df))
    config.add_column(
        dd.SamplerColumnConfig(
            name="difficulty",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=list(DIFFICULTY_WEIGHTS),
                weights=list(DIFFICULTY_WEIGHTS.values()),
            ),
        )
    )
    config.add_column(
        dd.LLMStructuredColumnConfig(
            name="qa",
            model_alias="gen",
            prompt=GEN_PROMPT,
            output_format=QA,
        )
    )
    config.add_column(
        dd.LLMJudgeColumnConfig(
            name="judge",
            model_alias="judge",
            prompt=JUDGE_PROMPT,
            scores=SCORES,
        )
    )

    designer = DataDesigner()
    if args.preview:
        designer.preview(config, num_records=5).display_sample_record()
        return

    results = designer.create(config, num_records=len(seed_df), dataset_name=args.dataset_name)
    df = results.load_dataset()

    seen: set[str] = set()
    accepted: list[dict[str, Any]] = []
    stats = {
        "generated": len(df),
        "bad_ids": 0,
        "low_score": 0,
        "duplicate": 0,
        "malformed": 0,
    }

    for _, row in df.iterrows():
        try:
            qa = parse_jsonish(row["qa"])
            seed_ids = json.loads(row["chunk_ids"])
            used_ids = [chunk_id for chunk_id in qa.get("used_ids", []) if chunk_id in seed_ids]
            question = str(qa["question"]).strip()
            answer = str(qa["answer"]).strip()
        except Exception:
            stats["malformed"] += 1
            continue

        is_multi = str(row["question_type"]).startswith("multi_chunk")
        if not used_ids or (is_multi and len(set(used_ids)) < 2):
            stats["bad_ids"] += 1
            continue

        try:
            if min(judge_scores(row["judge"]).values()) < args.min_score:
                stats["low_score"] += 1
                continue
        except Exception:
            stats["low_score"] += 1
            continue

        key = normalize_question_key(question)
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)

        accepted.append(
            {
                "question": question,
                "answer": answer,
                "expected_doc_ids": sorted(set(used_ids), key=used_ids.index),
                "type": row["question_type"],
                "difficulty": row["difficulty"],
            }
        )
        if len(accepted) >= args.n:
            break

    output_path = Path(args.out)
    with output_path.open("w", encoding="utf-8") as file:
        for item in accepted:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Kept {len(accepted)}/{args.n} requested -> {output_path}")
    print(f"Filtered: {json.dumps(stats, indent=2)}")
    if len(accepted) < args.n:
        print("Short of target: increase --overgen or lower --min-score, then rerun.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, help="Input JSONL/JSON chunks. Supports content, contents, or text.")
    parser.add_argument("--out", default="eval_questions.jsonl")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--min-score", type=float, default=4.0)
    parser.add_argument("--overgen", type=float, default=OVERGEN)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-name", default="medx_rag_eval")
    args = parser.parse_args()
    create_dataset(args)


if __name__ == "__main__":
    main()
