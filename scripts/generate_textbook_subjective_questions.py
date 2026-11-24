"""Generate subjective evaluation questions FROM the 18-textbook corpus.

Unlike BioASQ (whose questions target PubMed journal knowledge), these
questions are written from and answered by the actual textbook passages,
so every question has ground-truth chunk IDs and a corpus-matched answer.

The generator:
  1. Samples chunks from each textbook proportionally.
  2. Groups chunks into single-chunk or multi-chunk contexts.
  3. Uses an LLM to write a standalone free-form question + reference answer
     from the passages, with used chunk IDs.
  4. Uses a separate judge LLM to score groundedness, standalone quality,
     and retrieval value.
  5. Filters by quality, deduplicates, and writes a frozen JSONL benchmark.

The output includes:
  - ~20-25 questions per textbook → ~400 total (configurable)
  - Each record has: question, answer, expected_doc_ids, textbook, type, difficulty
  - Plus ~50 BioASQ out-of-corpus questions for abstention testing

Usage:
  # Preview mode (5 questions, no file written)
  python scripts/generate_textbook_subjective_questions.py --preview

  # Full generation
  python scripts/generate_textbook_subjective_questions.py \
      --out datasets/subjective/textbook_subjective_benchmark_v1.jsonl \
      --per-book 25 --seed 42

  # With specific LLM provider
  python scripts/generate_textbook_subjective_questions.py \
      --out datasets/subjective/textbook_subjective_benchmark_v1.jsonl \
      --generator-llm gemini --judge-llm gemini
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from medragent.data import FULL_TEXTBOOK_NAMES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MIN_CHUNK_WORDS = 60
MAX_CONTEXT_CHARS = 6000
MULTI_CHUNK_FRACTION = 0.25

SKIP_PATTERNS = [
    "all rights reserved", "library of congress", "board of trustees",
    "manufactured in the united states", "correct citation", "index of",
    "copyright", "isbn", "printed in", "table of contents",
]

# Domain-agnostic question types (work across all 18 textbooks)
QUESTION_TYPES = {
    "factual": (
        0.20,
        "asks for a specific fact, definition, mechanism, or value that is "
        "directly stated in the passages",
    ),
    "explanation": (
        0.25,
        "asks the user to explain a process, mechanism, pathophysiology, or "
        "concept using information from the passages",
    ),
    "comparison": (
        0.15,
        "asks how two or more concepts, conditions, drugs, structures, or "
        "processes differ or relate to each other",
    ),
    "clinical_reasoning": (
        0.20,
        "presents a brief clinical scenario or situation and asks what "
        "conclusion, diagnosis, or next step follows from the passage content",
    ),
    "integration": (
        0.10,
        "requires combining information from multiple parts of the passage(s) "
        "to form a complete answer",
    ),
    "safety_limitation": (
        0.10,
        "asks about contraindications, adverse effects, limitations, caveats, "
        "or when the information would be insufficient",
    ),
}

DIFFICULTY_WEIGHTS = {"easy": 0.25, "medium": 0.50, "hard": 0.25}

# Friendly display names for textbooks
TEXTBOOK_DISPLAY = {
    "Anatomy_Gray": "Anatomy (Gray's)",
    "Biochemistry_Lippincott": "Biochemistry (Lippincott)",
    "Cell_Biology_Alberts": "Cell Biology (Alberts)",
    "First_Aid_Step1": "First Aid Step 1",
    "First_Aid_Step2": "First Aid Step 2",
    "Gynecology_Novak": "Gynecology (Novak)",
    "Histology_Ross": "Histology (Ross)",
    "Immunology_Janeway": "Immunology (Janeway)",
    "InternalMed_Harrison": "Internal Medicine (Harrison)",
    "Neurology_Adams": "Neurology (Adams)",
    "Obstentrics_Williams": "Obstetrics (Williams)",
    "Pathology_Robbins": "Pathology (Robbins)",
    "Pathoma_Husain": "Pathoma (Husain)",
    "Pediatrics_Nelson": "Pediatrics (Nelson)",
    "Pharmacology_Katzung": "Pharmacology (Katzung)",
    "Physiology_Levy": "Physiology (Levy)",
    "Psichiatry_DSM-5": "Psychiatry (DSM-5)",
    "Surgery_Schwartz": "Surgery (Schwartz)",
}

# Map textbook names to medical specialties for domain tagging
TEXTBOOK_DOMAIN = {
    "Anatomy_Gray": "anatomy",
    "Biochemistry_Lippincott": "biochemistry",
    "Cell_Biology_Alberts": "cell_biology",
    "First_Aid_Step1": "general_medicine",
    "First_Aid_Step2": "general_medicine",
    "Gynecology_Novak": "gynecology",
    "Histology_Ross": "histology",
    "Immunology_Janeway": "immunology",
    "InternalMed_Harrison": "internal_medicine",
    "Neurology_Adams": "neurology",
    "Obstentrics_Williams": "obstetrics",
    "Pathology_Robbins": "pathology",
    "Pathoma_Husain": "pathology",
    "Pediatrics_Nelson": "pediatrics",
    "Pharmacology_Katzung": "pharmacology",
    "Physiology_Levy": "physiology",
    "Psichiatry_DSM-5": "psychiatry",
    "Surgery_Schwartz": "surgery",
}


# ---------------------------------------------------------------------------
# Chunk loading
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_chunks(textbook: str, chunk_dir: Path) -> list[dict]:
    """Load chunks for a single textbook."""
    path = chunk_dir / f"{textbook}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    chunks = []
    with path.open("r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            content = raw.get("content") or raw.get("contents") or ""
            title = raw.get("title", "")
            if title and content.startswith(f"{title}. "):
                content = content[len(title) + 2:]
            content = normalize_text(content)
            if not content or len(content.split()) < MIN_CHUNK_WORDS:
                continue
            # Skip boilerplate
            lower = content.lower()
            if any(p in lower for p in SKIP_PATTERNS):
                continue
            chunks.append({
                "id": raw.get("id", f"{textbook}_{i}"),
                "content": content,
                "title": title,
                "textbook": textbook,
            })
    return chunks


def load_all_chunks(chunk_dir: Path) -> dict[str, list[dict]]:
    """Load chunks grouped by textbook."""
    all_chunks = {}
    for tb in FULL_TEXTBOOK_NAMES:
        chunks = load_chunks(tb, chunk_dir)
        all_chunks[tb] = chunks
        print(f"  {tb}: {len(chunks)} usable chunks", file=sys.stderr)
    total = sum(len(v) for v in all_chunks.values())
    print(f"  Total usable: {total}", file=sys.stderr)
    return all_chunks


# ---------------------------------------------------------------------------
# Chunk grouping (single and multi-chunk)
# ---------------------------------------------------------------------------

def sample_groups(
    chunks: list[dict],
    count: int,
    rng: random.Random,
) -> list[list[dict]]:
    """Create `count` chunk groups: ~75% single, ~25% multi-chunk (adjacent)."""
    if len(chunks) < 3:
        return [[c] for c in rng.sample(chunks, min(count, len(chunks)))]

    groups = []
    pool = chunks[:]
    rng.shuffle(pool)
    idx = 0

    while len(groups) < count and idx < len(pool):
        chunk = pool[idx]
        idx += 1
        if rng.random() < MULTI_CHUNK_FRACTION:
            # Find an adjacent chunk by ID
            doc_match = re.match(r"(.+)_(\d+)$", chunk["id"])
            if doc_match:
                prefix, num = doc_match.group(1), int(doc_match.group(2))
                partner_id = f"{prefix}_{num + 1}"
                partner = next((c for c in chunks if c["id"] == partner_id), None)
                if partner and len(partner["content"].split()) >= MIN_CHUNK_WORDS:
                    groups.append([chunk, partner])
                    continue
        groups.append([chunk])

    return groups


def truncate_context(group: list[dict], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Format chunk group as labeled passages for the LLM prompt."""
    parts = []
    used = 0
    per_budget = max(800, max_chars // max(len(group), 1))
    for chunk in group:
        text = chunk["content"][:per_budget]
        part = f"[{chunk['id']}]\n{text}"
        if used + len(part) > max_chars:
            break
        parts.append(part)
        used += len(part)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM interaction (uses the project's existing providers)
# ---------------------------------------------------------------------------

GENERATE_SYSTEM = """You are building a retrieval benchmark for a medical textbook RAG system.
You will be given passages from a {textbook_display} textbook, each tagged with an ID.

Write exactly one high-quality, open-ended (subjective/free-form) question and a reference
answer based ONLY on the passages.

Question type to write: {type_instruction}
Difficulty level: {difficulty}
  - easy: direct factual lookup from one passage
  - medium: requires understanding, paraphrase, or connecting ideas within a passage
  - hard: requires synthesis, comparison, or integration across passage parts

RULES:
- The question must be fully STANDALONE — a medical student should understand it without
  seeing the passages. Never mention "passage", "excerpt", "text", "source", "context",
  or "according to".
- Paraphrase heavily; do NOT copy distinctive wording from the passages.
- The answer must be 2-5 sentences, fully supported by the passages, and NOT use outside
  knowledge beyond what's in the given text.
- For used_ids, list ONLY the passage IDs required to answer. If you used only one
  passage, list only that one.
- Do NOT ask about headings, page numbers, formatting, or OCR artifacts.
- Write a question that a medical professional or student would actually ask.

Respond with ONLY a JSON object:
{{"question": "...", "answer": "...", "used_ids": ["id1", ...]}}"""

GENERATE_USER = """PASSAGES:
{context}

Write one {difficulty} {question_type} question based on these passages."""

JUDGE_SYSTEM = """You evaluate question-answer pairs generated from medical textbook passages.
Score each criterion from 1 (worst) to 5 (best).

PASSAGES:
{context}

QUESTION:
{question}

ANSWER:
{answer}

Criteria:
- groundedness: Is every claim in the answer fully supported by the passages? (1=unsupported, 5=fully supported)
- standalone: Does the question make sense without seeing the passages? (1=depends on passages, 5=fully standalone)
- retrieval_value: Would this question be useful for testing a retrieval system? (1=trivial/malformed, 5=excellent)
- answer_quality: Is the answer clear, complete, and well-written? (1=poor, 5=excellent)

Respond with ONLY a JSON object:
{{"groundedness": n, "standalone": n, "retrieval_value": n, "answer_quality": n}}"""


def _extract_json(text: str) -> dict:
    """Parse first JSON object from LLM reply."""
    # Strip thinking tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"No JSON in reply: {text[:200]}")
    return json.loads(text[start:end + 1])


def _call_llm(
    system: str, user: str, provider: str, temperature: float = 0.6
) -> dict:
    """Call an LLM via the iterative_retrival LLM client or fall back to
    a simple Gemini API call."""
    # Try using the project's medx_llm module
    try:
        iterative_dir = PROJECT_ROOT / "iterative_retrival"
        if str(iterative_dir) not in sys.path:
            sys.path.insert(0, str(iterative_dir))
        from medx_llm import get_llm, load_project_env
        load_project_env()
        llm = get_llm("answer", provider if provider else None)
        if llm is not None:
            reply = llm.chat(system, user, temperature=temperature,
                             max_tokens=2048, json_output=True)
            return _extract_json(reply)
    except ImportError:
        pass

    # Fallback: direct Gemini API
    try:
        from google import genai
        from google.genai import types
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
                contents=f"{system}\n\n{user}",
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=2048,
                    response_mime_type="application/json",
                ),
            )
            return _extract_json(response.text)
    except ImportError:
        pass

    raise RuntimeError(
        "No LLM provider available. Set GEMINI_API_KEY, NVIDIA_API_KEY, or "
        "GROQ_API_KEY in your .env file."
    )


# ---------------------------------------------------------------------------
# Question generation pipeline
# ---------------------------------------------------------------------------

def generate_question(
    group: list[dict],
    textbook: str,
    rng: random.Random,
    generator_provider: str,
    judge_provider: str,
    min_score: float = 3.5,
) -> Optional[dict]:
    """Generate and judge one question from a chunk group. Returns None on failure."""
    # Pick type and difficulty
    type_names = list(QUESTION_TYPES)
    type_weights = [QUESTION_TYPES[t][0] for t in type_names]
    qtype = rng.choices(type_names, weights=type_weights)[0]

    # For multi-chunk groups, override to integration type
    if len(group) > 1:
        qtype = "integration"

    diff_names = list(DIFFICULTY_WEIGHTS)
    diff_weights = list(DIFFICULTY_WEIGHTS.values())
    difficulty = rng.choices(diff_names, weights=diff_weights)[0]

    type_instruction = QUESTION_TYPES.get(qtype, QUESTION_TYPES["explanation"])[1]
    context = truncate_context(group)
    seed_ids = [c["id"] for c in group]

    # Generate
    try:
        system = GENERATE_SYSTEM.format(
            textbook_display=TEXTBOOK_DISPLAY.get(textbook, textbook),
            type_instruction=type_instruction,
            difficulty=difficulty,
        )
        user = GENERATE_USER.format(
            context=context, difficulty=difficulty, question_type=qtype,
        )
        qa = _call_llm(system, user, generator_provider, temperature=0.7)
    except Exception as e:
        print(f"  Generation failed: {e}", file=sys.stderr)
        return None

    question = str(qa.get("question", "")).strip()
    answer = str(qa.get("answer", "")).strip()
    used_ids = [uid for uid in qa.get("used_ids", []) if uid in seed_ids]

    if not question or not answer or not used_ids:
        return None

    # For multi-chunk, require at least 2 used IDs
    if len(group) > 1 and len(set(used_ids)) < 2:
        return None

    # Judge
    try:
        judge_prompt = JUDGE_SYSTEM.format(
            context=context, question=question, answer=answer,
        )
        scores = _call_llm(judge_prompt, "Score this Q&A pair.", judge_provider,
                           temperature=0.1)
        score_values = []
        for key in ("groundedness", "standalone", "retrieval_value", "answer_quality"):
            v = scores.get(key)
            if isinstance(v, dict):
                v = v.get("score", v.get("value", 0))
            try:
                score_values.append(float(v))
            except (TypeError, ValueError):
                score_values.append(0.0)

        if min(score_values) < min_score:
            return None
    except Exception as e:
        print(f"  Judging failed: {e}", file=sys.stderr)
        # Accept without judging rather than discarding
        score_values = [0.0, 0.0, 0.0, 0.0]

    return {
        "question_id": f"textbook_{textbook}_{hashlib.md5(question.encode()).hexdigest()[:8]}",
        "question": question,
        "reference_answer": answer,
        "expected_doc_ids": sorted(set(used_ids), key=used_ids.index),
        "textbook": textbook,
        "domain": TEXTBOOK_DOMAIN.get(textbook, "other"),
        "question_type": qtype,
        "difficulty": difficulty,
        "multi_chunk": len(group) > 1,
        "source_chunk_count": len(group),
        "judge_scores": {
            "groundedness": score_values[0],
            "standalone": score_values[1],
            "retrieval_value": score_values[2],
            "answer_quality": score_values[3],
        },
        "benchmark_version": "textbook_subjective_v1",
        "review_status": "auto_generated",
    }


def add_abstention_questions(
    bioasq_path: Path, count: int, rng: random.Random
) -> list[dict]:
    """Add BioASQ questions as out-of-corpus abstention test cases."""
    if not bioasq_path.exists():
        print(f"  BioASQ file not found at {bioasq_path}; skipping abstention set",
              file=sys.stderr)
        return []

    with bioasq_path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Pick questions likely NOT answerable from textbooks
    # (those that got expected_abstention in the model review, or random from unreviewed)
    abstention_pool = [
        r for r in records
        if r.get("expected_evidence_status") == "expected_abstention"
        or r.get("review_status") == "unreviewed"
    ]
    if not abstention_pool:
        abstention_pool = records

    sampled = rng.sample(abstention_pool, min(count, len(abstention_pool)))
    out = []
    for r in sampled:
        out.append({
            "question_id": f"abstention_{r.get('question_id', r.get('source_question_id', ''))}",
            "question": r["question"],
            "reference_answer": r.get("reference_answer", r.get("reference_answers", [""])[0]
                                       if isinstance(r.get("reference_answers"), list) else ""),
            "expected_doc_ids": [],  # No textbook should answer this
            "textbook": "none",
            "domain": "out_of_corpus",
            "question_type": "abstention_test",
            "difficulty": "n/a",
            "multi_chunk": False,
            "source_chunk_count": 0,
            "expected_abstention": True,
            "judge_scores": {},
            "benchmark_version": "textbook_subjective_v1",
            "review_status": "auto_generated",
            "source_dataset": "BioASQ-13b",
        })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate subjective QA benchmark from 18 textbooks."
    )
    parser.add_argument(
        "--out", type=Path,
        default=PROJECT_ROOT / "datasets" / "subjective" / "textbook_subjective_benchmark_v1.jsonl",
    )
    parser.add_argument("--chunk-dir", type=Path,
                        default=PROJECT_ROOT / "datasets" / "MedRAG" / "chunk")
    parser.add_argument("--per-book", type=int, default=25,
                        help="Questions to generate per textbook (default 25)")
    parser.add_argument("--abstention-count", type=int, default=50,
                        help="BioASQ abstention test questions (default 50)")
    parser.add_argument("--bioasq-path", type=Path,
                        default=PROJECT_ROOT / "datasets" / "subjective" / "bioasq13b_summary_unreviewed.jsonl")
    parser.add_argument("--min-score", type=float, default=3.5,
                        help="Minimum judge score per criterion (default 3.5)")
    parser.add_argument("--overgen", type=float, default=2.0,
                        help="Overgeneration factor (default 2.0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generator-llm", default="",
                        help="Generator LLM provider (e.g., gemini, nvidia)")
    parser.add_argument("--judge-llm", default="",
                        help="Judge LLM provider (e.g., gemini, nvidia)")
    parser.add_argument("--preview", action="store_true",
                        help="Generate 2 questions from 2 books only")
    parser.add_argument("--books", nargs="*",
                        help="Specific textbook names to process (default: all 18)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load .env
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env", override=True)
        # Also try iterative_retrival .env
        env2 = PROJECT_ROOT / "iterative_retrival" / ".env"
        if env2.exists():
            load_dotenv(env2, override=True)
    except ImportError:
        pass

    books = args.books or list(FULL_TEXTBOOK_NAMES)
    if args.preview:
        books = books[:2]
        args.per_book = 2
        args.abstention_count = 2

    print(f"Loading chunks from {args.chunk_dir}...", file=sys.stderr)
    all_chunks = {}
    for tb in books:
        if tb not in FULL_TEXTBOOK_NAMES:
            print(f"  Warning: '{tb}' not in FULL_TEXTBOOK_NAMES, skipping",
                  file=sys.stderr)
            continue
        all_chunks[tb] = load_chunks(tb, args.chunk_dir)
        print(f"  {tb}: {len(all_chunks[tb])} usable chunks", file=sys.stderr)

    # Generate questions per textbook
    all_questions = []
    seen_keys = set()
    gen_target = int(args.per_book * args.overgen)
    stats = defaultdict(lambda: {"generated": 0, "accepted": 0, "failed": 0,
                                  "low_score": 0, "duplicate": 0})

    for tb, chunks in all_chunks.items():
        if len(chunks) < 5:
            print(f"  Skipping {tb}: too few chunks ({len(chunks)})",
                  file=sys.stderr)
            continue

        print(f"\nGenerating for {TEXTBOOK_DISPLAY.get(tb, tb)} "
              f"(target: {args.per_book}, generating: {gen_target})...",
              file=sys.stderr)

        groups = sample_groups(chunks, gen_target, rng)
        accepted = 0

        for gi, group in enumerate(groups):
            if accepted >= args.per_book:
                break

            result = generate_question(
                group, tb, rng,
                args.generator_llm, args.judge_llm,
                min_score=args.min_score,
            )
            stats[tb]["generated"] += 1

            if result is None:
                stats[tb]["failed"] += 1
                continue

            # Dedup by normalized question text
            key = re.sub(r"\W+", " ", result["question"].lower()).strip()
            if key in seen_keys:
                stats[tb]["duplicate"] += 1
                continue
            seen_keys.add(key)

            all_questions.append(result)
            accepted += 1
            stats[tb]["accepted"] += 1

            if (gi + 1) % 5 == 0:
                print(f"  [{tb}] {gi+1}/{len(groups)} attempts, "
                      f"{accepted}/{args.per_book} accepted", file=sys.stderr)

            # Rate limiting
            time.sleep(0.5)

        print(f"  {tb}: {accepted}/{args.per_book} questions accepted",
              file=sys.stderr)

    # Add abstention questions
    abstention = add_abstention_questions(
        args.bioasq_path, args.abstention_count, rng
    )
    all_questions.extend(abstention)

    # Write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        for q in all_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # Summary
    textbook_qs = [q for q in all_questions if q["domain"] != "out_of_corpus"]
    abstention_qs = [q for q in all_questions if q["domain"] == "out_of_corpus"]
    by_domain = Counter(q["domain"] for q in textbook_qs)
    by_type = Counter(q["question_type"] for q in textbook_qs)
    by_diff = Counter(q["difficulty"] for q in textbook_qs)

    summary = {
        "output": str(args.out),
        "total_questions": len(all_questions),
        "textbook_questions": len(textbook_qs),
        "abstention_questions": len(abstention_qs),
        "books_covered": len(set(q["textbook"] for q in textbook_qs)),
        "by_domain": dict(sorted(by_domain.items())),
        "by_type": dict(sorted(by_type.items())),
        "by_difficulty": dict(sorted(by_diff.items())),
        "multi_chunk": sum(q["multi_chunk"] for q in textbook_qs),
        "per_book_stats": {
            tb: dict(s) for tb, s in sorted(stats.items())
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
