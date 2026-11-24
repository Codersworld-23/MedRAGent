"""Create a transparent, model-assisted technical review subset.

This removes the engineering dependency on external human review while keeping
the provenance honest: labels made here are suitable for retrieval development
and regression tests, not clinician validation or medical-safety claims.
"""

from __future__ import annotations

import argparse
import heapq
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from medragent.data import FULL_TEXTBOOK_NAMES
from src.subjective_benchmark import validate_records

TOKEN = re.compile(r"[a-z][a-z0-9-]{2,}")
STOPWORDS = {
    "about", "after", "against", "among", "answer", "are", "been", "between",
    "can", "does", "during", "each", "explain", "from", "have", "into", "is",
    "its", "medical", "more", "most", "must", "of", "on", "or", "the", "their",
    "these", "this", "those", "was", "what", "when", "which", "with", "would",
}
REVIEWER = "Codex model-assisted lexical review v1 (non-clinical)"


def terms(text: str) -> set[str]:
    return {term for term in TOKEN.findall(text.lower()) if term not in STOPWORDS}


def evenly_spaced(records: list[dict], count: int) -> list[dict]:
    if count > len(records):
        raise ValueError("sample size exceeds available records")
    if count == len(records):
        return records.copy()
    positions = {round(index * (len(records) - 1) / (count - 1)) for index in range(count)}
    return [record for index, record in enumerate(records) if index in positions]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=250)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-dir", type=Path, default=PROJECT_ROOT / "datasets" / "MedRAG" / "chunk")
    args = parser.parse_args()
    if args.sample_size < 2 or args.top_k <= 0:
        parser.error("--sample-size must be at least 2 and --top-k must be positive")

    with args.input.open("r", encoding="utf-8") as handle:
        source = [json.loads(line) for line in handle if line.strip()]
    validate_records(source)
    records = evenly_spaced(source, args.sample_size)
    query_terms = [terms(record["question"]) for record in records]
    by_term: dict[str, list[int]] = defaultdict(list)
    for index, query in enumerate(query_terms):
        for term in query:
            by_term[term].append(index)
    heaps: list[list[tuple[float, str, dict]]] = [[] for _ in records]

    scanned = 0
    for textbook in FULL_TEXTBOOK_NAMES:
        path = args.chunk_dir / f"{textbook}.jsonl"
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                text = raw.get("content") or raw.get("contents") or ""
                if not isinstance(text, str) or not text.strip():
                    continue
                scanned += 1
                frequencies = Counter(terms(text))
                applicable: set[int] = set()
                for term in frequencies:
                    applicable.update(by_term.get(term, ()))
                for index in applicable:
                    overlap = sum(min(frequencies[term], 2) for term in query_terms[index])
                    coverage = overlap / max(len(query_terms[index]), 1)
                    score = overlap + coverage
                    candidate = {
                        "chunk_id": raw["id"],
                        "textbook_name": textbook,
                        "score": round(score, 6),
                        "query_term_coverage": round(coverage, 6),
                        "text_excerpt": text[:500],
                    }
                    item = (score, raw["id"], candidate)
                    if len(heaps[index]) < args.top_k:
                        heapq.heappush(heaps[index], item)
                    elif item[:2] > heaps[index][0][:2]:
                        heapq.heapreplace(heaps[index], item)

    for index, record in enumerate(records):
        candidates = [item[2] for item in sorted(heaps[index], reverse=True)]
        top = candidates[0] if candidates else None
        record["candidate_textbook_evidence"] = candidates
        record["candidate_retrieval"] = {
            "method": "streaming_lexical_overlap",
            "top_k": args.top_k,
            "corpus": "MedRAG_18_textbooks_125847_chunks",
            "gold_evidence": False,
        }
        if top and top["score"] >= 4 and top["query_term_coverage"] >= 0.12:
            record["supporting_chunk_ids"] = [top["chunk_id"]]
            record["expected_evidence_status"] = "partial"
            confidence = "low" if top["query_term_coverage"] < 0.25 else "medium"
            note = "Candidate has lexical question overlap; label is model-assisted and requires no clinical interpretation."
        else:
            record["supporting_chunk_ids"] = []
            record["expected_evidence_status"] = "expected_abstention"
            confidence = "low"
            note = "No candidate met the transparent lexical-support threshold; abstention is the technical default."
        record["review_status"] = "model_reviewed"
        record["reviewer"] = REVIEWER
        record["reviewer_confidence"] = confidence
        record["review_notes"] = note

    known_ids = {
        candidate["chunk_id"]
        for record in records
        for candidate in record["candidate_textbook_evidence"]
    }
    summary = validate_records(records, known_chunk_ids=known_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "scanned_chunks": scanned, **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
