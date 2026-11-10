"""Package the generated evaluation questions for sharing with other RAG projects.

Writes, next to the input:
  eval_questions_shared.jsonl   one question per line, with reference answer and source passages
  eval_questions_shared.csv     the same, for spreadsheets

Each row carries the full text of its source chunks, so a system that does not use this corpus
can still judge answers against the evidence.

Usage:
  python export_eval_set.py --questions eval_questions.jsonl --data Psichiatry_DSM-5.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from medx_retrieval import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the generated evaluation set for sharing.")
    parser.add_argument("--questions", default="eval_questions.jsonl")
    parser.add_argument("--data", default="Psichiatry_DSM-5.jsonl")
    parser.add_argument("--out-prefix", default="eval_questions_shared")
    args = parser.parse_args()

    chunks = {document.id: document for document in load_jsonl(args.data)}
    with open(args.questions, encoding="utf-8") as file:
        items = [json.loads(line) for line in file if line.strip()]

    rows = []
    for number, item in enumerate(items, start=1):
        source_ids = item.get("expected_doc_ids", [])
        missing = [doc_id for doc_id in source_ids if doc_id not in chunks]
        if missing:
            raise SystemExit(f"Question {number} cites chunks not in {args.data}: {missing}")
        rows.append(
            {
                "id": f"q{number:03d}",
                "question": item["question"],
                "reference_answer": item.get("answer", ""),
                "source_chunk_ids": source_ids,
                "source_passages": [chunks[doc_id].text for doc_id in source_ids],
                "question_type": item.get("type", ""),
                "difficulty": item.get("difficulty", ""),
                "corpus": f"MedRAG textbooks: {Path(args.data).stem}",
            }
        )

    jsonl_path = Path(f"{args.out_prefix}.jsonl")
    with jsonl_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = Path(f"{args.out_prefix}.csv")
    # utf-8-sig so Excel opens symbols such as "≥" correctly.
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "source_chunk_ids": "; ".join(row["source_chunk_ids"]),
                    "source_passages": "\n\n---\n\n".join(row["source_passages"]),
                }
            )

    print(f"Wrote {len(rows)} questions to {jsonl_path} and {csv_path}")


if __name__ == "__main__":
    main()
