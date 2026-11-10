"""Run the no-retrieval LLM-only baseline against MedQA-US."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medragent.data import load_medqa_jsonl
from medragent.llm_only import OllamaClient, run_record, write_results
from medragent.scope import load_annotations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Exact Ollama model tag, e.g. llama3.2:3b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "MedQA" / "questions" / "US" / "dev.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "llm_only_dev.jsonl",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "annotations" / "medqa_scope_dev.jsonl",
        help="Versioned question-scope annotations to attach to result rows",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    records = load_medqa_jsonl(args.input)
    annotations = load_annotations(args.annotations)
    missing_annotations = [record.id for record in records if record.id not in annotations]
    if missing_annotations:
        parser.error(
            f"{len(missing_annotations)} input records have no scope annotation; "
            "annotate the exact input split first"
        )
    if args.limit is not None:
        records = records[: args.limit]
    client = OllamaClient(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        timeout_seconds=args.timeout,
    )
    results = [
        run_record(
            record,
            client,
            model=args.model,
            temperature=args.temperature,
            annotation=annotations[record.id],
        )
        for record in records
    ]
    write_results(args.output, results)
    failures = sum(result.error is not None for result in results)
    correct = sum(result.is_correct is True for result in results)
    print(f"Wrote {len(results)} results to {args.output}")
    print(f"Correct: {correct}/{len(results)}; failures/abstentions: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
