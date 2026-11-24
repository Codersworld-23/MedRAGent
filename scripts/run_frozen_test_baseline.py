"""Execute or preflight the single immutable MedQA-US test baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medragent.data import load_medqa_jsonl, load_text_chunks_jsonl
from medragent.scope import annotate_records, write_annotations


MODEL = "llama3.2:latest"
EMBEDDING_MODEL = "nomic-embed-text:latest"
TEMPERATURE = 0.0
TOP_K = 3
RRF_CONSTANT = 60
PROMPT_VERSION = "baseline_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def installed_ollama_models(base_url: str) -> set[str]:
    url = base_url.rstrip("/") + "/api/tags"
    with urlopen(url, timeout=10) as response:
        payload = json.load(response)
    return {
        str(model["name"])
        for model in payload.get("models", [])
        if isinstance(model, dict) and model.get("name")
    }


def run_once(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(process.stdout)
    if process.returncode != 0:
        raise RuntimeError(
            f"Frozen test command failed once; no retry was attempted. "
            f"See {log_path}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments" / "results" / "test")
    args = parser.parse_args()

    python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        raise FileNotFoundError("Project virtual environment was not found")

    question_path = PROJECT_ROOT / "datasets" / "MedQA" / "questions" / "US" / "test.jsonl"
    corpus_path = PROJECT_ROOT / "datasets" / "MedRAG" / "chunk" / "Psichiatry_DSM-5.jsonl"
    annotation_path = PROJECT_ROOT / "experiments" / "annotations" / "medqa_scope_test.jsonl"
    records = load_medqa_jsonl(question_path)
    chunks = load_text_chunks_jsonl(corpus_path)

    models = installed_ollama_models(args.base_url)
    missing_models = [
        model for model in (MODEL, EMBEDDING_MODEL)
        if model not in models
    ]
    if missing_models:
        raise RuntimeError(
            "Required Ollama models are not installed: " + ", ".join(missing_models)
        )

    if args.output_dir.exists():
        existing = [
            path for path in args.output_dir.rglob("*")
            if path.is_file()
            and path.name != "freeze.json"
        ]
        if existing:
            raise RuntimeError(
                f"Refusing to overwrite frozen test output directory: {args.output_dir}"
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_annotations(
        annotation_path,
        annotate_records(records),
    )
    freeze = {
        "freeze_status": "test_ready",
        "execute_status": "preflight_only",
        "split": "test",
        "question_count": len(records),
        "question_path": str(question_path),
        "question_sha256": sha256(question_path),
        "corpus_path": str(corpus_path),
        "corpus_sha256": sha256(corpus_path),
        "corpus_chunk_count": len(chunks),
        "annotation_path": str(annotation_path),
        "annotation_sha256": sha256(annotation_path),
        "annotation_policy": "deterministic lexical candidate; not expert-reviewed",
        "generator_model": MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "rrf_constant": RRF_CONSTANT,
        "prompt_version": PROMPT_VERSION,
        "base_url": args.base_url,
        "output_dir": str(args.output_dir),
    }
    freeze_path = args.output_dir / "freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(freeze, indent=2))

    if not args.execute:
        print("Preflight passed; use --execute for the one immutable test run.")
        return 0

    freeze["execute_status"] = "executing"
    freeze["output_files"] = []
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")

    outputs = {
        "llm_only": args.output_dir / "llm_only_test.jsonl",
        "bm25": args.output_dir / "bm25_rag_test.jsonl",
        "dense": args.output_dir / "dense_rag_test.jsonl",
        "hybrid": args.output_dir / "hybrid_rag_test.jsonl",
    }

    commands = {
        "llm_only": [
            str(python),
            "scripts/run_llm_only.py",
            "--model", MODEL,
            "--base-url", args.base_url,
            "--temperature", str(TEMPERATURE),
            "--input", str(question_path),
            "--annotations", str(annotation_path),
            "--output", str(outputs["llm_only"]),
        ],
        "bm25": [
            str(python),
            "scripts/run_bm25_rag.py",
            "--model", MODEL,
            "--base-url", args.base_url,
            "--temperature", str(TEMPERATURE),
            "--top-k", str(TOP_K),
            "--input", str(question_path),
            "--corpus", str(corpus_path),
            "--annotations", str(annotation_path),
            "--output", str(outputs["bm25"]),
        ],
        "dense": [
            str(python),
            "scripts/run_dense_rag.py",
            "--model", MODEL,
            "--embedding-model", EMBEDDING_MODEL,
            "--base-url", args.base_url,
            "--temperature", str(TEMPERATURE),
            "--top-k", str(TOP_K),
            "--input", str(question_path),
            "--corpus", str(corpus_path),
            "--annotations", str(annotation_path),
            "--output", str(outputs["dense"]),
        ],
        "hybrid": [
            str(python),
            "scripts/run_hybrid_rag.py",
            "--model", MODEL,
            "--embedding-model", EMBEDDING_MODEL,
            "--base-url", args.base_url,
            "--temperature", str(TEMPERATURE),
            "--top-k", str(TOP_K),
            "--rrf-constant", str(RRF_CONSTANT),
            "--input", str(question_path),
            "--corpus", str(corpus_path),
            "--annotations", str(annotation_path),
            "--output", str(outputs["hybrid"]),
        ],
    }

    for name, command in commands.items():
        run_once(command, args.output_dir / f"{name}_run.log")

    freeze["execute_status"] = "completed"
    freeze["output_files"] = [str(path) for path in outputs.values()]
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
