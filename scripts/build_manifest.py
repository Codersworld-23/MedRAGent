"""Create a reproducibility manifest for the Step 0 and Step 1 datasets."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medragent.data import load_medqa_jsonl, load_text_chunks_jsonl


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def package_versions() -> dict[str, str | None]:
    names = ("numpy", "rank-bm25", "faiss-cpu", "sentence-transformers", "pyyaml")
    return {name: _safe_version(name) for name in names}


def _safe_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def duplicate_count(values: list[str]) -> int:
    """Count repeated values beyond their first occurrence."""
    return len(values) - len(set(values))


def build_manifest(medqa_path: Path, chunk_path: Path) -> dict[str, Any]:
    medqa = load_medqa_jsonl(medqa_path)
    chunks = load_text_chunks_jsonl(chunk_path)
    return {
        "manifest_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "dependencies": package_versions(),
        "datasets": {
            "medqa_us": {
                "file": file_metadata(medqa_path),
                "record_count": len(medqa),
                "schema": ["id", "question", "options", "answer_index", "split", "source_file"],
                "split": medqa[0].split if medqa else medqa_path.stem,
                "answer_index_min": min(
                    (record.answer_index for record in medqa if record.answer_index is not None),
                    default=None,
                ),
                "answer_index_max": max(
                    (record.answer_index for record in medqa if record.answer_index is not None),
                    default=None,
                ),
                "duplicate_questions": duplicate_count([record.question for record in medqa]),
            },
            "medrag_psichiatry_dsm5": {
                "file": file_metadata(chunk_path),
                "record_count": len(chunks),
                "schema": ["chunk_id", "title", "text", "source_file", "metadata"],
                "unique_titles": len({chunk.title for chunk in chunks}),
                "total_text_characters": sum(len(chunk.text) for chunk in chunks),
                "duplicate_texts": duplicate_count([chunk.text for chunk in chunks]),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--medqa",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "MedQA" / "questions" / "US" / "dev.jsonl",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=PROJECT_ROOT
        / "datasets"
        / "MedRAG"
        / "chunk"
        / "Psichiatry_DSM-5.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "manifests" / "dataset_manifest.json",
    )
    args = parser.parse_args()
    manifest = build_manifest(args.medqa, args.chunks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
