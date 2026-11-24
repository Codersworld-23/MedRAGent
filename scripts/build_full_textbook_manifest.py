"""Validate and manifest the complete 18-textbook MedRAG corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medragent.data import load_text_chunks_jsonl


EXPECTED_TEXTBOOKS = (
    "Anatomy_Gray",
    "Biochemistry_Lippincott",
    "Cell_Biology_Alberts",
    "First_Aid_Step1",
    "First_Aid_Step2",
    "Gynecology_Novak",
    "Histology_Ross",
    "Immunology_Janeway",
    "InternalMed_Harrison",
    "Neurology_Adams",
    "Obstentrics_Williams",
    "Pathology_Robbins",
    "Pathoma_Husain",
    "Pediatrics_Nelson",
    "Pharmacology_Katzung",
    "Physiology_Levy",
    "Psichiatry_DSM-5",
    "Surgery_Schwartz",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "MedRAG" / "chunk",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "manifests" / "full_textbook_corpus_manifest.json",
    )
    args = parser.parse_args()

    textbooks = []
    all_chunk_ids = set()
    all_text = Counter()
    missing = []
    for name in EXPECTED_TEXTBOOKS:
        path = args.chunk_dir / f"{name}.jsonl"
        if not path.exists():
            missing.append(str(path))
            continue
        chunks = load_text_chunks_jsonl(path)
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        duplicate_ids = len(chunk_ids) - len(set(chunk_ids))
        overlap = all_chunk_ids.intersection(chunk_ids)
        if overlap:
            raise ValueError(
                f"{path}: duplicate chunk IDs across textbooks: "
                f"{', '.join(sorted(overlap)[:10])}"
            )
        all_chunk_ids.update(chunk_ids)
        all_text.update(chunk.text for chunk in chunks)
        textbooks.append({
            "name": name,
            "file": str(path.relative_to(PROJECT_ROOT)),
            "sha256": sha256(path),
            "chunk_count": len(chunks),
            "characters": sum(len(chunk.text) for chunk in chunks),
            "unique_titles": len({chunk.title for chunk in chunks}),
            "duplicate_chunk_ids": duplicate_ids,
            "duplicate_text_count": None,
        })

    if missing:
        raise FileNotFoundError(
            "Missing expected textbook chunks: " + ", ".join(missing)
        )
    present_names = [item["name"] for item in textbooks]
    unexpected = sorted(
        path.stem
        for path in args.chunk_dir.glob("*.jsonl")
        if path.stem not in EXPECTED_TEXTBOOKS
    )

    for item in textbooks:
        chunks = load_text_chunks_jsonl(PROJECT_ROOT / item["file"])
        item["duplicate_text_count"] = sum(
            all_text[chunk.text] > 1 for chunk in chunks
        )

    manifest = {
        "manifest_version": "full_textbooks_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_textbook_count": 18,
        "textbook_count": len(textbooks),
        "total_chunk_count": len(all_chunk_ids),
        "total_characters": sum(item["characters"] for item in textbooks),
        "unexpected_chunk_files": unexpected,
        "textbooks": textbooks,
        "source_scope": "18 medical textbook chunks",
        "hierarchy_status": "not trusted; title/chunk order is not hierarchy",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
