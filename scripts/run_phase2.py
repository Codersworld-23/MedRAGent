"""
MedRAGent — Phase 2 Orchestrator

Runs the complete Phase 2 pipeline:
  1. Loads raw textbooks
  2. Parses hierarchical structure (Chapters & Sections)
  3. Generates Baseline chunks
  4. Generates Hierarchical chunks

Usage:
    python scripts/run_phase2.py
"""

import sys
from pathlib import Path
import json

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.hierarchy_parser import parse_all_textbooks
from src.preprocessing.chunk_text import run_chunking_pipeline
from src.utils.io import ensure_dir


# ============================================================
# PATHS
# ============================================================

PROCESSED = ROOT / "data" / "processed"

TEXTBOOKS_FILE = PROCESSED / "textbooks" / "textbooks.jsonl"
HIERARCHIES_DIR = PROCESSED / "hierarchies"
CHUNKS_DIR = PROCESSED / "chunks"

# ============================================================
# MAIN PIPELINE
# ============================================================

def _load_jsonl(path):
    """Load a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main():

    print("\n")
    print("#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  MedRAGent -- PHASE 2: HIERARCHICAL PREPROCESSING".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    print()

    # Ensure output directories exist
    ensure_dir(HIERARCHIES_DIR)
    ensure_dir(CHUNKS_DIR)

    if not TEXTBOOKS_FILE.exists():
        print(f"ERROR: Textbooks file not found at {TEXTBOOKS_FILE}")
        print("Please run Phase 1 first.")
        return

    # ----------------------------------------------------------
    # STEP 1: LOAD TEXTBOOKS
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 1/3 -- LOAD TEXTBOOK CORPUS")
    print("=" * 70 + "\n")
    
    textbook_records = _load_jsonl(TEXTBOOKS_FILE)
    print(f"  Loaded {len(textbook_records)} textbooks.")

    # ----------------------------------------------------------
    # STEP 2: HIERARCHICAL PARSING
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 2/3 -- HIERARCHICAL PARSING")
    print("=" * 70 + "\n")
    
    hierarchies = parse_all_textbooks(textbook_records, HIERARCHIES_DIR)

    # ----------------------------------------------------------
    # STEP 3: CHUNKING EXPERIMENTS
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 3/3 -- GENERATE CHUNKS")
    print("=" * 70 + "\n")
    
    run_chunking_pipeline(textbook_records, hierarchies, CHUNKS_DIR)

    # ----------------------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------------------
    print("\n")
    print("#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  PHASE 2 COMPLETE".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    print("\n  Output directories:")
    print(f"    Hierarchies: {HIERARCHIES_DIR}")
    print(f"    Chunks     : {CHUNKS_DIR}")

    print("\n  Next step:")
    print("    -> Phase 3: Baseline Retrieval Implementation")
    print()


if __name__ == "__main__":
    main()
