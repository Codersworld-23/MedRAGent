"""
MedRAGent — Phase 1 Orchestrator

Runs the complete Phase 1 pipeline:
  1. Comprehensive dataset audit
  2. Textbook corpus build
  3. Dataset normalization
  4. Duplicate detection (within + cross-dataset)
  5. Question/textbook leakage check
  6. Dataset manifest creation

Usage:
    python scripts/run_phase1.py
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.audit_datasets import (
    run_full_audit,
)
from src.preprocessing.build_corpus import (
    build_textbook_corpus,
)
from src.preprocessing.normalize_pipeline import (
    run_normalization,
)
from src.preprocessing.detect_duplicates import (
    run_duplicate_detection,
)
from src.preprocessing.leakage_check import (
    run_leakage_check,
)
from src.preprocessing.build_manifest import (
    build_manifest,
)
from src.utils.io import save_json, ensure_dir


# ============================================================
# PATHS
# ============================================================

DATASETS = ROOT / "datasets"

PROCESSED = ROOT / "data" / "processed"

QUESTIONS_DIR = PROCESSED / "questions"

TEXTBOOKS_DIR = PROCESSED / "textbooks"

METADATA_DIR = PROCESSED / "metadata"

TEXTBOOK_SOURCE = (
    DATASETS / "MedQA" / "textbooks" / "en"
)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("\n")
    print("#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  MedRAGent -- PHASE 1: DATASET FINALIZATION".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    print()

    # Ensure output directories exist
    ensure_dir(QUESTIONS_DIR)
    ensure_dir(TEXTBOOKS_DIR)
    ensure_dir(METADATA_DIR)

    # ----------------------------------------------------------
    # STEP 1: COMPREHENSIVE AUDIT
    # ----------------------------------------------------------

    print("\n" + "=" * 70)
    print("  STEP 1/6 -- COMPREHENSIVE DATASET AUDIT")
    print("=" * 70 + "\n")

    audit_results = run_full_audit(DATASETS)

    # Save audit report
    save_json(
        audit_results,
        METADATA_DIR / "audit_report.json"
    )

    print(f"\n  [OK] Audit report saved to: {METADATA_DIR / 'audit_report.json'}")

    # ----------------------------------------------------------
    # STEP 2: TEXTBOOK CORPUS BUILD
    # ----------------------------------------------------------

    print("\n" + "=" * 70)
    print("  STEP 2/6 -- TEXTBOOK CORPUS BUILD")
    print("=" * 70 + "\n")

    if not TEXTBOOK_SOURCE.exists():

        print(
            f"  ERROR: Textbook directory not found: "
            f"{TEXTBOOK_SOURCE}"
        )

        return

    textbook_output = TEXTBOOKS_DIR / "textbooks.jsonl"

    build_textbook_corpus(
        TEXTBOOK_SOURCE,
        textbook_output,
    )

    print(f"\n  [OK] Textbooks saved to: {textbook_output}")

    # ----------------------------------------------------------
    # STEP 3: DATASET NORMALIZATION
    # ----------------------------------------------------------

    print("\n" + "=" * 70)
    print("  STEP 3/6 -- DATASET NORMALIZATION")
    print("=" * 70 + "\n")

    all_datasets = run_normalization(
        DATASETS, QUESTIONS_DIR
    )

    # ----------------------------------------------------------
    # STEP 4: DUPLICATE DETECTION
    # ----------------------------------------------------------

    print("\n" + "=" * 70)
    print("  STEP 4/6 -- DUPLICATE DETECTION")
    print("=" * 70 + "\n")

    duplicate_report = run_duplicate_detection(
        all_datasets, METADATA_DIR
    )

    # ----------------------------------------------------------
    # STEP 5: LEAKAGE CHECK
    # ----------------------------------------------------------

    print("\n" + "=" * 70)
    print("  STEP 5/6 -- QUESTION/TEXTBOOK LEAKAGE CHECK")
    print("=" * 70 + "\n")

    leakage_report = run_leakage_check(
        all_datasets, TEXTBOOK_SOURCE, METADATA_DIR
    )

    # ----------------------------------------------------------
    # STEP 6: DATASET MANIFEST
    # ----------------------------------------------------------

    print("\n" + "=" * 70)
    print("  STEP 6/6 -- DATASET MANIFEST")
    print("=" * 70 + "\n")

    manifest = build_manifest(
        all_datasets,
        audit_results,
        duplicate_report,
        leakage_report,
        METADATA_DIR,
    )

    # ----------------------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------------------

    print("\n")
    print("#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  PHASE 1 COMPLETE".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    print("\n  Output files:")
    print(f"    Audit report    : {METADATA_DIR / 'audit_report.json'}")
    print(f"    Duplicate report: {METADATA_DIR / 'duplicate_report.json'}")
    print(f"    Leakage report  : {METADATA_DIR / 'leakage_report.json'}")
    print(f"    Dataset manifest: {METADATA_DIR / 'dataset_manifest.json'}")

    print(f"\n    Normalized questions:")

    for f in sorted(QUESTIONS_DIR.glob("*.jsonl")):
        print(f"      {f.name}")

    print(f"\n    Textbooks:")

    for f in sorted(TEXTBOOKS_DIR.glob("*.jsonl")):
        print(f"      {f.name}")

    print("\n  Next step:")
    print("    -> Phase 2: Textbook Preprocessing & Chunking")
    print()


if __name__ == "__main__":
    main()