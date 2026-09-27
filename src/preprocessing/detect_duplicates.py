"""
Phase 1 -- Duplicate Detection

Detects duplicate questions both within each dataset
and across datasets. Uses normalized question text
comparison with fuzzy matching fallback.
"""

from pathlib import Path
from collections import defaultdict
import hashlib
import json
import re

from src.utils.io import save_json


# ============================================================
# TEXT NORMALIZATION FOR COMPARISON
# ============================================================

def _normalize_for_comparison(text):
    """
    Aggressively normalize question text for duplicate detection.

    Lowercases, strips punctuation and whitespace, collapses spaces.
    """

    if not text:
        return ""

    text = text.lower().strip()

    # Remove common preambles
    text = re.sub(
        r"^(question\s*:?\s*)", "", text
    )

    # Remove all non-alphanumeric except spaces
    text = re.sub(r"[^a-z0-9\s]", "", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _hash_text(text):
    """Get a stable hash for normalized text."""

    return hashlib.md5(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================
# WITHIN-DATASET DUPLICATES
# ============================================================

def find_within_dataset_duplicates(records, dataset_name):
    """
    Find exact duplicates within a single dataset.

    Parameters
    ----------
    records : list[dict]
        Normalized records (common schema).
    dataset_name : str
        Name of the dataset.

    Returns
    -------
    dict
        Duplicate report for this dataset.
    """

    seen = {}  # normalized_text -> list of record ids

    duplicates = []

    for rec in records:

        question = rec.get("question", "")

        normalized = _normalize_for_comparison(question)

        if not normalized:
            continue

        text_hash = _hash_text(normalized)

        if text_hash in seen:

            seen[text_hash].append(rec.get("id", "unknown"))

        else:

            seen[text_hash] = [rec.get("id", "unknown")]

    for text_hash, ids in seen.items():

        if len(ids) > 1:

            duplicates.append({
                "count": len(ids),
                "ids": ids[:10],  # limit for report
            })

    return {
        "dataset": dataset_name,
        "total_records": len(records),
        "unique_questions": len(seen),
        "duplicate_groups": len(duplicates),
        "total_duplicate_records": sum(
            d["count"] - 1 for d in duplicates
        ),
        "examples": duplicates[:20],  # limit examples
    }


# ============================================================
# CROSS-DATASET DUPLICATES
# ============================================================

def find_cross_dataset_duplicates(all_datasets):
    """
    Find questions that appear across different datasets.

    Parameters
    ----------
    all_datasets : dict[str, list[dict]]
        Mapping of dataset name to list of normalized records.

    Returns
    -------
    dict
        Cross-dataset overlap report.
    """

    # Build hash -> (dataset, id) index
    hash_index = defaultdict(list)

    for dataset_name, records in all_datasets.items():

        for rec in records:

            question = rec.get("question", "")

            normalized = _normalize_for_comparison(question)

            if not normalized:
                continue

            text_hash = _hash_text(normalized)

            hash_index[text_hash].append({
                "dataset": dataset_name,
                "id": rec.get("id", "unknown"),
                "split": rec.get("split", "unknown"),
            })

    # Find entries with records from different datasets
    overlaps = []

    pair_counts = defaultdict(int)

    for text_hash, entries in hash_index.items():

        datasets_in_group = set(
            e["dataset"] for e in entries
        )

        if len(datasets_in_group) > 1:

            overlaps.append({
                "datasets": sorted(datasets_in_group),
                "count": len(entries),
                "entries": entries[:10],
            })

            # Count pairwise overlaps
            sorted_ds = sorted(datasets_in_group)

            for i in range(len(sorted_ds)):

                for j in range(i + 1, len(sorted_ds)):

                    pair_key = (
                        f"{sorted_ds[i]} <-> {sorted_ds[j]}"
                    )

                    pair_counts[pair_key] += 1

    return {
        "total_cross_dataset_overlaps": len(overlaps),
        "pairwise_overlap_counts": dict(pair_counts),
        "examples": overlaps[:30],
    }


# ============================================================
# FULL DUPLICATE DETECTION
# ============================================================

def run_duplicate_detection(all_datasets, output_dir):
    """
    Run within-dataset and cross-dataset duplicate detection.

    Parameters
    ----------
    all_datasets : dict[str, list[dict]]
        Mapping of dataset name to list of normalized records.
    output_dir : str or Path
        Path to data/processed/metadata/.

    Returns
    -------
    dict
        Complete duplicate report.
    """

    output_dir = Path(output_dir)

    print("=" * 70)
    print("MedRAGent -- DUPLICATE DETECTION")
    print("=" * 70)

    report = {
        "within_dataset": {},
        "cross_dataset": {},
    }

    # Within-dataset
    print("\n--- Within-Dataset Duplicates ---")

    for dataset_name, records in all_datasets.items():

        result = find_within_dataset_duplicates(
            records, dataset_name
        )

        report["within_dataset"][dataset_name] = result

        dup_count = result["total_duplicate_records"]

        print(
            f"  {dataset_name:25s}"
            f"  unique={result['unique_questions']:>10,}"
            f"  duplicates={dup_count:>6,}"
        )

    # Cross-dataset
    print("\n--- Cross-Dataset Overlaps ---")

    cross_result = find_cross_dataset_duplicates(
        all_datasets
    )

    report["cross_dataset"] = cross_result

    total_cross = cross_result["total_cross_dataset_overlaps"]

    print(f"  Total overlapping questions: {total_cross}")

    for pair, count in sorted(
        cross_result["pairwise_overlap_counts"].items()
    ):
        print(f"    {pair}: {count}")

    # Save report
    output_file = output_dir / "duplicate_report.json"

    save_json(report, output_file)

    print(f"\n  Report saved to: {output_file}")

    print("=" * 70)

    return report
