"""
Phase 1 -- Leakage Check

Checks whether evaluation/test questions have significant
textual overlap with the medical textbook corpus.

This matters because if a test question is nearly verbatim
in the textbooks, retrieval performance may be inflated.
"""

from pathlib import Path
import re
from collections import defaultdict

from src.utils.io import save_json


# ============================================================
# N-GRAM EXTRACTION
# ============================================================

def _tokenize(text):
    """Simple whitespace + lowered tokenization."""

    if not text:
        return []

    text = text.lower()

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    return text.split()


def _extract_ngrams(tokens, n=5):
    """Extract all n-grams from a token list."""

    if len(tokens) < n:
        return set()

    return set(
        tuple(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    )


# ============================================================
# TEXTBOOK NGRAM INDEX
# ============================================================

def build_textbook_ngram_index(textbook_dir, n=5):
    """
    Build an n-gram index from all textbook files.

    Returns a set of all n-grams found in the corpus,
    plus a mapping from n-gram to textbook source(s).
    """

    textbook_dir = Path(textbook_dir)

    all_ngrams = set()

    ngram_sources = defaultdict(set)

    for txt_file in sorted(textbook_dir.glob("*.txt")):

        print(f"    Indexing: {txt_file.name}")

        with open(
            txt_file, "r",
            encoding="utf-8", errors="ignore"
        ) as f:

            text = f.read()

        tokens = _tokenize(text)

        file_ngrams = _extract_ngrams(tokens, n)

        for ng in file_ngrams:
            ngram_sources[ng].add(txt_file.stem)

        all_ngrams.update(file_ngrams)

    return all_ngrams, ngram_sources


# ============================================================
# LEAKAGE DETECTION
# ============================================================

def check_question_leakage(
    records,
    textbook_ngrams,
    dataset_name,
    n=5,
    threshold=0.5
):
    """
    Check how many evaluation questions have significant
    n-gram overlap with the textbook corpus.

    Parameters
    ----------
    records : list[dict]
        Normalized records.
    textbook_ngrams : set
        Set of all textbook n-grams.
    dataset_name : str
        Name of the dataset.
    n : int
        N-gram size (default 5).
    threshold : float
        Fraction of question n-grams that must be in textbooks
        to flag as potential leakage (default 0.5 = 50%).

    Returns
    -------
    dict
        Leakage report for this dataset.
    """

    # Only check test/dev/val splits
    eval_records = [
        r for r in records
        if r.get("split") in {"test", "dev", "val", "labeled"}
    ]

    if not eval_records:
        return {
            "dataset": dataset_name,
            "eval_records": 0,
            "flagged": 0,
            "mean_overlap": 0.0,
        }

    overlaps = []

    flagged_count = 0

    for rec in eval_records:

        question = rec.get("question", "")

        tokens = _tokenize(question)

        q_ngrams = _extract_ngrams(tokens, n)

        if not q_ngrams:

            overlaps.append(0.0)
            continue

        matched = sum(
            1 for ng in q_ngrams
            if ng in textbook_ngrams
        )

        overlap_ratio = matched / len(q_ngrams)

        overlaps.append(overlap_ratio)

        if overlap_ratio >= threshold:
            flagged_count += 1

    mean_overlap = (
        sum(overlaps) / len(overlaps)
        if overlaps else 0.0
    )

    # Bucket distribution
    buckets = {
        "0%": 0,
        "1-10%": 0,
        "11-25%": 0,
        "26-50%": 0,
        "51-75%": 0,
        "76-100%": 0,
    }

    for ov in overlaps:

        pct = ov * 100

        if pct == 0:
            buckets["0%"] += 1
        elif pct <= 10:
            buckets["1-10%"] += 1
        elif pct <= 25:
            buckets["11-25%"] += 1
        elif pct <= 50:
            buckets["26-50%"] += 1
        elif pct <= 75:
            buckets["51-75%"] += 1
        else:
            buckets["76-100%"] += 1

    return {
        "dataset": dataset_name,
        "eval_records": len(eval_records),
        "flagged_above_threshold": flagged_count,
        "threshold": threshold,
        "mean_overlap": round(mean_overlap, 4),
        "overlap_distribution": buckets,
    }


# ============================================================
# FULL LEAKAGE CHECK
# ============================================================

def run_leakage_check(all_datasets, textbook_dir, output_dir):
    """
    Run the full leakage check across all evaluation datasets.

    Parameters
    ----------
    all_datasets : dict[str, list[dict]]
        Mapping of dataset name to normalized records.
    textbook_dir : str or Path
        Path to datasets/MedQA/textbooks/en/.
    output_dir : str or Path
        Path to data/processed/metadata/.

    Returns
    -------
    dict
        Complete leakage report.
    """

    textbook_dir = Path(textbook_dir)
    output_dir = Path(output_dir)

    print("=" * 70)
    print("MedRAGent -- QUESTION/TEXTBOOK LEAKAGE CHECK")
    print("=" * 70)

    # Build textbook n-gram index
    print("\n  Building textbook n-gram index (n=5)...")

    textbook_ngrams, _ = build_textbook_ngram_index(
        textbook_dir, n=5
    )

    print(f"  Total unique 5-grams: {len(textbook_ngrams):,}")

    # Check each dataset
    report = {}

    print("\n--- Leakage Results ---\n")

    for dataset_name, records in all_datasets.items():

        result = check_question_leakage(
            records,
            textbook_ngrams,
            dataset_name,
            n=5,
            threshold=0.5,
        )

        report[dataset_name] = result

        eval_n = result["eval_records"]
        flagged = result["flagged_above_threshold"]
        mean_ov = result["mean_overlap"]

        print(
            f"  {dataset_name:25s}"
            f"  eval={eval_n:>8,}"
            f"  flagged={flagged:>6,}"
            f"  mean_overlap={mean_ov:.4f}"
        )

        if eval_n > 0:

            dist = result["overlap_distribution"]

            for bucket, count in dist.items():

                pct = (count / eval_n) * 100

                print(
                    f"    {bucket:>10s}: "
                    f"{count:>6,} ({pct:5.1f}%)"
                )

    # Save report
    output_file = output_dir / "leakage_report.json"

    save_json(report, output_file)

    print(f"\n  Report saved to: {output_file}")

    print("=" * 70)

    return report
