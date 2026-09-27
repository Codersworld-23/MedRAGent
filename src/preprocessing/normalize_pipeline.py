"""
Phase 1 -- Normalization Pipeline

Loads every raw dataset, applies per-dataset normalizers,
and saves normalized JSONL files to data/processed/questions/.
"""

from pathlib import Path
import json

from src.preprocessing.normalize_questions import (
    normalize_medqa,
    normalize_mmlu,
    normalize_medmcqa,
    normalize_pubmedqa,
    normalize_bioasq,
)

from src.utils.io import (
    save_jsonl,
    ensure_dir,
)


# ============================================================
# GENERIC LOADERS
# ============================================================

def _load_jsonl(path):
    """Load a JSONL file."""

    records = []

    with open(path, "r", encoding="utf-8-sig") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    return records


def _load_json(path):
    """Load a JSON file."""

    with open(path, "r", encoding="utf-8-sig") as f:

        return json.load(f)


# ============================================================
# PER-DATASET NORMALIZATION RUNNERS
# ============================================================

def normalize_medqa_dataset(dataset_root, output_dir):
    """
    Normalize MedQA-US questions into common schema.

    Focuses on the US subset (primary benchmark).
    """

    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)

    us_dir = dataset_root / "questions" / "US"

    all_records = []

    split_counts = {}

    for split_name in ["train", "dev", "test"]:

        split_file = us_dir / f"{split_name}.jsonl"

        if not split_file.exists():
            print(f"  WARNING: {split_file} not found")
            continue

        raw_records = _load_jsonl(split_file)

        normalized = []

        for i, rec in enumerate(raw_records):

            normalized.append(
                normalize_medqa(rec, i, split_name, "US")
            )

        all_records.extend(normalized)

        split_counts[split_name] = len(normalized)

    # Save
    output_file = output_dir / "medqa_us.jsonl"

    save_jsonl(all_records, output_file)

    print(f"  MedQA-US: {len(all_records):,} records")

    for split, count in split_counts.items():
        print(f"    {split}: {count:,}")

    return all_records


def normalize_medmcqa_dataset(dataset_root, output_dir):
    """Normalize MedMCQA questions into common schema."""

    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)

    all_records = []

    split_counts = {}

    for split_name in ["train", "dev", "test"]:

        split_file = dataset_root / f"{split_name}.json"

        if not split_file.exists():
            print(f"  WARNING: {split_file} not found")
            continue

        # MedMCQA .json files are actually JSONL
        raw_records = _load_jsonl(split_file)

        normalized = []

        for i, rec in enumerate(raw_records):

            normalized.append(
                normalize_medmcqa(rec, i, split_name)
            )

        all_records.extend(normalized)

        split_counts[split_name] = len(normalized)

    # Save
    output_file = output_dir / "medmcqa.jsonl"

    save_jsonl(all_records, output_file)

    print(f"  MedMCQA: {len(all_records):,} records")

    for split, count in split_counts.items():
        print(f"    {split}: {count:,}")

    return all_records


def normalize_mmlu_dataset(dataset_root, output_dir):
    """Normalize MMLU-Medical questions into common schema."""

    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)

    all_records = []

    subject_counts = {}

    subjects = [
        "anatomy",
        "clinical_knowledge",
        "college_biology",
        "college_medicine",
        "medical_genetics",
        "professional_medicine",
    ]

    for subject in subjects:

        subject_dir = dataset_root / subject

        if not subject_dir.exists():
            print(f"  WARNING: {subject_dir} not found")
            continue

        subject_total = 0

        for split_file in sorted(subject_dir.glob("*.json")):

            split_name = split_file.stem

            # Map "validation" to "val" for consistency
            if split_name == "validation":
                split_name = "val"

            try:
                data = _load_json(split_file)
            except Exception:
                continue

            records = data if isinstance(data, list) else []

            for i, rec in enumerate(records):

                all_records.append(
                    normalize_mmlu(rec, i, split_name, subject)
                )

                subject_total += 1

        subject_counts[subject] = subject_total

    # Save
    output_file = output_dir / "mmlu_medical.jsonl"

    save_jsonl(all_records, output_file)

    print(f"  MMLU-Medical: {len(all_records):,} records")

    for subject, count in subject_counts.items():
        print(f"    {subject}: {count:,}")

    return all_records


def normalize_pubmedqa_dataset(dataset_root, output_dir):
    """
    Normalize PubMedQA questions into common schema.

    Focuses on the labeled subset (ori_pqal.json) for evaluation.
    Also processes the artificially generated (pqaa) subset.
    """

    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)

    all_records = []

    file_mapping = {
        "ori_pqal.json": "labeled",
        "ori_pqaa.json": "artificial",
        "ori_pqau.json": "unlabeled",
    }

    file_counts = {}

    for filename, split_label in file_mapping.items():

        filepath = dataset_root / filename

        if not filepath.exists():
            continue

        try:
            data = _load_json(filepath)
        except Exception:
            print(f"  WARNING: Failed to load {filename}")
            continue

        # PubMedQA is dict keyed by PMID
        if isinstance(data, dict):

            records = []

            for pmid, record in data.items():

                record["__pmid__"] = pmid

                records.append(record)

        else:
            records = data

        normalized = []

        for i, rec in enumerate(records):

            normalized.append(
                normalize_pubmedqa(rec, i, split_label)
            )

        all_records.extend(normalized)

        file_counts[filename] = len(normalized)

    # Save
    output_file = output_dir / "pubmedqa.jsonl"

    save_jsonl(all_records, output_file)

    print(f"  PubMedQA: {len(all_records):,} records")

    for fname, count in file_counts.items():
        print(f"    {fname}: {count:,}")

    return all_records


def normalize_bioasq_dataset(dataset_root, output_dir):
    """Normalize BioASQ questions into common schema."""

    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)

    all_records = []

    file_counts = {}

    # Training data
    training_file = (
        dataset_root
        / "BioASQ-training13b"
        / "training13b.json"
    )

    if training_file.exists():

        try:
            data = _load_json(training_file)
        except Exception:
            data = {}

        questions = data.get("questions", [])

        for i, rec in enumerate(questions):

            all_records.append(
                normalize_bioasq(rec, i, "train")
            )

        file_counts["training13b.json"] = len(questions)

    # Golden test files
    golden_dir = dataset_root / "Task13BGoldenEnriched"

    if golden_dir.exists():

        for golden_file in sorted(
            golden_dir.glob("*.json")
        ):

            try:
                data = _load_json(golden_file)
            except Exception:
                continue

            questions = data.get("questions", [])

            for i, rec in enumerate(questions):

                all_records.append(
                    normalize_bioasq(rec, i, "test")
                )

            file_counts[golden_file.name] = len(questions)

    # Save
    output_file = output_dir / "bioasq.jsonl"

    save_jsonl(all_records, output_file)

    print(f"  BioASQ: {len(all_records):,} records")

    for fname, count in file_counts.items():
        print(f"    {fname}: {count:,}")

    return all_records


# ============================================================
# FULL NORMALIZATION PIPELINE
# ============================================================

def run_normalization(dataset_root, output_dir):
    """
    Run the full normalization pipeline for all datasets.

    Parameters
    ----------
    dataset_root : str or Path
        Path to the datasets/ directory.
    output_dir : str or Path
        Path to data/processed/questions/.

    Returns
    -------
    dict
        Mapping of dataset name to list of normalized records.
    """

    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)

    ensure_dir(output_dir)

    print("=" * 70)
    print("MedRAGent -- DATASET NORMALIZATION")
    print("=" * 70)

    all_datasets = {}

    # MedQA
    print("\n[1/5] Normalizing MedQA-US...")
    all_datasets["MedQA"] = normalize_medqa_dataset(
        dataset_root / "MedQA", output_dir
    )

    # MedMCQA
    print("\n[2/5] Normalizing MedMCQA...")
    all_datasets["MedMCQA"] = normalize_medmcqa_dataset(
        dataset_root / "MedMCQA", output_dir
    )

    # MMLU-Medical
    print("\n[3/5] Normalizing MMLU-Medical...")
    all_datasets["MMLU-Medical"] = normalize_mmlu_dataset(
        dataset_root / "MMLU-Medical", output_dir
    )

    # PubMedQA
    print("\n[4/5] Normalizing PubMedQA...")
    all_datasets["PubMedQA"] = normalize_pubmedqa_dataset(
        dataset_root / "PubMedQA", output_dir
    )

    # BioASQ
    print("\n[5/5] Normalizing BioASQ...")
    all_datasets["BioASQ"] = normalize_bioasq_dataset(
        dataset_root / "BioASQ", output_dir
    )

    # Summary
    print("\n" + "=" * 70)
    print("NORMALIZATION SUMMARY")
    print("=" * 70)

    total = 0

    for name, records in all_datasets.items():

        count = len(records)

        total += count

        print(f"  {name:25s} {count:>10,}")

    print(f"  {'TOTAL':25s} {total:>10,}")

    print("=" * 70)

    return all_datasets
