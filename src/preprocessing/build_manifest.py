"""
Phase 1 -- Dataset Manifest Builder

Creates a comprehensive JSON manifest documenting
every dataset, its splits, record counts, file paths,
and the selected evaluation subsets.
"""

from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

from src.utils.io import save_json


# ============================================================
# EVALUATION SUBSET DEFINITIONS
# ============================================================

# These are the final evaluation subsets the project will use.
# The decision criteria are documented here.

EVAL_SUBSETS = {
    "primary": {
        "name": "MedQA-US Test",
        "dataset": "MedQA",
        "split": "test",
        "subset": "US",
        "description": (
            "Primary evaluation benchmark. "
            "Standard MedQA-US test set (1,273 questions). "
            "Medical textbook QA is the core research story."
        ),
    },
    "secondary": [
        {
            "name": "MMLU-Medical Test",
            "dataset": "MMLU-Medical",
            "split": "test",
            "subset": "all_subjects",
            "description": (
                "Cross-benchmark medical generalization. "
                "1,089 test questions across 6 medical subjects. "
                "Matches MIRAGE/MEDRAG evaluation setup."
            ),
        },
        {
            "name": "MedMCQA Dev",
            "dataset": "MedMCQA",
            "split": "dev",
            "subset": "all",
            "description": (
                "Medical-domain robustness check. "
                "4,183 dev questions (test has no labels). "
                "Indian medical exam questions."
            ),
        },
    ],
    "supplementary": [
        {
            "name": "PubMedQA Labeled",
            "dataset": "PubMedQA",
            "split": "labeled",
            "subset": "ori_pqal",
            "description": (
                "Biomedical literature QA. "
                "1,000 expert-labeled questions. "
                "Yes/no/maybe format."
            ),
        },
        {
            "name": "BioASQ Golden",
            "dataset": "BioASQ",
            "split": "test",
            "subset": "golden",
            "description": (
                "Additional biomedical research QA. "
                "~340 golden test questions. "
                "Mixed question types."
            ),
        },
    ],
}


# ============================================================
# MANIFEST BUILDING
# ============================================================

def build_manifest(
    all_datasets,
    audit_results,
    duplicate_report,
    leakage_report,
    output_dir,
):
    """
    Build the comprehensive dataset manifest.

    Parameters
    ----------
    all_datasets : dict[str, list[dict]]
        Mapping of dataset name to normalized records.
    audit_results : dict
        Results from the audit step.
    duplicate_report : dict
        Results from duplicate detection.
    leakage_report : dict
        Results from leakage check.
    output_dir : str or Path
        Path to data/processed/metadata/.

    Returns
    -------
    dict
        The complete manifest.
    """

    output_dir = Path(output_dir)

    print("=" * 70)
    print("MedRAGent -- DATASET MANIFEST")
    print("=" * 70)

    manifest = {
        "project": "MedRAGent",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "datasets": {},
        "evaluation_subsets": EVAL_SUBSETS,
        "summary": {},
    }

    total_records = 0
    total_eval = 0

    for dataset_name, records in all_datasets.items():

        # Split counts
        split_counts = defaultdict(int)

        for rec in records:

            split_counts[rec.get("split", "unknown")] += 1

        # Answer coverage
        has_answer = sum(
            1 for r in records
            if r.get("answer") is not None
        )

        # With options
        has_options = sum(
            1 for r in records
            if r.get("options") is not None
        )

        # Unique subsets
        subsets = set(
            r.get("subset")
            for r in records
            if r.get("subset")
        )

        dataset_info = {
            "total_records": len(records),
            "splits": dict(split_counts),
            "records_with_answer": has_answer,
            "records_with_options": has_options,
            "subsets": sorted(subsets) if subsets else [],
            "schema_fields": (
                list(records[0].keys())
                if records else []
            ),
        }

        # Add audit info if available
        if dataset_name in audit_results:

            audit = audit_results[dataset_name]

            dataset_info["audit"] = {
                "total_malformed": audit.get(
                    "total_malformed", 0
                ),
                "issues": audit.get("total_issues", {}),
            }

        # Add duplicate info if available
        within_dup = duplicate_report.get(
            "within_dataset", {}
        ).get(dataset_name, {})

        if within_dup:

            dataset_info["duplicates"] = {
                "unique_questions": within_dup.get(
                    "unique_questions", 0
                ),
                "duplicate_groups": within_dup.get(
                    "duplicate_groups", 0
                ),
                "total_duplicate_records": within_dup.get(
                    "total_duplicate_records", 0
                ),
            }

        manifest["datasets"][dataset_name] = dataset_info

        total_records += len(records)

    # Evaluation subset record counts
    for subset_def in (
        [EVAL_SUBSETS["primary"]]
        + EVAL_SUBSETS["secondary"]
        + EVAL_SUBSETS["supplementary"]
    ):

        ds_name = subset_def["dataset"]
        split = subset_def["split"]

        records = all_datasets.get(ds_name, [])

        count = sum(
            1 for r in records
            if r.get("split") == split
        )

        total_eval += count

    # Summary
    manifest["summary"] = {
        "total_datasets": len(all_datasets),
        "total_records": total_records,
        "total_evaluation_records": total_eval,
        "cross_dataset_overlaps": (
            duplicate_report.get(
                "cross_dataset", {}
            ).get("total_cross_dataset_overlaps", 0)
        ),
    }

    # Print
    print("\n  Datasets:")

    for name, info in manifest["datasets"].items():

        print(
            f"    {name:25s} -> "
            f"{info['total_records']:>10,} records"
        )

    print(f"\n  Total records: {total_records:,}")
    print(f"  Total evaluation records: {total_eval:,}")

    # Save
    output_file = output_dir / "dataset_manifest.json"

    save_json(manifest, output_file)

    print(f"\n  Manifest saved to: {output_file}")

    print("=" * 70)

    return manifest
