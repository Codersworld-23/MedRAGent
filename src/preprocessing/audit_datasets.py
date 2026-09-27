"""
Phase 1 -- Comprehensive Dataset Audit

Performs field-level validation, per-split record counting,
answer-field verification, and malformed record detection
for every dataset in the MedRAGent project.
"""

from pathlib import Path
import json
import csv
from collections import defaultdict


# ============================================================
# GENERIC FILE LOADERS
# ============================================================

def _load_jsonl_file(path):
    """Load a JSONL file (one JSON object per line)."""

    records = []

    with open(path, "r", encoding="utf-8-sig") as f:

        for line_no, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            try:

                records.append(json.loads(line))

            except json.JSONDecodeError:

                records.append({
                    "__malformed__": True,
                    "__line__": line_no,
                    "__raw__": line[:200]
                })

    return records


def _load_json_file(path):
    """Load a standard JSON file."""

    with open(path, "r", encoding="utf-8-sig") as f:

        return json.load(f)


def _load_csv_file(path):
    """Load a CSV file as list of dicts."""

    with open(
        path, "r", encoding="utf-8-sig",
        errors="replace", newline=""
    ) as f:

        reader = csv.DictReader(f)

        return list(reader)


def _load_data_file(path):
    """Load any supported data file."""

    path = Path(path)

    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        return "JSONL", _load_jsonl_file(path)

    if suffix == ".json":

        try:
            data = _load_json_file(path)

            # Test if it's really JSONL disguised as .json
            # MedMCQA .json files are actually JSONL
            if isinstance(data, (dict, list)):
                return "JSON", data

        except json.JSONDecodeError:

            # Fallback -- try as JSONL
            return "JSONL", _load_jsonl_file(path)

    if suffix == ".csv":
        return "CSV", _load_csv_file(path)

    return "UNKNOWN", None


# ============================================================
# RECORD EXTRACTION
# ============================================================

def _extract_records(data):
    """Extract the record list from various data shapes."""

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        # BioASQ: {"questions": [...]}
        if "questions" in data:
            return data["questions"]

        # PubMedQA: {PMID: {...}, ...}
        # Convert to list with the key as pmid field
        first_val = next(iter(data.values()), None)

        if isinstance(first_val, dict):

            records = []

            for key, val in data.items():

                val["__pmid__"] = key

                records.append(val)

            return records

    return []


# ============================================================
# FIELD VALIDATORS
# ============================================================

def _validate_medqa(record, index):
    """Validate a single MedQA record."""

    issues = []

    question = record.get("question")
    options = record.get("options")
    answer = record.get("answer")
    answer_idx = record.get("answer_idx")

    if not question or not isinstance(question, str):
        issues.append("missing_or_invalid_question")

    if not options or not isinstance(options, dict):
        issues.append("missing_or_invalid_options")
    elif len(options) < 2:
        issues.append("too_few_options")

    if not answer and not answer_idx:
        issues.append("missing_answer")

    if answer_idx and isinstance(options, dict):
        if answer_idx not in options:
            issues.append("answer_idx_not_in_options")

    if answer and isinstance(options, dict):
        if answer not in options.values():
            issues.append("answer_text_not_in_options")

    return issues


def _validate_medmcqa(record, index):
    """Validate a single MedMCQA record."""

    issues = []

    question = record.get("question")
    cop = record.get("cop")

    if not question or not isinstance(question, str):
        issues.append("missing_or_invalid_question")

    for key in ["opa", "opb", "opc", "opd"]:
        val = record.get(key)
        if val is None:
            issues.append(f"missing_{key}")

    # cop can be None in the test split (no ground truth)
    if cop is not None:

        try:
            cop_int = int(cop)

            if cop_int not in {1, 2, 3, 4}:
                issues.append("cop_out_of_range")

        except (ValueError, TypeError):
            issues.append("cop_not_integer")

    return issues


def _validate_mmlu(record, index):
    """Validate a single MMLU record."""

    issues = []

    question = record.get("question")
    choices = record.get("choices")
    answer = record.get("answer")

    if not question or not isinstance(question, str):
        issues.append("missing_or_invalid_question")

    if not choices or not isinstance(choices, list):
        issues.append("missing_or_invalid_choices")
    elif len(choices) < 2:
        issues.append("too_few_choices")

    if answer is None:
        issues.append("missing_answer")
    elif isinstance(answer, int):
        if choices and (answer < 0 or answer >= len(choices)):
            issues.append("answer_index_out_of_range")
    elif isinstance(answer, str):
        if len(answer) != 1 or answer not in "ABCDEFGH":
            issues.append("answer_not_valid_letter")

    return issues


def _validate_pubmedqa(record, index):
    """Validate a single PubMedQA record."""

    issues = []

    question = (
        record.get("QUESTION")
        or record.get("question")
    )

    if not question or not isinstance(question, str):
        issues.append("missing_or_invalid_question")

    answer = (
        record.get("final_decision")
        or record.get("LABEL")
        or record.get("label")
    )

    if answer is not None:
        if answer not in {"yes", "no", "maybe"}:
            issues.append("invalid_answer_value")

    return issues


def _validate_bioasq(record, index):
    """Validate a single BioASQ record."""

    issues = []

    body = record.get("body")

    if not body or not isinstance(body, str):
        issues.append("missing_or_invalid_body")

    qtype = record.get("type")

    if qtype not in {
        "yesno", "list", "factoid", "summary"
    }:
        issues.append("invalid_question_type")

    return issues


# Map dataset names to their validators
_VALIDATORS = {
    "MedQA": _validate_medqa,
    "MedMCQA": _validate_medmcqa,
    "MMLU-Medical": _validate_mmlu,
    "PubMedQA": _validate_pubmedqa,
    "BioASQ": _validate_bioasq,
}


# ============================================================
# DATASET-LEVEL AUDIT FUNCTIONS
# ============================================================

def audit_medqa(dataset_root):
    """Audit MedQA dataset."""

    dataset_root = Path(dataset_root)

    results = {
        "dataset": "MedQA",
        "subsets": {},
        "total_records": 0,
        "total_malformed": 0,
        "total_issues": defaultdict(int),
    }

    questions_dir = dataset_root / "questions"

    for subset_dir in sorted(questions_dir.iterdir()):

        if not subset_dir.is_dir():
            continue

        subset_name = subset_dir.name

        if subset_name.startswith("."):
            continue

        subset_result = {"splits": {}}

        for split_file in sorted(subset_dir.glob("*.jsonl")):

            split_name = split_file.stem

            # Skip 4_options subdirectory files
            if "4_options" in str(split_file):
                continue

            records = _load_jsonl_file(split_file)

            malformed = [
                r for r in records
                if r.get("__malformed__")
            ]

            valid = [
                r for r in records
                if not r.get("__malformed__")
            ]

            issue_counts = defaultdict(int)

            records_with_issues = 0

            for i, rec in enumerate(valid):

                issues = _validate_medqa(rec, i)

                if issues:
                    records_with_issues += 1

                for issue in issues:
                    issue_counts[issue] += 1

            subset_result["splits"][split_name] = {
                "file": str(split_file.relative_to(dataset_root)),
                "total": len(records),
                "valid": len(valid),
                "malformed": len(malformed),
                "records_with_issues": records_with_issues,
                "issues": dict(issue_counts),
                "sample_fields": (
                    list(valid[0].keys()) if valid else []
                ),
            }

            results["total_records"] += len(records)
            results["total_malformed"] += len(malformed)

            for k, v in issue_counts.items():
                results["total_issues"][k] += v

        results["subsets"][subset_name] = subset_result

    results["total_issues"] = dict(results["total_issues"])

    return results


def audit_medmcqa(dataset_root):
    """Audit MedMCQA dataset."""

    dataset_root = Path(dataset_root)

    results = {
        "dataset": "MedMCQA",
        "splits": {},
        "total_records": 0,
        "total_malformed": 0,
        "total_issues": defaultdict(int),
    }

    for split_file in sorted(dataset_root.glob("*.json")):

        split_name = split_file.stem

        # MedMCQA .json files are actually JSONL
        records = _load_jsonl_file(split_file)

        malformed = [
            r for r in records
            if r.get("__malformed__")
        ]

        valid = [
            r for r in records
            if not r.get("__malformed__")
        ]

        issue_counts = defaultdict(int)

        records_with_issues = 0

        for i, rec in enumerate(valid):

            issues = _validate_medmcqa(rec, i)

            if issues:
                records_with_issues += 1

            for issue in issues:
                issue_counts[issue] += 1

        # Count subjects
        subjects = defaultdict(int)

        for rec in valid:

            subj = rec.get("subject_name", "unknown")

            subjects[subj] += 1

        results["splits"][split_name] = {
            "file": split_file.name,
            "total": len(records),
            "valid": len(valid),
            "malformed": len(malformed),
            "records_with_issues": records_with_issues,
            "issues": dict(issue_counts),
            "subjects_count": len(subjects),
            "sample_fields": (
                list(valid[0].keys()) if valid else []
            ),
        }

        results["total_records"] += len(records)
        results["total_malformed"] += len(malformed)

        for k, v in issue_counts.items():
            results["total_issues"][k] += v

    results["total_issues"] = dict(results["total_issues"])

    return results


def audit_mmlu(dataset_root):
    """Audit MMLU-Medical dataset."""

    dataset_root = Path(dataset_root)

    results = {
        "dataset": "MMLU-Medical",
        "subjects": {},
        "total_records": 0,
        "total_malformed": 0,
        "total_issues": defaultdict(int),
    }

    for subject_dir in sorted(dataset_root.iterdir()):

        if not subject_dir.is_dir():
            continue

        subject_name = subject_dir.name

        subject_result = {"splits": {}}

        for split_file in sorted(subject_dir.glob("*.json")):

            split_name = split_file.stem

            try:
                data = _load_json_file(split_file)
            except Exception:
                subject_result["splits"][split_name] = {
                    "file": split_file.name,
                    "error": "failed_to_load"
                }
                continue

            records = data if isinstance(data, list) else []

            issue_counts = defaultdict(int)

            records_with_issues = 0

            for i, rec in enumerate(records):

                issues = _validate_mmlu(rec, i)

                if issues:
                    records_with_issues += 1

                for issue in issues:
                    issue_counts[issue] += 1

            subject_result["splits"][split_name] = {
                "file": split_file.name,
                "total": len(records),
                "records_with_issues": records_with_issues,
                "issues": dict(issue_counts),
                "sample_fields": (
                    list(records[0].keys())
                    if records else []
                ),
            }

            results["total_records"] += len(records)

            for k, v in issue_counts.items():
                results["total_issues"][k] += v

        results["subjects"][subject_name] = subject_result

    results["total_issues"] = dict(results["total_issues"])

    return results


def audit_pubmedqa(dataset_root):
    """Audit PubMedQA dataset."""

    dataset_root = Path(dataset_root)

    results = {
        "dataset": "PubMedQA",
        "files": {},
        "total_records": 0,
        "total_issues": defaultdict(int),
    }

    for json_file in sorted(dataset_root.glob("*.json")):

        file_name = json_file.name

        try:
            data = _load_json_file(json_file)
        except Exception:
            results["files"][file_name] = {
                "error": "failed_to_load"
            }
            continue

        records = _extract_records(data)

        issue_counts = defaultdict(int)

        records_with_issues = 0

        for i, rec in enumerate(records):

            issues = _validate_pubmedqa(rec, i)

            if issues:
                records_with_issues += 1

            for issue in issues:
                issue_counts[issue] += 1

        results["files"][file_name] = {
            "total": len(records),
            "records_with_issues": records_with_issues,
            "issues": dict(issue_counts),
            "sample_fields": (
                list(records[0].keys()) if records else []
            ),
        }

        results["total_records"] += len(records)

        for k, v in issue_counts.items():
            results["total_issues"][k] += v

    results["total_issues"] = dict(results["total_issues"])

    return results


def audit_bioasq(dataset_root):
    """Audit BioASQ dataset."""

    dataset_root = Path(dataset_root)

    results = {
        "dataset": "BioASQ",
        "files": {},
        "total_records": 0,
        "total_issues": defaultdict(int),
        "question_types": defaultdict(int),
    }

    for json_file in sorted(dataset_root.rglob("*.json")):

        file_name = str(
            json_file.relative_to(dataset_root)
        )

        try:
            data = _load_json_file(json_file)
        except Exception:
            results["files"][file_name] = {
                "error": "failed_to_load"
            }
            continue

        records = _extract_records(data)

        issue_counts = defaultdict(int)

        records_with_issues = 0

        for i, rec in enumerate(records):

            issues = _validate_bioasq(rec, i)

            if issues:
                records_with_issues += 1

            for issue in issues:
                issue_counts[issue] += 1

            qtype = rec.get("type", "unknown")

            results["question_types"][qtype] += 1

        results["files"][file_name] = {
            "total": len(records),
            "records_with_issues": records_with_issues,
            "issues": dict(issue_counts),
            "sample_fields": (
                list(records[0].keys()) if records else []
            ),
        }

        results["total_records"] += len(records)

        for k, v in issue_counts.items():
            results["total_issues"][k] += v

    results["total_issues"] = dict(results["total_issues"])
    results["question_types"] = dict(results["question_types"])

    return results


def audit_textbooks(textbook_dir):
    """Audit the English textbook corpus."""

    textbook_dir = Path(textbook_dir)

    results = {
        "dataset": "Textbooks",
        "books": [],
        "total_books": 0,
        "total_bytes": 0,
        "total_words": 0,
    }

    for txt_file in sorted(textbook_dir.glob("*.txt")):

        size = txt_file.stat().st_size

        with open(
            txt_file, "r",
            encoding="utf-8", errors="ignore"
        ) as f:

            text = f.read()

        word_count = len(text.split())
        char_count = len(text)
        line_count = text.count("\n") + 1

        results["books"].append({
            "filename": txt_file.name,
            "size_bytes": size,
            "characters": char_count,
            "words": word_count,
            "lines": line_count,
        })

        results["total_bytes"] += size
        results["total_words"] += word_count

    results["total_books"] = len(results["books"])

    return results


# ============================================================
# FULL AUDIT
# ============================================================

def run_full_audit(dataset_root):
    """
    Run the full Phase 1 audit across all datasets.

    Parameters
    ----------
    dataset_root : str or Path
        Path to the datasets/ directory.

    Returns
    -------
    dict
        Complete audit results for all datasets.
    """

    dataset_root = Path(dataset_root)

    print("=" * 70)
    print("MedRAGent -- PHASE 1 COMPREHENSIVE DATASET AUDIT")
    print("=" * 70)

    audit_results = {}

    # ----------------------------------------------------------
    # MedQA
    # ----------------------------------------------------------

    medqa_path = dataset_root / "MedQA"

    if medqa_path.exists():
        print("\n[1/6] Auditing MedQA...")
        audit_results["MedQA"] = audit_medqa(medqa_path)
        _print_summary("MedQA", audit_results["MedQA"])
    else:
        print("\n[1/6] MedQA -- NOT FOUND")

    # ----------------------------------------------------------
    # MedMCQA
    # ----------------------------------------------------------

    medmcqa_path = dataset_root / "MedMCQA"

    if medmcqa_path.exists():
        print("\n[2/6] Auditing MedMCQA...")
        audit_results["MedMCQA"] = audit_medmcqa(medmcqa_path)
        _print_summary("MedMCQA", audit_results["MedMCQA"])
    else:
        print("\n[2/6] MedMCQA -- NOT FOUND")

    # ----------------------------------------------------------
    # MMLU-Medical
    # ----------------------------------------------------------

    mmlu_path = dataset_root / "MMLU-Medical"

    if mmlu_path.exists():
        print("\n[3/6] Auditing MMLU-Medical...")
        audit_results["MMLU-Medical"] = audit_mmlu(mmlu_path)
        _print_summary("MMLU-Medical", audit_results["MMLU-Medical"])
    else:
        print("\n[3/6] MMLU-Medical -- NOT FOUND")

    # ----------------------------------------------------------
    # PubMedQA
    # ----------------------------------------------------------

    pubmedqa_path = dataset_root / "PubMedQA"

    if pubmedqa_path.exists():
        print("\n[4/6] Auditing PubMedQA...")
        audit_results["PubMedQA"] = audit_pubmedqa(pubmedqa_path)
        _print_summary("PubMedQA", audit_results["PubMedQA"])
    else:
        print("\n[4/6] PubMedQA -- NOT FOUND")

    # ----------------------------------------------------------
    # BioASQ
    # ----------------------------------------------------------

    bioasq_path = dataset_root / "BioASQ"

    if bioasq_path.exists():
        print("\n[5/6] Auditing BioASQ...")
        audit_results["BioASQ"] = audit_bioasq(bioasq_path)
        _print_summary("BioASQ", audit_results["BioASQ"])
    else:
        print("\n[5/6] BioASQ -- NOT FOUND")

    # ----------------------------------------------------------
    # Textbooks
    # ----------------------------------------------------------

    textbook_path = dataset_root / "MedQA" / "textbooks" / "en"

    if textbook_path.exists():
        print("\n[6/6] Auditing textbook corpus...")
        audit_results["Textbooks"] = audit_textbooks(textbook_path)
        tb = audit_results["Textbooks"]
        print(f"  Books       : {tb['total_books']}")
        print(f"  Total words : {tb['total_words']:,}")
        print(f"  Total bytes : {tb['total_bytes']:,}")
    else:
        print("\n[6/6] Textbooks -- NOT FOUND")

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    return audit_results


def _print_summary(name, result):
    """Print a quick summary for one dataset."""

    total = result.get("total_records", 0)
    malformed = result.get("total_malformed", 0)
    issues = result.get("total_issues", {})

    print(f"  Total records : {total:,}")

    if malformed:
        print(f"  Malformed     : {malformed}")

    if issues:

        print(f"  Issues        :")

        for issue, count in sorted(issues.items()):

            print(f"    {issue}: {count}")

    else:
        print(f"  Issues        : NONE [OK]")


# ============================================================
# STANDALONE
# ============================================================

if __name__ == "__main__":

    import sys

    root = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "datasets"
    )

    results = run_full_audit(root)

    # Save audit report
    output = Path("data/processed/metadata/audit_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nAudit report saved to: {output}")
