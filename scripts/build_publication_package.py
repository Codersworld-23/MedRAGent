"""Build a reproducibility package for the MedQA baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def package_results(directory: Path) -> list[dict]:
    packages = []
    for path in sorted(directory.glob("*_test.jsonl")):
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        packages.append({
            "path": str(path),
            "sha256": sha256(path),
            "row_count": len(rows),
            "system_name": rows[0].get("system_name") if rows else None,
        })
    return packages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "test",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "publication" / "baseline_package.json",
    )
    args = parser.parse_args()

    freeze_path = args.test_dir / "freeze.json"
    if not freeze_path.exists():
        raise FileNotFoundError(
            f"Frozen test preflight not found: {freeze_path}"
        )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("execute_status") != "completed":
        raise RuntimeError(
            "Publication package is blocked: the immutable test run has not completed"
        )

    requirements = (PROJECT_ROOT / "requirements.txt").read_text(
        encoding="utf-8"
    )
    package = {
        "package_version": "baseline_v1",
        "repository": "Codersworld-23/MedRAGent",
        "commit": git_output("rev-parse", "HEAD"),
        "branch": git_output("rev-parse", "--abbrev-ref", "HEAD"),
        "status": git_output("status", "--short"),
        "platform": platform.platform(),
        "python": sys.version,
        "requirements": requirements.splitlines(),
        "freeze": freeze,
        "test_result_files": package_results(args.test_dir),
        "dataset_manifest": (
            json.loads(
                (
                    PROJECT_ROOT
                    / "experiments"
                    / "manifests"
                    / "dataset_manifest.json"
                ).read_text(encoding="utf-8")
            )
            if (
                PROJECT_ROOT
                / "experiments"
                / "manifests"
                / "dataset_manifest.json"
            ).exists()
            else None
        ),
        "commands": {
            "preflight": (
                "& .\\.venv\\Scripts\\python.exe "
                "scripts\\run_frozen_test_baseline.py"
            ),
            "execute_once": (
                "& .\\.venv\\Scripts\\python.exe "
                "scripts\\run_frozen_test_baseline.py --execute"
            ),
            "tests": (
                "& .\\.venv\\Scripts\\python.exe "
                "-m unittest discover -s tests -v"
            ),
        },
        "claims_scope": (
            "Frozen test claims apply only to the tested MedQA-US test split "
            "and DSM-5 evidence corpus."
        ),
        "clinical_status": "Not clinically validated.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(package, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
