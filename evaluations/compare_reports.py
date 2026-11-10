"""Compare common-evaluation reports using one consistent metric schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    comparison: dict[str, Any] = {"systems": {}}
    for path in args.reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        name = report.get("system_name") or path.stem
        comparison["systems"][name] = {
            "source": str(path),
            "overall": report["overall"],
            "by_scope": report["by_scope"],
        }
    text = json.dumps(comparison, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
