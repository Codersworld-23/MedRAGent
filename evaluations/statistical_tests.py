"""Paired statistical tests for baseline system comparisons."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable


def mcnemar_exact(both_correct: int, both_wrong: int, rag_only: int, llm_only: int) -> dict:
    """Return exact McNemar counts and a two-sided binomial-discordant test."""
    discordant = rag_only + llm_only
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = sum(
            math.comb(discordant, k) * (0.5 ** discordant)
            for k in range(discordant + 1)
            if math.comb(discordant, k) * (0.5 ** discordant)
            <= math.comb(discordant, rag_only) * (0.5 ** discordant) + 1e-15
        )
        p_value = min(1.0, p_value)
    n = both_correct + both_wrong + rag_only + llm_only
    return {
        "total_questions": n,
        "both_correct": both_correct,
        "rag_only": rag_only,
        "llm_only": llm_only,
        "both_wrong": both_wrong,
        "discordant": discordant,
        "odds_ratio": (
            rag_only / llm_only if llm_only else None
        ),
        "exact_mcnemar_p": p_value,
    }


def _wilson_difference(values_a: list[bool], values_b: list[bool]) -> tuple[float, float]:
    differences = [1 if a != b else 0 for a, b in zip(values_a, values_b)]
    mean = sum(differences) / len(differences) if differences else 0.0
    z = 1.96
    margin = z * math.sqrt(mean * (1 - mean) / len(differences)) if differences else 0.0
    return mean, margin


def bootstrap_accuracy_difference(
    baseline: list[bool],
    system: list[bool],
    *,
    samples: int = 10000,
    seed: int = 42,
) -> dict:
    """Bootstrap the paired accuracy difference with deterministic resampling."""
    if len(baseline) != len(system):
        raise ValueError("paired correctness lists must have equal length")
    if not baseline:
        raise ValueError("paired correctness lists cannot be empty")
    rng = random.Random(seed)
    n = len(baseline)
    estimates = []
    for _ in range(samples):
        correct_base = 0
        correct_system = 0
        for _ in range(n):
            index = rng.randrange(n)
            correct_base += baseline[index]
            correct_system += system[index]
        estimates.append(correct_system / n - correct_base / n)
    estimates.sort()
    low = estimates[max(0, math.ceil(0.025 * samples) - 1)]
    high = estimates[max(0, math.ceil(0.975 * samples) - 1)]
    difference = sum(system) / n - sum(baseline) / n
    return {
        "accuracy_difference": difference,
        "bootstrap_95_ci": [low, high],
        "bootstrap_samples": samples,
        "seed": seed,
    }


def holm_bonferroni(p_values: Iterable[tuple[str, float]]) -> dict[str, float]:
    """Apply Holm-Bonferroni correction and preserve comparison labels."""
    ordered = sorted(p_values, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, p_value) in enumerate(ordered):
        corrected = min(1.0, (len(ordered) - index) * p_value)
        corrected = max(running, corrected)
        running = corrected
        adjusted[name] = corrected
    return dict(sorted(adjusted.items()))


def compare_paired_results(
    baseline_rows: list[dict],
    system_rows: list[dict],
) -> dict:
    baseline = {row["question_id"]: bool(row["is_correct"]) for row in baseline_rows}
    system = {row["question_id"]: bool(row["is_correct"]) for row in system_rows}
    if set(baseline) != set(system):
        raise ValueError("paired result files must contain the same question IDs")
    ids = sorted(baseline)
    both_correct = both_wrong = rag_only = llm_only = 0
    for question_id in ids:
        base = baseline[question_id]
        other = system[question_id]
        both_correct += base and other
        both_wrong += not base and not other
        rag_only += not base and other
        llm_only += base and not other
    test = mcnemar_exact(both_correct, both_wrong, rag_only, llm_only)
    bootstrap = bootstrap_accuracy_difference(
        [baseline[question_id] for question_id in ids],
        [system[question_id] for question_id in ids],
    )
    test.update(bootstrap)
    test["effect_size_g1_odds"] = (
        rag_only / llm_only if llm_only else None
    )
    return test


def load_result_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    ids = [row["question_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate question IDs")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--systems", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline = load_result_jsonl(args.baseline)
    comparisons = {}
    raw_p_values = []
    for path in args.systems:
        name = path.stem
        comparison = compare_paired_results(baseline, load_result_jsonl(path))
        comparisons[name] = comparison
        raw_p_values.append((name, comparison["exact_mcnemar_p"]))
    adjusted = holm_bonferroni(raw_p_values)
    for name in comparisons:
        comparisons[name]["holm_bonferroni_p"] = adjusted[name]
    report = {
        "baseline": str(args.baseline),
        "comparison_method": "exact McNemar for paired discordant correctness",
        "multiple_comparison_correction": "Holm-Bonferroni",
        "bootstrap": {"method": "paired, 10000 samples, seed 42"},
        "comparisons": comparisons,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
