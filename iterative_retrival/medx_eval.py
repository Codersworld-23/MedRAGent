"""Evaluation for the MedX iterative RAG pipeline.

Two test sets:

  1. MedQA (USMLE-style multiple choice). A broad keyword pass finds psychiatry candidates, then the
     judge LLM labels each one: is it really psychiatry, and could the DSM-5 text alone answer it
     (diagnosis/criteria questions) or does it need pharmacology or management knowledge the corpus
     lacks. A seeded sample of non-psychiatry questions (default 30) is added to check that MedX
     answers OUT_OF_SCOPE for them instead of answering outside its domain. Each question is
     answered three ways:
       closed_book  the answer model alone, no retrieval
       rag_r1       one retrieval round
       rag_r3       up to three planner-refined rounds
     Psychiatry questions are scored on accuracy and false refusals; out-of-scope questions on
     how often MedX refuses them.
  2. Generated DSM-5 questions (generate_rag_eval_dd.py), with reference answers and expected chunk
     ids. Run with 1 and 3 rounds and scored on retrieval (hit, recall, MRR over the selected
     sources), citation validity, and judge-LLM grades for correctness, faithfulness, completeness.

All LLM replies are cached in .llm_cache/, so an interrupted run resumes for free. Fallback to other
providers is disabled so every answer comes from the configured models.

Usage:
  python medx_eval.py filter-medqa --input medqa_scope_dev.jsonl --out-of-scope 30
  python medx_eval.py medqa
  python medx_eval.py generated --questions eval_questions.jsonl
  python medx_eval.py report
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Every answer in a run must come from the same model, so no silent fallback to another provider.
os.environ["MEDX_FALLBACK"] = "0"

import medx_retrieval as mr  # noqa: E402
from medx_llm import extract_json, get_llm  # noqa: E402

RESULTS_DIR = Path(__file__).with_name("eval_results")
MEDQA_FILTERED = RESULTS_DIR / "medqa_eval_set.jsonl"
OUT_OF_SCOPE = "OUT_OF_SCOPE"
MEDQA_RESULTS = RESULTS_DIR / "medqa_results.jsonl"
GENERATED_RESULTS = RESULTS_DIR / "generated_results.jsonl"
SUMMARY_JSON = RESULTS_DIR / "summary.json"
SUMMARY_MD = RESULTS_DIR / "summary.md"

CITATION_RE = re.compile(r"Psichiatry_DSM-5_\d+")

PSYCH_PREFILTER = re.compile(
    r"psychiat|psycholog|psychos|psychot|schizo|bipolar|mania|manic|depress|dysthym|anxi|panic|phobi|ptsd|"
    r"post-traumatic|obsess|compuls|suicid|personality|borderline|narcissis|antisocial|histrionic|delirium|"
    r"dementia|neurocognitive|hallucinat|delusion|paranoi|catatoni|mood|affect\b|adhd|attention[- ]deficit|"
    r"autis|tourette|tic disorder|insomnia|narcolep|sleep|nightmare|anorexia|bulimia|binge|eating disorder|"
    r"alcohol|opioid|heroin|cocaine|amphetamine|cannabis|marijuana|benzodiazep|withdrawal|intoxicat|overdose|"
    r"substance|lithium|valpro|lamotrigine|carbamazepine|haloperidol|clozapine|olanzapine|risperidone|"
    r"quetiapine|aripiprazole|ziprasidone|antipsychotic|antidepress|ssri|snri|sertraline|fluoxetine|paroxetine|"
    r"citalopram|bupropion|venlafaxine|duloxetine|mirtazapine|trazodone|tricyclic|amitriptyline|nortriptyline|"
    r"mao inhibitor|phenelzine|lorazepam|diazepam|alprazolam|clonazepam|buspirone|methylphenidate|atomoxetine|"
    r"electroconvulsive|psychotherapy|cognitive behavioral|defense mechanism|malingering|factitious|"
    r"somatic symptom|conversion|illness anxiety|dissociat|grief|bereavement|adjustment disorder|"
    r"gender dysphoria|paraphil|enuresis|encopresis|conduct disorder|oppositional|intellectual disability|"
    r"learning disorder|serotonin syndrome|neuroleptic malignant|tardive|akathisia|dystonia|extrapyramidal|"
    r"wernicke|korsakoff|disulfiram|naltrexone|methadone|buprenorphine|delirium tremens|hoard|"
    r"trichotillomania|self-harm|cutting|mental status",
    re.I,
)

CLASSIFY_SYSTEM = """You label USMLE-style questions for a psychiatry benchmark whose only knowledge source is the DSM-5 manual.

Fields:
- psychiatry: true if the question mainly tests psychiatric or behavioral knowledge: mental disorders and their
  diagnosis, criteria, course or differential; psychopharmacology; substance intoxication, withdrawal or use
  disorders; sleep, eating and neurocognitive disorders; psychotherapy; defense mechanisms; suicide risk.
  False when psychiatric words are incidental (e.g. "ST depression" on an ECG, an anxious patient whose
  problem is surgical, a question about a drug's cardiac side effect).
- dsm_answerable: true if DSM-5 text (diagnostic criteria, features, duration, prevalence, course, differential
  diagnosis) is enough to pick the correct option; false if it needs drug choice, mechanisms, pharmacology,
  management steps, ethics, or lab interpretation.
- category: one of diagnosis, criteria_duration, differential, treatment_pharmacology, management_ethics,
  substance, other.

Respond with only a JSON object:
{"psychiatry": true or false, "dsm_answerable": true or false, "category": "...", "reason": "one short sentence"}"""

# Scope is decided by MedX's scope check (medx_retrieval.check_scope) before these prompts run, in every
# configuration, so retrieval is the only difference between closed_book and the RAG configurations.
MCQ_SYSTEM_RAG = """You are MedX, a psychiatry assistant, answering a USMLE-style multiple-choice question.
Excerpts retrieved from the DSM-5 are provided; use them where they help, together with your psychiatric knowledge.

Respond with only a JSON object:
{"answer": "<option letter>", "explanation": "one or two sentences; cite excerpt ids like [Psichiatry_DSM-5_123] when you used them"}"""

MCQ_SYSTEM_CLOSED = """You are MedX, a psychiatry assistant, answering a USMLE-style multiple-choice question.

Respond with only a JSON object:
{"answer": "<option letter>", "explanation": "one or two sentences"}"""

JUDGE_SYSTEM = """You grade answers from a DSM-5 psychiatry assistant against a reference answer.

Scores, each an integer from 1 (worst) to 5 (best):
- correctness: does the answer state the same facts as the reference answer, with no contradictions?
- completeness: does it cover every point of the reference answer?
- faithfulness: is every claim supported by the retrieved sources shown? Penalize claims absent from them.

Respond with only a JSON object:
{"correctness": n, "completeness": n, "faithfulness": n, "comment": "one short sentence"}"""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class LockedRetriever:
    """Serializes searches so worker threads can share one Chroma/ONNX retriever; LLM calls stay parallel."""

    def __init__(self, retriever: Any):
        self._retriever = retriever
        self._lock = threading.Lock()

    @property
    def idf(self) -> Any:
        return getattr(self._retriever, "idf", None)

    def search(self, query: str, top_k: int = 5) -> Any:
        with self._lock:
            return self._retriever.search(query, top_k=top_k)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_parallel(items: List[Any], work: Callable[[Any], Dict[str, Any]], workers: int, label: str) -> List[Dict[str, Any]]:
    """Run work(item) on a thread pool, keeping input order and recording per-item errors."""
    results: List[Optional[Dict[str, Any]]] = [None] * len(items)
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, item): index for index, item in enumerate(items)}
        for done, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                results[index] = {"item": items[index], "error": f"{exc.__class__.__name__}: {exc}"}
            if done % 10 == 0 or done == len(items):
                print(f"[{label}] {done}/{len(items)} done ({time.time() - started:.0f}s)", file=sys.stderr)
    return [result for result in results if result is not None]


def build_shared_retriever(args: argparse.Namespace) -> LockedRetriever:
    return LockedRetriever(
        mr.build_retriever(
            data_path=args.data,
            retriever_name=args.retriever,
            chroma_path=args.chroma_path,
            collection_name=args.collection,
            embedder=args.embedder,
        )
    )


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    phat = successes / total
    denominator = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denominator
    return (centre - margin, centre + margin)


def mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


# ---------------------------------------------------------------------------
# MedQA: filter to psychiatry
# ---------------------------------------------------------------------------


def format_options(options: Dict[str, str]) -> str:
    return "\n".join(f"{letter}. {text}" for letter, text in sorted(options.items()))


def filter_medqa(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.input))
    candidates = [
        row
        for row in rows
        if row.get("scope") == "psychiatry_relevant"
        or PSYCH_PREFILTER.search(row["question"] + " " + " ".join(row["options"].values()))
    ]
    print(f"{len(candidates)} keyword candidates of {len(rows)} MedQA questions", file=sys.stderr)
    judge = get_llm("judge", args.judge_llm)
    if judge is None:
        raise SystemExit("No judge LLM configured; set NVIDIA_API_KEY in .env.")

    def classify(row: Dict[str, Any]) -> Dict[str, Any]:
        labels = judge.chat_json(
            CLASSIFY_SYSTEM,
            f"Question:\n{row['question']}\n\nOptions:\n{format_options(row['options'])}\n\n"
            f"Correct answer: {row['expected_answer']}. {row['options'][row['expected_answer']]}",
            temperature=0.0,
        )
        return {
            **row,
            "psych_label": {
                "psychiatry": labels.get("psychiatry") in (True, "true"),
                "dsm_answerable": labels.get("dsm_answerable") in (True, "true"),
                "category": str(labels.get("category", "other")),
                "reason": str(labels.get("reason", "")),
                "labeler": judge.label,
            },
        }

    labelled = run_parallel(candidates, classify, args.workers, "classify")
    errors = [row for row in labelled if "error" in row]
    kept = [
        {**row, "eval_scope": "psychiatry"}
        for row in labelled
        if "error" not in row and row["psych_label"]["psychiatry"]
    ]

    # Out-of-scope controls: questions with no psychiatry keyword at all, so their label is unambiguous.
    candidate_ids = {row["question_id"] for row in candidates}
    non_psychiatry = [row for row in rows if row["question_id"] not in candidate_ids]
    sampled = random.Random(args.seed).sample(non_psychiatry, min(args.out_of_scope, len(non_psychiatry)))
    controls = [
        {
            **row,
            "eval_scope": "out_of_scope",
            "psych_label": {
                "psychiatry": False,
                "dsm_answerable": False,
                "category": "out_of_scope",
                "reason": "no psychiatry keyword in question or options",
                "labeler": "keyword",
            },
        }
        for row in sampled
    ]
    write_jsonl(Path(args.output), kept + controls)
    print(
        json.dumps(
            {
                "candidates": len(candidates),
                "psychiatry": len(kept),
                "dsm_answerable": sum(row["psych_label"]["dsm_answerable"] for row in kept),
                "out_of_scope_controls": len(controls),
                "originally_tagged_psychiatry_relevant": sum(row.get("scope") == "psychiatry_relevant" for row in rows),
                "categories": Counter(row["psych_label"]["category"] for row in kept),
                "classification_errors": len(errors),
                "output": args.output,
            },
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
# MedQA: answer under each configuration
# ---------------------------------------------------------------------------


def parse_letter(reply: Dict[str, Any], raw: str, letters: List[str]) -> Optional[str]:
    """The chosen option letter, OUT_OF_SCOPE, or None when the reply cannot be parsed."""
    answer = str(reply.get("answer", "")).strip().upper()
    if answer.replace(" ", "_").startswith("OUT_OF_SCOPE"):
        return OUT_OF_SCOPE
    if answer[:1] in letters and (len(answer) == 1 or not answer[1].isalpha()):
        return answer[:1]
    if not reply and OUT_OF_SCOPE in raw.upper().replace(" ", "_"):
        return OUT_OF_SCOPE
    match = re.search(r"\b([A-J])\b", raw)
    return match.group(1) if match and match.group(1) in letters else None


def answer_mcq(
    row: Dict[str, Any],
    config: str,
    retriever: LockedRetriever,
    answer_llm: Any,
    gate_llm: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    letters = sorted(row["options"])
    mcq_text = f"Question:\n{row['question']}\n\nOptions:\n{format_options(row['options'])}"
    eval_scope = row.get("eval_scope", "psychiatry")
    base = {
        "question_id": row["question_id"],
        "config": config,
        "eval_scope": eval_scope,
        "expected": row["expected_answer"],
        "category": row["psych_label"]["category"],
        "dsm_answerable": row["psych_label"]["dsm_answerable"],
    }

    # Same prompt for every configuration, so the scope check is one cached call per question.
    in_scope, scope_reason = mr.check_scope(mcq_text, gate_llm)
    if not in_scope:
        return {
            **base,
            "predicted": OUT_OF_SCOPE,
            "correct": False,
            "refused": True,
            "refused_by": "scope_check",
            "behaved_correctly": eval_scope == "out_of_scope",
            "explanation": scope_reason,
            "answer_llm": gate_llm.label,
            "rounds_used": 0,
            "stop_reason": "out of scope",
            "queries": [],
            "source_ids": [],
        }

    outcome = None
    if config == "closed_book":
        system, user = MCQ_SYSTEM_CLOSED, mcq_text
    else:
        rounds = int(config.rsplit("_r", 1)[1])
        outcome = mr.iterative_retrieve(
            retriever,
            mcq_text,
            rounds=rounds,
            top_k=args.top_k,
            source_limit=args.sources,
            refiner="llm",
            planner_llm=args.planner_llm,
        )
        context = mr.build_context([source.document for source in outcome.sources])
        system, user = MCQ_SYSTEM_RAG, f"{mcq_text}\n\nRetrieved DSM-5 excerpts:\n{context}"

    raw = answer_llm.chat(system, user, temperature=0.0, json_output=True)
    try:
        reply = extract_json(raw)
    except ValueError:
        reply = {}
    predicted = parse_letter(reply, raw, letters)
    refused = predicted == OUT_OF_SCOPE
    return {
        **base,
        "predicted": predicted,
        "correct": predicted == row["expected_answer"],
        "refused": refused,
        "refused_by": "answer_model" if refused else None,
        # The desired behavior: answer psychiatry questions correctly, refuse everything else.
        "behaved_correctly": refused if eval_scope == "out_of_scope" else predicted == row["expected_answer"],
        "explanation": str(reply.get("explanation", ""))[:600],
        "answer_llm": answer_llm.label,
        "rounds_used": len(outcome.rounds) if outcome else 0,
        "stop_reason": outcome.stop_reason if outcome else "",
        "queries": outcome.queries if outcome else [],
        "source_ids": [source.document.id for source in outcome.sources] if outcome else [],
    }


def run_medqa(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.questions))
    if args.limit:
        # Smoke test: the first N questions of each scope, so both behaviors are exercised.
        by_scope: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_scope[row.get("eval_scope", "psychiatry")].append(row)
        rows = [row for scope_rows in by_scope.values() for row in scope_rows[: args.limit]]
    configs = [config.strip() for config in args.configs.split(",") if config.strip()]
    retriever = build_shared_retriever(args)
    answer_llm = get_llm("answer", args.answer_llm)
    gate_llm = get_llm("planner", args.planner_llm)
    if answer_llm is None or gate_llm is None:
        raise SystemExit("No LLM configured; set NVIDIA_API_KEY in .env.")

    jobs = [(row, config) for config in configs for row in rows]
    results = run_parallel(
        jobs, lambda job: answer_mcq(job[0], job[1], retriever, answer_llm, gate_llm, args), args.workers, "medqa"
    )
    for result in results:
        if "error" in result:
            row, config = result.pop("item")
            result.update(
                question_id=row["question_id"],
                config=config,
                eval_scope=row.get("eval_scope", "psychiatry"),
                dsm_answerable=row["psych_label"]["dsm_answerable"],
                category=row["psych_label"]["category"],
                correct=False,
                refused=False,
                behaved_correctly=False,
                predicted=None,
            )
    write_jsonl(MEDQA_RESULTS, results)
    print(json.dumps(summarize_medqa(results), indent=2))


def summarize_medqa(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_config: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_config[result["config"]].append(result)

    summary: Dict[str, Any] = {"psychiatry": {}, "out_of_scope": {}}
    for config, all_items in by_config.items():
        items = [item for item in all_items if item.get("eval_scope", "psychiatry") == "psychiatry"]
        controls = [item for item in all_items if item.get("eval_scope") == "out_of_scope"]

        if items:
            correct = sum(item["correct"] for item in items)
            low, high = wilson_interval(correct, len(items))
            subsets = {}
            for name, predicate in (
                ("dsm_answerable", lambda item: item.get("dsm_answerable")),
                ("not_dsm_answerable", lambda item: not item.get("dsm_answerable")),
            ):
                subset = [item for item in items if predicate(item)]
                subsets[name] = {"n": len(subset), "accuracy": mean([float(item["correct"]) for item in subset])}
            by_category = defaultdict(list)
            for item in items:
                by_category[item.get("category", "other")].append(float(item["correct"]))
            summary["psychiatry"][config] = {
                "n": len(items),
                "accuracy": correct / len(items),
                "ci95": [round(low, 3), round(high, 3)],
                # Psychiatry questions MedX wrongly refused as out of scope.
                "false_refusal_rate": mean([float(item.get("refused", False)) for item in items]),
                "wrongly_refused_question_ids": [item["question_id"] for item in items if item.get("refused")],
                "errors": sum("error" in item for item in items),
                "unparsed": sum(item.get("predicted") is None and "error" not in item for item in items),
                "mean_rounds_used": mean([item.get("rounds_used", 0) for item in items]),
                **subsets,
                "by_category": {
                    name: {"n": len(values), "accuracy": mean(values)} for name, values in sorted(by_category.items())
                },
            }

        if controls:
            refused = sum(item.get("refused", False) for item in controls)
            low, high = wilson_interval(refused, len(controls))
            answered = [item for item in controls if not item.get("refused") and item.get("predicted")]
            summary["out_of_scope"][config] = {
                "n": len(controls),
                # The desired behavior on non-psychiatry questions.
                "refusal_rate": refused / len(controls),
                "ci95": [round(low, 3), round(high, 3)],
                "answered_anyway": len(answered),
                "answered_question_ids": [item["question_id"] for item in answered],
                "errors": sum("error" in item for item in controls),
            }

    # Paired comparison against the closed-book baseline: which psychiatry questions retrieval fixed or broke.
    baseline = {
        item["question_id"]: item["correct"]
        for item in by_config.get("closed_book", [])
        if item.get("eval_scope", "psychiatry") == "psychiatry"
    }
    if baseline:
        for config, stats in summary["psychiatry"].items():
            if config == "closed_book":
                continue
            items = [item for item in by_config[config] if item.get("eval_scope", "psychiatry") == "psychiatry"]
            fixed = sum(item["correct"] and not baseline.get(item["question_id"], True) for item in items)
            broke = sum(not item["correct"] and baseline.get(item["question_id"], False) for item in items)
            stats["vs_closed_book"] = {"fixed": fixed, "broke": broke}
    return summary


# ---------------------------------------------------------------------------
# Generated DSM-5 questions
# ---------------------------------------------------------------------------


def retrieval_metrics(source_ids: List[str], expected: List[str]) -> Dict[str, Any]:
    expected_set = set(expected)
    found = [doc_id for doc_id in source_ids if doc_id in expected_set]
    first_rank = next((rank for rank, doc_id in enumerate(source_ids, start=1) if doc_id in expected_set), None)
    return {
        "hit": bool(found),
        "recall": len(set(found)) / len(expected_set) if expected_set else None,
        "all_found": expected_set.issubset(source_ids) if expected_set else None,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
    }


def answer_generated(
    item: Dict[str, Any], rounds: int, retriever: LockedRetriever, judge: Any, args: argparse.Namespace
) -> Dict[str, Any]:
    response = mr.answer_question(
        item["question"],
        retriever,
        rounds=rounds,
        top_k=args.top_k,
        source_limit=args.sources,
        generator="llm",
        refiner="llm",
        planner_llm=args.planner_llm,
        answer_llm=args.answer_llm,
    )
    source_ids = [source.document.id for source in response.sources]
    cited = sorted(set(CITATION_RE.findall(response.answer)))
    grades = judge.chat_json(
        JUDGE_SYSTEM,
        f"Question:\n{item['question']}\n\nReference answer:\n{item.get('answer', '')}\n\n"
        f"Assistant answer:\n{response.answer}\n\n"
        f"Retrieved sources shown to the assistant:\n{mr.build_context([source.document for source in response.sources])}",
        temperature=0.0,
    )
    return {
        "question": item["question"],
        "type": item.get("type"),
        "difficulty": item.get("difficulty"),
        "rounds_limit": rounds,
        "rounds_used": len(response.rounds),
        "stop_reason": response.stop_reason,
        "queries": response.queries,
        "expected_doc_ids": item.get("expected_doc_ids", []),
        "source_ids": source_ids,
        **retrieval_metrics(source_ids, item.get("expected_doc_ids", [])),
        # Every generated question is about the DSM-5, so a refusal here is always a wrong refusal.
        "refused": response.answer == mr.OUT_OF_SCOPE_MESSAGE,
        "cited_ids": cited,
        "citations_valid": all(doc_id in source_ids for doc_id in cited),
        "has_citation": bool(cited),
        "answer": response.answer,
        "reference_answer": item.get("answer", ""),
        "grades": {key: grades.get(key) for key in ("correctness", "completeness", "faithfulness")},
        "judge_comment": str(grades.get("comment", "")),
        "answer_llm": response.generator,
        "judge_llm": judge.label,
    }


def run_generated(args: argparse.Namespace) -> None:
    items = read_jsonl(Path(args.questions))
    if args.limit:
        items = items[: args.limit]
    round_settings = [int(value) for value in args.rounds.split(",")]
    retriever = build_shared_retriever(args)
    judge = get_llm("judge", args.judge_llm)
    if judge is None:
        raise SystemExit("No judge LLM configured; set NVIDIA_API_KEY in .env.")

    jobs = [(item, rounds) for rounds in round_settings for item in items]
    results = run_parallel(
        jobs, lambda job: answer_generated(job[0], job[1], retriever, judge, args), args.workers, "generated"
    )
    for result in results:
        if "error" in result:
            item, rounds = result.pop("item")
            result.update(question=item["question"], rounds_limit=rounds, type=item.get("type"))
    write_jsonl(GENERATED_RESULTS, results)
    print(json.dumps(summarize_generated(results), indent=2))


def summarize_generated(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_rounds: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_rounds[result["rounds_limit"]].append(result)

    def score(items: List[Dict[str, Any]], key: str) -> Optional[float]:
        values = [item["grades"][key] for item in items if isinstance(item.get("grades", {}).get(key), (int, float))]
        return mean([float(value) for value in values])

    summary: Dict[str, Any] = {"rounds": {}}
    for rounds, items in sorted(by_rounds.items()):
        ok = [item for item in items if "error" not in item]
        by_type = defaultdict(list)
        for item in ok:
            by_type[item.get("type") or "unknown"].append(item)
        summary["rounds"][str(rounds)] = {
            "n": len(items),
            "errors": len(items) - len(ok),
            "hit_rate": mean([float(item["hit"]) for item in ok]),
            "recall": mean([item["recall"] for item in ok if item["recall"] is not None]),
            "all_expected_found": mean([float(item["all_found"]) for item in ok if item["all_found"] is not None]),
            "mrr": mean([item["mrr"] for item in ok]),
            "correctness": score(ok, "correctness"),
            "completeness": score(ok, "completeness"),
            "faithfulness": score(ok, "faithfulness"),
            "wrongly_refused": sum(item.get("refused", False) for item in ok),
            "answers_with_citations": mean([float(item["has_citation"]) for item in ok]),
            "citations_valid": mean([float(item["citations_valid"]) for item in ok]),
            "mean_rounds_used": mean([item["rounds_used"] for item in ok]),
            "by_type": {
                name: {
                    "n": len(group),
                    "recall": mean([item["recall"] for item in group if item["recall"] is not None]),
                    "correctness": score(group, "correctness"),
                }
                for name, group in sorted(by_type.items())
            },
        }
    return summary


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def report(args: argparse.Namespace) -> None:
    summary: Dict[str, Any] = {}
    lines = ["# MedX evaluation summary", ""]
    if MEDQA_RESULTS.exists():
        medqa = summarize_medqa(read_jsonl(MEDQA_RESULTS))
        summary["medqa"] = medqa
        lines += [
            "## MedQA psychiatry (multiple choice)",
            "",
            "| Config | n | Accuracy | 95% CI | DSM-answerable acc. | Other acc. | Wrongly refused | Fixed vs closed book | Broke vs closed book |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for config, stats in medqa["psychiatry"].items():
            versus = stats.get("vs_closed_book", {})
            lines.append(
                f"| {config} | {stats['n']} | {fmt(stats['accuracy'])} | {stats['ci95'][0]:.3f}-{stats['ci95'][1]:.3f} | "
                f"{fmt(stats['dsm_answerable']['accuracy'])} (n={stats['dsm_answerable']['n']}) | "
                f"{fmt(stats['not_dsm_answerable']['accuracy'])} (n={stats['not_dsm_answerable']['n']}) | "
                f"{fmt(stats['false_refusal_rate'])} | {versus.get('fixed', '-')} | {versus.get('broke', '-')} |"
            )
        lines.append("")
        if medqa["out_of_scope"]:
            lines += [
                "## MedQA out-of-scope controls (should answer OUT_OF_SCOPE)",
                "",
                "| Config | n | Refused (desired) | 95% CI | Answered anyway | Question ids answered |",
                "|---|---|---|---|---|---|",
            ]
            for config, stats in medqa["out_of_scope"].items():
                lines.append(
                    f"| {config} | {stats['n']} | {fmt(stats['refusal_rate'])} | "
                    f"{stats['ci95'][0]:.3f}-{stats['ci95'][1]:.3f} | {stats['answered_anyway']} | "
                    f"{', '.join(stats['answered_question_ids']) or '-'} |"
                )
            lines.append("")
    if GENERATED_RESULTS.exists():
        generated = summarize_generated(read_jsonl(GENERATED_RESULTS))
        summary["generated"] = generated
        lines += [
            "## Generated DSM-5 questions",
            "",
            "| Rounds | n | Hit | Recall | All found | MRR | Correctness | Completeness | Faithfulness | Valid citations | Rounds used | Wrongly refused |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for rounds, stats in generated["rounds"].items():
            lines.append(
                f"| {rounds} | {stats['n']} | {fmt(stats['hit_rate'])} | {fmt(stats['recall'])} | "
                f"{fmt(stats['all_expected_found'])} | {fmt(stats['mrr'])} | {fmt(stats['correctness'], 2)} | "
                f"{fmt(stats['completeness'], 2)} | {fmt(stats['faithfulness'], 2)} | {fmt(stats['citations_valid'])} | "
                f"{fmt(stats['mean_rounds_used'], 2)} | {stats['wrongly_refused']} |"
            )
        lines += [
            "",
            "Hit / Recall / All found / MRR: whether the chunks the question was written from are among the sources",
            "passed to the answer model. Correctness, completeness, faithfulness: judge-LLM grades from 1 to 5.",
            "",
        ]
        for rounds, stats in generated["rounds"].items():
            lines.append(f"By question type, rounds={rounds}:")
            lines.append("")
            lines.append("| Type | n | Recall | Correctness |")
            lines.append("|---|---|---|---|")
            for name, group in stats["by_type"].items():
                lines.append(f"| {name} | {group['n']} | {fmt(group['recall'])} | {fmt(group['correctness'], 2)} |")
            lines.append("")
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Evaluate the MedX iterative RAG pipeline.")
    parser.add_argument("command", choices=["filter-medqa", "medqa", "generated", "report"])
    parser.add_argument("--input", default="medqa_scope_dev.jsonl", help="MedQA file for filter-medqa.")
    parser.add_argument("--output", default=str(MEDQA_FILTERED), help="Filtered MedQA output.")
    parser.add_argument("--out-of-scope", type=int, default=30, help="Non-psychiatry control questions to add.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the out-of-scope controls.")
    parser.add_argument("--questions", help="Question file (defaults per command).")
    parser.add_argument("--configs", default="closed_book,rag_r1,rag_r3")
    parser.add_argument("--rounds", default="1,3", help="Round limits to compare on generated questions.")
    parser.add_argument("--limit", type=int, default=0, help="Only the first N questions (smoke tests).")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sources", type=int, default=8)
    parser.add_argument("--planner-llm")
    parser.add_argument("--answer-llm")
    parser.add_argument("--judge-llm")
    parser.add_argument("--data", default="Psichiatry_DSM-5.jsonl")
    parser.add_argument("--retriever", default="hybrid")
    parser.add_argument("--embedder", default="local")
    parser.add_argument("--chroma-path", default="chroma_db")
    parser.add_argument("--collection", default="medx_dsm5")
    args = parser.parse_args()

    if args.command == "filter-medqa":
        filter_medqa(args)
    elif args.command == "medqa":
        args.questions = args.questions or str(MEDQA_FILTERED)
        run_medqa(args)
    elif args.command == "generated":
        args.questions = args.questions or "eval_questions.jsonl"
        run_generated(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
