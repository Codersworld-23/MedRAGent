"""
MedRAGent Benchmark Evaluator
=============================
Simple, fixed-benchmark evaluation function:
Pass any model/LLM -> Evaluates on fixed MedQA benchmark -> Returns structured report dict + markdown.

Usage:
    from evaluate_model import evaluate_model
    from llm_interface import create_llm

    # Example 1: With an LLM object
    model = create_llm('ollama', 'llama3.1:8b')
    report = evaluate_model(model, num_questions=20)

    # Example 2: With any custom callable/function
    def my_custom_rag(prompt: str) -> str:
        return "Answer is C because..."
    report = evaluate_model(my_custom_rag, num_questions=10)
"""

import os
import re
import json
import random
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

# Fixed benchmark configuration: data_clean.zip is in the major root directory
DEFAULT_BENCHMARK_ZIP = Path(__file__).resolve().parents[2] / "data_clean.zip"
DEFAULT_TEST_SET_PATH = "data_clean/questions/US/test.jsonl"
DEFAULT_BENCHMARK_SEED = 42


def _load_fixed_test_set(
    zip_path: Path = DEFAULT_BENCHMARK_ZIP,
    max_questions: Optional[int] = 50,
    seed: int = DEFAULT_BENCHMARK_SEED,
) -> List[Dict[str, Any]]:
    """Loads a deterministic, fixed test set from the benchmark archive."""
    if not zip_path.exists():
        # Fallback to parent folder if running from different location
        parent_zip = Path(__file__).resolve().parent / "data_clean.zip"
        if parent_zip.exists():
            zip_path = parent_zip
        else:
            raise FileNotFoundError(f"Benchmark dataset not found at {zip_path}")

    questions = []
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(DEFAULT_TEST_SET_PATH) as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line.decode("utf-8")))

    # Fixed seed guarantees same questions evaluated every time
    random.seed(seed)
    if max_questions and max_questions < len(questions):
        indices = list(range(len(questions)))
        random.shuffle(indices)
        selected_indices = sorted(indices[:max_questions])
        return [questions[i] for i in selected_indices]

    return questions


def _extract_answer_letter(text: str) -> str:
    """Extracts multiple-choice option letter (A-E) from model output."""
    if not text:
        return ""
    text_clean = text.strip()

    patterns = [
        r"(?:answer|choice|option)\s*(?:is|:)?\s*[\*#]*\s*([A-E])\b",
        r"\b([A-E])\s*(?:\.|\))\s*",
        r"\*\*([A-E])\*\*",
        r"^([A-E])\b",
        r"\b([A-E])$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text_clean, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    match = re.search(r"\b([A-E])\b", text_clean)
    return match.group(1).upper() if match else ""


def _format_question_prompt(q: Dict[str, Any]) -> str:
    """Standardized medical prompt for test benchmark."""
    question = q.get("question", "")
    options = q.get("options", {})
    options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])
    return (
        f"You are a medical expert taking the USMLE clinical examination.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Instructions: Think carefully and select the best option. "
        f"Clearly state the correct option letter (A, B, C, D, or E) and provide a brief clinical justification.\n"
        f"Final Answer:"
    )


def evaluate_model(
    model: Union[Any, Callable[[str], str]],
    num_questions: int = 50,
    output_dir: Optional[str] = "benchmark_reports",
    model_name: Optional[str] = None,
    seed: int = DEFAULT_BENCHMARK_SEED,
) -> Dict[str, Any]:
    """
    Evaluates any model or RAG pipeline on a fixed MedQA medical benchmark.

    Args:
        model: Any object with a `.generate(prompt: str) -> str` method,
               or any callable function `model(prompt: str) -> str`.
        num_questions: Number of questions from fixed benchmark (default 50).
        output_dir: Folder where report JSON/markdown will be saved.
        model_name: Name of model for reporting (inferred automatically if None).
        seed: Random seed for deterministic question sampling.

    Returns:
        dict: Complete structured evaluation report including:
              - accuracy (percentage)
              - total_questions, correct_count
              - per_question_results
              - execution metadata
    """
    # 1. Determine model callable & name
    if hasattr(model, "generate") and callable(getattr(model, "generate")):
        generate_fn = model.generate
        inferred_name = getattr(model, "get_name", lambda: getattr(model, "model_name", model.__class__.__name__))()
    elif callable(model):
        generate_fn = model
        inferred_name = getattr(model, "__name__", "Custom_Callable_Model")
    else:
        raise ValueError("Model must have a .generate(prompt) method or be a callable function!")

    final_model_name = model_name or str(inferred_name)

    # 2. Load fixed test set
    test_questions = _load_fixed_test_set(max_questions=num_questions, seed=seed)
    total_q = len(test_questions)

    print("=" * 60)
    print(f"  MedRAGent Benchmark: Evaluating '{final_model_name}'")
    print(f"  Fixed Test Set: {total_q} questions (seed={seed})")
    print("=" * 60)

    # 3. Run evaluation loop
    correct_count = 0
    results = []

    for idx, q in enumerate(test_questions, start=1):
        prompt = _format_question_prompt(q)
        ground_truth_letter = str(q.get("answer_idx", "")).strip().upper()
        ground_truth_text = q.get("answer", "")

        try:
            response = generate_fn(prompt)
            if not isinstance(response, str):
                response = str(response)
        except Exception as e:
            response = f"[ERROR: Generation failed: {e}]"

        pred_letter = _extract_answer_letter(response)
        is_correct = bool(pred_letter and pred_letter == ground_truth_letter)
        if is_correct:
            correct_count += 1

        results.append({
            "index": idx,
            "question": q.get("question"),
            "options": q.get("options"),
            "ground_truth_letter": ground_truth_letter,
            "ground_truth_text": ground_truth_text,
            "predicted_letter": pred_letter,
            "is_correct": is_correct,
            "model_response": response,
        })

        status = "CORRECT" if is_correct else "WRONG"
        print(f"[{idx:02d}/{total_q:02d}] GT: {ground_truth_letter} | Pred: {pred_letter or '?'} | {status}")

    # 4. Compute aggregate metrics
    accuracy = (correct_count / total_q) * 100.0 if total_q > 0 else 0.0

    report = {
        "benchmark_name": "MedQA-USMLE Fixed Test Benchmark",
        "model_name": final_model_name,
        "seed": seed,
        "total_questions": total_q,
        "correct_count": correct_count,
        "accuracy_pct": round(accuracy, 2),
        "summary": {
            "Accuracy": f"{accuracy:.2f}%",
            "Score": f"{correct_count}/{total_q}",
        },
        "per_question_results": results,
    }

    # 5. Save report files if output_dir specified
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        sanitized_name = re.sub(r"[^\w\-_\.]", "_", final_model_name)
        
        # JSON report
        json_file = out_path / f"report_{sanitized_name}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Markdown report
        md_file = out_path / f"report_{sanitized_name}.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(f"# MedRAGent Benchmark Report: {final_model_name}\n\n")
            f.write(f"- **Benchmark**: MedQA-USMLE (Fixed Test Split)\n")
            f.write(f"- **Questions Evaluated**: {total_q}\n")
            f.write(f"- **Correct Answers**: {correct_count}\n")
            f.write(f"- **Accuracy**: **{accuracy:.2f}%**\n\n")
            f.write("## Per-Question Breakdown\n\n")
            f.write("| # | Ground Truth | Predicted | Result |\n")
            f.write("|---|---|---|---|\n")
            for r in results:
                res_sym = "Passed" if r["is_correct"] else "Failed"
                f.write(f"| {r['index']} | {r['ground_truth_letter']} | {r['predicted_letter'] or 'None'} | {res_sym} |\n")

        print("=" * 60)
        print(f"BENCHMARK COMPLETE")
        print(f"  Final Accuracy: {accuracy:.2f}% ({correct_count}/{total_q})")
        print(f"  Report saved to: {json_file}")
        print(f"  Summary saved to: {md_file}")
        print("=" * 60)

    return report


if __name__ == "__main__":
    # Self-test using a dummy rule-based function
    print("Testing evaluate_model() with a dummy model...")
    def dummy_medical_model(prompt: str) -> str:
        return "Based on clinical review, the answer is C."

    test_report = evaluate_model(dummy_medical_model, num_questions=5, output_dir=None)
    print("\nSelf-test Output Report:")
    print(json.dumps(test_report["summary"], indent=2))
