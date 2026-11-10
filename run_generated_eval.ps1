# Generated DSM-5 question evaluation for MedX.
#
#   .\run_generated_eval.ps1 -Preview     generate 5 sample questions and print them (nothing saved)
#   .\run_generated_eval.ps1 -Limit 5     full flow, but evaluate only the first 5 questions
#   .\run_generated_eval.ps1              full run: 100 questions, rounds 1 vs 3
#
# Steps:
#   1. generate   generate_rag_eval_dd.py: Nemotron writes questions from DSM-5 chunks (some need 2-3
#                 chunks), GLM grades each one, low-scoring ones are dropped -> eval_questions.jsonl.
#                 Skipped when eval_questions.jsonl already exists, so a rerun never regenerates by
#                 accident; pass -Regenerate to replace it.
#   2. evaluate   medx_eval.py generated: answers every question with 1 and with 3 retrieval rounds and
#                 measures whether the source chunks were retrieved; GLM grades each answer.
#   3. report     eval_results\summary.md (MedQA tables plus the new generated-question tables)
#
# LLM replies in step 2 are cached in .llm_cache\, so rerunning after an interruption resumes.

param(
    [int]$N = 100,
    [int]$Limit = 0,
    [int]$Workers = 6,
    [string]$Rounds = "1,3",
    [switch]$Preview,
    [switch]$Regenerate
)

$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $workspace ".venv\Scripts\python.exe"
$questions = Join-Path $workspace "eval_questions.jsonl"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment not found at $python"
}

Push-Location $workspace
try {
    Write-Host "== Checking LLM configuration"
    & $python medx_llm.py ping
    if ($LASTEXITCODE -ne 0) { throw "LLM check failed" }

    if ($Preview) {
        Write-Host "`n== Preview: generating 5 sample questions (not saved)"
        & $python generate_rag_eval_dd.py --chunks Psichiatry_DSM-5.jsonl --preview
        if ($LASTEXITCODE -ne 0) { throw "preview failed" }
        return
    }

    if ($Regenerate -or -not (Test-Path -LiteralPath $questions)) {
        Write-Host "`n== Step 1/3: generating $N questions from the DSM-5 corpus"
        & $python generate_rag_eval_dd.py --chunks Psichiatry_DSM-5.jsonl --n $N --out eval_questions.jsonl
        if ($LASTEXITCODE -ne 0) { throw "question generation failed" }
    } else {
        $count = (Get-Content -LiteralPath $questions | Measure-Object -Line).Lines
        Write-Host "`n== Step 1/3: using existing eval_questions.jsonl ($count questions); -Regenerate to replace"
    }

    Write-Host "`n== Step 2/3: answering with rounds $Rounds and grading"
    & $python medx_eval.py generated --questions eval_questions.jsonl --rounds $Rounds --limit $Limit --workers $Workers
    if ($LASTEXITCODE -ne 0) { throw "evaluation failed" }

    Write-Host "`n== Step 3/3: report"
    & $python medx_eval.py report
    if ($LASTEXITCODE -ne 0) { throw "report failed" }

    Write-Host "`nDone. See eval_results\summary.md and eval_results\generated_results.jsonl (per question)."
} finally {
    Pop-Location
}
