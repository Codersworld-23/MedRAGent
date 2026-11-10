# MedQA evaluation for MedX: psychiatry questions + 30 non-psychiatry controls.
#
#   .\run_medqa_eval.ps1              full run
#   .\run_medqa_eval.ps1 -Limit 5     smoke test: 5 psychiatry + 5 out-of-scope questions
#
# Steps:
#   1. filter-medqa  keyword pass, then the judge LLM (GLM) labels each candidate as psychiatry or not;
#                    adds 30 seeded non-psychiatry questions as out-of-scope controls
#   2. medqa         answers every question three ways: closed_book, rag_r1, rag_r3
#   3. report        writes eval_results\summary.md and summary.json
#
# Every LLM reply is cached in .llm_cache\, so rerunning after an interruption only makes the
# calls that are still missing.

param(
    [int]$Limit = 0,
    [int]$OutOfScope = 30,
    [int]$Workers = 6,
    [string]$Configs = "closed_book,rag_r1,rag_r3"
)

$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $workspace ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment not found at $python"
}

Push-Location $workspace
try {
    Write-Host "== Checking LLM configuration"
    & $python medx_llm.py ping
    if ($LASTEXITCODE -ne 0) { throw "LLM check failed" }

    Write-Host "`n== Step 1/3: selecting psychiatry questions and out-of-scope controls"
    & $python medx_eval.py filter-medqa --input medqa_scope_dev.jsonl --out-of-scope $OutOfScope --workers $Workers
    if ($LASTEXITCODE -ne 0) { throw "filter-medqa failed" }

    Write-Host "`n== Step 2/3: answering ($Configs)"
    & $python medx_eval.py medqa --configs $Configs --limit $Limit --workers $Workers
    if ($LASTEXITCODE -ne 0) { throw "medqa run failed" }

    Write-Host "`n== Step 3/3: report"
    & $python medx_eval.py report
    if ($LASTEXITCODE -ne 0) { throw "report failed" }

    Write-Host "`nDone. See eval_results\summary.md (tables) and eval_results\medqa_results.jsonl (per question)."
} finally {
    Pop-Location
}
