param(
    [string]$Query = "criteria for major depressive disorder",
    [int]$Rounds = 3,
    [int]$TopK = 5
)

$ErrorActionPreference = "Stop"

$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$candidates = @(
    "$workspace\.venv\Scripts\python.exe",
    "$workspace\.venv-1\Scripts\python.exe",
    "C:\Users\ishit\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
    "python",
    "py"
)

$python = $null
foreach ($candidate in $candidates) {
    try {
        if ($candidate -like "*\*") {
            if (-not (Test-Path -LiteralPath $candidate)) {
                continue
            }
            & $candidate -c "import sys; print(sys.version)" *> $null
        } else {
            & $candidate -c "import sys; print(sys.version)" *> $null
        }
        if ($LASTEXITCODE -ne 0) {
            continue
        }
        $python = $candidate
        break
    } catch {
        continue
    }
}

if (-not $python) {
    throw "No working Python interpreter found. Install Python 3.8+ or fix the virtual environment."
}

Push-Location $workspace
try {
    & $python "$workspace\medx_retrieval.py" $Query --rounds $Rounds --top-k $TopK
} finally {
    Pop-Location
}
