param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Continue"

# This Anaconda-based environment links two OpenMP runtimes (PyTorch and the
# scientific stack).  The flag keeps the process alive instead of aborting.
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:HF_HUB_OFFLINE = "1"
$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTHONIOENCODING = "utf-8"

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (Test-Path $venvPython) {
    & $venvPython @RemainingArgs
} else {
    & python @RemainingArgs
}
exit $LASTEXITCODE
