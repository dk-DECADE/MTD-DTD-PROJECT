$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Create a Python 3.10 .venv and install requirements.txt first. See README.md.'
}
& $projectPython -m streamlit run app.py
exit $LASTEXITCODE
