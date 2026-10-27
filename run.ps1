param([Parameter(ValueFromRemainingArguments=$true)][string[]]$CliArgs)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path .\.venv\Scripts\python.exe)) { throw 'Run .\setup.ps1 first.' }
& .\.venv\Scripts\python.exe -m exec_feedback @CliArgs
exit $LASTEXITCODE
