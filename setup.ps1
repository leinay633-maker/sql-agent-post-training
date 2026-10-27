param([switch]$ML, [string]$Python = '')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = '1'
$candidates = @()
if ($Python) { $candidates += $Python }
else { $candidates += @('python', 'python3') }
$exe = $null
foreach ($candidate in $candidates) {
    $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($cmd) {
        & $cmd.Source -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 2)' 2>$null
        if ($LASTEXITCODE -eq 0) { $exe = $cmd.Source; break }
    }
}
if (-not $exe) { throw 'Python 3.11+ required. Use setup.ps1 -Python C:\path\python.exe. The py launcher is not required.' }
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    & $exe -m venv .venv
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
$venv = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$extras = '.[dev,server]'
if ($ML) { $extras = '.[dev,server,ml]' }
& $venv -m pip install -e $extras
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $venv -m exec_feedback check
exit $LASTEXITCODE
