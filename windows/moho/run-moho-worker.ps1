$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
$Runner = Join-Path $RepoRoot 'scripts/signal_room_moho_job_runner.py'
$QueueRoot = Join-Path $RepoRoot 'windows/moho/jobs'

Set-Location $RepoRoot

while ($true) {
  & $Python $Runner worker --queue-root $QueueRoot --max-jobs 1
  Start-Sleep -Seconds 10
}
