$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
$Runner = Join-Path $RepoRoot 'scripts/signal_room_cavalry_job_runner.py'
$QueueRoot = Join-Path $RepoRoot 'windows/cavalry/jobs'

Set-Location $RepoRoot

while ($true) {
  & $Python $Runner worker --queue-root $QueueRoot --max-jobs 1 --recover-stale-after-seconds 7200
  Start-Sleep -Seconds 10
}
