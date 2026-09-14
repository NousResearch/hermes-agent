param(
  [switch]$Strict
)

$ErrorActionPreference = "Continue"
$script:StrictFailureCount = 0
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvRoot = Join-Path $Root ".venv"
$VenvPython = Join-Path $VenvRoot "Scripts\python.exe"

Write-Host "Hermes Workstation Doctor" -ForegroundColor Cyan
Write-Host "Root: $Root"
Write-Host ""

function Register-DoctorFailure {
  if ($Strict) {
    $script:StrictFailureCount++
  }
}

function Show-CommandVersion {
  param(
    [Parameter(Mandatory = $true)][string]$Name,
    [string[]]$CommandArgs = @()
  )
  try {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
      Write-Host ("[MISSING] {0}" -f $Name) -ForegroundColor Yellow
      Register-DoctorFailure
      return
    }

    # Capture the native exit code BEFORE piping/selecting output. Windows
    # PowerShell 5.1 can otherwise leave LASTEXITCODE at -1 for a successful
    # native command that participates in a PowerShell pipeline.
    $raw = & $Name @CommandArgs 2>&1
    $exitCode = $LASTEXITCODE
    $out = $raw | Select-Object -First 1

    if ($exitCode -eq 0) {
      Write-Host ("[OK] {0}: {1}" -f $Name, $out) -ForegroundColor Green
    } else {
      Write-Host ("[FAIL] {0}: exit {1}" -f $Name, $exitCode) -ForegroundColor Red
      Register-DoctorFailure
    }
  } catch {
    Write-Host ("[FAIL] {0}: {1}" -f $Name, $_.Exception.Message) -ForegroundColor Red
    Register-DoctorFailure
  }
}

function Test-PythonCandidate($Command, $Prefix) {
  try {
    $candidateArgs = @()
    $candidateArgs += $Prefix
    $candidateArgs += @("-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))")
    $out = & $Command @candidateArgs 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $out) { return $null }
    return [pscustomobject]@{
      Command = $Command
      Prefix = @($Prefix)
      Version = [version]($out | Select-Object -Last 1)
    }
  } catch {
    return $null
  }
}

function Resolve-Python {
  if ($env:HERMES_PYTHON) {
    $candidate = Test-PythonCandidate $env:HERMES_PYTHON @()
    if ($candidate) { return $candidate }
  }
  if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($selector in @("-3.13", "-3.12", "-3.11")) {
      $candidate = Test-PythonCandidate "py" @($selector)
      if ($candidate) { return $candidate }
    }
  }
  if (Get-Command python -ErrorAction SilentlyContinue) {
    return Test-PythonCandidate "python" @()
  }
  return $null
}

function Invoke-PythonCheck($Python, $Script, $Arguments) {
  if (-not $Python) {
    Write-Host ("[FAIL] {0}: Python unavailable" -f (Split-Path $Script -Leaf)) -ForegroundColor Red
    Register-DoctorFailure
    return
  }
  $pythonArgs = @()
  $pythonArgs += $Python.Prefix
  $pythonArgs += $Script
  $pythonArgs += $Arguments
  & $Python.Command @pythonArgs
  if ($LASTEXITCODE -ne 0) {
    Write-Host ("[FAIL] {0}: exit {1}" -f (Split-Path $Script -Leaf), $LASTEXITCODE) -ForegroundColor Red
    Register-DoctorFailure
  }
}

Show-CommandVersion -Name "git" -CommandArgs @("--version")
Show-CommandVersion -Name "node" -CommandArgs @("--version")
Show-CommandVersion -Name "npm" -CommandArgs @("--version")

$Python = Resolve-Python
if ($Python) {
  Write-Host ("[OK] Bootstrap Python: {0} {1} ({2})" -f $Python.Command, ($Python.Prefix -join ' '), $Python.Version) -ForegroundColor Green
  if ($Python.Version -ge [version]"3.14.0") {
    Write-Host "[WARN] Python 3.14 can run bootstrap scripts, but Hermes dependency installation requires <3.14." -ForegroundColor Yellow
  }
} else {
  Write-Host "[MISSING] Bootstrap Python. Install Python 3.13, 3.12, or 3.11." -ForegroundColor Yellow
  Register-DoctorFailure
}

if (Test-Path $VenvPython) {
  try {
    $venvVersion = (& $VenvPython -c "import sys; print(sys.version.split()[0])" 2>$null | Select-Object -Last 1)
    $importProbe = & $VenvPython -c "import hermes_cli; print('ok')" 2>$null
    if ($LASTEXITCODE -eq 0 -and $importProbe) {
      Write-Host ("[OK] Workstation .venv: {0} ({1})" -f $VenvPython, $venvVersion) -ForegroundColor Green
    } else {
      Write-Host "[FAIL] Workstation .venv exists but cannot import hermes_cli." -ForegroundColor Red
      Register-DoctorFailure
    }
  } catch {
    Write-Host ("[FAIL] Workstation .venv: {0}" -f $_.Exception.Message) -ForegroundColor Red
    Register-DoctorFailure
  }
} else {
  Write-Host "[WARN] Workstation .venv is not installed yet. Run workstation\install.cmd -InstallDependencies." -ForegroundColor Yellow
  Register-DoctorFailure
}

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
  Write-Host "[FAIL] Node.js is unavailable." -ForegroundColor Red
  Register-DoctorFailure
} else {
  try {
    $nodeRaw = (& node -p "process.versions.node" 2>$null).Trim()
    if (-not $nodeRaw -or $LASTEXITCODE -ne 0) {
      Write-Host "[FAIL] Could not determine Node.js version." -ForegroundColor Red
      Register-DoctorFailure
    } else {
      $nodeVersion = [version]$nodeRaw
      $minimumNode = [version]"22.22.0"
      if ($nodeVersion -lt $minimumNode) {
        Write-Host ("[WARN] Node {0} is below Hermes Desktop minimum {1}." -f $nodeVersion, $minimumNode) -ForegroundColor Yellow
        Register-DoctorFailure
      } else {
        Write-Host ("[OK] Node satisfies Desktop minimum: {0}" -f $nodeVersion) -ForegroundColor Green
        if ($nodeVersion.Major -ne 26) {
          Write-Host ("[WARN] Repository .nvmrc selects Node 26; current Node is {0}. CI validates on Node 26." -f $nodeVersion) -ForegroundColor Yellow
        }
      }
    }
  } catch {
    Write-Host "[WARN] Could not validate Node minimum version" -ForegroundColor Yellow
    Register-DoctorFailure
  }
}

Invoke-PythonCheck $Python (Join-Path $PSScriptRoot "scripts\apply_core_integration.py") @("--root", $Root, "--check")
Invoke-PythonCheck $Python (Join-Path $PSScriptRoot "scripts\validate_lock.py") @()
Invoke-PythonCheck $Python (Join-Path $PSScriptRoot "scripts\verify_licenses.py") @()

function Resolve-WorkstationHome {
  if ($env:HERMES_WORKSTATION_HOME -and $env:HERMES_WORKSTATION_HOME.Trim()) {
    return [System.IO.Path]::GetFullPath($env:HERMES_WORKSTATION_HOME.Trim())
  }
  if ($env:LOCALAPPDATA) {
    return Join-Path $env:LOCALAPPDATA "HermesWorkstation"
  }
  return Join-Path $HOME ".hermes-workstation"
}

$WorkstationHome = Resolve-WorkstationHome
$RuntimePath = Join-Path $WorkstationHome "Runtime"
$BrowserPath = Join-Path $WorkstationHome "Browser"
Write-Host ("Workstation home: {0}" -f $WorkstationHome)
Write-Host ("Browser profile: {0}" -f (Join-Path $BrowserPath "User Data"))
if ($env:HERMES_WORKSTATION_HOME -and (-not (Test-Path -LiteralPath $RuntimePath) -or -not (Test-Path -LiteralPath $BrowserPath))) {
  Write-Host "[FAIL] Explicit Workstation home is missing Runtime or Browser directories." -ForegroundColor Red
  Register-DoctorFailure
}

Write-Host ""
try {
  git -C $Root status --short
} catch {
  Write-Host "[WARN] Could not read git status." -ForegroundColor Yellow
}

if ($Strict -and $script:StrictFailureCount -gt 0) {
  Write-Host ("[FAIL] Strict doctor found {0} required failure(s)." -f $script:StrictFailureCount) -ForegroundColor Red
  exit 1
}

if ($Strict) {
  Write-Host "[OK] Strict doctor passed all required checks." -ForegroundColor Green
}
exit 0
