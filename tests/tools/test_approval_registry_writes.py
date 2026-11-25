"""Registry WRITE commands require approval, like the registry deletes (#132202).

Only `reg delete` / `Remove-ItemProperty -Force` were gated; `reg add`, `setx` and
`Set-ItemProperty` mutated persistent machine state with no prompt. Read verbs must stay quiet,
and `hermes approvals suggest` must never propose a registry-write class for the allowlist.
"""

import os
import subprocess
import sys

import pytest

from hermes_cli.approvals_suggest import is_unsafe_class
from tools.approval import detect_dangerous_command


@pytest.mark.parametrize("cmd", [
    r"reg add HKCU\Environment /v HERMES_TEST /t REG_SZ /d x /f",
    r"reg.exe add HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Run /v x /d c:\x.exe",
    r"REG ADD HKCU\Software\Thing /f",
    r"reg import C:\tmp\settings.reg",
    r"reg copy HKCU\Software\A HKCU\Software\B /s",
    r"reg restore HKLM\SOFTWARE\Thing C:\tmp\hive.hiv",
    "setx HERMES_TEST x",
    "setx.exe PATH \"%PATH%;C:\\tools\" /M",
    r"Set-ItemProperty -Path HKCU:\Environment -Name X -Value y",
    r"New-ItemProperty -Path HKCU:\Software\Thing -Name X -Value 1",
    r"powershell -NoProfile -Command Set-ItemProperty -Path HKLM:\X -Name Y -Value 0",
    r"cmd /c reg add HKCU\Software\Thing /f",
    r"reg unload HKLM\TempHive",
    r"regedit /s C:\tmp\settings.reg",
    r"sp HKCU:\Environment X y",
    r"ni -Path 'HKCU:\Software\Thing'",
    r"New-Item -Path Registry::HKEY_CURRENT_USER\Software\Thing",
    # The provider path may follow any other parameter, and `si` is Set-Item's built-in alias.
    r"New-Item -Force -Path 'HKCU:\Software\Thing'",
    r"New-Item -Name Thing -Path HKCU:\Software",
    r"New-Item -ItemType Key -Path HKCU:\Software\Thing",
    r"New-Item -LiteralPath HKCU:\Software\Thing",
    r"New-Item -Path:HKCU:\Software\Thing",
    r"New-Item -Pa HKCU:\Software\Thing",
    r"Set-Item -LiteralPath HKCU:\Software\Thing -Value x",
    r"si HKCU:\Software\Thing -Value x",
])
def test_registry_write_requires_approval(cmd):
    is_dangerous, _key, description = detect_dangerous_command(cmd)
    assert is_dangerous, f"should be flagged: {cmd}"
    assert "registry" in description
    assert is_unsafe_class(description), "registry writes must never be proposed for the allowlist"


@pytest.mark.parametrize("cmd", [
    r"reg query HKCU\Environment",
    r"reg export HKCU\Software\Thing C:\tmp\out.reg",
    r"Get-ItemProperty -Path HKCU:\Environment",
    "setx=1",
    "sp notes.txt",
    "regedit",
    r"New-Item -Path C:\tmp\notes.txt",
    r"ni notes.txt; Get-ItemProperty HKCU:\Environment",
    r"ni notes.txt | Out-Null & Get-ItemProperty HKCU:\Environment",
    "echo $setx",
])
def test_registry_reads_do_not_prompt(cmd):
    assert not detect_dangerous_command(cmd)[0], f"should NOT be flagged: {cmd}"


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_registry_path_scan_is_linear_on_repeated_verbs():
    """Each verb token used to restart a scan to the segment end: 24k chars of `si ` took ~3 s."""
    code = '''
from tools.approval_detection import DANGEROUS_PATTERNS_COMPILED
rules = [rx for rx, desc in DANGEROUS_PATTERNS_COMPILED
         if desc == "registry write (sp/ni/Set-Item/New-Item on a registry path)"]
assert len(rules) == 1
for text in ("si " * 100_000, "ni\\t" * 100_000, "sp " + "a " * 150_000):
    assert rules[0].search(text) is None
'''
    # Pin the checkout under test: a bare `python -c` resolves `tools` through the venv's editable
    # install (the primary clone), not through the worktree pytest is running in.
    env = {**os.environ, "PYTHONPATH": REPO_ROOT}
    subprocess.run([sys.executable, "-c", code], check=True, timeout=5, cwd=REPO_ROOT, env=env)
