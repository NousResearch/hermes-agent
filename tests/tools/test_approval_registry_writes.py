"""Registry WRITE commands require approval, like the registry deletes (#132202).

Only `reg delete` / `Remove-ItemProperty -Force` were gated; `reg add`, `setx` and
`Set-ItemProperty` mutated persistent machine state with no prompt. Read verbs must stay quiet,
and `hermes approvals suggest` must never propose a registry-write class for the allowlist.
"""

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
    "echo $setx",
])
def test_registry_reads_do_not_prompt(cmd):
    assert not detect_dangerous_command(cmd)[0], f"should NOT be flagged: {cmd}"
