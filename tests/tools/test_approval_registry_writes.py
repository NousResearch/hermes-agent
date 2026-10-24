"""The approval gate catches registry WRITES, not just registry deletes.

Regression for #132202. The detection table gated the delete side (``reg delete``,
``Remove-ItemProperty -Force``) and nothing on the write side, so on a stock install
``reg add``, ``setx`` and ``Set-ItemProperty`` mutated persistent machine state with no prompt
at all -- the guardian never saw them, and there was no class to escalate on.
"""

import pytest

from hermes_cli.approvals_suggest import is_unsafe_class
from tools.approval import detect_dangerous_command


class TestRegistryWriteDetected:
    """Every registry write form asks for approval."""

    @pytest.mark.parametrize("command", [
        r"reg add HKCU\Environment /v HERMES_TEST /t REG_SZ /d x /f",
        r"reg.exe add HKCU\Environment /v HERMES_TEST /t REG_SZ /d x /f",
        r"reg import machine.reg",
        r"reg copy HKCU\Software\A HKCU\Software\B /s",
        r"reg rename HKCU\Software\A B",
        "setx HERMES_TEST x",
        "setx.exe HERMES_TEST x",
        "Set-ItemProperty -Path HKCU:\\Environment -Name HERMES_TEST -Value x",
        "New-ItemProperty -Path HKCU:\\Environment -Name HERMES_TEST -Value x -PropertyType String",
    ])
    def test_registry_writes_are_dangerous(self, command):
        dangerous, _key, description = detect_dangerous_command(command)
        assert dangerous is True, command
        assert "registry write" in description.lower(), command

    @pytest.mark.parametrize("command", [
        r"reg delete HKCU\Environment /v HERMES_TEST /f",
        "Remove-ItemProperty -Path HKCU:\\Environment -Name HERMES_TEST -Force",
    ])
    def test_registry_deletes_stay_dangerous(self, command):
        """The sibling delete class is unchanged by the new write class."""
        dangerous, _key, description = detect_dangerous_command(command)
        assert dangerous is True, command
        assert "registry" in description.lower(), command


class TestRegistryReadStaysUnprompted:
    """Reading the registry is not a write."""

    @pytest.mark.parametrize("command", [
        r"reg query HKCU\Environment /v HERMES_TEST",
        r"reg export HKCU\Environment out.reg",
        "Get-ItemProperty -Path HKCU:\\Environment -Name HERMES_TEST",
        "Get-ChildItem HKCU:\\Environment",
        "Get-Item Env:PATH",
    ])
    def test_registry_reads_are_not_dangerous(self, command):
        assert detect_dangerous_command(command) == (False, None, None)

    def test_a_setx_shell_assignment_is_not_a_write(self):
        """`setx=1` is a shell assignment, not the Windows `setx` tool."""
        assert detect_dangerous_command("setx=1") == (False, None, None)

    def test_grep_add_is_not_a_registry_write(self):
        assert detect_dangerous_command("grep add file.txt") == (False, None, None)


def test_the_write_classes_are_never_proposed_for_an_allowlist():
    """`hermes approvals suggest` must not mine these into a permanent allowlist entry."""
    for description in ("registry write (reg)", "registry write (setx)",
                        "registry write (Set-ItemProperty)", "registry write (New-ItemProperty)"):
        assert is_unsafe_class(description), description
