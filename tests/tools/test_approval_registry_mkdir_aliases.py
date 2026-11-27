"""PowerShell's mkdir/md functions can create registry keys via the Registry provider."""

from hermes_cli.approvals_suggest import is_unsafe_class
from tools.approval import detect_dangerous_command


def test_powershell_mkdir_md_registry_paths_require_approval():
    for command in (
        r"mkdir HKCU:\Software\HermesApprovalProbe",
        r"md HKCU:\Software\HermesApprovalProbe",
    ):
        dangerous, _key, description = detect_dangerous_command(command)
        assert dangerous, f"registry key creation must prompt: {command}"
        assert "registry" in description
        assert is_unsafe_class(description)

    # Filesystem mkdir is not a registry mutation, and neither is a `.md` file name before a registry path.
    assert not detect_dangerous_command(r"mkdir C:\tmp\HermesApprovalProbe")[0]
    assert not detect_dangerous_command(r"cat README.md HKCU:\Software\HermesApprovalProbe")[0]
    # A `.md` argument must not end the scan for an earlier verb either.
    assert detect_dangerous_command(r"New-Item -Name README.md -Path HKCU:\Software")[0]
