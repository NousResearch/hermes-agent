"""Windows hardline rules inside PowerShell -Command scripts and around attached redirects (#87419)."""

import pytest

from tools.approval import detect_dangerous_command, detect_hardline_command


@pytest.mark.parametrize("cmd", [
    # A later command of an inline -Command / -c script is a command start too.
    'pwsh -c "Get-Date; Stop-Computer"',
    'powershell -NoProfile -Command "Get-Date; Format-Volume -DriveLetter D"',
    "pwsh -c Get-Date; Clear-Disk -Number 0 -RemoveData",
    'pwsh -ExecutionPolicy Bypass -c "ls; Remove-Item -Recurse C:\\"',
    # A redirect glued to the drive root still ends the root operand.
    "rd /s /q C:\\>nul",
    "rd /s /q C:\\<nul",
    "rm -rf C:/>/dev/null",
])
def test_later_script_commands_and_redirected_roots_are_hardline(cmd):
    assert detect_hardline_command(cmd)[0], cmd


@pytest.mark.parametrize("cmd", [
    # -File runs a script path; its name is not PowerShell code.
    "pwsh -File Stop-Computer.ps1",
    "pwsh -NoProfile -File ./Format-Volume.ps1",
    "powershell -f Clear-Disk.ps1",
    "Stop-Computer.ps1",
    # A cmdlet-looking prefix of a longer name is another command.
    "Clear-Disk-Report -Path out.txt",
    # Script text after -Command that only mentions a cmdlet as data.
    'pwsh -c "Write-Output Stop-Computer"',
])
def test_script_paths_and_longer_names_are_not_hardline(cmd):
    assert not detect_hardline_command(cmd)[0], cmd


def test_script_files_still_need_approval():
    # Not hardline, but running a PowerShell script is still gated as script execution.
    assert detect_dangerous_command("pwsh -File Stop-Computer.ps1")[0]


B = "\\"


@pytest.mark.parametrize("cmd", [
    # A quoted drive root glued to the rest of its shell word names a subdirectory.
    'rm -rf "C:/"Users/me/build',
    "rm -rf 'C:/'Users/me/build",
    'rm -rf "C:/"../work',
    'rd /s /q "C:' + B + '"Users' + B + 'me',
    "Remove-Item -Recurse 'C:" + B + "'Users" + B + "me",
    'Remove-Item -Recurse "C:' + B + '"Users',
])
def test_quote_concatenated_subdirectory_delete_is_not_hardline(cmd):
    assert not detect_hardline_command(cmd)[0], cmd
    assert detect_dangerous_command(cmd)[0], cmd


@pytest.mark.parametrize("cmd", [
    # The quoted root itself, with root-collapse suffixes, globs or an attached redirect.
    'rd /s /q "C:' + B + '"',
    'rd /s /q "C:' + B + '">nul',
    'rd /s /q "C:' + B + '"*',
    'del /s /q "C:' + B + '"*.*',
    "ri -r 'C:" + B + "'",
    'Remove-Item -Recurse "C:' + B + '"; ls',
    'rm -rf "C:/"',
    'rm -rf "C:/"./',
    'rm -rf "C:/"./*',
    "rm -rf 'C:/'..",
    # Marked on the raw command: an escaped quote does not hide the later root delete.
    'cat "f' + B + '"n"; rd /s /q C:' + B + ' ',
    'ls; rd /s /q C:' + B + ' ',
])
def test_quoted_drive_roots_stay_hardline(cmd):
    assert detect_hardline_command(cmd)[0], cmd


def test_escaped_quote_in_prose_does_not_end_the_string():
    # One echo argument; the backslash-flattened text must not close the quote at `\"`.
    assert not detect_hardline_command('echo "x' + B + '"; rd /s /q C:' + B + ' later"')[0]


@pytest.mark.parametrize("cmd", [
    "Restart-Computer -WhatIf",
    "stop-computer -whatif",
    "Restart-Computer -ComputerName srv1 -WhatIf -Force",
    'pwsh -c "Restart-Computer -WhatIf"',
    'pwsh -c "Get-Date; Stop-Computer -WhatIf"',
    "Restart-Computer -ComputerName srv1, srv2 -WhatIf",
])
def test_power_off_whatif_preview_is_not_hardline(cmd):
    assert not detect_hardline_command(cmd)[0], cmd


@pytest.mark.parametrize("cmd", [
    "Restart-Computer -WhatIf:$false",
    "Restart-Computer -WhatIf -WhatIf:$false",
    'Restart-Computer -ComputerName " -WhatIf"',
    "Restart-Computer # -WhatIf",
    "Restart-Computer $args -WhatIf",
    "Restart-Computer @p -WhatIf",
    "Restart-Computer -WhatIfx",
    "Restart-Computer -WhatIf; Restart-Computer",
    "Restart-Computer -WhatIf && Stop-Computer",
    "Restart-Computer -WhatIf" + chr(10) + "Stop-Computer",
    'pwsh -c "Restart-Computer -WhatIf; Stop-Computer"',
    # After a comma `-WhatIf` is an array element (ComputerName @('localhost', '-WhatIf')), not the switch.
    "Restart-Computer localhost, -WhatIf",
    "Stop-Computer -ComputerName localhost,  -WhatIf -Force",
])
def test_whatif_data_or_a_later_real_power_off_stays_hardline(cmd):
    assert detect_hardline_command(cmd)[0], cmd
