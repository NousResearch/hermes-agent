"""Windows hardline rules inside PowerShell -Command scripts and around attached redirects (#87419)."""

import time

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


@pytest.mark.parametrize("cmd", [
    "Remove-Item C:" + B + "* -Recurse -Include *.tmp",
    "Remove-Item -Recurse C:" + B + " -Include:*.log -Force",
    "rm -r C:" + B + "* -Include *.tmp",
    'pwsh -c "Remove-Item -Recurse C:' + B + ' -Include *.tmp -Force"',
])
def test_include_filtered_recursive_delete_stays_approvable(cmd):
    # -Include deletes only the matching items, like `del /s /q C:\*.tmp`.
    assert not detect_hardline_command(cmd)[0], cmd
    assert detect_dangerous_command(cmd)[0], cmd


@pytest.mark.parametrize("cmd", [
    "Remove-Item C:" + B + "* -Recurse -Include *",
    "Remove-Item C:" + B + "* -Recurse -Include *.*",
    "Remove-Item C:" + B + "* -Recurse -Include *.tmp,*",
    "rm -r C:" + B + "* -Include *",
    # PowerShell expands these to `*`: only a literal pattern narrows the delete.
    'Remove-Item C:' + B + '* -Recurse -Include "*$null"',
    "Remove-Item C:" + B + "* -Recurse -Include *$x",
    "Remove-Item C:" + B + "* -Recurse -Include (Get-Pattern)",
    "Remove-Item -Recurse C:" + B + "; Get-ChildItem -Include *.tmp",
    # Not a PowerShell -Include of this delete: cmd `/s` and GNU `--` take it as a name, `#` comments it
    # out, a quote makes it another argument's data, and a spaced list still ends in `*`.
    "del /s /q C:" + B + "* -Include *.tmp",
    "cmd /c rd /s /q C:" + B + " -Include x",
    "rm -rf -- C:/ -Include x",
    "rm -r C:/* -- -Include x",
    "Remove-Item -Recurse C:" + B + " # -Include *.tmp",
    "Remove-Item -Recurse C:" + B + " -Exclude 'a -Include *.tmp'",
    "Remove-Item -Recurse C:" + B + ' -Exclude "a -Include *.tmp"',
    "Remove-Item -Recurse C:" + B + " -Include *.tmp , *",
    # The -Include belongs to the outer command, not to the delete inside the substitution/subshell.
    "echo $(rm -rf C:/) -Include x",
    "ls $(rm -rf C:/*) -Include *.tmp",
    "echo $(Remove-Item -Recurse C:/*) -Include x",
])
def test_include_that_matches_everything_stays_hardline(cmd):
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


# Clear-Disk's documented -WhatIf switch previews and does not wipe the disk.
# Keep this approvable, unlike Format-Volume whose -WhatIf switch is ineffective.
@pytest.mark.parametrize("cmd", [
    "Clear-Disk -Number 0 -WhatIf",
    "Clear-Disk -Number 0 -RemoveData -WhatIf",
    'pwsh -c "Get-Disk; Clear-Disk -Number 0 -WhatIf"',
])
def test_clear_disk_whatif_preview_is_not_hardline(cmd):
    assert not detect_hardline_command(cmd)[0], cmd
    assert detect_dangerous_command(cmd)[0], cmd


CR, LF = chr(13), chr(10)


# PowerShell reads a bare CR as a statement break, so the second cmdlet is a real command (and a quoted `pwsh -c`
# newline, masked to a space for detection, must not leave the first `-WhatIf` covering a later cmdlet).
@pytest.mark.parametrize("cmd", [
    "Clear-Disk -Number 0 -WhatIf" + CR + "Clear-Disk -Number 1 -RemoveData -Confirm:$false",
    'pwsh -c "Clear-Disk -Number 0 -WhatIf' + CR + 'Clear-Disk 1"',
    "Get-Date" + CR + "Clear-Disk 1",
    "Restart-Computer -WhatIf" + CR + "Restart-Computer",
    "Stop-Computer -WhatIf" + CR + " Stop-Computer",
    'pwsh -c "Clear-Disk -Number 0 -WhatIf' + LF + 'Clear-Disk 1"',
    'pwsh -c "Restart-Computer -WhatIf' + LF + 'Restart-Computer"',
])
def test_bare_cr_starts_a_new_powershell_statement(cmd):
    assert detect_hardline_command(cmd)[0], cmd


# A CR inside quotes is data, like a quoted newline.
@pytest.mark.parametrize("cmd", [
    "git commit -m 'a" + CR + "Clear-Disk 1'",
    "echo 'a" + CR + "Restart-Computer'",
])
def test_quoted_cr_is_not_a_statement_break(cmd):
    assert not detect_hardline_command(cmd)[0], cmd


# A -WhatIf only previews its own statement: in a quoted `pwsh -c` payload the CR/LF is masked to a space for
# detection, so a -WhatIf on a later destructive statement must not exempt an earlier unpreviewed Clear-Disk.
@pytest.mark.parametrize("cmd", [
    'pwsh -c "Clear-Disk -Number 0' + CR + 'Remove-Item -Recurse -Force C:\\ -WhatIf"',
    "powershell -Command 'Clear-Disk -WhatIf" + LF + "Format-Volume -DriveLetter C C:\\x,C:\\*'",
    'pwsh -c "Clear-Disk C:\\x,C:\\*' + CR + 'rd /s /q C:\\ -WhatIf"',
    'pwsh -c "Clear-Disk -Include *.tmp' + CR + 'Format-Volume -DriveLetter C -WhatIf"',
    "powershell -Command 'Clear-Disk,D:\\x" + LF + "Format-Volume -DriveLetter C -WhatIf -Confirm:$false'",
    'pwsh -c "Clear-Disk -Number 0' + CR + 'Get-Date -WhatIf"',
    # PowerShell also breaks statements at NEL, LINE SEPARATOR and PARAGRAPH SEPARATOR.
    "Clear-Disk,D:\\x\u2028Get-Date -WhatIf",
    "powershell -Command 'Clear-Disk -Include *.tmp\u2028Get-Date -WhatIf -Confirm:$false'",
    'pwsh -c "Clear-Disk -Number 0\u2029Get-Date -WhatIf"',
    'pwsh -c "Clear-Disk -Number 0\x85Get-Date -WhatIf"',
])
def test_whatif_of_another_statement_does_not_exempt_clear_disk(cmd):
    assert detect_hardline_command(cmd)[0], cmd


@pytest.mark.parametrize("cmd", [
    'pwsh -c "Get-Date' + CR + 'Clear-Disk -Number 0 -WhatIf"',
    "Get-Disk | Clear-Disk -WhatIf",
])
def test_every_statement_previewing_stays_approvable(cmd):
    assert not detect_hardline_command(cmd)[0], cmd
    assert detect_dangerous_command(cmd)[0], cmd


# A drive root later in a PowerShell/cmd argument list is still deleted.
@pytest.mark.parametrize("cmd", [
    "Remove-Item -Recurse C:\\x,C:\\*",
    "Remove-Item -Recurse -Path 'D:\\x','C:\\'",
    'Remove-Item -Recurse -Path "D:/x","C:/"',
    "Remove-Item -Recurse -Path D:/x, C:/*",
    "rd /s /q D:\\x,C:\\",
    "del /s /q D:\\x,C:\\*",
])
def test_drive_root_as_a_later_list_element_is_hardline(cmd):
    assert detect_hardline_command(cmd)[0], cmd


@pytest.mark.parametrize("cmd", [
    "Remove-Item -Recurse C:/build,C:/Users",
    "rd /s /q D:\\x,C:\\build",
    "Remove-Item -Recurse -Include *.tmp D:/x,C:/",
    "rm -rf x,C:/",
])
def test_later_list_element_below_the_root_stays_approvable(cmd):
    assert not detect_hardline_command(cmd)[0], cmd
    assert detect_dangerous_command(cmd)[0], cmd


@pytest.mark.parametrize("command", [
    "Get-Date" + "\r" * 8000 + "; x",
    "Get-Date" + " \r" * 4000 + "; x",
    "Get-Date" + "\rri -r" * 2000 + "; x",
    "Get-Date" + "\rpwsh -a" * 1500 + "; x",
], ids=["cr", "space-cr", "cr-delete-verb", "cr-launcher"])
def test_bare_cr_runs_scan_in_linear_time(command):
    # A bare CR starts a PowerShell statement; a run of them must not make each CR rescan the rest of the run.
    start = time.perf_counter()
    assert not detect_hardline_command(command)[0]
    assert time.perf_counter() - start < 2


@pytest.mark.parametrize("cmd", [
    "Get-Date\r  Clear-Disk -Number 1",
    "Get-Date\r \r\tStop-Computer",
    "x\r pwsh -NoProfile -c \"Format-Volume -DriveLetter D\"",
])
def test_command_after_a_cr_run_with_blanks_is_hardline(cmd):
    assert detect_hardline_command(cmd)[0], cmd
