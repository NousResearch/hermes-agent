"""Regression coverage for #140704: heredoc data is not shell source."""

import pytest

from tools import approval, approval_context, approval_floors
from tools.approval_detection import detect_dangerous_command, detect_hardline_command

# Dangerous-tier (prompts on main) but not hardline: a body the dangerous layer may drop as data.
PUSH = "git push --force"


def _body_adds_no_prompt(command: str) -> bool:
    """The dangerous verdict is the one the same command gets with a harmless body: the runner
    (`bash -c ...`, `pwsh -c ...`) may prompt on its own, the written doc must not add to it."""
    return detect_dangerous_command(command) == detect_dangerous_command(command.replace(PUSH, "echo hi"))


@pytest.fixture
def floors_off(monkeypatch):
    """Exercise the unconditional floors through the production gate."""
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "off")
    monkeypatch.setattr(
        approval_context,
        "_get_approval_config",
        lambda: {"mode": "off", "deny": ["rm", "rm *", "git push*"]},
    )


@pytest.mark.parametrize("body", ["it's data", 'say "hello'])
@pytest.mark.parametrize("followup", ["rm notes.txt", "git push --force"])
def test_heredoc_body_quotes_do_not_hide_following_denied_command(floors_off, body, followup):
    command = f"cat > notes.txt <<'EOF'\n{body}\nEOF\n{followup}"

    result = approval.check_all_command_guards(command, "local")

    assert result["approved"] is False
    assert approval_floors._match_user_deny_rule(command) is not None


@pytest.mark.parametrize("body", [PUSH, "Never run `rm -rf`."])
def test_heredoc_doc_body_does_not_prompt(body):
    command = f"cat > notes.txt <<'EOF'\n{body}\nEOF\nwc -l notes.txt"

    assert detect_dangerous_command(command)[0] is False


@pytest.mark.parametrize("body", ["rm is documentation", "Never run `rm -rf`."])
@pytest.mark.parametrize("followup", [
    "wc -l notes.txt", "mv notes.txt run.sh && bash run.sh", 'f=notes.txt; bash "$f"', "make", "crontab notes.txt",
])
def test_heredoc_body_stays_on_the_deny_floor_whatever_runs_next(floors_off, body, followup):
    # What can run a file written earlier in the string has no end, so the floors read every body,
    # as main does; only the dangerous-prompt layer drops a `cat` body as data.
    command = f"cat > notes.txt <<'EOF'\n{body}\nEOF\n{followup}"

    assert approval_floors._match_user_deny_rule(command) is not None, followup
    assert approval.check_all_command_guards(command, "local")["approved"] is False, followup


def test_heredoc_written_then_run_keeps_its_body_on_the_deny_list(floors_off):
    command = "cat > job.sh <<'EOF'\nrm notes.txt\nEOF\nbash job.sh"

    assert approval_floors._match_user_deny_rule(command) is not None


def test_absolute_shell_path_keeps_written_heredoc_on_floors(floors_off):
    # /bin/bash executes the file just written; its heredoc is code, not inert documentation.
    rm_root = "r" + "m -r" + "f /"
    command = f"cat > job.sh <<'EOF'\n{rm_root}\nEOF\n/bin/bash job.sh"

    assert detect_hardline_command(command)[0] is True
    assert approval_floors._match_user_deny_rule(command) is not None
    assert approval.check_all_command_guards(command, "local")["approved"] is False


@pytest.mark.parametrize("runner", [
    "/opt/homebrew/bin/bash", "./bin/bash", "/usr/bin/env bash", "/usr/bin/sudo -E sh", "/usr/bin/python3",
])
def test_any_program_path_keeps_written_heredoc_on_floors(floors_off, runner):
    # A directory before the shell / launcher / interpreter still names that program.
    rm_root = "r" + "m -r" + "f /"
    command = f"cat > job.sh <<'EOF'\n{rm_root}\nEOF\n{runner} job.sh"

    assert detect_hardline_command(command)[0] is True, runner
    assert approval.check_all_command_guards(command, "local")["approved"] is False, runner


def test_path_qualified_reader_of_a_written_doc_does_not_prompt():
    command = f"cat > notes.md <<'EOF'\n{PUSH}\nEOF\n/bin/cat notes.md"

    assert _body_adds_no_prompt(command)


@pytest.mark.parametrize("reader", [
    'echo "$(cat notes.md)"', "cat notes.md | bash -n", "bash -n <(cat notes.md)",
    "cat notes.md | python3 -m json.tool", "printf '%s' 'cat notes.md | bash'", "cat notes.md | bashate",
    "cat notes.md | bash other.sh", "cat notes.md | python3 -W ignore lint.py",
    'bash other.sh "$(cat notes.md)"',
])
def test_reading_a_written_doc_without_running_it_does_not_prompt(reader):
    command = f"cat > notes.md <<'EOF'\n{PUSH}\nEOF\n{reader}"

    assert _body_adds_no_prompt(command), reader


@pytest.mark.parametrize("runner", [
    'eval "$(cat job.sh)"', "cat job.sh | bash -s -- --prefix /opt", ". <(cat job.sh)", "cat job.sh | sudo bash",
    "cat job.sh | bash > run.log 2>&1", "cat job.sh | python3 - install", "cat job.sh | bash -euo pipefail",
    "cat job.sh | bash -o pipefail -x", "cat job.sh | bash +o posix", "cat job.sh | python3 -W ignore",
])
def test_running_a_written_file_s_content_keeps_it_on_the_deny_list(floors_off, runner):
    command = f"cat > job.sh <<'EOF'\nrm notes.txt\nEOF\n{runner}"

    assert approval_floors._match_user_deny_rule(command) is not None, runner


@pytest.mark.parametrize("command", [
    "cat > job.sh <<'EOF'\nrm -rf /\nEOF\nbash -euo pipefail job.sh",
    "cat > job.sh <<'EOF'\nrm -rf /\nEOF\nbash +o posix job.sh",
    "cat > j.ps1 <<'EOF'\nRemove-Item C:\\ -Recurse -Force\nEOF\npwsh -File j.ps1",
    "cat > j.ps1 <<'EOF'\nStop-Computer -Force\nEOF\npowershell -NoProfile -ExecutionPolicy Bypass -File .\\j.ps1",
    "cat > j.ps1 <<'EOF'\nClear-Disk -Number 0 -RemoveData\nEOF\npwsh -ep bypass j.ps1",
    "cat > j.bat <<'EOF'\nrd /s /q C:\\\nEOF\ncmd /c j.bat",
    "cat > j.bat <<'EOF'\nrd /s /q C:\\\nEOF\nCMD.EXE /Q /C call .\\j.bat",
])
def test_option_values_and_windows_runners_keep_written_script_on_floors(floors_off, command):
    command = command.replace("rm -rf /", "r" + "m -r" + "f /")
    assert detect_hardline_command(command)[0] is True
    assert approval.check_all_command_guards(command, "local")["approved"] is False


@pytest.mark.parametrize("script,runner", [
    ("j.ps1", "pwsh -File j.ps1"), ("job.py", "python3 -W ignore job.py"), ("job.sh", "bash -o pipefail job.sh"),
    ("j.bat", "cmd /c j.bat"),
    ("job.sh", "bash -o pipefail 2>err.log job.sh"),
    ("job.sh", "bash -o pipefail 2>&1 job.sh"), ("job.sh", "bash -o pipefail < job.sh"),
    ("job.sh", 'bash -o pipefail -c "$(cat job.sh)"'),
])
def test_option_value_or_powershell_run_keeps_written_script_on_the_deny_list(floors_off, script, runner):
    command = f"cat > {script} <<'EOF'\nrm notes.txt\nEOF\n{runner}"

    assert approval_floors._match_user_deny_rule(command) is not None, runner


@pytest.mark.parametrize("script,runner", [
    ("job.sh", "bash -c 'bash job.sh'"), ("job.sh", 'sudo sh -c "sh job.sh"'), ("job.sh", 'bash -lc "source job.sh"'),
    ("job.sh", "bash -euo pipefail -c 'bash job.sh'"), ("job.sh", "timeout 60 bash -c 'bash -o pipefail job.sh'"),
    ("job.sh", "bash -c 'cat job.sh | bash'"), ("j.bat", 'cmd /c "call j.bat"'), ("job.sh", "su -c 'bash job.sh' root"),
    ("job.sh", "sh -c 'sudo bash job.sh'"), ("job.sh", "bash -c ' bash job.sh'"), ("job.sh", "bash -c ' cat job.sh | bash'"),
])
def test_command_string_handed_to_a_shell_keeps_written_script_on_the_deny_list(floors_off, script, runner):
    command = f"cat > {script} <<'EOF'\nrm notes.txt\nEOF\n{runner}"

    assert approval_floors._match_user_deny_rule(command) is not None, runner


@pytest.mark.parametrize("runner", [
    "bash -c 'wc -l job.sh'", "bash -c 'bash other.sh'", "python3 -c 'print(1)' job.sh", "grep -c 'job.sh' list.txt",
])
def test_command_string_that_does_not_run_the_written_file_does_not_prompt(runner):
    command = f"cat > job.sh <<'EOF'\n{PUSH}\nEOF\n{runner}"

    assert _body_adds_no_prompt(command), runner


@pytest.mark.parametrize("runner", [
    'pwsh -c "iex (gc ps/j.ps1 -Raw)"', 'pwsh -Command "Invoke-Expression -Command (Get-Content -Path ps/j.ps1 -Raw)"',
    'powershell -NoProfile -Command "Get-Content ps/j.ps1 | Invoke-Expression"', 'pwsh -c "gc -Raw .\\ps\\j.ps1 | iex"',
    "pwsh -c 'iex $(type ps/j.ps1)'", 'pwsh -c "& ([scriptblock]::Create((gc ps/j.ps1 -Raw)))"', 'pwsh -c ". ./ps/j.ps1"',
    'pwsh -c ". .\\ps\\j.ps1"', 'pwsh -c "iex ( gc ps/j.ps1 -Raw )"', 'pwsh -c "& ([scriptblock]::Create( (gc ps/j.ps1) ))"',
])
def test_powershell_running_a_written_file_s_content_keeps_it_on_the_deny_list(floors_off, runner):
    command = f"cat > ps/j.ps1 <<'EOF'\nrm notes.txt\nEOF\n{runner}"

    assert approval_floors._match_user_deny_rule(command) is not None, runner


@pytest.mark.parametrize("reader", [
    'pwsh -c "gc ps/j.ps1 | Select-String rm"', 'pwsh -c "Write-Output (gc ps/j.ps1 -Raw)"',
    'pwsh -c "iex (gc other.ps1 -Raw)"', 'pwsh -c "Get-Content ps/j.ps1 | Out-File copy.txt"',
    "tar czf out.tgz -C . ps/j.ps1",
])
def test_powershell_reading_a_written_file_without_running_it_does_not_prompt(reader):
    command = f"cat > ps/j.ps1 <<'EOF'\n{PUSH}\nEOF\n{reader}"

    assert _body_adds_no_prompt(command), reader


@pytest.mark.parametrize("runner", [
    "bash -o pipefail other.sh", "pwsh -File other.ps1", "pwsh -Command Get-Date", "cmd /c other.bat",
])
def test_running_another_script_after_a_written_doc_does_not_prompt(runner):
    command = f"cat > notes.md <<'EOF'\n{PUSH}\nEOF\n{runner}"

    assert _body_adds_no_prompt(command), runner


@pytest.mark.parametrize("command", [
    "cat > job.sh <<'EOF'\ngrep -P \"x\nrm -rf /\nEOF\nmv job.sh run.sh && bash run.sh",
    "cat > job.sh <<'EOF'\ngrep -P 'x\nEOF\nbash job.sh",
    "cat <<'EOF'\nit's\nEOF\ngrep -P 'x",
])
def test_malformed_grep_reaches_the_hardline_floor_in_or_after_a_heredoc_body(floors_off, command):
    # The malformed-grep verdict reads the raw command like the other floors (a script body something runs
    # later), and the body-dropped copy (a body quote hid the grep after the terminator).
    command = command.replace("rm -rf /", "r" + "m -r" + "f /")
    assert detect_hardline_command(command)[0] is True
    assert approval.check_all_command_guards(command, "local")["approved"] is False


@pytest.mark.parametrize("grep", ["grep -P 'foo' n.md", 'grep -o "[^\\"]*" n.md'])
def test_valid_grep_after_a_doc_with_a_stray_quote_is_not_malformed(grep):
    assert detect_hardline_command(f"cat > n.md <<'EOF'\nit's here\nEOF\n{grep}")[0] is False


def test_unquoted_heredoc_keeps_command_substitution_visible_to_hardline():
    command = "cat > notes.txt <<EOF\n$(rm -rf /)\nEOF"

    assert detect_hardline_command(command)[0] is True


@pytest.mark.parametrize("command", [
    "cat <<-'EOF'\n\tit's data\n\tEOF\nrm -rf /",
    "cat <<'A' > a\nit's data\nA\ncat <<'B' > b\nok\nB\nrm -rf /",
])
def test_heredoc_forms_resume_hardline_scanning_after_their_bodies(command):
    assert detect_hardline_command(command)[0] is True


def test_unterminated_heredoc_body_stays_visible(floors_off):
    # Without a terminator line the parse is unproven (`$((a<<b))` looks the same), so nothing is
    # dropped: a false positive is acceptable, hiding a command from the floors is not.
    command = "cat > notes.txt <<'EOF'\nNever run `rm -rf`."

    assert approval.check_all_command_guards(command, "local")["approved"] is False


@pytest.mark.parametrize("delimiter", ["EOF", "'EOF'"])
@pytest.mark.parametrize("nested_command", ["rm notes.txt", "rm -rf /"])
def test_nested_heredoc_substitution_reaches_command_floors(floors_off, delimiter, nested_command):
    # Unquoted, the substitution runs; quoted, it is text the floors still read, as on main.
    command = f"cat > notes.txt <<{delimiter}\n$(printf '%s' `{nested_command}`)\nEOF"

    assert approval.check_all_command_guards(command, "local")["approved"] is False
    assert approval_floors._match_user_deny_rule(command) is not None
    if nested_command == "rm -rf /":
        assert detect_hardline_command(command)[0] is True
