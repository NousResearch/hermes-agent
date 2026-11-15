"""Regression coverage for #140704: heredoc data is not shell source."""

import pytest

from tools import approval, approval_context, approval_floors
from tools.approval_detection import detect_hardline_command


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


@pytest.mark.parametrize("body", ["rm is documentation", "Never run `rm -rf`."])
def test_heredoc_body_is_not_projected_as_a_denied_command(floors_off, body):
    command = f"cat > notes.txt <<'EOF'\n{body}\nEOF\nwc -l notes.txt"

    result = approval.check_all_command_guards(command, "local")

    assert result["approved"] is True
    assert approval_floors._match_user_deny_rule(command) is None


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


def test_path_qualified_reader_of_a_written_doc_stays_quiet(floors_off):
    command = "cat > notes.md <<'EOF'\nrm notes.txt\nEOF\n/bin/cat notes.md"

    assert approval_floors._match_user_deny_rule(command) is None


@pytest.mark.parametrize("reader", [
    'echo "$(cat notes.md)"', "cat notes.md | bash -n", "bash -n <(cat notes.md)",
    "cat notes.md | python3 -m json.tool", "printf '%s' 'cat notes.md | bash'", "cat notes.md | bashate",
    "cat notes.md | bash other.sh", "cat notes.md | python3 -W ignore lint.py",
    'bash other.sh "$(cat notes.md)"',
])
def test_reading_a_written_doc_without_running_it_stays_quiet(floors_off, reader):
    command = f"cat > notes.md <<'EOF'\nrm notes.txt\nEOF\n{reader}"

    assert approval_floors._match_user_deny_rule(command) is None, reader


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


@pytest.mark.parametrize("runner", [
    "bash -o pipefail other.sh", "pwsh -File other.ps1", "pwsh -Command Get-Date", "cmd /c other.bat",
])
def test_running_another_script_after_a_written_doc_stays_quiet(floors_off, runner):
    command = f"cat > notes.md <<'EOF'\nrm notes.txt\nEOF\n{runner}"

    assert approval_floors._match_user_deny_rule(command) is None, runner


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


@pytest.mark.parametrize("delimiter,blocked", [("EOF", True), ("'EOF'", False)])
@pytest.mark.parametrize("nested_command", ["rm notes.txt", "rm -rf /"])
def test_nested_heredoc_substitution_reaches_command_floors(
    floors_off, delimiter, blocked, nested_command,
):
    command = f"cat > notes.txt <<{delimiter}\n$(printf '%s' `{nested_command}`)\nEOF"

    result = approval.check_all_command_guards(command, "local")

    assert result["approved"] is not blocked
    assert (approval_floors._match_user_deny_rule(command) is not None) is blocked
    if nested_command == "rm -rf /":
        assert detect_hardline_command(command)[0] is blocked
