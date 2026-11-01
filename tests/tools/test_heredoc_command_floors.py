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


def test_unquoted_heredoc_keeps_command_substitution_visible_to_hardline():
    command = "cat > notes.txt <<EOF\n$(rm -rf /)\nEOF"

    assert detect_hardline_command(command)[0] is True


@pytest.mark.parametrize("command", [
    "cat <<-'EOF'\n\tit's data\n\tEOF\nrm -rf /",
    "cat <<'A' > a; cat <<'B' > b\nit's data\nA\nok\nB\nrm -rf /",
])
def test_heredoc_forms_resume_hardline_scanning_after_their_bodies(command):
    assert detect_hardline_command(command)[0] is True


def test_unterminated_heredoc_is_data_until_eof(floors_off):
    command = "cat > notes.txt <<'EOF'\nNever run `rm -rf`."

    assert approval.check_all_command_guards(command, "local")["approved"] is True


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
