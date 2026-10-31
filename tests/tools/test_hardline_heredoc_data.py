"""Regression for #124551. Heredoc data is not shell syntax; never execute these fixtures."""
import pytest

from tools.approval_detection import detect_hardline_command


@pytest.mark.parametrize("opener", [
    "cat > note.md <<'EOF'", 'cat <<"EOF" > note.md',
    "tee note.md <<'EOF'", "cat <<\\EOF > note.md",
])
@pytest.mark.parametrize("word", ["poweroff", "shutdown", "reboot", "halt"])
def test_quoted_data_sink_does_not_trip_hardline(opener, word):
    command = f"{opener}\nMaintenance notes\n{word} only after saving work\nEOF\n"
    assert detect_hardline_command(command) == (False, None)


@pytest.mark.parametrize("command", [
    "bash <<'EOF'\nreboot\nEOF",
    "sh <<'EOF'\npoweroff\nEOF",
    "cat <<'EOF' | sh\npoweroff\nEOF",
    "cat <<EOF\n$(reboot)\nEOF",
    "cat <<'EOF'\nnotes\nEOF\nreboot",
    "cat <<'EOF'\npoweroff\nWRONG",
    "python <<'EOF'\npoweroff\nEOF",
    "osascript <<'EOF'\npoweroff\nEOF",
])
def test_executable_or_uncertain_payload_stays_blocked(command):
    assert detect_hardline_command(command)[0] is True


@pytest.mark.parametrize("command", [
    "cat() { sh; }\ncat <<'EOF'\npoweroff\nEOF",
    "PATH=./bin\ncat <<'EOF'\npoweroff\nEOF",
    "cat > script <<'EOF'\npoweroff\nEOF\nsh script",
    "./cat <<'EOF'\npoweroff\nEOF",
    "/opt/custom/tee <<'EOF'\npoweroff\nEOF",
    "PATH=./bin cat <<'EOF'\npoweroff\nEOF",
    "env PATH=./bin cat <<'EOF'\npoweroff\nEOF",
    "cat <<'EOF' > /dev/sda\nnotes\nEOF",
    "cat <<'EOF' > >(sh)\npoweroff\nEOF",
    "cat <<'EOF'; sh\npoweroff\nEOF",
    "cat <<'EOF'\npoweroff\n EOF",
    "cat <<EOF\n`reboot`\nEOF",
])
def test_redirections_and_ambiguous_consumers_stay_blocked(command):
    assert detect_hardline_command(command)[0] is True


def test_guard_chain_distinguishes_data_and_execution():
    from tools.approval import check_all_command_guards, enable_session_yolo, disable_session_yolo
    from tools.approval_context import set_current_session_key, reset_current_session_key

    session = "heredoc-data-regression"
    token = set_current_session_key(session)
    enable_session_yolo(session)
    try:
        data = "cat > note.md <<'EOF'\nMaintenance notes\npoweroff after saving work\nEOF"
        assert check_all_command_guards(data, "local")["approved"]
        executable = "sh <<'EOF'\npoweroff\nEOF"
        assert not check_all_command_guards(executable, "local")["approved"]
    finally:
        disable_session_yolo(session)
        reset_current_session_key(token)


def test_data_only_mode_preserves_interpreter_and_default_contract():
    from tools.shell_heredoc import strip_inert_heredoc_bodies

    for receiver in ("python", "osascript"):
        command = f"{receiver} <<'EOF'\npoweroff\nEOF"
        assert strip_inert_heredoc_bodies(command, data_only=True) == command
        assert "poweroff" not in strip_inert_heredoc_bodies(command)


def test_empty_and_plain_commands_are_unchanged():
    assert detect_hardline_command("") == (False, None)
    assert detect_hardline_command("echo ready") == (False, None)
