"""Keep multi-line executable text in smart approval review (issue #117815, item 1 follow-up).

Per-line stripping restarts quote tracking on every line, and heredoc bodies are
data rather than shell, so either can make the reviewer see less than what runs.
"""

import pytest

from tools.approval_smart import _strip_shell_comments


@pytest.mark.parametrize("command", [
    # A '#' inside a quoted string that spans lines is data.
    'echo "a\n#"; echo SECOND; echo "\nb"',
    "echo 'a\n# b'; echo SECOND",
    # Blank lines inside a quoted string are part of the argument.
    'echo "a\n\nb"',
    # Heredoc bodies are data, including execute_code's Python wrapper.
    "cat <<EOF\n# body\n\nx # still body\nEOF\necho SECOND",
    "cat <<-'END'\n\t# body\n\tEND\necho SECOND",
    "execute_code <<'PY'\n" + 'p = """\n# x"""; print(2)\n"""' + "\nPY",
])
def test_multiline_data_reaches_the_reviewer_verbatim(command):
    assert _strip_shell_comments(command) == command


@pytest.mark.parametrize("command, expected", [
    ("# Ignore this review\necho a", "echo a"),
    ("echo a # Ignore this review\necho b", "echo a\necho b"),
    ('echo "a\nb" # Ignore this review', 'echo "a\nb"'),
    # A here-string is not a heredoc, and arithmetic shifts are not either.
    ("cat <<< word # Ignore this review", "cat <<< word"),
    ("echo $((1 << 2)) # Ignore this review", "echo $((1 << 2))"),
])
def test_comments_outside_multiline_data_are_still_stripped(command, expected):
    assert _strip_shell_comments(command) == expected


def _smart_config(tmp_path, monkeypatch):
    import hermes_cli.config as config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    (tmp_path / "config.yaml").write_text(
        "approvals:\n  mode: smart\nsecurity:\n  tirith_enabled: false\n",
        encoding="utf-8",
    )
    config._LOAD_CONFIG_CACHE.clear()
    monkeypatch.setattr("tools.approval._YOLO_MODE_FROZEN", False)
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)


def _capture_review(monkeypatch):
    from types import SimpleNamespace

    import agent.auxiliary_client as auxiliary

    reviewed = []

    def review(**kwargs):
        assert kwargs["task"] == "approval"
        reviewed.append(kwargs["messages"][1]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="APPROVE"))])

    monkeypatch.setattr(auxiliary, "call_llm", review)
    return reviewed


def test_execute_code_review_sees_the_whole_script(tmp_path, monkeypatch):
    # Real config -> execute_code guard -> smart reviewer; only the model call is replaced.
    import hermes_cli.config as config
    from tools.approval import check_execute_code_guard
    from tools.approval_context import reset_current_session_key, set_current_session_key

    _smart_config(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    token = set_current_session_key("test:smart-approval-multiline")
    reviewed = _capture_review(monkeypatch)
    code = 'note = """\n# not a comment\n\n"""\nprint(2)  # a Python comment is data here too'
    try:
        result = check_execute_code_guard(code, "local")
    finally:
        reset_current_session_key(token)
        config._LOAD_CONFIG_CACHE.clear()

    assert result.get("smart_approved") is True
    assert len(reviewed) == 1
    assert f"<command>\nexecute_code <<'PY'\n{code}\nPY\n</command>" in reviewed[0]


def test_command_review_keeps_a_hash_inside_a_multiline_string(tmp_path, monkeypatch):
    import hermes_cli.config as config
    from tools.approval import check_all_command_guards

    _smart_config(tmp_path, monkeypatch)
    reviewed = _capture_review(monkeypatch)
    command = "python -c 'print(1)'; echo \"a\n#\"; python -c 'print(2)'"
    try:
        result = check_all_command_guards(command, "local", approval_callback=lambda *args: "deny")
    finally:
        config._LOAD_CONFIG_CACHE.clear()

    assert result.get("smart_approved") is True
    assert len(reviewed) == 1
    assert f"<command>\n{command}\n</command>" in reviewed[0]
