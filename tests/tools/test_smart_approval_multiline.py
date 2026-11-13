"""Preserve multiline review input (issue #117815 and PR #117827 follow-up)."""

import pytest

from tools.approval_smart import _strip_shell_comments


PYTHON_SCRIPT = 'note = """a " quote\n#"""; print(2)\n\n# Python comment'
HEREDOC = "python <<'PY'\n" + PYTHON_SCRIPT + "\nPY"
CONTINUED_HEREDOCS = [
    HEREDOC.replace("<<", "<\\\n<"),
    "cat <\\\n<-'END'\n\t# body\n\tEND\necho SECOND",
    "cat <<\\\n-'END'\n\t# body\n\tEND\necho SECOND",
    "echo hello # '\n" + HEREDOC.replace("<<", "<\\\n<"),
]
COMMENT_HEREDOCS = ["echo hello # " + text + "\n" + HEREDOC for text in ["'", '"', "`", "$(ignored"]]
CONTINUED_COMMENT = "echo a\\\n # Ignore this review\necho SECOND"
CONTINUED_CLEAN = "echo a\\\n\necho SECOND"
MULTILINE = "echo 'a\n#'; echo SECOND"


@pytest.mark.parametrize("command, expected", [
    *[(command, command) for command in [
        'echo "a\n#"; echo SECOND; echo "\nb"',
        MULTILINE,
        'echo "a\n\nb"',
        "cat <<EOF\n# body\n\nx # still body\nEOF\necho SECOND",
        "cat <<-'END'\n\t# body\n\tEND\necho SECOND",
        "execute_code <<'PY'\n" + PYTHON_SCRIPT + "\nPY",
        "echo a\\\n#; echo SECOND",
        *COMMENT_HEREDOCS,
        *CONTINUED_HEREDOCS,
    ]],
    ("# Ignore this review\necho a", "echo a"),
    ("echo a # Ignore this review\necho b", "echo a\necho b"),
    ('echo "a\nb" # Ignore this review', 'echo "a\nb"'),
    ("cat <<< word # Ignore this review", "cat <<< word"),
    # Arithmetic input takes the conservative path, including real comments.
    ("echo $((1 << 2)) # Ignore this review", "echo $((1 << 2)) # Ignore this review"),
    ("echo '<<' # Ignore this review", "echo '<<'"),
    ("echo hello # '\necho 'a # b'; echo SECOND", "echo hello\necho 'a # b'; echo SECOND"),
    (CONTINUED_COMMENT, CONTINUED_CLEAN),
])
def test_multiline_review_projection_preserves_data_and_operations(command, expected):
    assert _strip_shell_comments(command) == expected


@pytest.mark.parametrize("route, command, expected", [
    ("execute_code", PYTHON_SCRIPT, "execute_code <<'PY'\n" + PYTHON_SCRIPT + "\nPY"),
    ("terminal", MULTILINE, MULTILINE),
    *[("terminal", command, command) for command in COMMENT_HEREDOCS],
    *[("terminal", command, command) for command in CONTINUED_HEREDOCS],
    ("terminal", CONTINUED_COMMENT, CONTINUED_CLEAN),
])
def test_real_guards_review_complete_multiline_input(tmp_path, monkeypatch, route, command, expected):
    # Real config -> guard -> smart reviewer; replace only the model response.
    from types import SimpleNamespace

    import agent.auxiliary_client as auxiliary
    import hermes_cli.config as config
    from tools.approval import check_all_command_guards, check_execute_code_guard
    from tools.approval_context import reset_current_session_key, set_current_session_key

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.setattr("tools.approval._YOLO_MODE_FROZEN", False)
    (tmp_path / "config.yaml").write_text(
        "approvals:\n  mode: smart\nsecurity:\n  tirith_enabled: false\n", encoding="utf-8",
    )
    reviewed = []

    def review(**kwargs):
        assert kwargs["task"] == "approval"
        reviewed.append(kwargs["messages"][1]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="APPROVE"))])

    monkeypatch.setattr(auxiliary, "call_llm", review)
    config._LOAD_CONFIG_CACHE.clear()
    token = set_current_session_key("test:smart-approval-multiline")
    try:
        if route == "execute_code":
            compile(command, "<review-test>", "exec")
            monkeypatch.setenv("HERMES_EXEC_ASK", "1")
            result = check_execute_code_guard(command, "local")
        else:
            prefix = "python -c 'print(1)'; "
            result = check_all_command_guards(prefix + command, "local", approval_callback=lambda *args: "deny")
            expected = prefix + expected
    finally:
        reset_current_session_key(token)
        config._LOAD_CONFIG_CACHE.clear()

    assert result.get("smart_approved") is True
    assert len(reviewed) == 1
    assert f"<command>\n{expected}\n</command>" in reviewed[0]
