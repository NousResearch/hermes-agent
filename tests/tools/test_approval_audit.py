import json
import os
import stat
from unittest.mock import patch

import pytest

import tools.approval as approval_module
from tools.approval import (
    check_all_command_guards,
    normalize_approval_choice,
    reset_current_session_key,
    set_current_session_key,
)


def _tirith_result(action="allow", findings=None, summary=""):
    return {"action": action, "findings": findings or [], "summary": summary}


def _events(audit_dir):
    files = sorted(audit_dir.glob("*.jsonl"))
    assert files
    rows = []
    for path in files:
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    return rows


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture
def approval_audit_env(tmp_path, monkeypatch):
    audit_dir = tmp_path / "audit"
    monkeypatch.setenv("HERMES_AUDIT_DIR", str(audit_dir))
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")
    token = set_current_session_key("approval-audit-session")
    try:
        yield audit_dir
    finally:
        reset_current_session_key(token)


def test_cli_denial_writes_private_redacted_audit_events(approval_audit_env):
    fake_key = "sk-" + "test1234567890abcdef"
    old_umask = os.umask(0)
    try:
        with patch("tools.tirith_security.check_command_security", return_value=_tirith_result()):
            result = check_all_command_guards(
                f"OPENAI_API_KEY={fake_key} rm -rf /tmp/private",
                "local",
                approval_callback=lambda *_args, **_kwargs: "deny",
            )
    finally:
        os.umask(old_umask)

    assert result["approved"] is False
    assert _mode(approval_audit_env) == 0o700
    audit_file = next(approval_audit_env.glob("*.jsonl"))
    assert _mode(audit_file) == 0o600
    events = _events(approval_audit_env)
    assert [event["decision"] for event in events] == ["skipped", "denied"]
    assert events[-1]["status"] == "blocked_user_denied"
    assert events[-1]["surface"] == "cli"
    assert events[-1]["session_key"]["present"] is True

    text = audit_file.read_text(encoding="utf-8")
    assert "approval-audit-session" not in text
    assert fake_key not in text


def test_cli_approval_writes_approved_event(approval_audit_env):
    with patch("tools.tirith_security.check_command_security", return_value=_tirith_result()):
        result = check_all_command_guards(
            "rm -rf /tmp/approval-audit",
            "local",
            approval_callback=lambda *_args, **_kwargs: "once",
        )

    assert result["approved"] is True
    events = _events(approval_audit_env)
    assert events[-1]["decision"] == "approved"
    assert events[-1]["approval_scope"] == "once"
    assert events[-1]["status"] == "allowed_user_approved"


def test_hardline_block_writes_blocked_audit_event(approval_audit_env):
    result = check_all_command_guards("rm -rf /", "local")

    assert result["approved"] is False
    events = _events(approval_audit_env)
    assert events[-1]["decision"] == "blocked"
    assert events[-1]["risk_tier"] == "R5"
    assert events[-1]["status"] == "blocked_hardline"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("once", "once"),
        (" o ", "once"),
        ("session", "session"),
        ("always", "always"),
        ("deny", "deny"),
        ("yes", "deny"),
        ("approve once", "deny"),
        ("always approve", "deny"),
        ("nope", "deny"),
    ],
)
def test_typed_approval_choice_rejects_near_misses(raw, expected):
    assert normalize_approval_choice(raw) == expected


def test_always_choice_without_permanent_scope_becomes_session():
    assert normalize_approval_choice("always", allow_permanent=False) == "session"
