from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

import tools.approval as approval_module
from hermes_cli.security_policy import classify_command, typed_confirmation_phrase
from tools.approval import (
    check_all_command_guards,
    check_dangerous_command,
    reset_current_session_key,
    set_current_session_key,
)


def _tirith_result(action="allow", findings=None, summary=""):
    return {"action": action, "findings": findings or [], "summary": summary}


def _events(audit_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(audit_dir.glob("*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    return rows


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture
def typed_audit_env(tmp_path, monkeypatch):
    audit_dir = tmp_path / "audit"
    monkeypatch.setenv("HERMES_AUDIT_DIR", str(audit_dir))
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")
    token = set_current_session_key("typed-confirm-session")
    try:
        yield audit_dir
    finally:
        reset_current_session_key(token)


def test_read_only_command_does_not_prompt_for_typed_confirmation(typed_audit_env):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("read-only command should not request approval")

    with patch("tools.tirith_security.check_command_security", return_value=_tirith_result()):
        result = check_all_command_guards("git status --short", "local", approval_callback=fail_if_called)

    assert result["approved"] is True


def test_high_risk_command_requires_exact_typed_confirmation(typed_audit_env):
    command = "git reset --hard HEAD~1"
    expected_phrase = typed_confirmation_phrase(classify_command(command))
    seen = {}

    def callback(_command, _description, **kwargs):
        seen.update(kwargs)
        return expected_phrase

    old_umask = os.umask(0)
    try:
        with patch("tools.tirith_security.check_command_security", return_value=_tirith_result()):
            result = check_all_command_guards(command, "local", approval_callback=callback)
    finally:
        os.umask(old_umask)

    assert result["approved"] is True
    assert seen["typed_confirmation_phrase"] == expected_phrase
    assert seen["allow_permanent"] is False
    assert _mode(typed_audit_env) == 0o700
    audit_file = next(typed_audit_env.glob("*.jsonl"))
    assert _mode(audit_file) == 0o600
    events = _events(typed_audit_env)
    assert events[-1]["decision"] == "approved"
    assert events[-1]["approval_scope"] == "once"
    assert events[-1]["extra"]["risk_class"] == "destructive"
    assert events[-1]["extra"]["typed_confirmation_required"] is True


@pytest.mark.parametrize("raw_value", ["confirm destructive action", "CONFIRM DESTRUCTIVE ACTION", "", "once"])
def test_high_risk_near_miss_confirmation_is_denied(typed_audit_env, raw_value):
    command = "git reset --hard HEAD~1"

    def callback(_command, _description, **kwargs):
        assert kwargs["typed_confirmation_phrase"] == typed_confirmation_phrase(classify_command(command))
        return raw_value

    with patch("tools.tirith_security.check_command_security", return_value=_tirith_result()):
        result = check_all_command_guards(command, "local", approval_callback=callback)

    assert result["approved"] is False
    events = _events(typed_audit_env)
    assert events[-1]["decision"] == "denied"
    assert events[-1]["status"] == "blocked_user_denied"
    assert events[-1]["extra"]["typed_confirmation_required"] is True


def test_noninteractive_high_risk_command_is_blocked_and_audited(tmp_path, monkeypatch):
    audit_dir = tmp_path / "audit"
    monkeypatch.setenv("HERMES_AUDIT_DIR", str(audit_dir))
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")
    token = set_current_session_key("typed-block-session")
    try:
        result = check_all_command_guards("git reset --hard HEAD~1", "local")
    finally:
        reset_current_session_key(token)

    assert result["approved"] is False
    events = _events(audit_dir)
    assert events[-1]["decision"] == "blocked"
    assert events[-1]["status"] == "blocked_typed_confirmation_required"
    assert events[-1]["extra"]["risk_class"] == "destructive"


def test_gateway_high_risk_command_blocks_button_approval_path(tmp_path, monkeypatch):
    audit_dir = tmp_path / "audit"
    monkeypatch.setenv("HERMES_AUDIT_DIR", str(audit_dir))
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")
    token = set_current_session_key("typed-gateway-session")
    try:
        with patch("tools.tirith_security.check_command_security", return_value=_tirith_result()):
            result = check_all_command_guards("git reset --hard HEAD~1", "local")
    finally:
        reset_current_session_key(token)

    assert result["approved"] is False
    assert result["status"] == "blocked_typed_confirmation_required"
    events = _events(audit_dir)
    assert events[-1]["decision"] == "blocked"
    assert events[-1]["surface"] == "gateway"
    assert events[-1]["extra"]["risk_class"] == "destructive"


def test_typed_confirmation_audit_redacts_secret_bearing_command(typed_audit_env):
    fake_key = "sk-" + "typedtest1234567890abcdef"
    command = f"echo OPENAI_API_KEY={fake_key} > .env"
    expected_phrase = typed_confirmation_phrase(classify_command(command))

    with patch("tools.tirith_security.check_command_security", return_value=_tirith_result()):
        result = check_all_command_guards(
            command,
            "local",
            approval_callback=lambda *_args, **_kwargs: expected_phrase,
        )

    assert result["approved"] is True
    audit_file = next(typed_audit_env.glob("*.jsonl"))
    audit_text = audit_file.read_text(encoding="utf-8")
    assert fake_key not in audit_text
    assert "typed-confirm-session" not in audit_text
    assert "OPENAI_API_KEY=***" in audit_text


@pytest.mark.parametrize("guard_name", ["legacy", "combined"])
def test_credential_path_write_requires_exact_typed_confirmation(typed_audit_env, guard_name):
    command = "printf '%s' placeholder > ~/.hermes/provider-token"
    expected_phrase = typed_confirmation_phrase(classify_command(command))
    guard = check_dangerous_command if guard_name == "legacy" else check_all_command_guards

    with patch("tools.tirith_security.check_command_security", return_value=_tirith_result()):
        result = guard(command, "local", approval_callback=lambda *_args, **_kwargs: expected_phrase)

    assert result["approved"] is True
    audit_file = next(typed_audit_env.glob("*.jsonl"))
    assert _mode(audit_file) == 0o600
    events = _events(typed_audit_env)
    assert events[-1]["decision"] == "approved"
    assert events[-1]["extra"]["risk_class"] == "credential_sensitive"
    assert events[-1]["extra"]["typed_confirmation_required"] is True


@pytest.mark.parametrize("guard_name", ["legacy", "combined"])
def test_credential_path_write_near_miss_is_denied_and_audited(typed_audit_env, guard_name):
    command = "printf '%s' placeholder > ~/.hermes/provider-token"
    guard = check_dangerous_command if guard_name == "legacy" else check_all_command_guards

    with patch("tools.tirith_security.check_command_security", return_value=_tirith_result()):
        result = guard(command, "local", approval_callback=lambda *_args, **_kwargs: "once")

    assert result["approved"] is False
    events = _events(typed_audit_env)
    assert events[-1]["decision"] == "denied"
    assert events[-1]["status"] == "blocked_user_denied"
    assert events[-1]["extra"]["risk_class"] == "credential_sensitive"
