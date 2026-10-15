"""Approval audit contracts for #90186 on the current approval entry points."""
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from tools import approval


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


def records(home):
    return [json.loads(line) for line in (home / "logs/approvals.jsonl").read_text().splitlines()]


def deny(monkeypatch):
    monkeypatch.setattr(approval, "_floor_block", lambda *a, **kw: {
        "approved": False, "message": "blocked", "outcome": "denied",
        "pattern_key": "test", "description": "test rule",
    })


def test_command_entry_points_log_without_changing_decision(tmp_path, monkeypatch):
    deny(monkeypatch)
    for entry in (approval.check_all_command_guards, approval.check_dangerous_command):
        assert entry("synthetic command", "local")["approved"] is False
    rows = records(tmp_path)
    assert [r["surface"] for r in rows] == ["terminal", "command"]
    assert all(r["outcome"] == "denied" for r in rows)


def test_plugin_gate_is_audited_even_for_plain_approval(tmp_path, monkeypatch):
    decision = {"approved": True, "message": None}
    monkeypatch.setattr(approval, "_run_approval_gate", lambda **kw: decision)
    assert approval.request_tool_approval("synthetic_tool", "synthetic reason") is decision
    assert records(tmp_path)[0]["surface"] == "tool"


def test_execute_code_denial_is_audited(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(approval, "_yolo_active", lambda: False)
    monkeypatch.setattr(approval.approval_context, "_get_approval_mode", lambda: "ask")
    monkeypatch.setattr(approval, "_unattended_contexts", lambda: [
        SimpleNamespace(mode=lambda: "deny", exec_tail="test unattended")
    ])
    assert not approval.check_execute_code_guard("print('synthetic')", "local")["approved"]
    assert records(tmp_path)[0]["surface"] == "execute_code"


def test_plain_approval_does_not_log_commands(tmp_path, monkeypatch):
    monkeypatch.setattr(approval, "_floor_block", lambda *a, **kw: None)
    monkeypatch.setattr(approval, "_yolo_active", lambda: True)
    approval.check_all_command_guards("synthetic safe", "local")
    assert not (tmp_path / "logs/approvals.jsonl").exists()


@pytest.mark.parametrize("broken", [False, True])
def test_all_free_text_is_redacted_and_session_is_hashed(tmp_path, monkeypatch, broken):
    from tools.approval_audit import log_approval_event
    def redact(text, **kw):
        if broken:
            raise ValueError("synthetic failure")
        return text.replace("SENSITIVE_SENTINEL", "[REDACTED]")
    monkeypatch.setattr("agent.redact.redact_sensitive_text", redact)
    monkeypatch.setattr("tools.approval_context.get_current_session_key", lambda **kw: "synthetic-session")
    log_approval_event("SENSITIVE_SENTINEL", "SENSITIVE_SENTINEL", "SENSITIVE_SENTINEL",
                       {"approved": False, "outcome": "denied"}, "test")
    text = (tmp_path / "logs/approvals.jsonl").read_text()
    assert "SENSITIVE_SENTINEL" not in text
    assert "synthetic-session" not in text
    assert records(tmp_path)[0]["session_key"].startswith("sha256:")


@pytest.mark.platforms("posix")
def test_symlink_refused_and_permissions_private(tmp_path):
    import os
    from tools.approval_audit import log_approval_event
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    victim = tmp_path / "victim"
    victim.write_text("unchanged")
    path = logs / "approvals.jsonl"
    path.symlink_to(victim)
    log_approval_event("test", "test", "test", {})
    assert victim.read_text() == "unchanged"
    path.unlink()
    path.write_text("")
    path.chmod(0o644)
    log_approval_event("test", "test", "test", {})
    assert os.stat(path).st_mode & 0o777 == 0o600


def test_logging_failure_does_not_change_guard(monkeypatch):
    deny(monkeypatch)
    monkeypatch.setattr("os.open", lambda *a, **kw: (_ for _ in ()).throw(OSError("synthetic")))
    assert not approval.check_all_command_guards("synthetic", "local")["approved"]


def test_concurrent_records_are_complete(tmp_path):
    from tools.approval_audit import log_approval_event
    def write(i):
        log_approval_event(str(i), "test", "test", {"approved": False})
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(40)))
    assert {r["pattern_key"] for r in records(tmp_path)} == {str(i) for i in range(40)}


def test_profile_override_owns_log(tmp_path):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from tools.approval_audit import log_approval_event
    owner = tmp_path / "owner"
    owner.mkdir()
    token = set_hermes_home_override(owner)
    try:
        log_approval_event("test", "test", "test", {})
    finally:
        reset_hermes_home_override(token)
    assert records(owner)
    assert not (tmp_path / "logs/approvals.jsonl").exists()
