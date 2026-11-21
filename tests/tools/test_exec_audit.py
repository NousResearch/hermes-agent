"""Exec-audit log: append-only, 0600, redacted body, never raises.

A terminal/execute_code invocation must leave one durable JSONL record whose body is
secret-scrubbed; the write path must never fail the observed tool, even when the log
location is unwritable.
"""

import json
import os
import stat

import pytest


@pytest.fixture
def audit_dir(tmp_path, monkeypatch):
    from tools import exec_audit

    path = tmp_path / "security" / "exec_audit.jsonl"
    monkeypatch.setattr(exec_audit, "exec_audit_path", lambda: path)
    return path


def _read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_append_only_two_records(audit_dir):
    from tools.exec_audit import record_exec

    record_exec(tool="terminal", body="echo hi", profile="tech", session_key="s1", task_id="t1")
    record_exec(tool="execute_code", body="print(1)", profile="tech")

    lines = _read_lines(audit_dir)
    assert len(lines) == 2
    assert lines[0]["tool"] == "terminal"
    assert lines[0]["body"] == "echo hi"
    assert lines[0]["profile"] == "tech"
    assert lines[0]["session_key"] == "s1"
    assert lines[0]["task_id"] == "t1"
    assert lines[1]["tool"] == "execute_code"
    assert all("ts" in line for line in lines)


def test_permissions_are_0600(audit_dir):
    from tools.exec_audit import record_exec

    record_exec(tool="terminal", body="true")
    mode = stat.S_IMODE(os.stat(audit_dir).st_mode)
    assert mode == 0o600


def test_body_is_redacted(audit_dir, monkeypatch):
    # Force redaction on so the secret assignment is definitely scrubbed.
    monkeypatch.setattr("os.environ", {**os.environ, "HERMES_REDACT_SECRETS": "true"})
    from tools.exec_audit import record_exec

    record_exec(tool="terminal", body="export API_KEY='super-secret-token-value'")
    lines = _read_lines(audit_dir)
    assert "super-secret-token-value" not in lines[0]["body"]


def test_never_raises_when_unwritable(audit_dir, monkeypatch):
    from tools import exec_audit
    from tools.exec_audit import record_exec

    # Point the log at a path whose parent is a regular file → open() must fail, not raise.
    blocker = audit_dir.parent.parent / "blocker"
    blocker.write_text("file not dir")
    monkeypatch.setattr(exec_audit, "exec_audit_path", lambda: blocker / "exec_audit.jsonl")

    # Must not raise; merely a no-op with a warning.
    record_exec(tool="terminal", body="true")


def test_code_body_uses_code_redaction(audit_dir):
    from tools.exec_audit import record_exec

    record_exec(tool="execute_code", body="url = 'https://user:pass@example.com/x'")
    lines = _read_lines(audit_dir)
    assert "user:pass" not in lines[0]["body"]
    assert "***" in lines[0]["body"]
