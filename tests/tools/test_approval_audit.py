"""Approval receipts exercise real config loading, guards, waits and filesystem."""
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.config import load_config
from tools import approval, approval_audit, approval_context, approval_smart
from tools.approval_gateway_wait import _await_gateway_decision


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("surface,choice,verdict", [
    ("cli", "once", "approved"), ("cli", "deny", "denied"),
    ("cli", "timeout", "timeout"),
    ("smart", "approve", "approved"), ("smart", "deny", "denied"),
    ("smart_redactor", "approve", "approved"), ("smart_redactor", "deny", "denied"),
    ("gateway", "once", "approved"), ("gateway", "deny", "denied"),
    ("gateway", "timeout", "timeout"), ("gateway", "notify_failed", "notify_failed"),
])
def test_real_approval_paths(tmp_path, monkeypatch, enabled, surface, choice, verdict):
    if surface == "smart_redactor":
        surface = "smart"
        from agent.redact import redact_sensitive_text
        def redact(text, *, force=False):
            if force:
                raise RuntimeError("redactor unavailable")
            return redact_sensitive_text(text)
        monkeypatch.setattr("agent.redact.redact_sensitive_text", redact)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        f"approvals:\n  mode: {'smart' if surface == 'smart' else 'manual'}\n"
        f"  timeout: {0 if choice == 'timeout' else 5}\n  audit_log:\n    enabled: {str(enabled).lower()}\n", encoding="utf-8")
    monkeypatch.setattr(approval_audit, "_settings", {})
    load_config()
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval, "_permanent_approved", set())
    monkeypatch.setattr(approval, "_session_approved", {})
    monkeypatch.setattr(approval, "_gateway_queues", {})
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    for name in ("HERMES_GATEWAY_SESSION", "HERMES_EXEC_ASK", "HERMES_SESSION_PLATFORM",
                 "HERMES_SINGLE_QUERY_SESSION", "HERMES_CRON_SESSION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("tools.tirith_security.check_command_security",
                        lambda _: {"action": "allow", "findings": [], "summary": ""})
    monkeypatch.setattr(approval_smart, "_smart_approve", lambda *_: choice)
    token = approval_context.set_current_observability_context(session_id="audit-session", tool_call_id="call")
    command = "rm -rf /tmp/private-audit-token"
    try:
        if surface == "gateway":
            def notify(_):
                if choice == "notify_failed":
                    raise RuntimeError("private-audit-token")
                if choice != "timeout":
                    approval.resolve_gateway_approval("audit", choice)
            result = _await_gateway_decision("audit", notify, {
                "command": command, "description": "private-audit-token",
                "pattern_key": "private-audit-token"})
            assert result["resolved"] is (choice not in {"timeout", "notify_failed"})
        else:
            result = approval.check_all_command_guards(command, "local", approval_callback=lambda *a, **k: choice)
            assert result["approved"] is (verdict == "approved")
    finally:
        approval_context.reset_current_observability_context(token)
        approval.clear_session("audit")
    path = tmp_path / "logs" / "approvals.jsonl"
    if not enabled:
        assert not path.exists()
        # No config read, path stat, directory creation, or file open in disabled observation.
        with patch("builtins.open") as builtin_open, \
             patch.object(Path, "open") as path_open, \
             patch.object(Path, "stat") as stat, \
             patch.object(Path, "mkdir") as mkdir, \
             patch("hermes_cli.lifecycle.invoke_hook"):
            approval_context._fire_approval_hook("post_approval_response", choice="once", surface=surface)
        for operation in (builtin_open, path_open, stat, mkdir):
            operation.assert_not_called()
        return
    text = path.read_text(encoding="utf-8")
    assert "private-audit-token" not in text
    records = [json.loads(line) for line in text.splitlines()]
    assert len(records) == (2 if surface == "smart" and choice == "deny" else 1)
    if len(records) == 2:
        assert records[1]["source"] == "interactive"
        assert records[1]["verdict"] == "denied"
    record = records[0]
    assert record["verdict"] == verdict
    assert record["session_id"] == "audit-session"
    assert record["tool_call_id"] == "call"
    assert record["source"] == {"cli": "interactive", "smart": "smart_approval", "gateway": "gateway_wait"}[surface]
    assert record["interactive"] is (surface == "cli")
    assert record["decided_by"] == ("aux_llm" if surface == "smart" else
                                    "timeout" if choice == "timeout" else
                                    "user" if surface == "cli" else "gateway")


def test_chain_survives_concurrent_writers_rotation_and_restart(tmp_path):
    path = tmp_path / "approvals.jsonl"
    def append(index):
        approval_audit._append(path, {"decision": index}, 450, 40)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(append, range(24)))
    # Independent processes must read the predecessor and rotate under the same lock.
    script = (
        "import sys; from pathlib import Path; from tools.approval_audit import _append; "
        "_append(Path(sys.argv[1]), {'decision': int(sys.argv[2])}, 450, 40)"
    )
    processes = [subprocess.Popen([sys.executable, "-c", script, str(path), str(index)])
                 for index in range(24, 28)]
    for process in processes:
        assert process.wait(timeout=30) == 0
    append(28)
    files = sorted(tmp_path.glob("approvals.jsonl.*"),
                   key=lambda p: int(p.suffix[1:]) if p.suffix[1:].isdigit() else -1, reverse=True)
    records = []
    for file in files + [path]:
        if file.suffix != ".lock":
            records.extend(json.loads(line) for line in file.read_text().splitlines())
    assert {record["decision"] for record in records} == set(range(29))
    previous = "0" * 64
    for record in records:
        digest = record.pop("hash")
        assert record["prev_hash"] == previous
        assert digest == hashlib.sha256(approval_audit._canonical(record)).hexdigest()
        previous = digest
    # Corrupt tails are never silently adopted or overwritten.
    original = path.read_bytes()
    path.write_bytes(original.replace(b'"decision":28', b'"decision":99'))
    with pytest.raises(ValueError, match="tail hash"):
        append(29)
    path.write_bytes(original[:-1])
    with pytest.raises(ValueError, match="Incomplete"):
        append(29)
