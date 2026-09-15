"""Audit coverage and failure isolation through native approval paths."""
import json
import subprocess
import sys
import time
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.config import load_config
from tools import approval, approval_audit as audit, approval_context, approval_prompt


@pytest.fixture
def audit_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(audit, "_settings", {})
    (tmp_path / "config.yaml").write_text(
        "approvals:\n  mode: manual\n  audit_log:\n    enabled: true\n", encoding="utf-8")
    load_config()
    monkeypatch.setattr(approval, "_gateway_notify_cbs", {})
    token = approval_context.set_current_session_key("platform:private-user-id")
    try:
        yield tmp_path
    finally:
        approval_context.reset_current_session_key(token)


def _records(home):
    return [json.loads(line) for line in (home / "logs" / "approvals.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("path", ["protected", "elicitation"])
@pytest.mark.parametrize("choice", ["once", "deny", "timeout"])
def test_missing_cli_paths_record_exactly_one_response(audit_home, monkeypatch, enabled, path, choice):
    from tools import terminal_tool
    from tools.file_tools_write_guards import _request_protected_instruction_approval

    if not enabled:
        (audit_home / "config.yaml").write_text("approvals:\n  audit_log:\n    enabled: false\n")
        load_config()
    monkeypatch.setattr(terminal_tool, "_get_approval_callback", lambda: lambda *a, **kw: choice)
    monkeypatch.setattr(approval_context, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(approval_prompt, "_read_choice", lambda *a: None if choice == "timeout" else choice)
    if path == "protected":
        result = _request_protected_instruction_approval([str(audit_home / "AGENTS.md")])
        assert (result is None) is (choice == "once")
    else:
        result = approval_prompt.request_elicitation_consent("private-content", "confirm")
        assert result == {"once": "accept", "deny": "decline", "timeout": "cancel"}[choice]
    if not enabled:
        assert not (audit_home / "logs" / "approvals.jsonl").exists()
        return
    record, = _records(audit_home)
    assert record["raw_choice"] == choice
    assert record["scope"] == ("once" if choice == "once" else None)
    assert record["verdict"] == {"once": "approved", "deny": "denied", "timeout": "timeout"}[choice]
    assert record["source"] == "interactive" and record["interactive"] is True
    assert record["session_id"] == audit._digest("platform:private-user-id")


@pytest.mark.parametrize("choice", ["once", "session", "always", "deny", "error", "timeout", "redactor", "unavailable",
                                    "invalid", "stale", "busy", "interrupted"])
@pytest.mark.parametrize("surface", ["cli", "gateway"])
def test_transport_receipts_keep_origin_scope_and_failures(audit_home, monkeypatch, choice, surface):
    from hermes_cli.plugins import PluginManager

    manager = PluginManager()
    def present(request):
        if choice == "error":
            raise RuntimeError("private-content")
        if choice == "invalid":
            return "invalid decision"
        if choice == "stale":
            from hermes_cli.approval_transport import ApprovalDecision
            return ApprovalDecision("stale", request.digest, "once")
        return request.respond(choice)
    if choice != "unavailable":
        manager.register_approval_transport("phone", present, plugin_id="test-audit")
    monkeypatch.setattr(approval_prompt, "get_plugin_manager", lambda: manager)
    # Publish the selection through the real config loader.
    with (audit_home / "config.yaml").open("a") as handle:
        handle.write("security:\n  approval:\n    transport: phone\n")
    if choice == "busy":
        monkeypatch.setattr("hermes_cli.approval_transport._transport_worker_slots", threading.BoundedSemaphore(0))
    if choice == "interrupted":
        monkeypatch.setattr(approval_prompt, "is_interrupted", lambda: True)
    if choice == "timeout":
        monkeypatch.setattr(approval_context, "_get_approval_timeout", lambda: 0)
    if choice == "redactor":
        def broken(*a, **kw):
            raise RuntimeError("private-content")
        monkeypatch.setattr("agent.redact.redact_sensitive_text", broken)
    result = approval_prompt._present_with_selected_transport(
        command="private-content", description="confirm", pattern_key="first",
        pattern_keys=["first", "second"], session_key="platform:private-user-id", surface=surface,
        allow_session=True, allow_permanent=True)
    record, = _records(audit_home)
    expected_choice = {"redactor": "transport_error", "error": "transport_error",
                       "timeout": "transport_timeout", "unavailable": "transport_unavailable",
                       **{failure: f"transport_{failure}" for failure in
                          ("invalid", "stale", "busy", "interrupted")}}.get(choice, choice)
    assert record["raw_choice"] == expected_choice
    assert record["source"] == "transport:phone"
    assert record["interactive"] is (surface == "cli")
    assert record["decided_by"] == ("timeout" if choice == "timeout" else "user" if surface == "cli" else "gateway")
    assert record["verdict"] == ("timeout" if choice == "timeout" else
                                  "approved" if choice in {"once", "session", "always"} else "denied")
    assert record["scope"] == (choice if choice in {"once", "session", "always"} else None)
    assert record["pattern_keys"] == [audit._digest("first"), audit._digest("second")]
    assert result["choice"] == (choice if choice in {"once", "session", "always"} else "deny")
    assert "private-content" not in json.dumps(record)


@pytest.mark.parametrize("holder", ["thread", "process", "filesystem"])
def test_sink_failure_never_blocks_real_approval(audit_home, monkeypatch, caplog, holder):
    from tools import terminal_tool
    from tools.file_tools_write_guards import _request_protected_instruction_approval

    monkeypatch.setattr(terminal_tool, "_get_approval_callback", lambda: lambda *a, **kw: "once")
    monkeypatch.setattr(audit, "_LOCK_TIMEOUT", 0.15)
    process = None
    if holder == "thread":
        audit._lock.acquire()
    elif holder == "process":
        logs = audit_home / "logs"
        logs.mkdir(exist_ok=True)
        script = (
            "import sys; from pathlib import Path; from tools.approval_audit import _writer_lock\n"
            "with _writer_lock(Path(sys.argv[1])):\n"
            " print('ready', flush=True)\n"
            " sys.stdin.readline()\n"
        )
        process = subprocess.Popen([sys.executable, "-c", script, str(logs / "approvals.jsonl.lock")],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        assert process.stdout.readline().strip() == "ready"
    else:
        (audit_home / "logs" / "approvals.jsonl").mkdir()
    try:
        started = time.monotonic()
        assert _request_protected_instruction_approval(["AGENTS.md"]) is None
        assert time.monotonic() - started < 3
        assert "Approval audit write failed" in caplog.text
        assert not (audit_home / "logs" / "approvals.jsonl").is_file()
    finally:
        if holder == "thread":
            audit._lock.release()
        if process:
            process.communicate("release\n", timeout=10)
            assert process.returncode == 0
    if holder != "filesystem":
        assert _request_protected_instruction_approval(["AGENTS.md"]) is None
        record, = _records(audit_home)
        assert record["verdict"] == "approved"


def test_disabled_config_and_hooks_are_inert_and_malformed_flags_warn_once(monkeypatch, caplog):
    monkeypatch.setattr(audit, "_settings", {})
    monkeypatch.setattr(audit, "_warned_enabled", False)
    with patch.object(audit, "get_hermes_home") as home, \
         patch("hermes_cli.lifecycle.invoke_hook") as hook:
        for config in ({}, {"approvals": {"audit_log": {"enabled": False}}}):
            audit.configure(config)
            approval_context._fire_approval_hook("post_approval_response", surface="cli", choice="once")
            assert "decided_by" not in hook.call_args.kwargs
        home.assert_not_called()
        for value in ("true", 1, None, []):
            audit.configure({"approvals": {"audit_log": {"enabled": value}}})
        home.assert_not_called()
    assert caplog.text.count("must be a boolean") == 1


@pytest.mark.parametrize("limit", ["max_size_mb", "backup_count"])
def test_overflow_limits_and_long_pattern_lists_resume_chain(audit_home, limit):
    audit.configure({"approvals": {"audit_log": {"enabled": True}}, "logging": {limit: float("inf")}})
    payload = {"surface": "cli", "choice": "session", "pattern_key": "primary",
               "pattern_keys": [f"pattern-{i}" for i in range(1100)]}
    audit.record_decision(payload)
    audit.record_decision({**payload, "choice": "always"})
    first, second = _records(audit_home)
    assert second["prev_hash"] == first["hash"]
    assert first["pattern_keys"] == [audit._digest(key) for key in payload["pattern_keys"]]
    assert second["scope"] == "always"


def test_rotation_interrupted_after_rename_resumes_from_backup(audit_home, monkeypatch):
    path = audit_home / "logs" / "approvals.jsonl"
    audit._append(path, {"decision": "first"}, 1, 3)
    first, = _records(audit_home)
    real_replace = audit.os.replace
    def interrupted(source, target):
        real_replace(source, target)
        if source == path:
            raise OSError("simulated crash after rename")
    with monkeypatch.context() as patcher:
        patcher.setattr(audit.os, "replace", interrupted)
        with pytest.raises(OSError, match="simulated crash"):
            audit._append(path, {"decision": "lost"}, 1, 3)
    audit._append(path, {"decision": "resumed"}, 1, 3)
    resumed, = _records(audit_home)
    assert resumed["prev_hash"] == first["hash"]
    assert json.loads(Path(f"{path}.1").read_text())["hash"] == first["hash"]
