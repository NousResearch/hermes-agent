"""execute_code results say who approved a flagged script, as terminal() results do.

The guardian's LLM call runs inside the tool dispatch, where Relay leaves nested calls
unmanaged (#77244), so the tool result is the only per-call record that reaches the session
store and the tool's trace span. The guard's ``smart_approved`` / ``user_approved`` flags
must survive into that result instead of being dropped after the approval check.
"""

from __future__ import annotations

import json

import pytest

import tools.code_execution_tool as cet
from tools import approval as A
from tools import approval_context
from tools import approval_smart
from tools import code_kernel
from tools import terminal_tool


@pytest.fixture
def gw_session(monkeypatch):
    """A gateway session with an isolated approval queue, so the real guard runs."""
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", False)
    session_key = "approval-note-session"
    token = approval_context.set_current_session_key(session_key)
    with A._lock:
        had_permanent = "execute_code" in A._permanent_approved
        A._gateway_queues.pop(session_key, None)
        A._gateway_notify_cbs.pop(session_key, None)
        A._permanent_approved.discard("execute_code")
        A._session_approved.pop(session_key, None)
    try:
        yield session_key
    finally:
        approval_context.reset_current_session_key(token)
        with A._lock:
            A._gateway_queues.pop(session_key, None)
            A._gateway_notify_cbs.pop(session_key, None)
            A._session_approved.pop(session_key, None)
            if had_permanent:
                A._permanent_approved.add("execute_code")


@pytest.fixture
def local_kernel(monkeypatch):
    """Run the real execute_code entry point up to the kernel, which returns a canned result."""
    from tools import process_registry

    ran: list[str] = []
    monkeypatch.setattr(cet, "SANDBOX_AVAILABLE", True)
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: False)
    # execute_code imports _get_env_config from terminal_tool at call time; patch it there.
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: {"env_type": "local"})

    def kernel(code, **_kwargs):
        ran.append(code)
        return json.dumps({"status": "success", "output": "hi\n"})

    monkeypatch.setattr(code_kernel, "execute_in_session_kernel", kernel)
    return ran


def _resolve_with(session_key: str, choice: str) -> None:
    def cb(_approval_data):
        with A._lock:
            entries = A._gateway_queues.get(session_key, [])
            if entries:
                entries[-1].result = choice
                entries[-1].event.set()

    with A._lock:
        A._gateway_notify_cbs[session_key] = cb


def test_smart_auto_approval_is_recorded_in_the_result(gw_session, local_kernel, monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "smart")
    monkeypatch.setattr(approval_smart, "_smart_approve", lambda _c, _d: "approve")

    result = json.loads(cet.execute_code("print('hi')"))

    assert local_kernel == ["print('hi')"]
    assert result["status"] == "success"
    assert result["approval"] == "Script was flagged (execute_code) and auto-approved by smart approval."


def test_escalated_then_human_approval_is_recorded_in_the_result(gw_session, local_kernel, monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "smart")
    monkeypatch.setattr(approval_smart, "_smart_approve", lambda _c, _d: "escalate")
    _resolve_with(gw_session, "once")

    result = json.loads(cet.execute_code("print('hi')"))

    assert local_kernel == ["print('hi')"]
    assert result["approval"] == "Script required approval (execute_code) and was approved by the user."


def test_manual_mode_human_approval_is_recorded_in_the_result(gw_session, local_kernel, monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    _resolve_with(gw_session, "once")

    result = json.loads(cet.execute_code("print('hi')"))

    assert result["approval"] == "Script required approval (execute_code) and was approved by the user."


def test_unflagged_run_carries_no_approval_note(local_kernel, monkeypatch):
    monkeypatch.setattr(A, "check_execute_code_guard", lambda *_a, **_k: {"approved": True, "message": None})

    result = json.loads(cet.execute_code("print('hi')"))

    assert local_kernel == ["print('hi')"]
    assert "approval" not in result


def test_remote_backend_result_carries_the_note(monkeypatch):
    from tools import process_registry

    monkeypatch.setattr(cet, "SANDBOX_AVAILABLE", True)
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: False)
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: {"env_type": "ssh"})
    monkeypatch.setattr(terminal_tool, "_docker_has_host_access", lambda _config: False)
    monkeypatch.setattr(A, "check_execute_code_guard",
                        lambda *_a, **_k: {"approved": True, "message": None, "smart_approved": True})
    remote_calls: list[str] = []

    def remote(code, *_a, **_k):
        remote_calls.append(code)
        return json.dumps({"status": "success", "output": "remote\n"})

    monkeypatch.setattr(cet, "_execute_remote", remote)
    monkeypatch.setattr(code_kernel, "execute_in_session_kernel",
                        lambda *_a, **_k: pytest.fail("remote env_type must not reach the local kernel"))

    result = json.loads(cet.execute_code("print('remote')"))

    assert remote_calls == ["print('remote')"]
    assert result["output"] == "remote\n"
    assert result["approval"] == "Script was flagged (execute_code) and auto-approved by smart approval."


@pytest.mark.parametrize("raw", ["not json", json.dumps(["a", "b"]), json.dumps({"approval": "kept"})])
def test_note_leaves_non_object_and_already_annotated_results_alone(raw):
    assert cet._with_approval_note(raw, "Script was flagged (execute_code) and auto-approved.") == raw


def test_no_note_returns_result_unchanged():
    raw = json.dumps({"status": "success"})
    assert cet._with_approval_note(raw, None) is raw
