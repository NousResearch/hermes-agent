"""Steering and failed admission must not revoke a still-running child's approvals."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from hermes_cli.backend_retirement import retirement
from tools import approval, async_delegation, delegate_tool
from tools.approval_context import reset_current_session_key, set_current_session_key
from tools.approval_gateway_wait import _await_gateway_decision
from tools.approval_ownership import _owners, gateway_approval_owner, register_gateway_approval_owner


class ApprovalChild:
    tool_progress_callback = None
    _credential_pool = None
    _delegate_saved_tool_names = []
    _delegate_role = "leaf"
    _delegate_depth = 1
    _subagent_id = None
    model = "test-model"
    session_prompt_tokens = session_completion_tokens = 0
    session_estimated_cost_usd = 0.0
    session_cost_status = "unknown"

    def __init__(self, interrupt):
        self._interrupt_requested = True
        hard = threading.Event()
        if interrupt == "hard_stop":
            hard.set()
        self._hard_interrupt_requested = None if interrupt == "legacy_stop" else hard
        self.decision = None

    def run_conversation(self, **kwargs):
        key = approval.get_current_session_key()
        self.decision = _await_gateway_decision(key, approval._gateway_notify_cb(key), {
            "command": "synthetic operation (never executed)", "description": "test approval",
            "pattern_key": "test",
        })
        return {"completed": True, "final_response": "finished", "api_calls": 1, "messages": []}

    def get_activity_summary(self):
        return {"api_call_count": 1}

    def close(self):
        pass


@pytest.mark.parametrize("interrupt", ["soft_steer", "hard_stop", "legacy_stop"])
def test_startup_interrupt_only_revokes_approval_for_a_hard_stop(interrupt, monkeypatch):
    monkeypatch.setattr(delegate_tool, "_get_worktree_isolation", lambda: False)
    key = "approval-child-startup"
    sent = []

    def notify(data):
        sent.append(data)
        assert approval.resolve_gateway_approval(key, "once", request_id=data["request_id"]) == 1

    owner = register_gateway_approval_owner(key, notify)
    owner_token = gateway_approval_owner.set(owner)
    session_token = set_current_session_key(key)
    child = ApprovalChild(interrupt)
    parent = SimpleNamespace(session_id="test-parent", _current_task_id=None,
                             _active_children=[], _active_children_lock=threading.Lock())
    try:
        result = delegate_tool._run_single_child(0, "synthetic", child=child, parent_agent=parent)
        assert child.decision is not None, result
        if interrupt == "soft_steer":
            assert child.decision["choice"] == "once"
            assert len(sent) == 1
        else:
            assert child.decision.get("cancelled")
            assert not sent
        assert {o for o in _owners if o.session_key == key} == {owner}
        assert not approval.has_blocking_approval(key)
    finally:
        approval.unregister_gateway_notify(key)
        gateway_approval_owner.reset(owner_token)
        reset_current_session_key(session_token)


@pytest.mark.parametrize("failure", ["monitor", "submit"])
def test_failed_admission_leaves_no_worker_or_approval_owner(failure, monkeypatch):
    """Fallible monitor startup must precede the ownership handoff to the executor."""
    async_delegation._reset_for_tests()
    key = "approval-admission-failure"
    owner = register_gateway_approval_owner(key, lambda data: None)
    owner_token = gateway_approval_owner.set(owner)
    session_token = set_current_session_key(key)
    initial_reservations = retirement.active_count()
    release = threading.Event()
    submitted = []
    executor = ThreadPoolExecutor(max_workers=1)
    real_submit = executor.submit

    def submit(fn):
        if failure == "submit":
            raise RuntimeError("injected submit failure")
        future = real_submit(fn)
        submitted.append(future)
        return future

    def start_monitor():
        if failure == "monitor":
            raise RuntimeError("injected monitor startup failure")

    def runner():
        assert release.wait(10)
        return {"status": "completed", "summary": "synthetic"}

    monkeypatch.setattr(executor, "submit", submit)
    monkeypatch.setattr(async_delegation, "_get_executor", lambda _: executor)
    monkeypatch.setattr(async_delegation, "_ensure_stale_monitor", start_monitor)
    try:
        result = async_delegation.dispatch_async_delegation(
            goal="synthetic", context=None, toolsets=None, role="leaf", model=None,
            session_key=key, runner=runner, progress_fn=lambda: (1, False),
        )
        assert result["status"] == "rejected"
        assert not submitted, "Rejected admission must not leave queued or running work"
        assert not async_delegation.has_live_for_session(session_key=key)
        with async_delegation._DB_LOCK, async_delegation._transaction() as conn:
            assert conn.execute("SELECT COUNT(*) FROM async_delegations").fetchone()[0] == 0
        assert {o for o in _owners if o.session_key == key} == {owner}
        assert approval._gateway_notify_cb(key) is owner
        assert retirement.active_count() == initial_reservations
    finally:
        release.set()
        executor.shutdown(wait=True)
        approval.unregister_gateway_notify(key)
        gateway_approval_owner.reset(owner_token)
        reset_current_session_key(session_token)
        async_delegation._reset_for_tests()
