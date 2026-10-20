"""HF-527: child cancellation must not revoke its authenticated parent request."""
from contextvars import copy_context
from threading import Event, Lock
from types import SimpleNamespace

import pytest

from agent.interrupt_control import InterruptControlMixin
from tools import approval as gate, approval_context as ctx
from tools.delegate_tool_child_run import _ChildRun
from tools.delegate_tool_dispatch import _dispatch_unit
from tools.approval_smart import _smart_approve


class Child(InterruptControlMixin):
    """No model/network: exercise the real native interrupt owner and child runner."""

    def __init__(self, conversation):
        self.run_conversation = conversation
        self.session_id = "child-session"
        self.model = "fixture"
        self._hard_interrupt_requested = Event()
        self._execution_thread_id = None
        self._active_children_lock = Lock()
        self._active_children = []
        self.quiet_mode = True


def dispatch_child(monkeypatch, child):
    captured = {}

    def dispatch(**kwargs):
        captured.update(kwargs)
        return {"status": "dispatched", "delegation_id": "unit"}

    monkeypatch.setattr("tools.async_delegation.dispatch_async_delegation_batch", dispatch)
    unit = SimpleNamespace(children=[(0, {"goal": "Inspect one file"}, child)],
                           task_list=[{"goal": "Inspect one file"}], context=None,
                           top_role="worker", creds={"model": "fixture"},
                           live_writers=[], live_home=None)
    routing = dict(session_key="s", origin_ui_session_id="", origin_session_id="s",
                   parent_session_id="s", max_async_children=1)
    assert _dispatch_unit(unit, "unit", None, routing)["status"] == "dispatched"
    return captured


def run_child(child):
    return _ChildRun(child=child, parent_agent=SimpleNamespace(), task_index=0,
                     goal="Inspect one file", subagent_id=None, child_progress_cb=None,
                     child_task_id="child").await_child()


def test_child_interrupt_keeps_parent_sibling_and_completion_authorized(monkeypatch):
    child = Child(lambda **_: {"final_response": "done"})
    with ctx.approval_task_context("Inspect two independent files", identity="u", profile="p",
                                   session="s", is_current=lambda: True):
        sibling = ctx.reserve_background_approval_task()
        accepted = dispatch_child(monkeypatch, child)
        try:
            accepted["interrupt_fn"]()
            assert child._hard_interrupt_requested.is_set()
            assert not ctx.approval_task_revoked()
            assert gate.check_all_command_guards("pwd", "local")["approved"]
            with ctx.background_approval_task(sibling, "sibling"):
                assert gate.check_all_command_guards("pwd", "local")["approved"]
            # A child stop report belongs to the still-live parent, not the stopped child.
            with ctx.completion_approval_task([accepted["approval_reservation"]], "report"):
                assert gate.check_all_command_guards("pwd", "local")["approved"]
        finally:
            sibling.release()
            accepted["on_unstarted"]()
            ctx.release_background_approval_task(accepted["approval_reservation"])


@pytest.mark.parametrize("cancel_unit", [False, True])
def test_native_child_stop_rejects_late_approval_and_copied_context(monkeypatch, cancel_unit):
    observations = {}
    accepted = {}
    monkeypatch.setattr("tools.delegate_tool_config._get_child_timeout", lambda: None)
    monkeypatch.setattr(ctx, "_get_approval_config", dict)

    def verdict(**_):
        (accepted["interrupt_fn"] if cancel_unit else child.hard_interrupt)()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="APPROVE"))])

    monkeypatch.setattr("agent.auxiliary_client.call_llm", verdict)

    def conversation(**_):
        assert ctx.current_approval_task_context().ancestry[-1] == "child"
        observations["copied"] = copy_context()
        observations["detached"] = ctx.reserve_background_approval_task()
        observations["verdict"] = _smart_approve("python -c 'print(1)'", "script execution")
        observations["command"] = gate.check_all_command_guards("pwd", "local")["approved"]
        observations["code"] = gate.check_execute_code_guard("print(1)", "local")["approved"]
        return {"final_response": "stopped"}

    child = Child(conversation)
    monkeypatch.setattr("tools.delegate_tool_dispatch._execute_and_aggregate", lambda *_a, **_k: run_child(child))
    with ctx.approval_task_context("Inspect files", identity="u", profile="p", session="s",
                                   is_current=lambda: True):
        accepted.update(dispatch_child(monkeypatch, child))
        try:
            result, error, deferred = accepted["runner"]()
            assert error is None and not deferred
            assert result["final_response"] == "stopped"
            assert observations["verdict"] == "deny"
            assert observations["command"] is False
            assert observations["code"] is False
            assert observations["copied"].run(ctx.approval_task_revoked)
            with ctx.background_approval_task(observations["detached"], "late-process"):
                assert ctx.approval_task_revoked()
            assert gate.check_all_command_guards("pwd", "local")["approved"]
        finally:
            accepted["on_unstarted"]()
            ctx.release_background_approval_task(accepted["approval_reservation"])
            ctx.release_background_approval_task(observations.get("detached"))


def test_native_stop_follows_nested_work_and_cannot_revive_after_observation():
    stopped = Event()
    owners = []
    with ctx.approval_task_context("Inspect files", identity="u", profile="p", session="s",
                                   is_current=lambda: True, owner=owners.append):
        sibling = ctx.reserve_background_approval_task()
        with ctx.delegated_approval_task("child", stop_event=stopped):
            with ctx.delegated_approval_task("grandchild", stop_event=Event()):
                retained = ctx.reserve_background_approval_task()
        try:
            # Ordinary return preserves a genuinely reserved background completion.
            with ctx.completion_approval_task([retained], "process-done"):
                assert ctx.current_approval_task_context().ancestry == ("child", "grandchild", "process-done")
                stopped.set()
                assert ctx.approval_task_revoked()
                stopped.clear()
                assert ctx.approval_task_revoked()
            assert not ctx.approval_task_revoked()
            # Whole-task Stop still revokes the parent and every sibling.
            owners[0].revoke()
            assert ctx.approval_task_revoked()
            with ctx.background_approval_task(sibling, "sibling"):
                assert ctx.approval_task_revoked()
        finally:
            ctx.release_background_approval_task(retained)
            ctx.release_background_approval_task(sibling)
    with ctx.approval_task_context("New request", identity="u", profile="p", session="s",
                                   is_current=lambda: True):
        assert not ctx.approval_task_revoked()
        with ctx.background_approval_task(retained, "old-process"):
            assert ctx.approval_task_revoked()
