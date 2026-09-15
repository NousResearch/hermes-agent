"""A delegated worker's own ``delegate_task(background=True)`` must not ride its coordinator's route.

A subagent inherits the spawning chat's session context, so ``_resolve_async_wake_sid`` used to hand
its nested batch to the async registry: the detached completion is then pushed at the COORDINATOR's
route while the event is stamped with the worker's internal session id.  The worker itself is gone by
then — nothing returns the nested result to the turn that asked for it.

Merged NousResearch/hermes-agent#103486 states the intended shape: "A subagent that fans out
(depth >= 1) calls ``delegate_task`` synchronously by design, since it needs its workers' results
inside its own turn."  Contract asserted here: the SAME batch dispatches detached from a normal
session and runs inline from inside ``delegated_child_context`` — only the child worker (which would
need a live model) is stubbed.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from agent.delegation_context import delegated_child_context
from gateway.session_context import clear_session_vars, set_session_vars
from tools import delegate_tool_dispatch as dispatch


@pytest.fixture()
def gateway_session():
    """Bind a messaging session that CAN receive a detached completion (the coordinator's)."""
    tokens = set_session_vars(
        platform="telegram", session_id="coordinator", session_key="agent:main:telegram:dm:42",
        async_delivery=True,
    )
    try:
        yield
    finally:
        clear_session_vars(tokens)


@pytest.fixture()
def batch(monkeypatch):
    """A one-task batch whose child run is stubbed (no model call), everything else real."""
    from tools import delegate_tool

    task = {"goal": "nested work"}
    child = SimpleNamespace(session_id="worker-child")
    monkeypatch.setattr(
        delegate_tool, "_run_single_child",
        lambda **kwargs: {"task_index": 0, "status": "completed", "summary": "nested result"},
    )
    monkeypatch.setattr(dispatch, "_finalize_child_results", lambda *args: None)
    return dispatch._Batch(
        task_list=[task], children=[(0, task, child)],
        parent_agent=SimpleNamespace(session_id="worker", _delegate_depth=1),
        creds={"model": "test-model"}, context=None, top_role="leaf", max_children=1,
        live_deleg_id=None, live_writers=[], live_paths=[],
        origin_wake_sid="", origin_ui_session_id="", origin_owner_transport=None,
        origin_owner_session_record=None, origin_session_history_delivery=False,
        overall_start=time.monotonic(),
    )


def test_background_delegation_detaches_for_a_normal_session(gateway_session, batch, monkeypatch) -> None:
    dispatched = []
    monkeypatch.setattr(
        "tools.async_delegation.dispatch_async_delegation_batch",
        lambda **kwargs: dispatched.append(kwargs) or {"status": "dispatched", "delegation_id": "deleg_test"},
    )

    result = json.loads(dispatch._dispatch_background(batch))

    assert dispatched, "a route-owning session still dispatches detached units"
    assert "results" not in result


def test_delegated_worker_gets_its_nested_batch_back_in_its_own_turn(
    gateway_session, batch, monkeypatch,
) -> None:
    def _must_not_dispatch(**kwargs):
        raise AssertionError("a delegated worker must not detach onto its coordinator's route")

    monkeypatch.setattr("tools.async_delegation.dispatch_async_delegation_batch", _must_not_dispatch)

    with delegated_child_context("worker"):
        result = json.loads(dispatch._dispatch_background(batch))

    assert [entry["summary"] for entry in result["results"]] == ["nested result"]
    assert "SYNCHRONOUSLY" in result["note"]
