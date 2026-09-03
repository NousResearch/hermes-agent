"""Bounded model-facing waits over the existing durable delegation ledger."""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

from tools import async_delegation as ad
from tools.delegate_tool import DELEGATE_TASK_SCHEMA, delegate_task


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ad._reset_for_tests()
    yield
    ad._reset_for_tests()


def _dispatch(delegation_id: str, parent: str) -> None:
    ad._persist_dispatch(
        {
            "delegation_id": delegation_id,
            "session_key": parent,
            "parent_session_id": parent,
            "origin_ui_session_id": "",
            "dispatched_at": time.time(),
            "goal": "test task",
            "role": "leaf",
            "model": "test-model",
        }
    )


def _complete(
    delegation_id: str,
    summary: str = "done",
    *,
    status: str = "completed",
) -> None:
    now = time.time()
    result = {"status": status, "summary": summary}
    ad._persist_completion(
        {
            "type": "async_delegation",
            "delegation_id": delegation_id,
            "status": status,
            "completed_at": now,
        },
        result,
    )


def test_wait_consumes_result_once_and_callback_loses():
    _dispatch("deleg_one", "parent-one")
    _complete("deleg_one")

    first = ad.wait_for_delegations(
        ["deleg_one"], parent_session_id="parent-one", timeout_seconds=1
    )
    assert first["status"] == "ready"
    assert first["results"][0]["result"]["summary"] == "done"
    assert ad.claim_completion_delivery("deleg_one", "callback-loser") is False

    repeated = ad.wait_for_delegations(
        ["deleg_one"], parent_session_id="parent-one", timeout_seconds=1
    )
    assert repeated["results"][0]["already_delivered"] is True
    assert "result" not in repeated["results"][0]


def test_wait_wakes_when_same_process_persists_completion():
    _dispatch("deleg_wake", "parent-wake")
    observed = []
    waiter = threading.Thread(
        target=lambda: observed.append(
            ad.wait_for_delegations(
                ["deleg_wake"],
                parent_session_id="parent-wake",
                timeout_seconds=2,
                heartbeat_seconds=0.02,
            )
        )
    )
    waiter.start()
    time.sleep(0.05)
    assert waiter.is_alive()

    _complete("deleg_wake", "woke")
    waiter.join(timeout=1)

    assert not waiter.is_alive()
    assert observed[0]["results"][0]["result"]["summary"] == "woke"


def test_wait_treats_timeout_as_terminal():
    _dispatch("deleg_timeout", "parent-timeout")
    _complete("deleg_timeout", "deadline exceeded", status="timeout")

    result = ad.wait_for_delegations(
        ["deleg_timeout"],
        parent_session_id="parent-timeout",
        timeout_seconds=1,
    )

    assert result["status"] == "ready"
    assert result["results"][0]["state"] == "timeout"
    assert result["results"][0]["result"]["summary"] == "deadline exceeded"


def test_return_when_any_leaves_unfinished_result_available():
    _dispatch("deleg_ready", "parent-any")
    _dispatch("deleg_running", "parent-any")
    _complete("deleg_ready", "first")

    result = ad.wait_for_delegations(
        ["deleg_ready", "deleg_running"],
        parent_session_id="parent-any",
        return_when="any",
        timeout_seconds=1,
    )

    assert [item["delegation_id"] for item in result["results"]] == ["deleg_ready"]
    assert result["still_running"] == ["deleg_running"]
    assert ad.get_durable_delegation("deleg_running")["delivery_state"] == "pending"


def test_wait_rejects_foreign_parent_without_consuming():
    _dispatch("deleg_foreign", "real-parent")
    _complete("deleg_foreign")

    result = ad.wait_for_delegations(
        ["deleg_foreign"], parent_session_id="other-parent", timeout_seconds=1
    )

    assert "not owned" in result["error"]
    assert ad.get_durable_delegation("deleg_foreign")["delivery_state"] == "pending"


def test_wait_accepts_owner_from_parent_compression_lineage():
    _dispatch("deleg_before_compression", "parent-root")
    _complete("deleg_before_compression", "lineage-visible")
    session_db = SimpleNamespace(
        get_compression_lineage=lambda _session_id: ["parent-root", "parent-tip"]
    )
    parent = SimpleNamespace(
        session_id="parent-tip",
        _session_db=session_db,
        _interrupt_requested=False,
    )

    result = json.loads(
        delegate_task(
            action="wait",
            delegation_ids=["deleg_before_compression"],
            timeout_seconds=10,
            parent_agent=parent,
        )
    )

    assert result["results"][0]["result"]["summary"] == "lineage-visible"


def test_wait_is_exposed_through_delegate_task_schema_and_handler():
    _dispatch("deleg_model", "parent-model")
    _complete("deleg_model", "model-visible")
    parent = SimpleNamespace(session_id="parent-model", _interrupt_requested=False)

    result = json.loads(
        delegate_task(
            action="wait",
            delegation_ids=["deleg_model"],
            timeout_seconds=10,
            parent_agent=parent,
        )
    )

    properties = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
    assert "wait" in properties["action"]["enum"]
    assert properties["timeout_seconds"]["maximum"] == 3600
    assert result["results"][0]["result"]["summary"] == "model-visible"
