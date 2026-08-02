"""Focused contracts for parent-scoped async delegation controls."""

from __future__ import annotations

import contextvars
import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tools import async_delegation as ad


@pytest.fixture(autouse=True)
def _clean_registry():
    ad._reset_for_tests()
    yield
    ad._reset_for_tests()


@pytest.fixture
def profile_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-a"))
    return tmp_path


def _db(root: Path, monkeypatch: pytest.MonkeyPatch, profile: str):
    from hermes_state import SessionDB

    home = root / profile
    monkeypatch.setenv("HERMES_HOME", str(home))
    return SessionDB(home / "state.db")


def _bind_session(session_id: str):
    from gateway.session_context import set_session_vars

    return set_session_vars(session_id=session_id)


def _create_session(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    session_id: str,
    **kwargs,
) -> None:
    db = _db(root, monkeypatch, profile)
    try:
        db.create_session(session_id, source=kwargs.pop("source", "cli"), **kwargs)
    finally:
        db.close()


def _current_owner(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: str = "profile-a",
    session_id: str = "parent",
) -> tuple[str, str]:
    _create_session(root, monkeypatch, profile, session_id)
    _bind_session(session_id)
    owner = ad._current_control_owner()
    assert owner is not None
    return owner


def _record(delegation_id: str, owner: tuple[str, str], **changes):
    record = {
        "delegation_id": delegation_id,
        "owner_profile": owner[0],
        "owner_session_id": owner[1],
        "status": "running",
        "goal": "private goal",
        "role": "leaf",
        "context": "private context",
        "session_key": "route",
        "interrupt_fn": None,
        "steer_fn": None,
        "progress_fn": None,
        "dispatched_at": 1.0,
        "delivery_state": "pending",
        "delivery_claim": "claim",
    }
    record.update(changes)
    with ad._records_lock:
        ad._records[delegation_id] = record
    return record


def _context_thread(target):
    context = contextvars.copy_context()
    return threading.Thread(target=lambda: context.run(target), daemon=True)


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def _delegate_env(monkeypatch, children):
    import tools.delegate_tool as dt

    child_iter = iter(children)
    monkeypatch.setattr(dt, "_build_child_agent", lambda **_: next(child_iter))
    monkeypatch.setattr(
        dt,
        "_resolve_delegation_credentials",
        lambda *_: {
            "model": "m",
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "command": None,
            "args": None,
        },
    )
    monkeypatch.setattr(dt, "_get_max_concurrent_children", lambda: 2)
    monkeypatch.setattr(dt, "_get_max_async_children", lambda: 2)
    monkeypatch.setattr(
        "gateway.session_context.async_delivery_supported", lambda: True
    )
    parent = MagicMock(
        session_id="parent",
        _delegate_depth=0,
        _interrupt_requested=False,
        _active_children=[],
        _active_children_lock=None,
    )
    return dt, parent


def _real_child(run):
    child = MagicMock(
        _credential_pool=None,
        session_id="",
        model="m",
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_estimated_cost_usd=0,
        tool_progress_callback=None,
    )
    child.run_conversation.side_effect = run
    child.get_activity_summary.return_value = {"api_call_count": 1}
    return child


def test_compression_only_owner_survives_rotation_and_rejects_noncontinuations(
    profile_root, monkeypatch
):
    db = _db(profile_root, monkeypatch, "profile-a")
    try:
        db.create_session("A", source="cli")
        db.end_session("A", "compression")
        db.create_session("B", source="cli", parent_session_id="A")
        db.create_session(
            "branch",
            source="cli",
            parent_session_id="A",
            model_config={"_branched_from": "A"},
        )
        db.create_session(
            "delegate",
            source="delegate",
            parent_session_id="A",
            model_config={"_delegate_from": "A"},
        )
        db.create_session("tool", source="tool", parent_session_id="A")
    finally:
        db.close()

    _bind_session("A")
    owner = ad._current_control_owner()
    assert owner is not None
    effects = []
    _record(
        "deleg_a11ce001",
        owner,
        interrupt_fn=lambda: effects.append("interrupt"),
        steer_fn=lambda message: effects.append(message) or True,
    )

    _bind_session("B")
    assert [
        item["delegation_id"]
        for item in ad.list_owned_async_delegations()["delegations"]
    ] == ["deleg_a11ce001"]
    assert ad.steer_owned_async_delegation("deleg_a11ce001", "focus") == {
        "status": "accepted"
    }
    assert ad.cancel_owned_async_delegation("deleg_a11ce001") == {"status": "accepted"}
    assert effects == ["focus", "interrupt"]

    for foreign_session in ("branch", "delegate", "tool"):
        _bind_session(foreign_session)
        assert ad.list_owned_async_delegations()["delegations"] == []
        assert ad.cancel_owned_async_delegation("deleg_a11ce001") == {
            "status": "not_found"
        }

    db = _db(profile_root, monkeypatch, "profile-a")
    try:
        db.create_session("sibling", source="cli", parent_session_id="A")
    finally:
        db.close()
    _bind_session("sibling")
    assert ad.list_owned_async_delegations()["delegations"] == []


def test_same_session_id_in_foreign_profile_is_opaque(profile_root, monkeypatch):
    owner = _current_owner(profile_root, monkeypatch)
    _record("deleg_a11ce001", owner, interrupt_fn=lambda: None, steer_fn=lambda _: True)

    _current_owner(profile_root, monkeypatch, "profile-b", "parent")
    assert ad.list_owned_async_delegations()["delegations"] == []
    assert ad.cancel_owned_async_delegation("deleg_a11ce001") == {"status": "not_found"}
    assert ad.steer_owned_async_delegation("deleg_a11ce001", "continue") == {
        "status": "not_found"
    }


def test_parentless_branch_delegate_and_tool_sessions_are_not_control_roots(
    profile_root, monkeypatch
):
    db = _db(profile_root, monkeypatch, "profile-a")
    try:
        db.create_session(
            "root-branch",
            source="cli",
            model_config={"_branched_from": "deleted-parent"},
        )
        db.create_session(
            "root-delegate",
            source="delegate",
            model_config={"_delegate_from": "deleted-parent"},
        )
        db.create_session("root-delegate-markerless", source="delegate")
        db.create_session("root-subagent-markerless", source="subagent")
        db.create_session("root-tool", source="tool")
        db.create_session(
            "root-null-branch-marker",
            source="cli",
            model_config={"_branched_from": None},
        )
        db.create_session(
            "root-null-delegate-marker",
            source="cli",
            model_config={"_delegate_from": None},
        )
        rejected = (
            "root-branch",
            "root-delegate",
            "root-delegate-markerless",
            "root-subagent-markerless",
            "root-tool",
            "root-null-branch-marker",
            "root-null-delegate-marker",
        )
        for session_id in rejected:
            assert db.get_compression_lineage_root(session_id) is None
    finally:
        db.close()

    for session_id in rejected:
        _bind_session(session_id)
        assert ad._current_control_owner() is None


def test_control_owner_opens_session_db_read_only(profile_root, monkeypatch):
    import hermes_state

    _create_session(profile_root, monkeypatch, "profile-a", "parent")
    _bind_session("parent")
    real_session_db = hermes_state.SessionDB
    read_only_values = []

    def session_db(*args, **kwargs):
        read_only_values.append(kwargs.get("read_only"))
        return real_session_db(*args, **kwargs)

    monkeypatch.setattr(hermes_state, "SessionDB", session_db)
    assert ad._current_control_owner() is not None
    assert read_only_values == [True]


def test_list_is_a_bounded_chooser_with_non_authorizing_pagination(
    profile_root, monkeypatch
):
    owner = _current_owner(profile_root, monkeypatch)
    for index in range(ad._MAX_OWNER_LIVE_SNAPSHOTS + 3):
        _record(
            f"deleg_{index:08x}",
            owner,
            goal=(f"goal-{index}-" + "x" * 500),
            role="orchestrator" + "x" * 500,
            dispatched_at=float(index),
        )
    _record("deleg_f0e1a001", (owner[0], "other"), dispatched_at=999.0)
    _record("deleg_malformed", owner, dispatched_at=998.0)

    first_page = ad.list_owned_async_delegations()
    assert first_page["total_live"] == ad._MAX_OWNER_LIVE_SNAPSHOTS + 3
    assert first_page["truncated"] is True
    assert first_page["next_cursor"] == ad._MAX_OWNER_LIVE_SNAPSHOTS
    assert len(first_page["delegations"]) == ad._MAX_OWNER_LIVE_SNAPSHOTS
    assert all(
        set(item)
        == {"delegation_id", "status", "is_batch", "dispatched_at", "goal", "role"}
        and len(item["goal"]) <= ad._MAX_OWNER_GOAL_CHARS
        and len(item["role"]) <= ad._MAX_OWNER_ROLE_CHARS
        and item["delegation_id"] != "deleg_f0e1a001"
        for item in first_page["delegations"]
    )

    second_page = ad.list_owned_async_delegations(cursor=first_page["next_cursor"])
    assert second_page["total_live"] == ad._MAX_OWNER_LIVE_SNAPSHOTS + 3
    assert second_page["truncated"] is False
    assert second_page["next_cursor"] is None
    assert [item["delegation_id"] for item in second_page["delegations"]] == [
        "deleg_00000002",
        "deleg_00000001",
        "deleg_00000000",
    ]


def test_model_dispatch_exposes_only_paged_safe_metadata(profile_root, monkeypatch):
    import model_tools

    owner = _current_owner(profile_root, monkeypatch)
    _record("deleg_a11ce001", owner, goal="choose me", role="leaf")
    result = json.loads(
        model_tools.handle_function_call(
            "delegate_control",
            {"action": "list", "cursor": 0},
            session_id="parent",
            skip_pre_tool_call_hook=True,
            skip_tool_execution_middleware=True,
        )
    )
    assert result == {
        "status": "ok",
        "delegations": [
            {
                "delegation_id": "deleg_a11ce001",
                "status": "running",
                "is_batch": False,
                "dispatched_at": 1.0,
                "goal": "choose me",
                "role": "leaf",
            }
        ],
        "total_live": 1,
        "truncated": False,
        "next_cursor": None,
    }


def test_control_tool_is_deferred_minimal_and_excluded_from_leaf_surface():
    import model_tools
    from tools.delegate_tool import DELEGATE_CONTROL_SCHEMA
    from tools.registry import registry
    from toolsets import _HERMES_CORE_TOOLS

    properties = DELEGATE_CONTROL_SCHEMA["parameters"]["properties"]
    assert set(properties) == {"action", "id", "message", "cursor"}
    assert properties["id"]["maxLength"] == ad._MAX_DELEGATION_ID_CHARS
    assert "delegate_control" not in _HERMES_CORE_TOOLS
    control_entry = registry.get_entry("delegate_control")
    delegate_entry = registry.get_entry("delegate_task")
    assert control_entry is not None
    assert delegate_entry is not None
    assert control_entry.check_fn is delegate_entry.check_fn

    raw = model_tools.get_tool_definitions(
        ["delegation"], quiet_mode=True, skip_tool_search_assembly=True
    )
    normal = model_tools.get_tool_definitions(["delegation"], quiet_mode=True)
    leaf = model_tools.get_tool_definitions(
        ["delegation"],
        ["delegation"],
        quiet_mode=True,
        skip_tool_search_assembly=True,
    )
    assert "delegate_control" in {item["function"]["name"] for item in raw}
    assert "delegate_control" not in {
        item["function"]["name"] for item in normal + leaf
    }
    assert "tool_search" in {item["function"]["name"] for item in normal}
    search_result = json.loads(
        model_tools.handle_function_call(
            "tool_search",
            {"query": "delegate control"},
            enabled_toolsets=["delegation"],
        )
    )
    assert "delegate_control" in {item["name"] for item in search_result["matches"]}


def test_steering_bounds_charge_only_accepted_messages(profile_root, monkeypatch):
    owner = _current_owner(profile_root, monkeypatch)
    calls = []
    record = _record(
        "deleg_a11ce001",
        owner,
        steer_fn=lambda message: calls.append(message) or False,
    )

    oversized = "x" * (ad._MAX_DELEGATION_ID_CHARS + 1)
    huge = "🍣" * (ad._MAX_STEER_MESSAGE_CHARS + 1)
    assert ad.steer_owned_async_delegation("deleg_a11ce001", " ") == {
        "status": "rejected"
    }
    assert ad.steer_owned_async_delegation("deleg_a11ce001", huge) == {
        "status": "rejected"
    }
    assert ad.cancel_owned_async_delegation(oversized) == {"status": "not_found"}
    assert ad.steer_owned_async_delegation(oversized, "ok") == {"status": "not_found"}
    for malformed in ("deleg_owned", "deleg_ABCDEF12", "deleg_1234567g"):
        assert ad.cancel_owned_async_delegation(malformed) == {"status": "not_found"}
        assert ad.steer_owned_async_delegation(malformed, "ok") == {
            "status": "not_found"
        }
    assert ad.steer_owned_async_delegation("deleg_a11ce001", "try") == {
        "status": "rejected"
    }
    assert record.get("_steer_message_count", 0) == 0

    record["steer_fn"] = lambda message: calls.append(message) or True
    assert ad.steer_owned_async_delegation("deleg_a11ce001", "ok") == {
        "status": "accepted"
    }
    assert record["_steer_message_count"] == 1
    assert record["_steer_character_count"] == 2
    record["_steer_message_count"] = ad._MAX_STEER_MESSAGES
    assert ad.steer_owned_async_delegation("deleg_a11ce001", "again") == {
        "status": "rejected"
    }
    record["_steer_message_count"] = 1
    record["_steer_character_count"] = ad._MAX_STEER_TOTAL_CHARS - 1
    assert ad.steer_owned_async_delegation("deleg_a11ce001", "xx") == {
        "status": "rejected"
    }
    assert calls == ["try", "ok"]


def test_control_callbacks_are_serialized_without_holding_registry_lock(
    profile_root, monkeypatch
):
    owner = _current_owner(profile_root, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    effects = []

    def interrupt():
        effects.append(("interrupt", ad._records_lock.locked()))
        entered.set()
        release.wait(2)

    _record(
        "deleg_a11ce001",
        owner,
        interrupt_fn=interrupt,
        steer_fn=lambda message: (
            effects.append((message, ad._records_lock.locked())) or True
        ),
    )
    result = []
    thread = _context_thread(
        lambda: result.append(ad.cancel_owned_async_delegation("deleg_a11ce001"))
    )
    thread.start()
    assert entered.wait(2)
    assert ad.steer_owned_async_delegation("deleg_a11ce001", "late") == {
        "status": "pending"
    }
    assert effects == [("interrupt", False)]
    release.set()
    thread.join(2)
    assert result == [{"status": "accepted"}]
    assert ad.steer_owned_async_delegation("deleg_a11ce001", "late") == {
        "status": "rejected"
    }


def test_steer_first_serializes_competing_cancel_then_cancel_retry(
    profile_root, monkeypatch
):
    owner = _current_owner(profile_root, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    effects = []

    def steer(message):
        effects.append(message)
        entered.set()
        release.wait(2)
        return True

    _record(
        "deleg_a11ce001",
        owner,
        interrupt_fn=lambda: effects.append("interrupt"),
        steer_fn=steer,
    )
    result = []
    thread = _context_thread(
        lambda: result.append(
            ad.steer_owned_async_delegation("deleg_a11ce001", "focus")
        )
    )
    thread.start()
    assert entered.wait(2)
    assert ad.cancel_owned_async_delegation("deleg_a11ce001") == {"status": "pending"}
    assert effects == ["focus"]
    release.set()
    thread.join(2)
    assert result == [{"status": "accepted"}]
    assert ad.cancel_owned_async_delegation("deleg_a11ce001") == {"status": "accepted"}
    assert effects == ["focus", "interrupt"]
    assert ad.steer_owned_async_delegation("deleg_a11ce001", "late") == {
        "status": "rejected"
    }


def test_interrupt_exception_is_indeterminate_and_cannot_be_reissued(
    profile_root, monkeypatch
):
    owner = _current_owner(profile_root, monkeypatch)
    effects = []

    def interrupt():
        effects.append("issued")
        raise RuntimeError("side effect happened before transport failed")

    _record(
        "deleg_a11ce001",
        owner,
        interrupt_fn=interrupt,
        steer_fn=lambda _: pytest.fail("steer"),
    )
    assert ad.cancel_owned_async_delegation("deleg_a11ce001") == {
        "status": "indeterminate"
    }
    assert ad.cancel_owned_async_delegation("deleg_a11ce001") == {
        "status": "indeterminate"
    }
    assert ad.steer_owned_async_delegation("deleg_a11ce001", "late") == {
        "status": "rejected"
    }
    assert effects == ["issued"]


def test_two_child_partial_interrupt_failure_remains_indeterminate(
    profile_root, monkeypatch
):
    child_one = _real_child(lambda **_: {"final_response": "ok", "completed": True})
    child_two = _real_child(lambda **_: {"final_response": "ok", "completed": True})
    child_two.interrupt.side_effect = RuntimeError("transport failed")
    pending = []
    monkeypatch.setattr(
        ad,
        "_get_executor",
        lambda _: MagicMock(submit=lambda callback: pending.append(callback)),
    )
    _current_owner(profile_root, monkeypatch)
    dt, parent = _delegate_env(monkeypatch, [child_one, child_two])
    delegation_id = json.loads(
        dt.delegate_task(
            tasks=[{"goal": "one"}, {"goal": "two"}],
            background=True,
            parent_agent=parent,
        )
    )["delegation_id"]

    assert ad.cancel_owned_async_delegation(delegation_id) == {
        "status": "indeterminate"
    }
    assert ad.cancel_owned_async_delegation(delegation_id) == {
        "status": "indeterminate"
    }
    assert ad.steer_owned_async_delegation(delegation_id, "late") == {
        "status": "rejected"
    }
    assert child_one.interrupt.call_count == 1
    assert child_two.interrupt.call_count == 1
    assert pending


def test_blocked_batch_callback_does_not_hold_the_lifecycle_lock(
    profile_root, monkeypatch
):
    child_started, child_release, interrupt_entered, interrupt_release = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )

    def run(**_):
        child_started.set()
        child_release.wait(2)
        return {"final_response": "ok", "completed": True}

    child = _real_child(run)
    child.interrupt.side_effect = lambda _: (
        interrupt_entered.set(),
        interrupt_release.wait(2),
    )
    _current_owner(profile_root, monkeypatch)
    dt, parent = _delegate_env(monkeypatch, [child])
    delegation_id = json.loads(
        dt.delegate_task(goal="one", background=True, parent_agent=parent)
    )["delegation_id"]
    assert child_started.wait(2)

    cancellation = []
    thread = _context_thread(
        lambda: cancellation.append(ad.cancel_owned_async_delegation(delegation_id))
    )
    thread.start()
    assert interrupt_entered.wait(2)
    child_release.set()
    assert _wait_until(
        lambda: (
            ad._records.get(delegation_id, {}).get("status")
            not in {"running", "stalling"}
        )
    )
    interrupt_release.set()
    thread.join(2)
    assert cancellation == [{"status": "not_found"}]


@pytest.mark.parametrize("interrupt_kind", ["session", "all"])
def test_lifecycle_interrupt_supersedes_a_blocked_batch_steer(
    profile_root, monkeypatch, interrupt_kind
):
    child_started, child_release, steer_entered, steer_release = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )

    def run(**_):
        child_started.set()
        child_release.wait(2)
        return {"final_response": "ok", "completed": True}

    def steer(_message):
        steer_entered.set()
        steer_release.wait(2)
        return True

    child = _real_child(run)
    child.steer.side_effect = steer
    _current_owner(profile_root, monkeypatch)
    dt, parent = _delegate_env(monkeypatch, [child])
    delegation_id = json.loads(
        dt.delegate_task(goal="one", background=True, parent_agent=parent)
    )["delegation_id"]
    assert child_started.wait(2)

    steer_result = []
    steer_thread = _context_thread(
        lambda: steer_result.append(
            ad.steer_owned_async_delegation(delegation_id, "focus")
        )
    )
    steer_thread.start()
    assert steer_entered.wait(2)
    interrupted = (
        ad.interrupt_for_session(parent_session_id="parent")
        if interrupt_kind == "session"
        else ad.interrupt_all()
    )
    assert interrupted == 1
    child.interrupt.assert_called_once()

    steer_release.set()
    steer_thread.join(2)
    child_release.set()
    assert steer_result == [{"status": "rejected"}]


def test_foreign_and_terminal_controls_are_opaque(profile_root, monkeypatch):
    owner = _current_owner(profile_root, monkeypatch)
    terminal = _record("deleg_dead0001", owner, status="completed")
    _record("deleg_f0e1a001", (owner[0], "other"))
    before = dict(terminal)
    for delegation_id in ("deleg_dead0001", "deleg_f0e1a001", "deleg_0000ffff"):
        assert ad.cancel_owned_async_delegation(delegation_id) == {
            "status": "not_found"
        }
        assert ad.steer_owned_async_delegation(delegation_id, "continue") == {
            "status": "not_found"
        }
    assert terminal == before
