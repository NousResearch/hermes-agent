"""Unit tests for durable owner_session_key reclaim (#114909) — no tui_gateway.server import."""

from types import SimpleNamespace

from tools.delegate_tool_child_run import _register_child
from tools.delegate_tool_registry import (
    _active_subagents,
    _active_subagents_lock,
    _unregister_subagent,
    reclaim_subagent_owners_for_session,
    rewrite_subagent_owner_session_keys,
)


def test_reclaim_rebinds_ui_sid_for_matching_durable_key_only():
    transport = object()
    owner = {"session_key": "durable-parent", "transport": transport}
    foreign = {"session_key": "other-parent", "transport": transport}
    child = SimpleNamespace(_subagent_id="child", _delegate_depth=1, model="test")
    _register_child(
        child,
        None,
        "owned",
        owner_session_id="ui-old",
        owner_transport=transport,
        owner_session_record=owner,
        owner_session_key="durable-parent",
    )
    try:
        reclaim_subagent_owners_for_session("ui-new", owner)
        with _active_subagents_lock:
            record = _active_subagents["child"]
        assert record["owner_session_id"] == "ui-new"
        assert record["owner_session_record"] is owner

        reclaim_subagent_owners_for_session("ui-foreign", foreign)
        with _active_subagents_lock:
            record = _active_subagents["child"]
        # Foreign key must not steal the child.
        assert record["owner_session_id"] == "ui-new"
        assert record["owner_session_record"] is owner
    finally:
        _unregister_subagent("child")


def test_rewrite_owner_session_key_follows_compression_rotation():
    transport = object()
    owner = {"session_key": "parent-v1", "transport": transport}
    child = SimpleNamespace(_subagent_id="child", _delegate_depth=1, model="test")
    _register_child(
        child,
        None,
        "owned",
        owner_session_id="ui-owner",
        owner_transport=transport,
        owner_session_record=owner,
        owner_session_key="parent-v1",
    )
    try:
        rewrite_subagent_owner_session_keys("parent-v1", "parent-v2")
        with _active_subagents_lock:
            record = _active_subagents["child"]
        assert record["owner_session_key"] == "parent-v2"
        owner["session_key"] = "parent-v2"
        reclaim_subagent_owners_for_session("ui-rotated", owner)
        with _active_subagents_lock:
            record = _active_subagents["child"]
        assert record["owner_session_id"] == "ui-rotated"
    finally:
        _unregister_subagent("child")


def test_missing_owner_session_key_is_not_reclaimed_by_empty_match():
    transport = object()
    owner = {"session_key": "durable-parent", "transport": transport}
    child = SimpleNamespace(_subagent_id="child", _delegate_depth=1, model="test")
    _register_child(
        child,
        None,
        "owned",
        owner_session_id="ui-old",
        owner_transport=transport,
        owner_session_record=owner,
        owner_session_key="",
    )
    try:
        reclaim_subagent_owners_for_session("ui-new", owner)
        with _active_subagents_lock:
            record = _active_subagents["child"]
        # Empty key must not match; exact sid remains frozen.
        assert record["owner_session_id"] == "ui-old"
    finally:
        _unregister_subagent("child")
