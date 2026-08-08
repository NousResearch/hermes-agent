"""Regression tests for dispatch-time async delegation routing."""

import json
from unittest.mock import MagicMock


def test_background_delegation_captures_threaded_gateway_route(monkeypatch):
    """Detached completion routing must retain the exact inbound thread."""
    import tools.delegate_tool as delegate_tool
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.approval import reset_current_session_key, set_current_session_key

    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "parent-session"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = None

    child = MagicMock()
    child._delegate_role = "leaf"
    child.get_activity_summary.return_value = {}

    credentials = {
        "model": "m",
        "provider": None,
        "base_url": None,
        "api_key": None,
        "api_mode": None,
        "command": None,
        "args": None,
    }
    captured = {}

    def capture_dispatch(**kwargs):
        captured.update(kwargs)
        return {"status": "dispatched", "delegation_id": "deleg_route"}

    monkeypatch.setattr(delegate_tool, "_build_child_agent", lambda **kwargs: child)
    monkeypatch.setattr(
        delegate_tool,
        "_resolve_delegation_credentials",
        lambda *args, **kwargs: credentials,
    )
    monkeypatch.setattr(
        "tools.async_delegation.dispatch_async_delegation_batch",
        capture_dispatch,
    )

    approval_token = set_current_session_key(
        "agent:main:buzz:group:channel-id:user-id"
    )
    session_tokens = set_session_vars(
        platform="buzz",
        source="buzz",
        chat_id="channel-id",
        chat_type="group",
        thread_id="thread-root-event-id",
        user_id="user-id",
        user_name="cmyk",
        session_key="agent:main:buzz:group:channel-id:user-id",
        message_id="triggering-event-id",
        profile="work",
    )
    try:
        result = delegate_tool.delegate_task(
            goal="audit branches",
            background=True,
            parent_agent=parent,
        )
    finally:
        reset_current_session_key(approval_token)
        clear_session_vars(session_tokens)

    assert json.loads(result)["status"] == "dispatched"
    assert captured["origin_route"] == {
        "platform": "buzz",
        "chat_id": "channel-id",
        "chat_type": "group",
        "thread_id": "thread-root-event-id",
        "user_id": "user-id",
        "user_name": "cmyk",
        "message_id": "triggering-event-id",
        "profile": "work",
    }


def test_completion_route_outranks_stale_session_origin():
    """A persisted channel origin must not discard a captured thread route."""
    from types import SimpleNamespace
    from typing import Any, cast

    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    session_key = "agent:main:slack:group:channel-id:user-id"
    stale_origin = SessionSource(
        platform=Platform("slack"),
        chat_id="channel-id",
        chat_type="group",
        user_id="user-id",
        thread_id=None,
    )
    runner = cast(Any, object.__new__(GatewayRunner))
    runner.session_store = SimpleNamespace(
        _ensure_loaded=lambda: None,
        _entries={session_key: SimpleNamespace(origin=stale_origin)},
    )
    runner._get_cached_session_source = lambda session_key: None

    source = runner._build_process_event_source(
        {
            "type": "async_delegation",
            "session_key": session_key,
            "platform": "slack",
            "chat_id": "channel-id",
            "chat_type": "group",
            "thread_id": "thread-root-event-id",
            "user_id": "user-id",
            "user_name": "cmyk",
            "message_id": "triggering-event-id",
            "profile": "work",
        }
    )

    assert source is not None
    assert source.thread_id == "thread-root-event-id"
    assert source.user_id == "user-id"
    assert source.profile == "work"


def test_batch_completion_event_carries_dispatch_route(monkeypatch):
    """The production batch happy path must publish the captured route."""
    from tools import async_delegation as ad
    from tools.process_registry import process_registry

    monkeypatch.setattr(ad, "_persist_completion", lambda *args: None)
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()

    try:
        ad._push_batch_completion_event(
            {
                "delegation_id": "deleg_route",
                "session_key": "agent:main:buzz:group:channel-id:user-id",
                "origin_route": {
                    "platform": "buzz",
                    "chat_id": "channel-id",
                    "chat_type": "group",
                    "thread_id": "thread-root-event-id",
                    "user_id": "user-id",
                    "message_id": "triggering-event-id",
                },
            },
            {"results": [{"status": "completed", "summary": "done"}]},
            "completed",
        )
        evt = process_registry.completion_queue.get_nowait()
    finally:
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()

    assert evt["platform"] == "buzz"
    assert evt["chat_id"] == "channel-id"
    assert evt["chat_type"] == "group"
    assert evt["thread_id"] == "thread-root-event-id"
    assert evt["user_id"] == "user-id"
    assert evt["message_id"] == "triggering-event-id"


def test_explicit_completion_route_overlays_persisted_origin():
    """Thread freshness must not discard multiplex/workspace routing fields."""
    from types import SimpleNamespace
    from typing import Any, cast

    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    session_key = "agent:main:slack:group:channel-id:user-id"
    persisted_origin = SessionSource(
        platform=Platform("slack"),
        chat_id="channel-id",
        chat_name="Engineering",
        chat_type="group",
        user_id="old-user",
        user_name="Old Name",
        thread_id=None,
        scope_id="workspace-id",
        parent_chat_id="parent-channel-id",
        message_id="old-message-id",
        profile="work",
    )
    runner = cast(Any, object.__new__(GatewayRunner))
    runner.session_store = SimpleNamespace(
        _ensure_loaded=lambda: None,
        _entries={session_key: SimpleNamespace(origin=persisted_origin)},
    )
    runner._get_cached_session_source = lambda session_key: None

    source = runner._build_process_event_source(
        {
            "type": "async_delegation",
            "session_key": session_key,
            "platform": "slack",
            "chat_id": "channel-id",
            "chat_type": "group",
            "thread_id": "fresh-thread-id",
            "user_id": "fresh-user",
            "user_name": "Fresh Name",
            "message_id": "fresh-message-id",
        }
    )

    assert source is not None
    assert source.thread_id == "fresh-thread-id"
    assert source.user_id == "fresh-user"
    assert source.user_name == "Fresh Name"
    assert source.message_id == "fresh-message-id"
    assert source.profile == "work"
    assert source.scope_id == "workspace-id"
    assert source.chat_name == "Engineering"
    assert source.parent_chat_id == "parent-channel-id"


def test_explicit_completion_route_overlays_cached_origin_on_store_miss():
    """A cache-only origin must retain workspace routing metadata too."""
    from types import SimpleNamespace
    from typing import Any, cast

    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    cached_origin = SessionSource(
        platform=Platform("slack"),
        chat_id="channel-id",
        chat_name="Engineering",
        chat_type="group",
        scope_id="workspace-id",
        parent_chat_id="parent-channel-id",
        profile="work",
    )
    runner = cast(Any, object.__new__(GatewayRunner))
    runner.session_store = SimpleNamespace(
        _ensure_loaded=lambda: None,
        _entries={},
    )
    runner._get_cached_session_source = lambda session_key: cached_origin

    source = runner._build_process_event_source(
        {
            "type": "async_delegation",
            "session_key": "agent:main:slack:group:channel-id:user-id",
            "platform": "slack",
            "chat_id": "channel-id",
            "chat_type": "group",
            "thread_id": "fresh-thread-id",
            "message_id": "fresh-message-id",
            "profile": "work",
        }
    )

    assert source is not None
    assert source.thread_id == "fresh-thread-id"
    assert source.scope_id == "workspace-id"
    assert source.chat_name == "Engineering"
    assert source.parent_chat_id == "parent-channel-id"
