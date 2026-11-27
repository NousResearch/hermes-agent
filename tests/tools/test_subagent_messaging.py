"""Tests for tools/subagent_messaging.py - in-process subagent message bus."""

import time

import pytest

from tools.subagent_messaging import (
    SubagentMessageBus,
    drain_inbox,
    peek_inbox,
    register,
    reset_bus,
    send_message,
    stats,
    unregister,
)


@pytest.fixture(autouse=True)
def _reset_global_bus():
    """Each test gets a clean global bus."""
    reset_bus()
    yield
    reset_bus()


# --- Construction & registration -----------------------------------------


def test_construct_rejects_bad_budgets():
    with pytest.raises(ValueError):
        SubagentMessageBus(max_messages=0)
    with pytest.raises(ValueError):
        SubagentMessageBus(ttl_seconds=0)


def test_register_is_idempotent_and_validates():
    bus = SubagentMessageBus()
    assert bus.register("a") is True
    assert bus.register("a") is True  # idempotent
    assert bus.is_registered("a")
    assert bus.register("") is False
    assert bus.register("   ") is False
    assert bus.is_registered("") is False


def test_unregister_returns_presence():
    bus = SubagentMessageBus()
    bus.register("a")
    assert bus.unregister("a") is True
    assert bus.unregister("a") is False
    assert bus.unregister("") is False


# --- send_message --------------------------------------------------------


def test_send_to_registered_recipient_succeeds():
    bus = SubagentMessageBus()
    bus.register("a")
    bus.register("b")
    ok, reason = bus.send_message("a", "b", "hello")
    assert ok is True
    assert reason == ""


def test_send_rejects_empty_inputs():
    bus = SubagentMessageBus()
    bus.register("a")
    bus.register("b")
    assert bus.send_message("", "b", "x")[0] is False
    assert bus.send_message("a", "", "x")[0] is False
    assert bus.send_message("a", "b", "")[0] is False


def test_send_rejects_oversize_message():
    bus = SubagentMessageBus()
    bus.register("a")
    bus.register("b")
    ok, reason = bus.send_message("a", "b", "x" * 70000)
    assert ok is False
    assert "too long" in reason


def test_send_rejects_unregistered_sender():
    bus = SubagentMessageBus()
    bus.register("b")
    ok, reason = bus.send_message("ghost", "b", "hi")
    assert ok is False
    assert "sender" in reason
    assert "ghost" in reason


def test_send_rejects_unregistered_recipient():
    bus = SubagentMessageBus()
    bus.register("a")
    ok, reason = bus.send_message("a", "ghost", "hi")
    assert ok is False
    assert "recipient" in reason


def test_send_message_with_kind_and_metadata():
    bus = SubagentMessageBus()
    bus.register("a")
    bus.register("b")
    ok, _ = bus.send_message("a", "b", "stop", kind="abort", metadata={"reason": "deadline"})
    assert ok
    msgs = bus.peek_inbox("b")
    assert msgs[0]["kind"] == "abort"
    assert msgs[0]["metadata"]["reason"] == "deadline"


# --- peek / drain --------------------------------------------------------


def test_peek_inbox_does_not_drain():
    bus = SubagentMessageBus()
    bus.register("a")
    bus.register("b")
    bus.send_message("a", "b", "one")
    bus.send_message("a", "b", "two")
    first = bus.peek_inbox("b")
    second = bus.peek_inbox("b")
    assert len(first) == 2
    assert len(second) == 2


def test_drain_inbox_returns_all_and_clears():
    bus = SubagentMessageBus()
    bus.register("a")
    bus.register("b")
    bus.send_message("a", "b", "one")
    bus.send_message("a", "b", "two")
    drained = bus.drain_inbox("b")
    assert len(drained) == 2
    assert bus.peek_inbox("b") == []


def test_peek_unknown_recipient_returns_empty():
    bus = SubagentMessageBus()
    bus.register("a")
    assert bus.peek_inbox("ghost") == []
    assert bus.drain_inbox("ghost") == []


def test_peek_respects_limit():
    bus = SubagentMessageBus()
    bus.register("a")
    bus.register("b")
    for i in range(10):
        bus.send_message("a", "b", f"msg-{i}")
    last3 = bus.peek_inbox("b", limit=3)
    assert len(last3) == 3
    assert last3[-1]["message"] == "msg-9"


def test_peek_respects_ttl():
    bus = SubagentMessageBus(ttl_seconds=1)
    bus.register("a")
    bus.register("b")
    bus.send_message("a", "b", "stale")
    time.sleep(1.2)
    msgs = bus.peek_inbox("b")
    assert msgs == []


def test_peek_with_drop_expired_false_keeps_expired():
    bus = SubagentMessageBus(ttl_seconds=1)
    bus.register("a")
    bus.register("b")
    bus.send_message("a", "b", "stale")
    time.sleep(1.2)
    msgs = bus.peek_inbox("b", drop_expired=False)
    # Expired but not dropped; consumer can decide what to do.
    assert len(msgs) == 1


# --- Budget enforcement --------------------------------------------------


def test_max_messages_caps_per_recipient():
    bus = SubagentMessageBus(max_messages=3)
    bus.register("a")
    bus.register("b")
    for i in range(10):
        bus.send_message("a", "b", f"m{i}")
    msgs = bus.peek_inbox("b")
    assert len(msgs) == 3
    # Oldest should be dropped; newest should remain.
    assert msgs[-1]["message"] == "m9"
    # Dropped counter incremented.
    s = bus.stats()
    assert s["dropped"] == 7


# --- Stats ---------------------------------------------------------------


def test_stats_reflects_activity():
    bus = SubagentMessageBus()
    bus.register("a")
    bus.register("b")
    bus.send_message("a", "b", "x")
    bus.send_message("a", "b", "y")
    bus.peek_inbox("b")
    s = bus.stats()
    assert s["recipients"] == 2
    assert s["sent"] == 2
    assert s["received"] == 2
    assert s["dropped"] == 0
    assert s["max_messages_per_recipient"] == 256
    assert s["ttl_seconds"] == 3600


def test_unregistered_recipients_dont_count():
    bus = SubagentMessageBus()
    bus.register("a")
    assert bus.stats()["recipients"] == 1
    bus.unregister("a")
    assert bus.stats()["recipients"] == 0


# --- Module-level wrappers -----------------------------------------------


def test_module_wrappers_use_global_bus():
    register("alpha")
    register("beta")
    ok, _ = send_message("alpha", "beta", "hello via globals")
    assert ok
    msgs = peek_inbox("beta")
    assert msgs[0]["message"] == "hello via globals"
    drained = drain_inbox("beta")
    assert drained[0]["message"] == "hello via globals"
    assert peek_inbox("beta") == []
    assert stats()["sent"] >= 1
    assert unregister("beta") is True


def test_reset_bus_clears_global_state():
    register("a")
    register("b")
    send_message("a", "b", "before reset")
    reset_bus()
    # After reset, 'a' is no longer registered.
    ok, reason = send_message("a", "b", "after reset")
    assert ok is False
    assert "not registered" in reason
