"""Tests for tools/clarify_state.py -- gateway clarify bridge."""

import contextvars
import threading
import time

import pytest

from tools.clarify_state import (
    gateway_clarify_callback,
    get_clarify_session_key,
    has_blocking_clarify,
    register_clarify_notify,
    resolve_gateway_clarify,
    set_clarify_session_key,
    reset_clarify_session_key,
    unregister_clarify_notify,
    _gateway_queues,
    _lock,
)


class TestClarifySessionKey:
    """Test session key context management."""

    def test_default_key(self):
        key = get_clarify_session_key()
        assert key == "default"

    def test_set_and_get(self):
        token = set_clarify_session_key("sess_abc")
        try:
            assert get_clarify_session_key() == "sess_abc"
        finally:
            reset_clarify_session_key(token)

    def test_reset_restores_previous(self):
        token1 = set_clarify_session_key("first")
        token2 = set_clarify_session_key("second")
        try:
            assert get_clarify_session_key() == "second"
            reset_clarify_session_key(token2)
            assert get_clarify_session_key() == "first"
        finally:
            reset_clarify_session_key(token1)

    def test_empty_string_falls_through(self):
        token = set_clarify_session_key("")
        try:
            result = get_clarify_session_key("fallback")
            assert result == "fallback"
        finally:
            reset_clarify_session_key(token)


class TestRegisterNotify:
    """Test notify callback registration."""

    def test_register_and_unregister(self):
        register_clarify_notify("sess_x", lambda d: None)
        unregister_clarify_notify("sess_x")

    def test_unregister_wakes_blocked_threads(self):
        token = set_clarify_session_key("sess_y")
        register_clarify_notify("sess_y", lambda d: None)

        result = [None]
        # Each thread needs its own context copy
        ctx = contextvars.copy_context()

        def blocking():
            # Set the session key inside the thread directly
            inner_token = set_clarify_session_key("sess_y")
            try:
                result[0] = gateway_clarify_callback("Question?")
            finally:
                reset_clarify_session_key(inner_token)

        t = threading.Thread(target=blocking)
        t.daemon = True
        t.start()
        time.sleep(0.15)

        unregister_clarify_notify("sess_y")
        t.join(timeout=2)

        assert not t.is_alive()
        assert result[0] == ""
        reset_clarify_session_key(token)


class TestResolveGatewayClarify:
    """Test resolve_gateway_clarify."""

    def test_resolve_no_pending(self):
        count = resolve_gateway_clarify("nonexistent_session", "answer")
        assert count == 0

    def test_resolve_single(self):
        """Basic resolve of a single pending clarify."""
        token = set_clarify_session_key("test_single")
        register_clarify_notify("test_single", lambda d: None)

        result = [None]

        def blocking():
            inner_token = set_clarify_session_key("test_single")
            try:
                result[0] = gateway_clarify_callback("Q?")
            finally:
                reset_clarify_session_key(inner_token)

        t = threading.Thread(target=blocking)
        t.daemon = True
        t.start()
        time.sleep(0.15)

        count = resolve_gateway_clarify("test_single", "my_answer")
        assert count == 1
        t.join(timeout=2)
        assert result[0] == "my_answer"

        unregister_clarify_notify("test_single")
        reset_clarify_session_key(token)

    def test_resolve_with_clarify_id(self):
        """Resolve a specific entry by clarify_id."""
        token = set_clarify_session_key("test_id")
        register_clarify_notify("test_id", lambda d: None)

        results = [None, None]

        def block(idx):
            inner_token = set_clarify_session_key("test_id")
            try:
                gateway_clarify_callback(f"Q{idx}?")
                results[idx] = "done"
            finally:
                reset_clarify_session_key(inner_token)

        threads = []
        for i in range(2):
            t = threading.Thread(target=block, args=(i,))
            t.daemon = True
            t.start()
            threads.append(t)
            time.sleep(0.1)

        # Peek at the queue to get the second entry's clarify_id
        with _lock:
            queue = _gateway_queues.get("test_id", [])
            second_id = queue[1].clarify_id if len(queue) >= 2 else None

        # Resolve only the second entry by clarify_id
        count = resolve_gateway_clarify("test_id", "targeted_answer", second_id)
        assert count == 1
        threads[1].join(timeout=2)

        # First entry is still pending
        assert threads[0].is_alive()

        # Clean up
        resolve_gateway_clarify("test_id", "cleanup")
        threads[0].join(timeout=2)

        unregister_clarify_notify("test_id")
        reset_clarify_session_key(token)

    def test_resolve_fifo_multiple(self):
        """FIFO: resolve resolves the oldest pending entry first."""
        token = set_clarify_session_key("test_fifo")
        register_clarify_notify("test_fifo", lambda d: None)

        results = [None, None]

        def block(idx):
            inner_token = set_clarify_session_key("test_fifo")
            try:
                results[idx] = gateway_clarify_callback(f"Q{idx}?")
            finally:
                reset_clarify_session_key(inner_token)

        threads = []
        for i in range(2):
            t = threading.Thread(target=block, args=(i,))
            t.daemon = True
            t.start()
            threads.append(t)
            time.sleep(0.15)

        # Resolve the first one (oldest)
        count1 = resolve_gateway_clarify("test_fifo", "first_answer")
        assert count1 == 1
        threads[0].join(timeout=2)
        assert results[0] == "first_answer"

        # Resolve the second one
        count2 = resolve_gateway_clarify("test_fifo", "second_answer")
        assert count2 == 1
        threads[1].join(timeout=2)
        assert results[1] == "second_answer"

        unregister_clarify_notify("test_fifo")
        reset_clarify_session_key(token)


class TestGatewayClarifyCallback:
    """Test the blocking callback passed to AIAgent."""

    def test_callback_with_choices(self):
        token = set_clarify_session_key("cb_test")
        notify_data = [None]

        def notify(data):
            notify_data[0] = data

        register_clarify_notify("cb_test", notify)

        result = [None]

        def blocking():
            inner_token = set_clarify_session_key("cb_test")
            try:
                result[0] = gateway_clarify_callback("Pick one", ["A", "B", "C"])
            finally:
                reset_clarify_session_key(inner_token)

        t = threading.Thread(target=blocking)
        t.daemon = True
        t.start()
        time.sleep(0.15)

        assert notify_data[0] is not None
        assert notify_data[0]["question"] == "Pick one"
        assert notify_data[0]["choices"] == ["A", "B", "C"]
        assert notify_data[0]["clarify_id"].startswith("clr_")

        resolve_gateway_clarify("cb_test", "B")
        t.join(timeout=2)
        assert result[0] == "B"

        unregister_clarify_notify("cb_test")
        reset_clarify_session_key(token)

    def test_callback_open_ended(self):
        token = set_clarify_session_key("cb_open")
        register_clarify_notify("cb_open", lambda d: None)

        result = [None]

        def blocking():
            inner_token = set_clarify_session_key("cb_open")
            try:
                result[0] = gateway_clarify_callback("What's your name?")
            finally:
                reset_clarify_session_key(inner_token)

        t = threading.Thread(target=blocking)
        t.daemon = True
        t.start()
        time.sleep(0.15)

        resolve_gateway_clarify("cb_open", "Alice")
        t.join(timeout=2)
        assert result[0] == "Alice"

        unregister_clarify_notify("cb_open")
        reset_clarify_session_key(token)


class TestHasBlockingClarify:
    """Test has_blocking_clarify."""

    def test_no_pending(self):
        assert not has_blocking_clarify("nonexistent")

    def test_with_pending(self):
        token = set_clarify_session_key("has_test")
        register_clarify_notify("has_test", lambda d: None)

        def blocking():
            inner_token = set_clarify_session_key("has_test")
            try:
                gateway_clarify_callback("Q?")
            finally:
                reset_clarify_session_key(inner_token)

        t = threading.Thread(target=blocking)
        t.daemon = True
        t.start()
        time.sleep(0.15)

        assert has_blocking_clarify("has_test")

        resolve_gateway_clarify("has_test", "A")
        t.join(timeout=2)
        assert not has_blocking_clarify("has_test")

        unregister_clarify_notify("has_test")
        reset_clarify_session_key(token)
