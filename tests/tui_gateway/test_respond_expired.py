"""Tests for the blocking prompt factory (_block) and respond (_respond) functions.

Covers:
- _respond returns expired status when allow_expired=True and request is missing
- _respond returns 4009 error when allow_expired=False (default) and request is missing
- _respond works normally when request IS in _pending
- _block emits {event}.expire on timeout
- _block does NOT emit expire on normal answer
"""

import threading
from unittest.mock import patch

import pytest

from tui_gateway.server import (
    _answers,
    _block,
    _ok,
    _err,
    _pending,
    _pending_prompt_payloads,
    _prompt_lock,
    _respond,
)


@pytest.fixture(autouse=True)
def _clean_prompt_state():
    """Ensure _pending and _answers are clean before each test."""
    with _prompt_lock:
        _pending.clear()
        _answers.clear()
        _pending_prompt_payloads.clear()
    yield
    with _prompt_lock:
        _pending.clear()
        _answers.clear()
        _pending_prompt_payloads.clear()


class TestRespondExpired:
    """_respond with allow_expired handles stale clarify answers gracefully."""

    def test_stale_clarify_returns_expired_status(self):
        """When allow_expired=True and request_id is missing, return ok with status=expired."""
        result = _respond("r1", {"request_id": "nonexistent"}, "answer", allow_expired=True)
        assert result == _ok("r1", {"status": "expired"})

    def test_stale_default_returns_4009(self):
        """When allow_expired=False (default) and request_id is missing, return 4009."""
        result = _respond("r2", {"request_id": "nonexistent"}, "answer")
        assert result == _err("r2", 4009, "no pending answer request")

    def test_stale_sudo_returns_4009(self):
        """sudo.respond does NOT pass allow_expired, so stale requests get 4009."""
        result = _respond("r3", {"request_id": "nonexistent"}, "password")
        assert result == _err("r3", 4009, "no pending password request")

    def test_normal_answer_succeeds(self):
        """When the request IS in _pending, _respond sets the answer and signals."""
        ev = threading.Event()
        with _prompt_lock:
            _pending["rid123"] = ("sid1", ev)

        result = _respond("r4", {"request_id": "rid123", "answer": "hello"}, "answer")
        assert result == _ok("r4", {"status": "ok"})
        assert _answers.get("rid123") == "hello"
        assert ev.is_set()

    def test_normal_answer_with_allow_expired_succeeds(self):
        """Even with allow_expired=True, a valid request still works normally."""
        ev = threading.Event()
        with _prompt_lock:
            _pending["rid456"] = ("sid2", ev)

        result = _respond(
            "r5", {"request_id": "rid456", "answer": "world"}, "answer", allow_expired=True
        )
        assert result == _ok("r5", {"status": "ok"})
        assert _answers.get("rid456") == "world"
        assert ev.is_set()


class TestBlockExpire:
    """_block emits {event}.expire on timeout and does not emit on normal answer."""

    def test_timeout_emits_expire_event(self):
        """When _block times out, it should emit {event}.expire with the request_id."""
        emitted = []

        def fake_emit(event, sid, payload=None):
            emitted.append((event, sid, payload))

        with patch("tui_gateway.server._emit", side_effect=fake_emit):
            result = _block("clarify", "s1", {"question": "pick one", "choices": ["a", "b"]}, timeout=0.1)

        # Should have emitted the original clarify event and the expire event
        assert len(emitted) == 2
        assert emitted[0][0] == "clarify"
        assert emitted[1][0] == "clarify.expire"
        expire_payload = emitted[1][2]
        assert "request_id" in expire_payload
        # The answer should be empty (timeout)
        assert result == ""

    def test_normal_answer_no_expire_event(self):
        """When _block receives an answer before timeout, no expire event is emitted."""
        emitted = []

        def fake_emit(event, sid, payload=None):
            emitted.append((event, sid, payload))
            # When the original event is emitted, simulate a quick answer
            if event == "clarify":
                rid = payload.get("request_id", "")
                with _prompt_lock:
                    if rid in _pending:
                        _answers[rid] = "my answer"
                        _, ev = _pending[rid]
                        ev.set()

        with patch("tui_gateway.server._emit", side_effect=fake_emit):
            result = _block("clarify", "s2", {"question": "yes?", "choices": ["y", "n"]}, timeout=5)

        # Only the original event, no expire
        assert len(emitted) == 1
        assert emitted[0][0] == "clarify"
        assert result == "my answer"

    def test_expire_event_carries_correct_request_id(self):
        """The expire event payload must contain the same request_id as the original."""
        emitted = []

        def fake_emit(event, sid, payload=None):
            emitted.append((event, sid, payload))

        with patch("tui_gateway.server._emit", side_effect=fake_emit):
            _block("clarify", "s3", {"question": "pick"}, timeout=0.05)

        original_rid = emitted[0][2]["request_id"]
        expire_rid = emitted[1][2]["request_id"]
        assert original_rid == expire_rid
