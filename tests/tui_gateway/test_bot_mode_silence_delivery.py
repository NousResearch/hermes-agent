"""Regression coverage for Bot Mode's shared completed-reply delivery boundary."""

import contextlib
from types import SimpleNamespace

import pytest

import tui_gateway.server as srv
from tui_gateway.prompt_turn import _bot_mode_delivery_text, _is_bot_mode_session


@pytest.mark.parametrize("response", [
    "NO_REPLY", " [silent] ", "silent", "no reply", "*NO_REPLY*",
])
def test_bot_mode_delivery_hides_successful_canonical_silence_markers(response):
    assert _bot_mode_delivery_text(response, successful=True) == ""


@pytest.mark.parametrize("response", [
    "The NO_REPLY marker means do not answer.",
    "[SILENT] is mentioned here, but this is a real answer.",
])
def test_bot_mode_delivery_keeps_substantive_marker_mentions(response):
    assert _bot_mode_delivery_text(response, successful=True) == response


def test_bot_mode_delivery_keeps_failed_marker_response_visible():
    assert _bot_mode_delivery_text("NO_REPLY", successful=False) == "NO_REPLY"


def test_only_canonical_bot_chat_sessions_use_the_live_delivery_boundary():
    assert _is_bot_mode_session({"pending_title": "Bot Chat"})
    assert _is_bot_mode_session({"title": "Bot Chat"})
    assert not _is_bot_mode_session({"pending_title": "Scratch"})


def test_live_bot_chat_completion_suppresses_markers_but_failed_turns_fail_open(monkeypatch):
    """The prompt.submit completion path applies the shared delivery boundary."""
    monkeypatch.setattr(srv, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(srv, "render_message", lambda _text, _cols: None)
    monkeypatch.setattr(srv, "_clear_inflight_turn", lambda _session: None)

    session = {"pending_title": "Bot Chat", "history_lock": contextlib.nullcontext()}
    turn = SimpleNamespace(
        result={"final_response": "NO_REPLY"}, agent=object(), terminal_callback=None,
        receipt_committed=True, receipt_attempted=False, marker_key="", error_retained=False,
        error_detail="", prompt_text="ping",
    )
    payload, _, status = srv._complete_turn_payload(session, turn, None, 80)
    assert status == "complete"
    assert payload["text"] == ""

    turn.result = {"final_response": "NO_REPLY", "error": "provider failed", "failed": True}
    payload, _, status = srv._complete_turn_payload(session, turn, None, 80)
    assert status == "error"
    assert payload["text"] == "NO_REPLY"
