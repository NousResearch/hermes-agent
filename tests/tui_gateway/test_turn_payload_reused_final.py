"""message.complete names a final that reuses a response the turn already delivered.

The housekeeping-tool fallback (turn_empty_response, ``fallback_prior_turn_content``) ends a
turn on the answer the model streamed BEFORE its tool call. The renderer bounds "the current
response" at the bubble's last tool row, so without an identity it appended that answer again:
the same reply twice in one bubble, nothing between the copies when the tool is silent
(``todo_list``). ``response_reused`` is set where the reuse happens, never inferred from equal
text: a model that says the same words in two responses keeps both.
"""

import contextlib
from types import SimpleNamespace

import tui_gateway.server as srv
from agent.stream_delivery import StreamDeliveryMixin


class _Agent(StreamDeliveryMixin):
    _session_title_hint = "Scratch"


def _payload(monkeypatch, result):
    monkeypatch.setattr(srv, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(srv, "render_message", lambda _text, _cols: None)
    monkeypatch.setattr(srv, "_clear_inflight_turn", lambda _session: None)
    agent = _Agent()
    session = {"pending_title": None, "session_key": "k", "history_lock": contextlib.nullcontext(), "agent": agent}
    st = SimpleNamespace(
        result=result, agent=agent, terminal_callback=None, receipt_committed=True, receipt_attempted=False,
        marker_key="", error_retained=False, error_detail="", prompt_text="ping",
    )
    payload, _, _ = srv._complete_turn_payload(session, st, None, 80)
    return payload


def test_reused_final_is_named_on_the_wire(monkeypatch):
    payload = _payload(monkeypatch, {"final_response": "Here is the answer.", "response_previewed": True,
                                     "response_reused": True})
    assert payload["response_reused"] is True
    assert srv._event_frame("message.complete", "sid", payload)["params"]["payload"]["response_reused"] is True


def test_a_fresh_final_is_not_named_reused(monkeypatch):
    assert "response_reused" not in _payload(monkeypatch, {"final_response": "Here is the answer."})
