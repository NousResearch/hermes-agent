"""message.complete marks a final already delivered as an interim as previewed (#125951).

The Codex app-server bridge sends every completed agentMessage, the final one included,
through the interim path and never sets ``response_previewed``. Without the flag the
TUI and Desktop keep the sealed interim AND render the final: the reply shows twice.
"""

import contextlib
from types import SimpleNamespace

import tui_gateway.server as srv
from agent.stream_delivery import StreamDeliveryMixin


class _Agent(StreamDeliveryMixin):
    _session_title_hint = "Scratch"


def _turn(agent, text):
    return SimpleNamespace(
        result={"final_response": text}, agent=agent, terminal_callback=None,
        receipt_committed=True, receipt_attempted=False, marker_key="", error_retained=False,
        error_detail="", prompt_text="ping",
    )


def _payload(monkeypatch, agent, text):
    monkeypatch.setattr(srv, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(srv, "render_message", lambda _text, _cols: None)
    monkeypatch.setattr(srv, "_clear_inflight_turn", lambda _session: None)
    session = {"pending_title": None, "session_key": "k", "history_lock": contextlib.nullcontext(), "agent": agent}
    payload, _, _ = srv._complete_turn_payload(session, _turn(agent, text), None, 80)
    return payload


def test_final_delivered_as_interim_is_previewed(monkeypatch):
    agent = _Agent()
    agent._record_delivered_interim_text("Here is the answer.\n")

    payload = _payload(monkeypatch, agent, "Here is the answer.")

    assert payload["response_previewed"] is True
    assert srv._event_frame("message.complete", "sid", payload)["params"]["payload"]["response_previewed"] is True


def test_final_distinct_from_interim_commentary_is_not_previewed(monkeypatch):
    agent = _Agent()
    agent._record_delivered_interim_text("Looking at the config first.")

    assert "response_previewed" not in _payload(monkeypatch, agent, "Here is the answer.")
