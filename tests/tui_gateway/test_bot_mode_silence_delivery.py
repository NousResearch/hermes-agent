"""Bot Mode never renders or relays a bare intentional-silence marker (#110782).

Silence is a delivery decision shared with the gateway (``gateway/response_filters``):
the assistant row stays persisted, only the outbound text is emptied; failed turns and
prose that merely mentions a marker are delivered unchanged.
"""

import contextlib
from types import SimpleNamespace

import tui_gateway.server as srv


def _turn(result):
    return SimpleNamespace(
        result=result, agent=SimpleNamespace(_session_title_hint="Bot Chat"), terminal_callback=None,
        receipt_committed=True, receipt_attempted=False, marker_key="", error_retained=False,
        error_detail="", prompt_text="ping", gateway_system_event=None,
    )


def test_live_bot_chat_completion_empties_marker_only_for_successful_turns(monkeypatch):
    monkeypatch.setattr(srv, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(srv, "render_message", lambda _text, _cols: None)
    monkeypatch.setattr(srv, "_clear_inflight_turn", lambda _session: None)
    session = {"pending_title": None, "session_key": "k", "history_lock": contextlib.nullcontext(),
               "agent": SimpleNamespace(_session_title_hint="Bot Chat")}

    payload, _, status = srv._complete_turn_payload(session, _turn({"final_response": " *NO_REPLY* "}), None, 80)
    assert (status, payload["text"]) == ("complete", "")

    prose = "[SILENT] is mentioned here, but this is a real answer."
    payload, _, _ = srv._complete_turn_payload(session, _turn({"final_response": prose}), None, 80)
    assert payload["text"] == prose

    failed = {"final_response": "NO_REPLY", "error": "provider failed", "failed": True}
    payload, _, status = srv._complete_turn_payload(session, _turn(failed), None, 80)
    assert (status, payload["text"]) == ("error", "NO_REPLY")

    # A plain (non-Bot-Chat) desktop session keeps the marker: the gate is the canonical title.
    session["agent"] = SimpleNamespace(_session_title_hint="Scratch")
    monkeypatch.setattr(srv, "_session_live_title", lambda _s, _k: "Scratch")
    payload, _, _ = srv._complete_turn_payload(session, _turn({"final_response": "NO_REPLY"}), None, 80)
    assert payload["text"] == "NO_REPLY"


def test_live_bot_chat_stream_holds_back_partial_silence_marker(monkeypatch):
    """Mirror of stream_consumer's hold-back: a marker never reaches message.delta, prose that
    diverges from every marker is flushed intact once it diverges."""
    events = []
    monkeypatch.setattr(srv, "_emit", lambda event, _sid, payload=None: events.append((event, payload)))
    monkeypatch.setattr(srv, "_load_interim_assistant_messages", lambda: False)
    monkeypatch.setattr(srv, "_start_usage_ticker", lambda _sid, _agent: (SimpleNamespace(set=lambda: None), SimpleNamespace(join=lambda: None)))

    def _run(final, chunks):
        events.clear()

        def run_conversation(_message, **kwargs):
            for chunk in chunks:
                kwargs["stream_callback"](chunk)
            return {"final_response": final}

        agent = SimpleNamespace(_session_title_hint="Bot Chat", run_conversation=run_conversation)
        session = {"pending_title": None, "session_key": "k", "history_lock": contextlib.nullcontext(), "agent": agent}
        st = srv._TurnRun(agent=agent, one_turn_restore=None, terminal_callback=None, receipt_committed=True)
        srv._invoke_agent("sid", session, st, "ping", "ping", None, [], None, None)
        return [p["text"] for e, p in events if e == "message.delta"], (session.get("inflight_turn") or {}).get("assistant", "")

    assert _run("NO_REPLY", ["NO_", "REPLY"]) == ([], "")
    assert _run("NO way, here is the answer.", ["NO", " way,", " here is the answer."]) == (
        ["NO way,", " here is the answer."], "NO way, here is the answer.")
