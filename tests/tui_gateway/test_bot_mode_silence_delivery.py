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
        error_detail="", prompt_text="ping",
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


def test_bot_chat_receipt_still_names_the_saved_row_when_a_trailing_marker_is_dropped(monkeypatch):
    """The delivered copy drops the marker; the stored row keeps it, and the receipt must
    still bind this turn to that row."""
    from agent.context_compressor import _DB_PERSISTED_MARKER
    monkeypatch.setattr(srv, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(srv, "render_message", lambda _text, _cols: None)
    monkeypatch.setattr(srv, "_clear_inflight_turn", lambda _session: None)
    session = {"pending_title": None, "session_key": "k", "history_lock": contextlib.nullcontext(),
               "agent": SimpleNamespace(_session_title_hint="Bot Chat")}
    raw = "Paris.\n\nNO_REPLY"
    rows = [{"role": "user", "content": "q", "_row_id": 7, _DB_PERSISTED_MARKER: True},
            {"role": "assistant", "content": raw, "_row_id": 8, _DB_PERSISTED_MARKER: True}]
    st = _turn({"final_response": raw, "messages": rows})
    st.agent._persist_user_message_idx = 0
    st.history, st.compression_count = [], None

    payload, _, _ = srv._complete_turn_payload(session, st, None, 80)

    assert payload["text"] == "Paris."
    assert payload["persisted_turn"]["final_assistant_row_id"] == 8


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


def test_live_bot_chat_stream_never_shows_or_speaks_a_closing_silence_marker(monkeypatch):
    """A marker closing a real reply is held line-wise: no delta and no TTS chunk carries it,
    while a last line that only starts like one ("NO problem", "NOTE: done", "NO") arrives whole."""
    events = []
    monkeypatch.setattr(srv, "_emit", lambda event, _sid, payload=None: events.append((event, payload)))
    monkeypatch.setattr(srv, "_load_interim_assistant_messages", lambda: False)
    monkeypatch.setattr(srv, "_start_usage_ticker", lambda _sid, _agent: (SimpleNamespace(set=lambda: None), SimpleNamespace(join=lambda: None)))

    def _run(chunks):
        events.clear()
        spoken = []

        def run_conversation(_message, **kwargs):
            for chunk in chunks:
                kwargs["stream_callback"](chunk)
            return {"final_response": "".join(chunks)}

        agent = SimpleNamespace(_session_title_hint="Bot Chat", run_conversation=run_conversation)
        session = {"pending_title": None, "session_key": "k", "history_lock": contextlib.nullcontext(), "agent": agent}
        st = srv._TurnRun(agent=agent, one_turn_restore=None, terminal_callback=None, receipt_committed=True)
        st.tts_queue = SimpleNamespace(put=spoken.append)
        srv._invoke_agent("sid", session, st, "ping", "ping", None, [], None, None)
        return [p["text"] for e, p in events if e == "message.delta"], spoken

    for chunks in (["Paris.", "\n\nNO", "_RE", "PLY"], ["Paris.\n", "\n[SIL", "ENT]\n"]):
        deltas, spoken = _run(chunks)
        assert "".join(deltas).strip() == "Paris."
        assert "".join(spoken).strip() == "Paris."
        assert not any(c in "".join(deltas + spoken) for c in ("NO", "SIL", "["))

    for tail in ("NO problem", "NOTE: done", "NO"):
        chunks = ["Paris.", "\n\nNO", tail[2:]] if len(tail) > 2 else ["Paris.", "\n\nNO"]
        deltas, spoken = _run(chunks)
        assert "".join(deltas) == "".join(spoken) == f"Paris.\n\n{tail}"
