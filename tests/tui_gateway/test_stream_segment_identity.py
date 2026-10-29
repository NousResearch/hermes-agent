"""A prompt turn's message.* frames carry its wire identity (tui_gateway/stream_segments.py).

The desktop upserts a re-sent frame by this identity instead of painting a second copy (#144609):
message.start names the turn, every delta its turn, segment (interim seals before it) and UTF-16
offset inside that segment, every interim the segment it seals. The stamp goes on beneath ``_emit``,
on the frame that reaches the client, so the turn's ``_emit`` calls are the same with turn isolation
off and on (test_prompt_submit_golden_transcript_matches_flag_off_and_on).
"""

import threading
import types

import tui_gateway.server as srv


class _ImmediateThread:
    def __init__(self, target=None, daemon=None, **_kwargs):
        self._target = target

    def start(self):
        self._target()


def _submit_turns(monkeypatch, run_conversation, turns):
    """Client frames (``write_json``) of ``turns`` in-process prompt.submit turns, and the session."""
    frames = []
    monkeypatch.setattr(srv, "write_json", lambda obj: frames.append(obj["params"]) or True)
    monkeypatch.setattr(srv.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(srv, "_load_cfg", lambda: {"dashboard": {"turn_isolation": False}})
    monkeypatch.setattr(srv, "_load_interim_assistant_messages", lambda: True)
    monkeypatch.setattr(srv, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(srv, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(srv, "_session_info", lambda _agent, _session=None: {"model": "m"})
    monkeypatch.setattr(srv, "make_stream_renderer", lambda _cols: None)
    monkeypatch.setattr(srv, "render_message", lambda _raw, _cols: None)
    fake_title = types.ModuleType("agent.title_generator")
    fake_title.maybe_auto_title = lambda *args, **kwargs: None
    monkeypatch.setitem(srv.sys.modules, "agent.title_generator", fake_title)

    agent = types.SimpleNamespace(
        model="m", provider="p", session_id="session-key", context_compressor=None,
        clear_interrupt=lambda: None)
    agent.run_conversation = lambda prompt, **kwargs: run_conversation(agent, prompt, **kwargs)
    session = {
        "agent": agent, "session_key": "session-key", "history": [], "history_lock": threading.Lock(),
        "history_version": 0, "running": False, "attached_images": [], "image_counter": 0, "cols": 80,
        "slash_worker": None, "show_reasoning": False, "tool_progress_mode": "all",
        "model_override": {"model": "m", "provider": "p"}}
    srv._sessions["sid"] = session
    try:
        for n in range(turns):
            response = srv.handle_request(
                {"id": f"turn-{n}", "method": "prompt.submit", "params": {"session_id": "sid", "text": "ping"}})
            assert response["result"]["status"] == "streaming"
    finally:
        srv._sessions.pop("sid", None)
    return [(f["type"], f.get("payload") or {}) for f in frames if f.get("session_id") == "sid"], session


def test_deltas_and_seals_carry_turn_segment_and_utf16_offset(monkeypatch):
    def run_conversation(agent, prompt, stream_callback=None, **_kwargs):
        stream_callback("Looking 👀 ")
        stream_callback("now.")
        agent.interim_assistant_callback("Looking 👀 now.", already_streamed=True)
        stream_callback("Done.")
        return {"final_response": "Done.", "messages": [
            {"role": "user", "content": prompt}, {"role": "assistant", "content": "Done."}]}

    frames, session = _submit_turns(monkeypatch, run_conversation, turns=2)

    reply = [(e, p) for e, p in frames if e in {"message.start", "message.delta", "message.interim"}]
    stamps = [(e, p.get("segment"), p.get("offset")) for e, p in reply]
    # The eye emoji is one code point but two UTF-16 units: "Looking 👀 " advances the offset by 11.
    one_turn = [("message.start", None, None), ("message.delta", 0, 0), ("message.delta", 0, 11),
                ("message.interim", 0, None), ("message.delta", 1, 0)]
    assert stamps == one_turn * 2
    first, second = {p.get("turn_id") for _e, p in reply[:5]}, {p.get("turn_id") for _e, p in reply[5:]}
    assert len(first) == len(second) == 1 and None not in first | second and first != second
    # A finished turn leaves no cursor behind, so a later non-turn emitter's frame stays unstamped.
    assert srv.stream_segments.SESSION_KEY not in session
