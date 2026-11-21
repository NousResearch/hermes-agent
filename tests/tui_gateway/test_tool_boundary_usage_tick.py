"""Tool-boundary ``session.usage`` ticks keep the status-bar context % current mid-turn (#77744).

A fast tool loop can out-run the 1 Hz usage ticker — or finish inside its first
window — leaving the context gauge at the turn-start value until
``message.complete``. Each ``tool.complete`` now emits an immediate deduped
``session.usage`` frame, sharing the ticker's dedupe cell so slow loops never
see duplicates. No prompt/context mutation: display timing only.
"""

import time
from types import SimpleNamespace

import tui_gateway.server as server


def _compressor(prompt_tokens, context_length=100_000):
    return SimpleNamespace(
        last_prompt_tokens=prompt_tokens, context_length=context_length, compression_count=0
    )


def _agent(prompt_tokens):
    return SimpleNamespace(
        model="test-model",
        session_input_tokens=10,
        session_output_tokens=5,
        session_reasoning_tokens=0,
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_total_tokens=15,
        session_api_calls=2,
        context_compressor=_compressor(prompt_tokens),
    )


def _session(agent, **extra):
    return {
        "agent": agent,
        "session_key": "session-key",
        "running": True,
        "tool_started_at": {},
        "tool_progress_mode": "all",
        **extra,
    }


def test_tool_completion_ticks_usage_between_tool_calls(monkeypatch):
    sid = "boundary-tick"
    agent = _agent(20_000)
    monkeypatch.setitem(server._sessions, sid, _session(agent))
    events: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        server, "_emit", lambda event, esid, payload=None: events.append((event, esid, payload))
    )

    # Two tool completions in one turn; the context reading grows between them
    # exactly as update_from_response() bumps the compressor per API response.
    server._on_tool_complete(sid, "call-1", "read_file", {"path": "a"}, "ok")
    agent.context_compressor = _compressor(40_000)
    server._on_tool_complete(sid, "call-2", "read_file", {"path": "b"}, "ok")

    ticks = [e for e in events if e[0] == "session.usage"]
    assert [e[2]["usage"]["context_percent"] for e in ticks] == [20, 40]
    # Each tick lands after its own tool.complete, in tool order — the bar
    # updates between tool calls, not in one jump at turn end.
    assert [e[0] for e in events] == [
        "tool.complete", "session.usage", "tool.complete", "session.usage",
    ]
    # The client-visible snapshot equals what message.complete would report.
    assert ticks[-1][2]["usage"] == server._get_usage(agent)


def test_tool_usage_tick_is_idle_gated_and_deduped_with_ticker(monkeypatch):
    sid = "boundary-dedupe"
    agent = _agent(20_000)
    monkeypatch.setitem(server._sessions, sid, _session(agent))
    events: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        server, "_emit", lambda event, esid, payload=None: events.append((event, esid, payload))
    )

    # Idle: message.complete owns the authoritative usage — no boundary tick.
    server._sessions[sid]["running"] = False
    server._on_tool_complete(sid, "call-0", "read_file", {"path": "x"}, "ok")
    assert [e for e in events if e[0] == "session.usage"] == []

    # Running: first completion ticks; an unchanged second doesn't re-emit.
    server._sessions[sid]["running"] = True
    server._on_tool_complete(sid, "call-1", "read_file", {"path": "a"}, "ok")
    server._on_tool_complete(sid, "call-2", "read_file", {"path": "b"}, "ok")
    ticks = [e for e in events if e[0] == "session.usage"]
    assert len(ticks) == 1

    # The boundary tick advances the SAME dedupe cell the 1 Hz ticker uses:
    # the ticker must not re-emit the snapshot the boundary already sent.
    stop, thread = server._start_usage_ticker(sid, agent, interval=0.01)
    try:
        time.sleep(0.15)  # ~15 ticks with unchanged counters
    finally:
        stop.set()
        thread.join(timeout=2.0)
    assert [e for e in events if e[0] == "session.usage"] == ticks
