"""The TUI gateway forwards per-call meters as their own events, whatever the display mode."""

from types import SimpleNamespace

from tui_gateway import server


def test_model_call_meters_reach_the_tui_stream_in_answer_only_mode(monkeypatch):
    events = []
    monkeypatch.setattr(server, "_emit", lambda event_type, sid, payload=None: events.append((event_type, payload)))
    monkeypatch.setitem(server._sessions, "meters", {
        "show_reasoning": False, "tool_progress_mode": "off", "tool_started_at": {}, "edit_snapshots": {},
        "agent": SimpleNamespace(reasoning_config={"enabled": False}),
    })

    server._on_tool_progress("meters", "model.rate_limited", None, None, None, model="m", provider="p",
                             attempt=2, max_attempts=3, retry_after_ms=1500, wait_ms=1500, api_request_id=None)
    server._on_tool_progress("meters", "usage.recorded", None, None, None, call_index=3, model="m",
                             provider="p", input_tokens=10, output_tokens=4, cache_read_tokens=0,
                             cache_write_tokens=0, reasoning_tokens=2, cost_usd=None, first_token_ms=None,
                             duration_ms=420, used_tokens=900, window_tokens=200000)

    assert events == [
        ("model.rate_limited", {"model": "m", "provider": "p", "attempt": 2, "max_attempts": 3,
                                "retry_after_ms": 1500, "wait_ms": 1500}),
        ("usage.recorded", {"call_index": 3, "model": "m", "provider": "p", "input_tokens": 10,
                            "output_tokens": 4, "cache_read_tokens": 0, "cache_write_tokens": 0,
                            "reasoning_tokens": 2, "duration_ms": 420, "used_tokens": 900,
                            "window_tokens": 200000}),
    ]
