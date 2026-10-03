"""``session.usage.reset_throughput`` clears the rolling throughput histories.

The status-bar tokens/sec average divides recent output tokens by recent call
durations; after a cold-start load the average drags until enough new calls
dilute it. Resetting re-seeds from current conditions. Behavior contracts on
that relationship, not snapshots.
"""
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch


def _session(server, sid, agent):
    session = {"agent": agent, "history": [], "running": False, "session_key": f"key-{sid}",
               "history_lock": __import__("threading").Lock()}
    server._sessions[sid] = session
    return session


def test_reset_clears_live_histories_and_returns_snapshot():
    from tui_gateway import server

    agent = SimpleNamespace(
        model="m", provider="p", base_url="", api_key="",
        session_input_tokens=0, session_prompt_tokens=0,
        session_output_tokens=0, session_completion_tokens=0,
        session_total_tokens=0, session_api_calls=2,
        session_cache_read_tokens=0, session_cache_write_tokens=0,
        session_reasoning_tokens=0, context_compressor=None,
        _api_latency_history=deque([30.0, 1.0], maxlen=10),
        _api_decode_duration_history=deque([3.0, 1.0], maxlen=10),
        _api_output_history=deque([10, 20], maxlen=10),
    )
    sid = "sid-reset-tps"
    _session(server, sid, agent)
    try:
        r = server._methods["session.usage.reset_throughput"]("r1", {"session_id": sid})
    finally:
        server._sessions.pop(sid, None)

    assert "error" not in r, r
    assert list(agent._api_latency_history) == []
    assert list(agent._api_decode_duration_history) == []
    assert list(agent._api_output_history) == []
    assert r["result"]["calls"] == 2
    assert "avg_tps" not in r["result"]


def test_reset_before_agent_build_returns_zeroed_usage():
    from tui_gateway import server

    sid = "sid-reset-tps-no-agent"
    _session(server, sid, None)
    try:
        r = server._methods["session.usage.reset_throughput"]("r1", {"session_id": sid})
    finally:
        server._sessions.pop(sid, None)

    assert "error" not in r, r
    assert r["result"]["calls"] == 0


def test_reset_on_unknown_session_is_not_found():
    from tui_gateway import server

    r = server._methods["session.usage.reset_throughput"]("r1", {"session_id": "no-such-sid"})
    assert r["error"]["code"] == 4001


def test_tps_divides_by_decode_phase_not_wall_clock():
    """The user's scenario: a cold local model loads 8s then decodes 20 tokens in 2s.

    Throughput reports the 10 tok/s decode rate, while latency still reports the
    10s wall clock.
    """
    from tui_gateway import server

    agent = SimpleNamespace(
        model="m", provider="p", base_url="", api_key="",
        session_input_tokens=0, session_prompt_tokens=0,
        session_output_tokens=0, session_completion_tokens=0,
        session_total_tokens=0, session_api_calls=1,
        session_cache_read_tokens=0, session_cache_write_tokens=0,
        session_reasoning_tokens=0, context_compressor=None,
        _api_latency_history=deque([10.0], maxlen=10),
        _api_decode_duration_history=deque([2.0], maxlen=10),
        _api_output_history=deque([20], maxlen=10),
    )
    sid = "sid-decode-tps"
    _session(server, sid, agent)
    try:
        with (
            patch("agent.account_usage.fetch_account_usage", lambda *a, **k: None),
            patch("agent.account_usage.nous_credits_lines", lambda **kw: []),
        ):
            r = server._methods["session.usage"]("r1", {"session_id": sid})
    finally:
        server._sessions.pop(sid, None)

    assert "error" not in r, r
    assert r["result"]["avg_latency_s"] == 10.0
    assert r["result"]["avg_tps"] == 10.0
