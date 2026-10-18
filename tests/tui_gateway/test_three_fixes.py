
"""Fix for three issues (2026-10-18):
1. Session model display mismatch -- added working_model/prov.
2. Token speed not working for Mac Studio -- _session_usage_snapshot now
   reads local agent when _compute_host_active.
3. Reset model doesn't update UI -- _reset_session_agent now clears
   pending_model_switch."""

import types
from tui_gateway import server


def test_session_info_includes_working_model_and_provider():
    """Issue 1 fix: when a model is queued mid-turn the session's
    working_model should always reflect the agent's actual model,
    not the pending pick."""
    agent = types.SimpleNamespace(tools=[], model="old/model", provider="openai")
    session = {
        "session_key": "",
        "history": [],
        "pending_model_switch": {
            "raw": "new/model --provider anthropic",
            "display_model": "new/model",
            "display_provider": "anthropic",
        },
    }

    info = server._session_info(agent, session)

    # Existing behaviour: model shows the pending pick
    assert info["model"] == "new/model", f"expected 'new/model', got {info['model']!r}"
    assert info["provider"] == "anthropic"

    # NEW: working_model always shows what the agent is ACTUALLY running
    assert info["working_model"] == "old/model", f"expected 'old/model', got {info.get('working_model')!r}"
    assert info["working_provider"] == "openai"

    # When nothing is queued, working_model == model
    session.pop("pending_model_switch")
    info = server._session_info(agent, session)
    assert info["model"] == "old/model"
    assert info["working_model"] == "old/model"
    assert info["working_provider"] == "openai"


def test_session_usage_snapshot_with_compute_host_active():
    """Issue 2 fix: _session_usage_snapshot reads from local agent
    even when _compute_host_active is True (Mac Studio sessions).
    The desktop shows avg_tps (avg tokens/second) from the local
    agent's _api_latency_history/_api_output_history."""
    agent = types.SimpleNamespace(
        tools=[], model="test/model",
        _api_latency_history=[0.1, 0.15, 0.12],
        _api_output_history=[10, 12, 11],
    )
    session = {
        "session_key": "sid",
        "agent": agent,
        "_compute_host_active": True,
        "history": [],
    }

    usage = server._session_usage_snapshot(session)
    assert "avg_tps" in usage, f"missing avg_tps: {usage}"
    assert usage["avg_tps"] > 0, f"expected positive tps, got {usage.get('avg_tps')}"

