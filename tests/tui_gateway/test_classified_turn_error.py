"""Classified turn-error messages for TUI/desktop frames (#64182 item 3)."""

from __future__ import annotations

import types

from tui_gateway.server import _classify_turn_error_message, _fail_inflight_turn


def test_classify_enriches_exception_with_agent_route():
    class _Boom(Exception):
        status_code = 429

    agent = types.SimpleNamespace(
        provider="xai",
        model="grok-4.5",
        base_url="https://api.x.ai/v1",
        _summarize_api_error=lambda e: "rate limited",
    )
    msg = _classify_turn_error_message(_Boom("request failed"), agent)
    assert "rate limited" in msg
    assert "HTTP 429" in msg
    assert "provider=xai" in msg
    assert "model=grok-4.5" in msg


def test_classify_result_dict_keeps_failure_reason():
    agent = types.SimpleNamespace(
        provider="openrouter",
        model="foo",
        base_url="https://openrouter.ai/api/v1",
    )
    msg = _classify_turn_error_message(
        {
            "error": "credits exhausted",
            "failure_reason": "billing",
            "failed": True,
        },
        agent,
    )
    assert "credits exhausted" in msg
    assert "reason=billing" in msg
    assert "provider=openrouter" in msg


def test_fail_inflight_uses_classified_message():
    session = {
        "agent": types.SimpleNamespace(
            provider="ollama",
            model="qwen",
            base_url="http://127.0.0.1:11434",
            _summarize_api_error=lambda e: "connection refused",
        ),
        "inflight_turn": {"user": "hi", "assistant": "", "started_at": 0},
    }
    _fail_inflight_turn(session, ConnectionError("request failed"))
    err = session["inflight_turn"]["error"]
    assert "connection refused" in err
    assert "provider=ollama" in err
