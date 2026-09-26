"""Tests for ``llm.oneshot`` ``reasoning_effort`` param handling at the gateway."""

from unittest.mock import patch

import tui_gateway.server as server


def _ok_resp():
    return "t"


def test_reasoning_effort_forwards_to_run_oneshot():
    seen = {}

    def fake_oneshot(**kwargs):
        seen.update(kwargs)
        return _ok_resp()

    with patch("agent.oneshot.run_oneshot", fake_oneshot):
        r = server._methods["llm.oneshot"](
            "r1", {"instructions": "x", "input": "y", "reasoning_effort": "high"})

    assert r["result"]["text"] == "t"
    assert seen["reasoning_config"] == {"enabled": True, "effort": "high"}


def test_reasoning_effort_none_forwards_none():
    seen = {}

    def fake_oneshot(**kwargs):
        seen.update(kwargs)
        return _ok_resp()

    with patch("agent.oneshot.run_oneshot", fake_oneshot):
        r = server._methods["llm.oneshot"]("r2", {"instructions": "x", "input": "y"})

    assert r["result"]["text"] == "t"
    assert seen["reasoning_config"] is None


def test_reasoning_effort_none_word_disables():
    seen = {}

    def fake_oneshot(**kwargs):
        seen.update(kwargs)
        return _ok_resp()

    with patch("agent.oneshot.run_oneshot", fake_oneshot):
        r = server._methods["llm.oneshot"](
            "r3", {"instructions": "x", "input": "y", "reasoning_effort": "none"})

    assert r["result"]["text"] == "t"
    assert seen["reasoning_config"] == {"enabled": False}


def test_reasoning_effort_invalid_is_an_error():
    r = server._methods["llm.oneshot"](
        "r4", {"instructions": "x", "input": "y", "reasoning_effort": "hgih"})
    assert r["error"]["code"] == 4002
    assert "reasoning_effort" in r["error"]["message"]
