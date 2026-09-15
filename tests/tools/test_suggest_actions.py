"""Tests for the suggest_actions tool + gateway registry.

Covers the tool's validation/normalization and the non-blocking gateway
registry (register / resolve / eviction / session cleanup).
"""

import json

import pytest

from tools.suggest_actions_tool import (
    MAX_ACTIONS,
    _normalize_actions,
    suggest_actions_tool,
)
from tools import suggested_actions_gateway as sag


# --- normalization -------------------------------------------------------

def test_normalize_strings():
    out = _normalize_actions(["Book it", "See more"])
    assert out == [
        {"label": "Book it", "payload": "Book it"},
        {"label": "See more", "payload": "See more"},
    ]


def test_normalize_dicts_with_payload():
    out = _normalize_actions([{"label": "More", "payload": "show me more options"}])
    assert out == [{"label": "More", "payload": "show me more options"}]


def test_normalize_dict_defaults_payload_to_label():
    out = _normalize_actions([{"label": "Yes"}])
    assert out == [{"label": "Yes", "payload": "Yes"}]


def test_normalize_skips_blank_and_caps():
    many = [f"a{i}" for i in range(MAX_ACTIONS + 5)]
    out = _normalize_actions(["", "  ", *many])
    assert len(out) == MAX_ACTIONS


def test_normalize_rejects_non_list():
    assert _normalize_actions("nope") is None
    assert _normalize_actions([]) is None


# --- tool dispatch -------------------------------------------------------

def test_tool_requires_message():
    res = json.loads(suggest_actions_tool("", ["A"]))
    assert "error" in res


def test_tool_requires_valid_actions():
    res = json.loads(suggest_actions_tool("Pick", []))
    assert "error" in res


def test_tool_no_callback_returns_data():
    res = json.loads(suggest_actions_tool("Pick", ["A", "B"]))
    assert res["delivered"] is False
    assert len(res["actions"]) == 2


def test_tool_with_callback_invoked():
    captured = {}

    def cb(message, actions):
        captured["message"] = message
        captured["actions"] = actions
        return {"delivered": True}

    res = json.loads(suggest_actions_tool("Next?", ["A", "B"], callback=cb))
    assert res["delivered"] is True
    assert captured["message"] == "Next?"
    assert len(captured["actions"]) == 2


def test_tool_callback_exception_reported():
    def cb(message, actions):
        raise RuntimeError("send failed")

    res = json.loads(suggest_actions_tool("Next?", ["A"], callback=cb))
    assert "error" in res


# --- gateway registry ----------------------------------------------------

def test_register_and_resolve():
    sag.register("set1", "sess1", "msg", [
        {"label": "Yes", "payload": "do yes"},
        {"label": "No", "payload": "do no"},
    ])
    assert sag.resolve("set1", 0) == "do yes"
    assert sag.resolve("set1", 1) == "do no"
    assert sag.resolve("set1", 5) is None  # out of range
    sag.clear_session("sess1")


def test_resolve_unknown_set():
    assert sag.resolve("nope", 0) is None


def test_resolve_does_not_consume():
    sag.register("set2", "sess2", "msg", [{"label": "X", "payload": "x"}])
    assert sag.resolve("set2", 0) == "x"
    # Still resolvable after first tap (a message may offer multiple taps).
    assert sag.resolve("set2", 0) == "x"
    sag.clear_session("sess2")


def test_clear_session():
    sag.register("set3", "sess3", "m", [{"label": "A", "payload": "a"}])
    assert sag.has_pending("sess3") is True
    n = sag.clear_session("sess3")
    assert n == 1
    assert sag.has_pending("sess3") is False
    assert sag.resolve("set3", 0) is None


def test_per_session_eviction():
    session = "sess-evict"
    # Register more than the cap; oldest should be evicted.
    for i in range(sag._MAX_SETS_PER_SESSION + 5):
        sag.register(f"s{i}", session, "m", [{"label": "L", "payload": f"p{i}"}])
    # The very first sets are gone.
    assert sag.resolve("s0", 0) is None
    # The most recent one survives.
    last = sag._MAX_SETS_PER_SESSION + 4
    assert sag.resolve(f"s{last}", 0) == f"p{last}"
    sag.clear_session(session)


def test_label_for():
    entry = sag.register("set4", "sess4", "m", [{"label": "Tap me", "payload": "go"}])
    assert entry.label_for(0) == "Tap me"
    assert entry.label_for(9) is None
    sag.clear_session("sess4")
