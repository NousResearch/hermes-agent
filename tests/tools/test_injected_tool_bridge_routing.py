"""Regression: an injected-but-unregistered native tool must not be reported as an
unknown name by the tool_search/tool_call bridge.

``message_agent`` is INJECTED into a managed bot's canonical Bot Chat session by the
auth gate in ``tools.bot_mode_dm`` — it is deliberately never in the tool registry and
never in ``toolsets._HERMES_CORE_TOOLS``. So ``not_deferrable_error()`` took its
*unknown-name* branch for a tool that was, at that very moment, sitting in
``agent.tools`` / ``agent.valid_tool_names`` and advertised to the provider.

The resulting correction was actively harmful: "'message_agent' is not a known tool
name ... Use tool_search to find the exact name." The model dutifully ran tool_search
(msg 6025), which can never find an unregistered name, and then concluded the DM tool
was not loaded — while the tool sat in its own tool list the entire time.

Contract fixed here: when the name IS present in this session's live tool surface, the
bridge must give the "call it directly" correction. The tool_search path additionally
gets an explicit scoped-reason instead of a bare "not found". This mirrors the existing
GUI-tool precedent (``out_of_scope_reason``, #120413).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _td(name: str, description: str = "") -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {"target": {"type": "string"}}},
        },
    }


# The live Bot Chat tool surface: message_agent injected alongside ordinary core tools.
_LIVE_SURFACE: List[Dict[str, Any]] = [_td("terminal"), _td("web_search"), _td("message_agent")]


class _ScopedName:
    """Minimal stand-in for the ``resolve_underlying_call`` name resolver seam.

    The real bridge asks a session-scoped question ("is this name in MY tools?"), so the
    tests inject a resolver here rather than building a full AIAgent. Production code
    supplies this from the agent's published ``(tools, valid_tool_names)`` pair.
    """

    def __init__(self, surface: List[Dict[str, Any]]):
        self._names = {t["function"]["name"] for t in surface}

    def __call__(self, name: str) -> bool:
        return name in self._names


@pytest.fixture
def scoped(monkeypatch):
    resolver = _ScopedName(_LIVE_SURFACE)
    monkeypatch.setattr("tools.tool_search._session_direct_names", resolver, raising=False)
    return resolver


# ---------------------------------------------------------------------------
# 1. tool_call bridge: the exact correction text Jarvis received
# ---------------------------------------------------------------------------


def test_bridge_tells_model_to_call_injected_tool_directly(scoped):
    """The regression guard: an advertised native tool gets 'call it directly'.

    Before the fix this returned "'message_agent' is not a known tool name ... Use
    tool_search to find the exact name", which sent Jarvis hunting for a name that
    tool_search can never return.
    """
    from tools.tool_search_validation import not_deferrable_error

    err = not_deferrable_error("message_agent")

    assert "not a known tool name" not in err, (
        f"advertised native tool reported as unknown: {err}"
    )
    assert "Call it directly instead of via tool_call" in err
    assert "Use tool_search to find the exact name" not in err, (
        "must not send the model to a search that cannot find an unregistered name"
    )


def test_bridge_keeps_unknown_name_error_for_a_real_unknown(monkeypatch):
    """Fail-open: a genuinely unknown name still gets the search hint.

    The session-scoped resolver must not turn every typo into "call it directly".
    """
    from tools.tool_search_validation import not_deferrable_error

    monkeypatch.setattr(
        "tools.tool_search_validation._session_direct_names", lambda name: False, raising=False
    )

    err = not_deferrable_error("mcp__nope__ghost")

    assert "not a known tool name" in err
    assert "Call it directly instead of via tool_call" not in err


def test_bridge_still_reports_unregistered_unknown_as_unknown(monkeypatch):
    """Fail-open: a genuinely unknown name keeps the search hint.

    The session-scoped resolver must not turn every typo into "call it directly".
    """
    from tools.tool_search_validation import not_deferrable_error

    monkeypatch.setattr(
        "tools.tool_search_validation._session_direct_names", lambda name: False, raising=False
    )

    err = not_deferrable_error("mcp__nope__ghost")

    assert "not a known tool name" in err
    assert "Call it directly instead of via tool_call" not in err


def test_bridge_does_not_publish_names_the_gate_never_injected():
    """Before any gate runs, the resolver must not claim message_agent is a direct door.

    Guards the fail-closed default: publishing is driven solely by the injection gate, so a
    session that never injected it (an ordinary CLI chat) gets today's behaviour unchanged.
    """
    from tools import bot_mode_dm
    from tools.tool_search import _session_direct_names

    bot_mode_dm._injected_native_names.clear()

    assert _session_direct_names("message_agent") is False


def _managed_home(tmp_path):
    """A minimal Bot-Mode-MANAGED install: one profile carrying the bots ui_meta block.

    ``is_bot_mode_managed`` is a real gate (``_any_managed`` over the profile roots), so the
    end-to-end test drives a genuine install rather than stubbing the gate to True.
    """
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    d = home / "profiles" / "jarvis"
    d.mkdir(parents=True, exist_ok=True)
    (d / "profile.yaml").write_text(
        "description: test bot\nui_meta:\n  hermes-bots:\n    shape: cloud\n",
        encoding="utf-8",
    )
    return home


def test_gate_publication_makes_the_bridge_route_to_the_direct_door(tmp_path):
    """End-to-end on the real seam: once the gate injects, the bridge correction flips.

    This is the exact Jarvis failure sequence — the per-turn gate injects before the model
    ever calls the bridge, so by then the name is published and must be reported as a
    directly-listed tool rather than an unknown one.
    """
    from tools import bot_mode_dm, bot_mode_probe
    from tools.tool_search import _session_direct_names
    from tools.tool_search_validation import not_deferrable_error

    bot_mode_dm._injected_native_names.clear()
    bot_mode_probe._reset_cache_for_tests()
    home = _managed_home(tmp_path)

    class _DB:
        def __init__(self):
            self.db_path = str(home / "state.db")

        def get_session_title(self, _sid):
            return "Bot Chat"

    class _Agent:
        tools: list = []
        valid_tool_names: set = set()
        _bot_mode_protocol = True
        _session_title_hint = "Bot Chat"
        session_id = "sess-1"
        _session_db = _DB()

    try:
        assert bot_mode_dm.ensure_message_agent_tool(_Agent()) is True
        assert bot_mode_dm.is_injected_native_tool("message_agent") is True

        assert _session_direct_names("message_agent") is True
        assert "Call it directly instead of via tool_call" in not_deferrable_error("message_agent")
    finally:
        bot_mode_dm._injected_native_names.clear()
        bot_mode_probe._reset_cache_for_tests()


# ---------------------------------------------------------------------------
# 2. resolve_underlying_call: must not resolve an injected name for the bridge
# ---------------------------------------------------------------------------


def test_resolve_underlying_call_rejects_injected_tool_with_direct_instruction(monkeypatch):
    """tool_call(name='message_agent') must fail fast with the DIRECT instruction."""
    from tools import tool_search

    monkeypatch.setattr(
        tool_search, "_session_direct_names", lambda name: name == "message_agent", raising=False
    )

    underlying, args, err = tool_search.resolve_underlying_call(
        {"calls": [{"name": "message_agent", "arguments": {"target": "atlas", "message": "hi"}}]}
    )

    assert underlying is None
    assert args == {}
    assert err is not None
    assert "Call it directly instead of via tool_call" in err


# ---------------------------------------------------------------------------
# 3. tool_describe: an advertised native tool must not be a bare "not found"
# ---------------------------------------------------------------------------


def test_tool_describe_reports_injected_tool_as_direct_not_found(monkeypatch):
    """tool_describe on the native name explains the direct door rather than hiding it."""
    from tools import tool_search

    monkeypatch.setattr(
        tool_search, "_session_direct_names", lambda name: name == "message_agent", raising=False
    )
    monkeypatch.setattr(
        tool_search, "remote_schemas_for",
        lambda names, defs, describe: ({}, None),
        raising=False,
    )

    raw = tool_search.dispatch_tool_describe(
        {"names": ["message_agent"]}, current_tool_defs=[_td("message_agent")]
    )
    payload = json.loads(raw)

    assert payload.get("not_found") is None, (
        f"advertised native tool listed as not found: {raw}"
    )
    assert "Call it directly instead of via tool_call" in json.dumps(payload)