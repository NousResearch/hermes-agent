"""``message_agent``'s dispatch-boundary ``pre_tool_call`` policy pass (tools/bot_mode_dm.py):
hooks must see the RESOLVED canonical recipient (never a raw alias), a ``block`` directive must
refuse the send before any delivery starts, and a policy hook that raises or times out must fail
closed (the ``pre_tool_call`` contract), never silently let the message through."""
import json
import textwrap
from pathlib import Path

import pytest

from tests.tools.test_bot_mode_dm import _FakeAgent, _canonical_target, _capture_spawn, _managed_home, _rename
from tools import bot_mode_dm, bot_relay


@pytest.fixture(autouse=True)
def _fresh_probe_cache():
    from tools import bot_mode_probe

    bot_mode_probe._reset_cache_for_tests()
    yield
    bot_mode_probe._reset_cache_for_tests()


def _patch_directive(monkeypatch, directive, message=None):
    """Stand in for the whole plugin stack: ``get_pre_tool_call_directive`` is the one function
    the dispatch-boundary helper calls, so patching it exercises the real call site."""
    captured = {}

    def fake(tool_name, args, **hook_kwargs):
        captured["tool_name"] = tool_name
        captured["args"] = args
        captured["hook_kwargs"] = hook_kwargs
        return directive, message

    monkeypatch.setattr("hermes_cli.plugins.get_pre_tool_call_directive", fake)
    return captured


def test_policy_sees_the_resolved_recipient_not_the_raw_alias(tmp_path, monkeypatch):
    """A plugin blocking 'watch_tracker' must still catch a DM sent to its friendly-name alias."""
    calls = _capture_spawn(monkeypatch)
    monkeypatch.setattr(bot_relay, "_hermes_cli", lambda: "hermes")
    home = _managed_home(tmp_path, teammates=("watch_tracker",))
    _rename(home, "watch_tracker", display_name="Tracker")
    _canonical_target(monkeypatch, home / "profiles" / "watch_tracker")
    captured = _patch_directive(monkeypatch, None)

    result = json.loads(bot_mode_dm.message_agent_tool(target="Tracker", message="hi", agent=_FakeAgent(home)))

    assert result["status"] == "queued", result
    assert calls, "delivery must proceed when no hook blocks"
    assert captured["tool_name"] == "message_agent"
    assert captured["args"]["target"] == "Tracker"  # the model's raw alias is still visible
    assert captured["args"]["bot_recipient"]["id"] == "bot:watch_tracker"  # but policy sees the canonical id
    assert captured["args"]["bot_recipient"]["profile"] == "watch_tracker"
    assert captured["args"]["bot_sender"]["is_bot"] is True


def test_block_directive_refuses_the_send_before_dispatch(tmp_path, monkeypatch):
    calls = _capture_spawn(monkeypatch)
    monkeypatch.setattr(bot_relay, "_hermes_cli", lambda: "hermes")
    home = _managed_home(tmp_path, teammates=("watch_tracker",))
    _canonical_target(monkeypatch, home / "profiles" / "watch_tracker")
    _patch_directive(monkeypatch, "block", "not an authorized recipient for this sender")

    result = json.loads(bot_mode_dm.message_agent_tool(target="watch_tracker", message="hi", agent=_FakeAgent(home)))

    assert "not an authorized recipient" in result["error"]
    assert result["reason"] == "policy_refused"
    assert calls == [], "a blocked message must never start delivery"


def test_policy_hook_failure_fails_closed(tmp_path, monkeypatch):
    """A policy gate that cannot even be consulted must not silently open the send."""
    calls = _capture_spawn(monkeypatch)
    monkeypatch.setattr(bot_relay, "_hermes_cli", lambda: "hermes")
    home = _managed_home(tmp_path, teammates=("watch_tracker",))
    _canonical_target(monkeypatch, home / "profiles" / "watch_tracker")

    def boom(*a, **kw):
        raise RuntimeError("policy backend unreachable")

    monkeypatch.setattr("hermes_cli.plugins.get_pre_tool_call_directive", boom)

    result = json.loads(bot_mode_dm.message_agent_tool(target="watch_tracker", message="hi", agent=_FakeAgent(home)))

    assert "policy check" in result["error"]
    assert calls == [], "a policy check that errors must fail closed, never dispatch"


def test_peer_target_policy_sees_the_canonical_peer_recipient(tmp_path, monkeypatch):
    calls = _capture_spawn(monkeypatch)
    monkeypatch.setattr(bot_relay, "_hermes_cli", lambda: "hermes")
    home = _managed_home(tmp_path, peers=("spark",))
    captured = _patch_directive(monkeypatch, None)

    result = json.loads(
        bot_mode_dm.message_agent_tool(target="spark/researcher", message="hi", agent=_FakeAgent(home)))

    assert result["status"] == "queued", result
    assert calls
    assert captured["args"]["bot_recipient"]["kind"] == "peer"
    assert captured["args"]["bot_recipient"]["peer"] == "spark"
    assert captured["args"]["bot_recipient"]["profile"] == "researcher"
    assert captured["args"]["bot_sender"]["id"].startswith("bot:")


def test_peer_target_block_directive_refuses_before_delivery(tmp_path, monkeypatch):
    calls = _capture_spawn(monkeypatch)
    monkeypatch.setattr(bot_relay, "_hermes_cli", lambda: "hermes")
    home = _managed_home(tmp_path, peers=("spark",))
    _patch_directive(monkeypatch, "block", "peer messaging is disabled for this sender")

    result = json.loads(
        bot_mode_dm.message_agent_tool(target="spark/researcher", message="hi", agent=_FakeAgent(home)))

    assert "peer messaging is disabled" in result["error"]
    assert calls == []
