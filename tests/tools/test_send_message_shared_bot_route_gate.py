"""#147602: a shared-bot satellite borrows the primary's bots unscoped (``_adapters_for_profile``),
so every send_message path that hands it the live adapter for a caller-chosen target must apply the
route gate. ``_live_adapter`` is that single choke point: send, react and cron's standalone fallback
all resolve through it. Real ``GatewayRunner`` authz mixin, real ``profile_routes``, real handlers;
only the transports are fakes.
"""

import asyncio
import json
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.profile_routing import parse_profile_routes


class _Primary:
    """The default profile's live bot: records every outbound act."""

    def __init__(self, platform):
        self.platform = platform
        self.sent, self.reactions = [], []

    async def send(self, chat_id, content, metadata=None, **_kw):
        self.sent.append((chat_id, (metadata or {}).get("thread_id")))
        return SimpleNamespace(success=True, message_id="m1", error=None)

    async def add_reaction(self, chat_id, emoji, message_id=None):
        self.reactions.append(chat_id)
        return {"success": True}


@pytest.fixture
def satellite(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    home = tmp_path / "hh"
    (home / "profiles" / "sec").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.config.platforms = {p: PlatformConfig(enabled=True, extra={}) for p in (Platform.SLACK, Platform.TELEGRAM)}
    runner.config.profile_routes = parse_profile_routes([
        {"name": "sec-slack", "platform": "slack", "profile": "sec", "chat_id": "C0ROUTED1"},
        {"name": "sec-tg", "platform": "telegram", "profile": "sec", "chat_id": "-100111"},
    ])
    runner.config._runtime_profile_homes = [("default", home), ("sec", home / "profiles" / "sec")]
    runner._primary_profile_name = "default"
    runner._profile_failed_platforms = {}
    runner._profile_bot_platforms = {}
    primary = {p: _Primary(p) for p in (Platform.SLACK, Platform.TELEGRAM)}
    runner.adapters = dict(primary)
    runner._profile_adapters = {"sec": {}}  # no bot of its own: borrows the primary's
    with patch("gateway.run._gateway_runner_ref", return_value=runner), \
            patch("hermes_cli.profiles.get_active_profile_name", return_value="sec"):
        yield SimpleNamespace(runner=runner, primary=primary)


def _satellite_config():
    """The satellite's own config enables the platforms (the cron fallback precondition)."""
    return SimpleNamespace(
        platforms={p: PlatformConfig(enabled=True, token="xoxb-sat", extra={}) for p in (Platform.SLACK, Platform.TELEGRAM)},
        get_home_channel=lambda _p: None)


def _tool(args):
    from tools.send_message_tool import send_message_tool

    with patch("gateway.config.load_gateway_config", return_value=_satellite_config()), \
            patch("tools.send_message_tool._authorize_relay_target", return_value=None), \
            patch("tools.interrupt.is_interrupted", return_value=False), \
            patch("model_tools._run_async", side_effect=lambda c: asyncio.run(c)), \
            patch("gateway.mirror.mirror_to_session", return_value=False):
        return json.loads(send_message_tool(args))


def test_send_to_unrouted_chat_never_uses_the_primary_bot(satellite):
    slack = satellite.primary[Platform.SLACK]
    result = _tool({"action": "send", "target": "slack:C0UNROUTED", "message": "hi"})
    assert slack.sent == []
    assert "error" in result and "profile_routes" in result["error"]
    # The routed chat still goes out through the shared bot: the feature survives the gate.
    result = _tool({"action": "send", "target": "slack:C0ROUTED1", "message": "hi"})
    assert result.get("success") is True, result
    assert slack.sent == [("C0ROUTED1", None)]


def test_react_on_unrouted_chat_never_uses_the_primary_bot(satellite):
    tg = satellite.primary[Platform.TELEGRAM]
    result = _tool({"action": "react", "target": "telegram:-100999", "emoji": "👍", "message_id": "5"})
    assert tg.reactions == []
    assert "error" in result and "profile_routes" in result["error"]
    result = _tool({"action": "react", "target": "telegram:-100111", "emoji": "👍", "message_id": "5"})
    assert result.get("success") is True, result
    assert tg.reactions == ["-100111"]


def test_cron_standalone_fallback_cannot_reach_the_primary_bot(satellite):
    """A ``SharedRouteAdapters`` miss with the satellite's own ``platforms.slack`` enabled resolves
    no error, so the job falls to ``_deliver_standalone`` -> ``_send_to_platform`` ->
    ``_live_adapter``: that hop must not hand out the primary bot for the unrouted chat."""
    from cron.scheduler import _deliver_result
    from cron.scheduler_preflight import SharedRouteAdapters

    slack = satellite.primary[Platform.SLACK]
    routes = [r for r in satellite.runner.config.profile_routes if r.profile == "sec"]
    shared = SharedRouteAdapters({Platform.SLACK: slack}, routes)
    loop = MagicMock()
    loop.is_running.return_value = True

    def fake_run_coro(coro, _loop):
        future = Future()
        future.set_result(asyncio.run(coro))
        return future

    job = {"id": "j147602", "name": "brief", "deliver": "slack:C0UNROUTED"}
    with patch("gateway.config.load_gateway_config", return_value=_satellite_config()), \
            patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
            patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
        error = _deliver_result(job, "hello", adapters=shared, loop=loop)
    assert slack.sent == []
    assert error is not None and "profile_routes" in error
