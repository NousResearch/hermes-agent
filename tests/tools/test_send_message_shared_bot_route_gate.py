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


_PLATFORMS = (Platform.SLACK, Platform.TELEGRAM, Platform.DISCORD)


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
    runner.config.platforms = {p: PlatformConfig(enabled=True, extra={}) for p in _PLATFORMS}
    runner.config.profile_routes = parse_profile_routes([
        {"name": "sec-slack", "platform": "slack", "profile": "sec", "chat_id": "C0ROUTED1"},
        {"name": "sec-tg", "platform": "telegram", "profile": "sec", "chat_id": "-100111"},
    ])
    runner.config._runtime_profile_homes = [("default", home), ("sec", home / "profiles" / "sec")]
    runner._primary_profile_name = "default"
    runner._profile_failed_platforms = {}
    runner._profile_bot_platforms = {}
    primary = {p: _Primary(p) for p in _PLATFORMS}
    runner.adapters = dict(primary)
    runner._profile_adapters = {"sec": {}}  # no bot of its own: borrows the primary's
    with patch("gateway.run._gateway_runner_ref", return_value=runner), \
            patch("hermes_cli.profiles.get_active_profile_name", return_value="sec"):
        yield SimpleNamespace(runner=runner, primary=primary)


def _satellite_config():
    """The satellite's own config enables the platforms (the cron fallback precondition)."""
    return SimpleNamespace(
        platforms={p: PlatformConfig(enabled=True, token="xoxb-sat", extra={}) for p in _PLATFORMS},
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


# (routes, the turn's inbound origin): the inbound matcher hands each conversation to ``sec`` though
# none is a target-exact ``SharedRouteAdapters`` route: guild-only, a thread under a routed channel,
# a non-Telegram ``user_id`` route.
_ORIGIN_ROUTED = {
    "guild": ([{"name": "g", "platform": "discord", "guild_id": "900", "profile": "sec"}],
              dict(platform="discord", chat_id="111", scope_id="900")),
    "thread-under-channel": ([{"name": "c", "platform": "discord", "chat_id": "111", "profile": "sec"}],
                             dict(platform="discord", chat_id="555", thread_id="555", parent_chat_id="111")),
    "slack-user": ([{"name": "u", "platform": "slack", "user_id": "U1", "profile": "sec"}],
                   dict(platform="slack", chat_id="D1", user_id="U1")),
}


@pytest.mark.parametrize("shape", sorted(_ORIGIN_ROUTED))
def test_satellite_acts_in_its_inbound_routed_conversation_only(satellite, shape):
    """Its own conversation stays reachable; another chat, a foreign thread, or an origin whose most
    specific route names another profile does not."""
    from gateway.session_context import clear_session_vars, set_session_vars

    routes, origin = _ORIGIN_ROUTED[shape]
    satellite.runner.config.profile_routes = parse_profile_routes(routes)
    adapter = satellite.primary[Platform(origin["platform"])]

    def react(target):
        return _tool({"action": "react", "target": target, "emoji": "👍", "message_id": "5"})

    tokens = set_session_vars(profile="sec", **origin)
    try:
        assert react(f"{origin['platform']}:{origin['chat_id']}").get("success") is True
        assert "profile_routes" in react(f"{origin['platform']}:777").get("error", "")
        if origin["platform"] == "discord":  # thread ids are global there
            assert "profile_routes" in react("discord:111:999").get("error", "")
        # The same ids on another platform are not this conversation, whatever routes exist there.
        elsewhere = "slack" if origin["platform"] == "discord" else "discord"
        satellite.runner.config.profile_routes = parse_profile_routes(
            routes + [{**r, "name": f"{r['name']}-x", "platform": elsewhere} for r in routes])
        assert "profile_routes" in react(f"{elsewhere}:{origin['chat_id']}").get("error", "")
        more_specific = {k: origin[k] for k in ("chat_id", "thread_id", "user_id") if k in origin}
        satellite.runner.config.profile_routes = parse_profile_routes(routes + [dict(
            name="other", platform=origin["platform"], profile="other", **more_specific)])
        assert "profile_routes" in react(f"{origin['platform']}:{origin['chat_id']}").get("error", "")
    finally:
        clear_session_vars(tokens)
    assert adapter.reactions == [origin["chat_id"]]


def test_origin_topic_does_not_vouch_for_the_rest_of_its_chat(satellite):
    """A Telegram topic id is scoped to its chat, so a target without it is the General topic,
    which a topic-pinned route does not dispatch to ``sec``: the origin rule must match what the send
    reaches, not the origin's topic."""
    from gateway.session_context import clear_session_vars, set_session_vars

    satellite.runner.config.profile_routes = parse_profile_routes([
        {"name": "t", "platform": "telegram", "chat_id": "-100222", "thread_id": "7", "profile": "sec"}])
    tg = satellite.primary[Platform.TELEGRAM]
    tokens = set_session_vars(profile="sec", platform="telegram", chat_id="-100222", thread_id="7")
    try:
        refused = _tool({"action": "react", "target": "telegram:-100222", "emoji": "👍", "message_id": "5"})
        sent = _tool({"action": "react", "target": "telegram:-100222:7", "emoji": "👍", "message_id": "5"})
    finally:
        clear_session_vars(tokens)
    assert "profile_routes" in refused.get("error", ""), refused
    assert sent.get("success") is True, sent
    assert tg.reactions == ["-100222"]


def test_shared_channel_origin_matches_on_the_turn_sender(satellite):
    """A shared channel binds no route user (``session_env_source``), but dispatch matched the
    sender: a ``user_id`` route admits that sender's turn, and a turn other senders joined (no single
    actor) falls back to the strict rule."""
    from gateway.session_context import clear_session_vars, set_session_vars

    satellite.runner.config.profile_routes = parse_profile_routes([
        {"name": "u", "platform": "slack", "user_id": "U1", "profile": "sec"}])
    slack = satellite.primary[Platform.SLACK]
    react = {"action": "react", "target": "slack:C0SHARED", "emoji": "👍", "message_id": "5"}
    tokens = set_session_vars(profile="sec", platform="slack", chat_id="C0SHARED", actor_id="U1")
    try:
        assert _tool(react).get("success") is True
    finally:
        clear_session_vars(tokens)
    tokens = set_session_vars(profile="sec", platform="slack", chat_id="C0SHARED", actor_id="U1",
                              turn_event=SimpleNamespace(user_id=""))
    try:
        assert "profile_routes" in _tool(react).get("error", "")
    finally:
        clear_session_vars(tokens)
    assert slack.reactions == ["C0SHARED"]


# Platforms whose thread is a sub-conversation of a chat that keeps its own id: a thread-less target
# is that chat's top level, not the origin thread (unlike Discord, where the thread's chat id IS it).
_GROUP_PLUS_THREAD = {
    "feishu": ("oc_1", "omt_7"),
    "google_chat": ("spaces/AAA", "spaces/AAA/threads/T7"),
}


@pytest.mark.parametrize("platform_name", sorted(_GROUP_PLUS_THREAD))
def test_thread_pinned_origin_does_not_admit_its_groups_top_level(satellite, platform_name):
    """A route pinned to one Feishu/Google Chat thread dispatches only that thread to ``sec``; its
    group's top level (a thread-less send/react) is not ``sec``'s conversation."""
    from gateway.profile_routing import match_profile_route
    from gateway.session_context import clear_session_vars, set_session_vars

    platform = Platform(platform_name)
    chat, thread = _GROUP_PLUS_THREAD[platform_name]
    satellite.runner.config.profile_routes = parse_profile_routes([
        {"name": "t", "platform": platform.value, "chat_id": chat, "thread_id": thread, "profile": "sec"}])
    assert match_profile_route(satellite.runner.config.profile_routes, platform.value,
                               chat_id=chat, thread_id=None) is None  # dispatch premise
    allowed = satellite.runner._shared_bot_target_allowed
    tokens = set_session_vars(profile="sec", platform=platform.value, chat_id=chat, thread_id=thread)
    try:
        assert allowed(platform, "sec", chat, None) is False
        assert allowed(platform, "sec", chat, thread) is True
        assert allowed(platform, "sec", chat, thread + "x") is False
    finally:
        clear_session_vars(tokens)


def test_discord_thread_origin_still_admits_itself_without_a_thread_id(satellite):
    """A Discord thread's chat id is the thread id, so a thread-less target there IS the thread."""
    from gateway.session_context import clear_session_vars, set_session_vars

    satellite.runner.config.profile_routes = parse_profile_routes([
        {"name": "t", "platform": "discord", "chat_id": "111", "thread_id": "555", "profile": "sec"}])
    allowed = satellite.runner._shared_bot_target_allowed
    tokens = set_session_vars(profile="sec", platform="discord", chat_id="555", thread_id="555",
                              parent_chat_id="111")
    try:
        assert allowed(Platform.DISCORD, "sec", "555", None) is True
        assert allowed(Platform.DISCORD, "sec", "555", "555") is True
        assert allowed(Platform.DISCORD, "sec", "555", "999") is False
    finally:
        clear_session_vars(tokens)


# Chat-scoped thread ids (Telegram topic, Slack thread_ts, Matrix thread root, Mattermost root post):
# a chat-only route matches every thread of its chat, so only dispatch's most specific pick says
# whose a pinned thread is.
_CHAT_SCOPED = {
    "telegram": ("-100333", "7", "8"),
    "slack": ("C0SHARED", "1700000000.000100", "1700000000.000200"),
    "matrix": ("!room:hs", "$root1", "$root2"),
    "mattermost": ("chan1", "post1", "post2"),
}


def _sibling_pinned_routes(platform, chat, pinned):
    return [{"name": "c", "platform": platform, "chat_id": chat, "profile": "sec"},
            {"name": "t", "platform": platform, "chat_id": chat, "thread_id": pinned, "profile": "other"}]


@pytest.mark.parametrize("platform_name", sorted(_CHAT_SCOPED))
@pytest.mark.parametrize("in_turn", [True, False], ids=["in-turn", "no-origin"])
def test_chat_route_does_not_reach_a_thread_pinned_to_another_profile(satellite, platform_name, in_turn):
    """``chat -> sec`` plus ``chat + T2 -> other``: dispatch hands T2 to ``other``, so the gate must
    refuse it, from a turn in a sibling thread and with no origin bound (cron)."""
    from gateway.profile_routing import match_profile_route
    from gateway.session_context import clear_session_vars, set_session_vars

    platform = Platform(platform_name)
    chat, own, pinned = _CHAT_SCOPED[platform_name]
    routes = satellite.runner.config.profile_routes = parse_profile_routes(
        _sibling_pinned_routes(platform_name, chat, pinned))
    assert match_profile_route(routes, platform_name, chat_id=chat, thread_id=pinned).profile == "other"
    allowed = satellite.runner._shared_bot_target_allowed
    tokens = (set_session_vars(profile="sec", platform=platform_name, chat_id=chat, thread_id=own)
              if in_turn else None)
    try:
        assert allowed(platform, "sec", chat, pinned) is False
        assert allowed(platform, "sec", chat, own) is True
        assert allowed(platform, "sec", chat, None) is True
    finally:
        if tokens is not None:
            clear_session_vars(tokens)


@pytest.mark.parametrize("platform_name", sorted(_CHAT_SCOPED))
def test_cron_shared_view_does_not_reach_a_thread_pinned_to_another_profile(tmp_path, monkeypatch,
                                                                           platform_name):
    """Cron's view (routes read from the primary home, as the ticker builds it): a chat route does not
    lend the shared bot for a thread a more specific route dispatches to another profile."""
    import hermes_yaml as yaml
    from cron.scheduler_preflight import SharedRouteAdapters, _primary_profile_routes_for_current_home
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    chat, own, pinned = _CHAT_SCOPED[platform_name]
    root = tmp_path / "root"
    (root / "profiles" / "sec").mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump({"gateway": {
        "multiplex_profiles": True, "profile_routes": _sibling_pinned_routes(platform_name, chat, pinned)}}),
        encoding="utf-8")
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    primary = object()
    platform = Platform(platform_name)
    token = set_hermes_home_override(str(root / "profiles" / "sec"))
    try:
        shared = SharedRouteAdapters({platform: primary}, _primary_profile_routes_for_current_home())
    finally:
        reset_hermes_home_override(token)
    assert shared.get(platform, {"chat_id": chat, "thread_id": pinned}) is None
    assert shared.get(platform, {"chat_id": chat, "thread_id": own}) is primary
    assert shared.get(platform, {"chat_id": chat}) is primary


def test_cron_shared_view_refuses_every_target_when_the_route_read_fails(tmp_path, monkeypatch):
    """A route-read error must not look like "no other profile has routes": cron's view then lends
    the shared bot for nothing, not by the satellite's routes alone (which would admit a topic a
    more specific route pins to another profile)."""
    import hermes_yaml as yaml
    from cron import scheduler_preflight
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    chat, own, pinned = _CHAT_SCOPED["telegram"]
    root = tmp_path / "root"
    (root / "profiles" / "sec").mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump({"gateway": {
        "multiplex_profiles": True, "profile_routes": _sibling_pinned_routes("telegram", chat, pinned)}}),
        encoding="utf-8")
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    real_read, reads = scheduler_preflight._primary_routes, []

    def read_once_then_fail(home):
        reads.append(home)
        if len(reads) > 1:
            raise OSError("config.yaml rewritten mid-read")
        return real_read(home)

    monkeypatch.setattr(scheduler_preflight, "_primary_routes", read_once_then_fail)
    primary = object()
    token = set_hermes_home_override(str(root / "profiles" / "sec"))
    try:
        sec_routes = scheduler_preflight._primary_profile_routes_for_current_home()
        assert [r.profile for r in sec_routes] == ["sec"]
        # the satellite's routes came from a read that succeeded; the default dispatch read raises
        shared = scheduler_preflight.SharedRouteAdapters({Platform.TELEGRAM: primary}, sec_routes)
    finally:
        reset_hermes_home_override(token)
    for thread in (pinned, own, None):
        assert shared.get(Platform.TELEGRAM, {"chat_id": chat, "thread_id": thread}) is None


def test_cron_shared_view_is_built_from_one_route_read(tmp_path, monkeypatch):
    """``shared_route_adapters_for_current_home`` reads the routes once: a failing read refuses
    every target, a good one keeps the satellite's chat and refuses the topic pinned elsewhere."""
    import hermes_yaml as yaml
    from cron import scheduler_preflight
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    chat, own, pinned = _CHAT_SCOPED["telegram"]
    root = tmp_path / "root"
    (root / "profiles" / "sec").mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump({"gateway": {
        "multiplex_profiles": True, "profile_routes": _sibling_pinned_routes("telegram", chat, pinned)}}),
        encoding="utf-8")
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    real_read, reads = scheduler_preflight._primary_routes, []
    monkeypatch.setattr(scheduler_preflight, "_primary_routes",
                        lambda home: reads.append(home) or real_read(home))
    primary = object()
    token = set_hermes_home_override(str(root / "profiles" / "sec"))
    try:
        good = scheduler_preflight.shared_route_adapters_for_current_home({Platform.TELEGRAM: primary})
        assert len(reads) == 1
        monkeypatch.setattr(scheduler_preflight, "_primary_routes",
                            lambda home: reads.append(home) or (_ for _ in ()).throw(OSError("unreadable")))
        failed = scheduler_preflight.shared_route_adapters_for_current_home({Platform.TELEGRAM: primary})
        assert len(reads) == 2  # the failed read is not retried for the dispatch table
    finally:
        reset_hermes_home_override(token)
    assert good.get(Platform.TELEGRAM, {"chat_id": chat, "thread_id": own}) is primary
    assert good.get(Platform.TELEGRAM, {"chat_id": chat, "thread_id": pinned}) is None
    assert not failed
    assert failed.get(Platform.TELEGRAM, {"chat_id": chat, "thread_id": own}) is None


@pytest.mark.parametrize("in_turn", [True, False], ids=["in-turn", "no-origin"])
def test_google_chat_space_top_level_refused_when_another_profile_owns_a_thread(satellite, in_turn):
    """Outside cron, a thread-less Google Chat send replies under the space's last inbound thread
    (``_resolve_thread_id``), which may be the thread pinned to ``other``: the gate cannot vouch for
    it. An explicit thread of ``sec``'s still goes, and a space with no foreign thread is unaffected."""
    from gateway.session_context import clear_session_vars, set_session_vars

    space, pinned = "spaces/AAA", "spaces/AAA/threads/T2"
    satellite.runner.config.profile_routes = parse_profile_routes([
        {"name": "s", "platform": "google_chat", "chat_id": space, "profile": "sec"},
        {"name": "b", "platform": "google_chat", "chat_id": "spaces/BBB", "profile": "sec"},
        {"name": "t", "platform": "google_chat", "chat_id": space, "thread_id": pinned, "profile": "other"}])
    allowed = satellite.runner._shared_bot_target_allowed
    gchat = Platform("google_chat")
    tokens = (set_session_vars(profile="sec", platform="google_chat", chat_id=space) if in_turn else None)
    try:
        assert allowed(gchat, "sec", space, None) is False
        assert allowed(gchat, "sec", space, pinned) is False
        assert allowed(gchat, "sec", "spaces/BBB", None) is True
    finally:
        if tokens is not None:
            clear_session_vars(tokens)


@pytest.mark.parametrize("platform_name,routes,chat", [
    # A Telegram DM's chat id is its sender's id: dispatch hands U's DM to the ``user_id`` route.
    ("telegram", [{"name": "dm", "platform": "telegram", "chat_id": "4242", "profile": "sec"},
                  {"name": "u", "platform": "telegram", "user_id": "4242", "profile": "other"}], "4242"),
    # A ``guild_id + chat_id`` route outranks a bare ``chat_id`` route on the same channel.
    ("discord", [{"name": "c", "platform": "discord", "chat_id": "111", "profile": "sec"},
                 {"name": "gc", "platform": "discord", "guild_id": "900", "chat_id": "111", "profile": "other"}],
     "111"),
], ids=["telegram-dm-user-route", "discord-guild-chat-route"])
def test_more_specific_route_of_another_profile_wins_without_an_origin(satellite, platform_name, routes, chat):
    """No origin bound (cron, webhook): the satellite's chat route lends the shared bot only where
    dispatch would also pick it."""
    from gateway.profile_routing import match_profile_route

    satellite.runner.config.profile_routes = parse_profile_routes(routes)
    inbound = {"user_id": chat} if platform_name == "telegram" else {"guild_id": "900"}
    assert match_profile_route(satellite.runner.config.profile_routes, platform_name,
                               chat_id=chat, **inbound).profile == "other"
    assert satellite.runner._shared_bot_target_allowed(Platform(platform_name), "sec", chat, None) is False


def test_google_chat_space_guard_ignores_own_threads_and_matches_thread_only_routes(satellite):
    """The satellite's own thread routes don't close its space's top level; another profile's
    ``thread_id``-only route (no ``chat_id``) does."""
    gchat = Platform("google_chat")
    allowed = satellite.runner._shared_bot_target_allowed
    satellite.runner.config.profile_routes = parse_profile_routes([
        {"name": "s", "platform": "google_chat", "chat_id": "spaces/AAA", "profile": "sec"},
        {"name": "st", "platform": "google_chat", "chat_id": "spaces/AAA",
         "thread_id": "spaces/AAA/threads/T1", "profile": "sec"}])
    assert allowed(gchat, "sec", "spaces/AAA", None) is True
    satellite.runner.config.profile_routes = parse_profile_routes([
        {"name": "s", "platform": "google_chat", "chat_id": "spaces/AAA", "profile": "sec"},
        {"name": "t", "platform": "google_chat", "thread_id": "spaces/AAA/threads/T2", "profile": "other"}])
    assert allowed(gchat, "sec", "spaces/AAA", None) is False
