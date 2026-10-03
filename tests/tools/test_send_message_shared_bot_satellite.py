"""``send`` from a credentialless multiplex satellite borrows its host profile's bot (t_ab4225a1).

A satellite served through the host's ``gateway.profile_routes`` (``bot_profile`` unset) holds no
platform token of its own. ``hermes send`` / ``send_message`` must resolve the HOST's token from the
host's own secret scope — never writing it into ``os.environ``, the installed scope or any ``.env``
— while a profile with its own credential keeps using it and an ambiguous host fails closed.
"""
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, load_gateway_config
from tools.send_message_tool import _resolve_platform_config, send_message_tool

HOST_TOKEN = "xoxb-host-0000-test"
OWN_TOKEN = "xoxb-own-1111-test"


def _route_cfg(profile, platform="slack", chat_id="C0BVBHM4AS0", **extra):
    route = {"name": f"{platform}-{profile}", "profile": profile, "platform": platform, "chat_id": chat_id}
    route.update(extra)
    return route


def _write_host(home: Path, routes, token=HOST_TOKEN):
    home.mkdir(parents=True, exist_ok=True)
    import yaml
    (home / "config.yaml").write_text(yaml.safe_dump(
        {"gateway": {"multiplex_profiles": True, "profile_routes": routes},
         "platforms": {"slack": {"extra": {"unfurl_links": False}}}}), encoding="utf-8")
    (home / ".env").write_text(f"SLACK_BOT_TOKEN={token}\nSLACK_APP_TOKEN=xapp-host\n" if token else "X=1\n",
                               encoding="utf-8")


@pytest.fixture
def estate(tmp_path, monkeypatch):
    """Root with default + host 'gen' + satellite 'sat'; HERMES_HOME = sat (standalone CLI shape)."""
    root = tmp_path / ".hermes"
    root.mkdir()
    (root / "config.yaml").write_text("{}\n", encoding="utf-8")
    profiles = root / "profiles"
    gen, sat = profiles / "gen", profiles / "sat"
    _write_host(gen, [_route_cfg("sat")])
    sat.mkdir(parents=True)
    (sat / "config.yaml").write_text("{}\n", encoding="utf-8")
    (sat / ".env").write_text("SLACK_HOME_CHANNEL=C0SATHOME\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(sat))
    for name in ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_HOME_CHANNEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLACK_HOME_CHANNEL", "C0SATHOME")
    return root, gen, sat


def _env_snapshot():
    return {k: v for k, v in os.environ.items() if k.startswith("SLACK_")}


def test_satellite_resolves_host_token_without_leaking(estate):
    root, gen, sat = estate
    before_env = _env_snapshot()
    before_sat_env = (sat / ".env").read_text()
    from agent.secret_scope import current_secret_scope
    before_scope = dict(current_secret_scope() or {})

    platform, pconfig, _entry, err = _resolve_platform_config("slack", load_gateway_config())

    assert err is None and platform is Platform.SLACK
    assert pconfig.token == HOST_TOKEN and pconfig.enabled
    assert pconfig.extra.get("unfurl_links") is False  # host's platform block rides along
    assert pconfig.home_channel is None  # satellite home channel stays authoritative
    assert _env_snapshot() == before_env
    assert HOST_TOKEN not in os.environ.values()
    assert (sat / ".env").read_text() == before_sat_env
    assert dict(current_secret_scope() or {}) == before_scope


def test_satellite_send_uses_host_token_and_own_home_channel(estate):
    """End-to-end dry path: ``send_message_tool`` reaches the Slack sender with the host's token and
    the satellite's home channel; the mirror stays with the satellite."""
    captured = {}

    async def fake_send(platform, pconfig, chat_id, message, **kw):
        captured.update(token=pconfig.token, chat_id=chat_id, message=message)
        return {"success": True, "message_id": "1"}

    with patch("tools.send_message_tool._send_to_platform", side_effect=fake_send), \
            patch("tools.send_message_tool._mirror_sent_message", return_value=True) as mirror:
        result = json.loads(send_message_tool({"action": "send", "target": "slack", "message": "hi"}))
        explicit = json.loads(send_message_tool({"action": "send", "target": "slack:C0BTM8L69HQ", "message": "x"}))

    assert result["success"] is True, result
    assert captured["token"] == HOST_TOKEN
    assert explicit.get("success") is True, explicit
    assert captured["chat_id"] == "C0BTM8L69HQ"  # any chat on the platform
    assert mirror.call_args_list[0].args[:2] == ("slack", "C0SATHOME")
    assert HOST_TOKEN not in os.environ.values()


def test_own_token_profile_unchanged(estate):
    root, gen, sat = estate
    (sat / ".env").write_text(f"SLACK_BOT_TOKEN={OWN_TOKEN}\n", encoding="utf-8")
    os.environ["SLACK_BOT_TOKEN"] = OWN_TOKEN
    try:
        _, pconfig, _, err = _resolve_platform_config("slack", load_gateway_config())
    finally:
        os.environ.pop("SLACK_BOT_TOKEN", None)
    assert err is None and pconfig.token == OWN_TOKEN


def test_own_disabled_credential_never_borrows_host(estate):
    """A profile holding its own token with the platform disabled is its own credential boundary."""
    from gateway.config import PlatformConfig
    config = GatewayConfig()
    config.platforms[Platform.SLACK] = PlatformConfig(enabled=False, token=OWN_TOKEN)
    _, pconfig, _, err = _resolve_platform_config("slack", config)
    assert pconfig is None and "not configured" in err and "own slack credential" in err


def test_ambiguous_host_fails_closed(estate):
    root, gen, sat = estate
    _write_host(root / "profiles" / "other", [_route_cfg("sat")], token="xoxb-other")
    _, pconfig, _, err = _resolve_platform_config("slack", load_gateway_config())
    assert pconfig is None
    assert "is not configured" in err and "ambiguous" in err
    assert HOST_TOKEN not in err and "xoxb-other" not in err


def test_ambiguity_broken_by_live_gateway_serving_profile(estate, monkeypatch):
    root, gen, sat = estate
    _write_host(root / "profiles" / "other", [_route_cfg("sat")], token="xoxb-other")
    import tools.send_message_shared_bot as sb
    monkeypatch.setattr(sb, "_live_gateway_serves", lambda home, name: Path(home).name == "gen")
    _, pconfig, _, err = _resolve_platform_config("slack", load_gateway_config())
    assert err is None and pconfig.token == HOST_TOKEN


@pytest.mark.parametrize("route_extra, reason", [
    ({"enabled": False}, "no profile has an enabled"),
    ({"bot_profile": "gen"}, "no profile has an enabled"),
    ({"platform": "telegram"}, "no profile has an enabled"),
    ({"profile": "someone-else"}, "no profile has an enabled"),
])
def test_ineligible_routes_fail_closed(estate, route_extra, reason):
    root, gen, sat = estate
    route = _route_cfg("sat")
    route.update(route_extra)
    _write_host(gen, [route])
    _, pconfig, _, err = _resolve_platform_config("slack", load_gateway_config())
    assert pconfig is None and "is not configured" in err and reason in err


def test_host_without_token_fails_closed(estate):
    root, gen, sat = estate
    _write_host(gen, [_route_cfg("sat")], token=None)
    _, pconfig, _, err = _resolve_platform_config("slack", load_gateway_config())
    assert pconfig is None and "has no SLACK_BOT_TOKEN" in err


def test_lookup_fault_fails_closed(estate, monkeypatch):
    import tools.send_message_shared_bot as sb

    def boom(*a, **k):
        raise RuntimeError("broken")
    monkeypatch.setattr(sb, "resolve_shared_bot_host", boom)
    _, pconfig, _, err = _resolve_platform_config("slack", load_gateway_config())
    assert pconfig is None and "host lookup failed (RuntimeError)" in err


def test_in_process_satellite_turn_prefers_host_live_adapter(estate, monkeypatch):
    """Inside the multiplexed host gateway the token is only a fallback: the satellite's send goes
    through the host's live adapter (``_adapters_for_profile`` shared-bot rule)."""
    root, gen, sat = estate
    from gateway.run import GatewayRunner, _profile_runtime_scope
    from agent.secret_scope import build_profile_secret_scope
    from gateway.profile_routing import parse_profile_routes
    from types import SimpleNamespace
    import gateway.run as gateway_run

    host_adapter = SimpleNamespace(send=AsyncMock(return_value=SimpleNamespace(
        success=True, message_id="m1", error=None)))
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.SLACK: host_adapter}
    runner._profile_adapters = {"sat": {}}
    runner._primary_profile_name = "gen"
    runner.config = SimpleNamespace(multiplex_profiles=True,
                                    profile_routes=parse_profile_routes([_route_cfg("sat")]))
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)
    monkeypatch.setattr(gateway_run, "_multiplex_profile_homes", lambda cfg: [("sat", sat)])

    with patch("tools.send_message_tool._mirror_sent_message", return_value=False):
        with _profile_runtime_scope(sat, build_profile_secret_scope(sat)):
            result = json.loads(send_message_tool({"action": "send", "target": "slack:C0BVBHM4AS0", "message": "hey"}))
    assert result.get("success") is True, result
    host_adapter.send.assert_awaited_once()
    assert host_adapter.send.await_args.kwargs["chat_id"] == "C0BVBHM4AS0"


# --- Review round 1 (Gurney, 68d749d1f8): enabled-but-credentialless block; unreadable census ---

def test_enabled_block_without_own_token_still_borrows(estate):
    """Removing the duplicate token while ``platforms.slack.enabled: true`` stays must still borrow."""
    _, _, sat = estate
    (sat / "config.yaml").write_text("platforms:\n  slack:\n    enabled: true\n", encoding="utf-8")
    before = _env_snapshot()
    _, pconfig, _, err = _resolve_platform_config("slack", load_gateway_config())
    assert err is None and pconfig.token == HOST_TOKEN
    assert _env_snapshot() == before


def test_enabled_block_without_route_keeps_existing_behaviour(estate):
    """No route anywhere: an enabled credentialless block is returned as before (no new refusal)."""
    root, gen, sat = estate
    _write_host(gen, [])
    (sat / "config.yaml").write_text("platforms:\n  slack:\n    enabled: true\n", encoding="utf-8")
    _, pconfig, _, err = _resolve_platform_config("slack", load_gateway_config())
    assert err is None and pconfig is not None and not pconfig.token


def test_enabled_block_with_ambiguous_host_fails_closed(estate):
    root, gen, sat = estate
    _write_host(root / "profiles" / "other", [_route_cfg("sat")], token="xoxb-other")
    (sat / "config.yaml").write_text("platforms:\n  slack:\n    enabled: true\n", encoding="utf-8")
    _, pconfig, _, err = _resolve_platform_config("slack", load_gateway_config())
    assert pconfig is None and "ambiguous" in err


def test_enabled_block_with_own_token_unchanged(estate, monkeypatch):
    _, _, sat = estate
    (sat / "config.yaml").write_text("platforms:\n  slack:\n    enabled: true\n", encoding="utf-8")
    monkeypatch.setenv("SLACK_BOT_TOKEN", OWN_TOKEN)
    _, pconfig, _, err = _resolve_platform_config("slack", load_gateway_config())
    assert err is None and pconfig.token == OWN_TOKEN


def _unreadable(monkeypatch, target):
    import tools.send_message_shared_bot as sb
    original = sb._raw_config

    def raw(home):
        if Path(home) == target:
            raise PermissionError("fixture unreadable config")
        return original(home)
    monkeypatch.setattr(sb, "_raw_config", raw)
    return sb


def test_unreadable_competing_host_fails_closed(estate, monkeypatch):
    root, gen, sat = estate
    other = root / "profiles" / "other"
    _write_host(other, [_route_cfg("sat")], token="xoxb-other")
    _unreadable(monkeypatch, other)
    _, pconfig, _, err = _resolve_platform_config("slack", load_gateway_config())
    assert pconfig is None and "is not configured" in err
    assert "unproven" in err and "other" in err
    assert HOST_TOKEN not in err and "xoxb-other" not in err and "fixture unreadable" not in err


def test_unreadable_profile_with_no_readable_route_fails_closed(estate, monkeypatch):
    """An unreadable profile is uncertainty, never proof that no route exists."""
    root, gen, sat = estate
    _unreadable(monkeypatch, gen)
    _, pconfig, _, err = _resolve_platform_config("slack", load_gateway_config())
    assert pconfig is None and "unproven" in err


def test_unreadable_census_resolved_by_verified_live_host(estate, monkeypatch):
    root, gen, sat = estate
    other = root / "profiles" / "other"
    _write_host(other, [_route_cfg("sat")], token="xoxb-other")
    sb = _unreadable(monkeypatch, other)
    monkeypatch.setattr(sb, "_live_gateway_serves", lambda home, name: Path(home) == gen)
    _, pconfig, _, err = _resolve_platform_config("slack", load_gateway_config())
    assert err is None and pconfig.token == HOST_TOKEN


def test_unreadable_census_not_resolved_by_non_serving_host(estate, monkeypatch):
    root, gen, sat = estate
    other = root / "profiles" / "other"
    _write_host(other, [_route_cfg("sat")], token="xoxb-other")
    sb = _unreadable(monkeypatch, other)
    monkeypatch.setattr(sb, "_live_gateway_serves", lambda home, name: False)
    _, pconfig, _, err = _resolve_platform_config("slack", load_gateway_config())
    assert pconfig is None and "unproven" in err


def test_malformed_host_yaml_is_uncertainty(estate):
    """Real (unmocked) read fault: a malformed competing config fails closed."""
    root, gen, sat = estate
    other = root / "profiles" / "other"
    other.mkdir(parents=True)
    (other / "config.yaml").write_text("gateway: [unclosed\n  profile_routes: {\n", encoding="utf-8")
    _, pconfig, _, err = _resolve_platform_config("slack", load_gateway_config())
    assert pconfig is None and "unproven" in err and "other" in err
