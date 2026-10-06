"""profiles.describe / profiles.configure resolve credentials from the described profile's scope.

``_hermes_home_scope`` used to bind only the HERMES_HOME override. Under multi-profile hosting the
toolset snapshot's ``XAI_API_KEY`` read then raised ``UnscopedSecretError``, which the best-effort
``_try`` turned into an empty enabled set: every unpinned profile described as "all toolsets off"
(#120726), and a Desktop editor save from that snapshot pinned whatever subset the user re-checked.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import tui_gateway.server as server
from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def hosted(tmp_path, monkeypatch):
    """Launch home with no xAI key + named profile ``bot`` whose own ``.env`` carries one; multi-profile
    hosting active the way ``hermes serve`` activates it at boot."""
    import tui_gateway.launch_profile_policy as policy
    from agent.secret_scope import set_multiplex_active

    launch = tmp_path / ".hermes"
    bot = launch / "profiles" / "bot"
    bot.mkdir(parents=True)
    (bot / ".env").write_text("XAI_API_KEY=xai-bot-only\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setattr(policy, "_snapshot", None)
    policy.activate_multi_profile_hosting()
    try:
        yield launch, bot
    finally:
        set_multiplex_active(False)


def _enabled_under_own_scope(home: Path) -> set:
    """What `hermes -p <profile> tools list` resolves: that profile's config under its own secrets."""
    from hermes_cli.config import load_config
    from hermes_cli.tools_config import _get_platform_tools

    home_token = set_hermes_home_override(str(home))
    secret_token = set_secret_scope(build_profile_secret_scope(home), profile_home=str(home))
    try:
        return set(_get_platform_tools(load_config() or {}, "cli", include_default_mcp_servers=False))
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


def _described(name: str) -> dict:
    resp = server._methods["profiles.describe"](1, {"name": name})
    assert "error" not in resp, resp.get("error")
    return {t["name"]: t["enabled"] for t in resp["result"]["toolsets"]}


def test_describe_reports_each_profiles_own_enabled_toolsets_under_multiplex(hosted):
    launch, bot = hosted
    environ_before = dict(os.environ)

    for name, home in (("default", launch), ("bot", bot)):
        described = _described(name)
        expected = _enabled_under_own_scope(home)
        assert expected, "the platform default must enable something for the comparison to mean anything"
        assert {ts for ts, on in described.items() if on} == expected & set(described)

    # The secondary's own key resolves (x_search auto-enables on xAI creds); the launch profile's does not.
    assert _described("bot").get("x_search") is True
    assert _described("default").get("x_search") is not True
    assert dict(os.environ) == environ_before
