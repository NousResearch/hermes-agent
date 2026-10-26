"""@hermes is the default profile's handle only while no real profile is named 'hermes'.

A pre-reservation ``profiles/hermes`` keeps the handle: a local ``message_agent`` to @hermes
reaches it, and the roster shows exactly one @hermes (the default answers to @default).
"""

import json

import pytest

from tests.tools.test_bot_mode_dm import _canonical_target, _capture_spawn, _FakeAgent, _managed_home
from tools import bot_mode_dm, bot_mode_probe, bot_relay


@pytest.fixture(autouse=True)
def _fresh_probe_cache():
    bot_mode_probe._reset_cache_for_tests()
    yield
    bot_mode_probe._reset_cache_for_tests()


@pytest.mark.parametrize("real_hermes", [False, True])
def test_local_hermes_dm_lands_once_in_the_profile_that_owns_the_handle(tmp_path, monkeypatch, real_hermes):
    _capture_spawn(monkeypatch)
    monkeypatch.setattr(bot_relay, "_hermes_cli", lambda: "hermes")
    monkeypatch.setattr(bot_mode_dm, "_dm_dir", lambda: tmp_path)
    root = _managed_home(tmp_path, teammates=("researcher", "hermes") if real_hermes else ("researcher",))
    owner = root / "profiles" / "hermes" if real_hermes else root
    authority = _canonical_target(monkeypatch, root, root / "profiles" / "hermes")

    ack = json.loads(bot_mode_dm.message_agent_tool(
        target="@hermes", message="ping", agent=_FakeAgent(root / "profiles" / "researcher")))

    assert ack["status"] == "queued" and ack["to"] == "@hermes"
    assert [home for home, _params in authority.calls] == [owner.resolve()]
