"""send_message must deliver through the ACTIVE PROFILE's gateway adapter under multiplex.

``runner.adapters`` holds the default profile's bots; a secondary profile's turn that resolved the
live adapter by bare platform posted (and reacted) with the default bot's identity.  The lookup must
honour ``_profile_adapters[profile]`` and fail closed (``None`` → scoped standalone sender / error)
when the profile has no adapter for that platform — never the default bot.
"""
from pathlib import Path

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner, _profile_runtime_scope
from tools.send_message_senders import _live_adapter


@pytest.fixture
def mux_runner(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "profiles" / "sec").mkdir(parents=True)
    (home / "profiles" / "nobot").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    default_slack, sec_slack = object(), object()
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.SLACK: default_slack}
    runner._profile_adapters = {"sec": {Platform.SLACK: sec_slack}, "nobot": {}}
    runner._primary_profile_name = "default"
    import gateway.run as gateway_run
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)
    return home, default_slack, sec_slack


def test_secondary_profile_turn_resolves_its_own_adapter(mux_runner):
    home, default_slack, sec_slack = mux_runner
    with _profile_runtime_scope(home / "profiles" / "sec", {}):
        _, adapter = _live_adapter(Platform.SLACK)
    assert adapter is sec_slack
    _, adapter = _live_adapter(Platform.SLACK)  # default scope still gets the default bot
    assert adapter is default_slack


def test_profile_without_adapter_fails_closed_never_default_bot(mux_runner):
    home, default_slack, _ = mux_runner
    with _profile_runtime_scope(home / "profiles" / "nobot", {}):
        _, adapter = _live_adapter(Platform.SLACK)
    assert adapter is None
