"""The standalone backend serves the Desktop Daybreak switch (`agent.daybreak`) like the native owner.

Desktop shows the switch on catalog-eligible models from any backend, including `hermes serve` over
SSH, so `config.get` / `config.set` must know the key here too, and a flip must reach this profile's
live agents (requests check eligibility themselves: agent/codex_service_tiers.py).
"""

from __future__ import annotations

from types import SimpleNamespace

import hermes_yaml as yaml
import pytest

from tui_gateway import server


@pytest.fixture
def homes(tmp_path, monkeypatch):
    launch, worker = tmp_path / "launch", tmp_path / "profiles" / "code"
    launch.mkdir(parents=True)
    worker.mkdir(parents=True)
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(server, "_profile_home", lambda name: worker if (name or "").strip() == "code" else None)
    server._cfg_cache = server._cfg_sig = server._cfg_path = None
    yield launch, worker
    server._cfg_cache = server._cfg_sig = server._cfg_path = None


def _call(method, **params):
    return server._methods[method](1, params)


def test_the_switch_round_trips_through_agent_daybreak(homes):
    launch, _worker = homes
    assert _call("config.get", key="daybreak")["result"] == {"value": False}  # default off

    assert _call("config.set", key="daybreak", value=True)["result"] == {"key": "daybreak", "value": True}
    assert yaml.safe_load((launch / "config.yaml").read_text())["agent"]["daybreak"] is True
    assert _call("config.get", key="daybreak")["result"] == {"value": True}

    assert _call("config.set", key="daybreak", value="false")["result"] == {"key": "daybreak", "value": False}
    assert _call("config.get", key="daybreak")["result"] == {"value": False}


def test_a_non_boolean_is_refused_rather_than_written(homes):
    launch, _worker = homes
    assert _call("config.set", key="daybreak", value="sometimes")["error"]["code"] == 4002
    assert not (launch / "config.yaml").exists()


def test_a_flip_reaches_only_this_profiles_local_agents(homes, monkeypatch):
    _launch, worker = homes
    launch_agent = SimpleNamespace(daybreak=False, platform="desktop")
    worker_agent = SimpleNamespace(daybreak=False, platform="tui")
    bot_agent = SimpleNamespace(daybreak=False, platform="telegram")  # a plugin session in the same profile
    monkeypatch.setitem(server._sessions, "launch-sid", {"agent": launch_agent, "profile_home": None})
    monkeypatch.setitem(server._sessions, "worker-sid", {"agent": worker_agent, "profile_home": str(worker)})
    monkeypatch.setitem(server._sessions, "bot-sid", {"agent": bot_agent, "profile_home": str(worker)})
    monkeypatch.setitem(server._sessions, "unbuilt-sid", {"agent": None, "profile_home": str(worker)})

    assert _call("config.set", key="daybreak", value=True, profile="code")["result"]["value"] is True
    assert worker_agent.daybreak is True
    assert launch_agent.daybreak is False  # another profile's setting
    assert bot_agent.daybreak is False  # someone else's chat: OpenAI allows no third-party access
    assert yaml.safe_load((worker / "config.yaml").read_text())["agent"]["daybreak"] is True

    assert _call("config.set", key="daybreak", value=True)["result"]["value"] is True  # launch profile
    assert launch_agent.daybreak is True


def test_only_the_owners_local_surfaces_carry_it():
    from agent.session_source import DAYBREAK_SOURCES
    assert DAYBREAK_SOURCES == {"tui", "desktop", "cli"}


@pytest.mark.parametrize(("source", "expected"), [(None, True), ("desktop", True), ("telegram", False)])
def test_new_agents_read_the_switch_only_for_local_surfaces(monkeypatch, source, expected):
    """``_make_agent`` end to end with a stand-in agent class: the switch is on in config.yaml, and a plugin
    session tagged ``telegram`` built by this backend still gets standard requests."""
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
    monkeypatch.delenv("HERMES_TUI_PROVIDER", raising=False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"agent": {"daybreak": True},
                                                       "model": {"default": "gpt-6-sol", "provider": "openai-codex"}})
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider",
                        lambda requested=None, target_model=None: {
                            "provider": "openai-codex", "base_url": "https://chatgpt.com/backend-api/codex",
                            "api_key": "token", "api_mode": "codex_responses", "credential_pool": None})
    monkeypatch.setattr("run_agent.AIAgent", lambda **kwargs: SimpleNamespace(model=kwargs.get("model")))
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda platform=None: ["file"])
    monkeypatch.setattr(server, "_get_db", lambda: None)

    agent = server._make_agent("sid", "session-key", platform_override=source)
    assert agent.daybreak is expected
