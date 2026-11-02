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


def test_a_flip_reaches_only_this_profiles_live_agents(homes, monkeypatch):
    _launch, worker = homes
    launch_agent, worker_agent = SimpleNamespace(daybreak=False), SimpleNamespace(daybreak=False)
    monkeypatch.setitem(server._sessions, "launch-sid", {"agent": launch_agent, "profile_home": None})
    monkeypatch.setitem(server._sessions, "worker-sid", {"agent": worker_agent, "profile_home": str(worker)})
    monkeypatch.setitem(server._sessions, "unbuilt-sid", {"agent": None, "profile_home": str(worker)})

    assert _call("config.set", key="daybreak", value=True, profile="code")["result"]["value"] is True
    assert worker_agent.daybreak is True
    assert launch_agent.daybreak is False  # another profile's setting
    assert yaml.safe_load((worker / "config.yaml").read_text())["agent"]["daybreak"] is True

    assert _call("config.set", key="daybreak", value=True)["result"]["value"] is True  # launch profile
    assert launch_agent.daybreak is True
