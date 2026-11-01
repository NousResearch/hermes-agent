"""Owned Browser Use lifecycle composes with profile-scoped backend publication."""

import json
import subprocess

import pytest

from gateway.run import _profile_runtime_scope
from hermes_constants import hermes_home_key
from tools import browser_session
from tools import browser_use_cli as cli


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setattr(cli, "_browser_use_sessions", {})
    monkeypatch.setattr(cli, "_driven_daemons", {})
    monkeypatch.setattr(cli, "_find_cli", lambda: ["browser-use"])
    monkeypatch.setattr(browser_session, "build_browser_env", dict)
    monkeypatch.setattr(cli, "_code_url_verdict", lambda code: cli._CodeUrlVerdict(None, False))
    monkeypatch.setattr(cli, "_route_backend", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_attach_vault_supervisor", lambda *args: None)
    monkeypatch.setattr(cli, "_workspace_dir", lambda task: None)
    monkeypatch.setattr(cli, "_read_browser_cfg", dict)
    monkeypatch.setattr(cli, "_ensure_browser_use_cleanup_thread", lambda: None)
    monkeypatch.setattr(cli, "_find_screenshot", lambda *args: None)
    monkeypatch.setattr(cli, "_live_generation", lambda: 0)
    calls = []

    def run(cmd, code, env, timeout):
        calls.append((cmd, dict(env)))
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    monkeypatch.setattr(cli, "_run_cli_killing_process_group", run)
    return calls


def test_inactivity_timeout_reads_the_active_profile(tmp_path):
    (tmp_path / "config.yaml").write_text("browser:\n  inactivity_timeout: 95\n")
    with _profile_runtime_scope(tmp_path, hydrate_secrets=False):
        assert cli._browser_use_inactivity_timeout() == 95


def test_profile_reload_keeps_other_owned_runtime(tmp_path, harness):
    a, b = tmp_path / "a", tmp_path / "b"
    for home in (a, b):
        home.mkdir()
        with _profile_runtime_scope(home, hydrate_secrets=False):
            result = cli.browser_exec("print(1)")
            assert isinstance(result, str)
            assert json.loads(result)["success"]
    states = list(cli._browser_use_sessions.values())
    assert len(states) == 2
    assert len({state["env"]["BH_RUNTIME_DIR"] for state in states}) == 2

    cli.stop_harness_daemons(hermes_home_key(a))

    remaining = list(cli._browser_use_sessions.values())
    assert len(remaining) == 1
    assert remaining[0]["home_key"] == hermes_home_key(b)
    assert harness[-1][0][-1] == "--reload"
    assert harness[-1][1]["BH_RUNTIME_DIR"] == states[0]["env"]["BH_RUNTIME_DIR"]


def test_republish_reloads_captured_runtime_under_operation_lock(tmp_path, harness, monkeypatch):
    generation = [0]
    monkeypatch.setattr(cli, "_live_generation", lambda: generation[0])
    reloads = []

    def run(cmd, code, env, timeout):
        if cmd[-1] == "--reload":
            state = next(iter(cli._browser_use_sessions.values()))
            assert state["operation_lock"].locked()
            assert env == state["env"]
            reloads.append(env["BH_RUNTIME_DIR"])
        else:
            generation[0] += 1
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    monkeypatch.setattr(cli, "_run_cli_killing_process_group", run)
    with _profile_runtime_scope(tmp_path, hydrate_secrets=False):
        result = cli.browser_exec("print(1)")
        assert isinstance(result, str)
        assert json.loads(result)["success"]
    assert len(reloads) == 1
    assert cli._browser_use_sessions == {}
