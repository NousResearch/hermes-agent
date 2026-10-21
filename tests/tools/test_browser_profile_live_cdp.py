"""CLI-9 (#134889): a live ``/browser connect`` belongs to ONE profile on a multiplexed gateway.

The override map is keyed by ``hermes_home_key()``; browser sessions and CDP supervisors remember
the profile that created them, so reaping one profile's browsers never closes another's.
"""
import types

import pytest

from gateway.run import _profile_runtime_scope
from hermes_constants import hermes_home_key
from tools import browser_tool, browser_tool_cdp as bt_cdp, browser_tool_lifecycle as lifecycle
from tools import browser_tool_session as bt_session
from tools.browser_supervisor import SUPERVISOR_REGISTRY


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (b / "config.yaml").write_text("browser:\n  cdp_url: ws://b.invalid/devtools/browser/PROFILE-B\n")
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    monkeypatch.setattr(bt_cdp, "_live_overrides", {})
    return a, b


def test_a_live_override_is_this_profiles_alone(profiles, monkeypatch):
    a, b = profiles
    with _profile_runtime_scope(a, hydrate_secrets=False):
        bt_cdp.set_live_cdp_override("http://127.0.0.1:9333")
        assert bt_cdp._get_cdp_override_raw() == "http://127.0.0.1:9333"
    with _profile_runtime_scope(b, hydrate_secrets=False):
        assert bt_cdp.live_cdp_override() == ""
        assert bt_cdp._get_cdp_override_raw().endswith("/PROFILE-B")
    with _profile_runtime_scope(a, hydrate_secrets=False):
        bt_cdp.set_live_cdp_override(None)
        assert bt_cdp._get_cdp_override_raw() == ""


def test_disconnect_falls_back_to_this_profiles_own_env_never_the_launch_env(profiles, monkeypatch):
    """After a disconnect the BROWSER_CDP_URL tier is the profile's own ``.env``; the launch process's
    ``os.environ`` value belongs to the launch profile and must not reach a served one."""
    from tui_gateway.server import _resolve_browser_cdp_url  # bound onto server.py globals
    from tools.browser_camofox import is_camofox_mode

    a, b = profiles
    (b / ".env").write_text("BROWSER_CDP_URL=http://127.0.0.1:7777\n")
    monkeypatch.setenv("BROWSER_CDP_URL", "http://127.0.0.1:6666")  # the launch profile's
    with _profile_runtime_scope(b, hydrate_secrets=False):
        bt_cdp.set_live_cdp_override("http://127.0.0.1:4444")
        assert bt_cdp._get_cdp_override_raw() == "http://127.0.0.1:4444"
        bt_cdp.set_live_cdp_override(None)
        assert bt_cdp._get_cdp_override_raw() == _resolve_browser_cdp_url() == "http://127.0.0.1:7777"
    with _profile_runtime_scope(a, hydrate_secrets=False):  # no BROWSER_CDP_URL in A's .env
        assert bt_cdp._get_cdp_override_raw() == _resolve_browser_cdp_url() == ""
        monkeypatch.setattr("tools.browser_camofox._config_cdp_url", lambda: "")
        monkeypatch.setattr("tools.tool_backend_helpers.read_selection", lambda _: "camofox")
        assert is_camofox_mode() is True  # the launch env's CDP URL does not switch A off Camofox
    assert bt_cdp._get_cdp_override_raw() == "http://127.0.0.1:6666"  # single-profile: env as before


def test_a_new_session_records_its_profile(profiles, monkeypatch):
    a, _b = profiles
    monkeypatch.setattr(browser_tool, "_active_sessions", {})
    monkeypatch.setattr(lifecycle, "_start_browser_cleanup_thread", lambda: None)
    monkeypatch.setattr(bt_session, "_create_session_for_key", lambda key, local: {"session_name": "s"})
    monkeypatch.setattr(bt_cdp, "_ensure_cdp_supervisor", lambda key: None)
    with _profile_runtime_scope(a, hydrate_secrets=False):
        info = bt_session._get_session_info("task-a")
    assert info["home_key"] == hermes_home_key(a)


def test_reaping_one_profile_leaves_the_other_profiles_browser_and_supervisor(profiles, monkeypatch):
    a, b = profiles
    key_a, key_b = hermes_home_key(a), hermes_home_key(b)
    sessions = {"ta": {"home_key": key_a}, "ta::local": {"home_key": key_a}, "tb": {"home_key": key_b}}
    monkeypatch.setattr(browser_tool, "_active_sessions", sessions)
    closed = []

    def close(task_id):
        closed.append(task_id)
        sessions.pop(task_id, None)
    monkeypatch.setattr(lifecycle, "_cleanup_single_browser_session", close)
    stopped = []
    sup = lambda name: types.SimpleNamespace(stop=lambda: stopped.append(name))
    monkeypatch.setattr(SUPERVISOR_REGISTRY, "_by_task", {"ta": sup("ta"), "vault-a": sup("vault-a"), "tb": sup("tb")})
    monkeypatch.setattr(SUPERVISOR_REGISTRY, "_home_by_task", {"ta": key_a, "vault-a": key_a, "tb": key_b})
    lifecycle.cleanup_profile_browsers(key_a)
    assert sorted(closed) == ["ta", "ta::local"] and list(sessions) == ["tb"]
    assert sorted(stopped) == ["ta", "vault-a"] and list(SUPERVISOR_REGISTRY._by_task) == ["tb"]


def test_each_profile_drives_and_reaps_its_own_harness_daemon(profiles, monkeypatch, tmp_path):
    """The harness names a daemon by BU_NAME alone (unnamed exec: ``default``) and reuses a live one, so
    with one shared runtime dir B's exec drove A's connected browser and A's reap stopped B's daemon."""
    import json
    import stat
    from tools import browser_use_cli as bu_cli

    a, b = profiles
    shared = tmp_path / "harness-runtime"
    cli = tmp_path / "browser-harness"  # one daemon per (runtime dir, BU_NAME) latches its first endpoint
    cli.write_text(f'''#!/bin/sh
dir="${{BH_RUNTIME_DIR:-{shared}}}"; mkdir -p "$dir"; latch="$dir/bu-${{BU_NAME:-default}}"
if [ "${{1:-}}" = "--reload" ]; then rm -f "$latch"; exit 0; fi
cat > /dev/null
[ -f "$latch" ] || echo "$BU_CDP_URL" > "$latch"
cat "$latch"
''')
    cli.chmod(cli.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("BH_RUNTIME_DIR", str(shared))
    monkeypatch.setattr(bu_cli, "_find_cli", lambda: [str(cli)])
    monkeypatch.setattr(bu_cli, "_driven_daemons", {})
    monkeypatch.setattr(bu_cli, "_attach_vault_supervisor", lambda env, task_id: None)
    monkeypatch.setattr(bt_cdp, "_resolve_cdp_override", lambda url: url)

    def drive(home, url=None):
        with _profile_runtime_scope(home, hydrate_secrets=False):
            if url:
                bt_cdp.set_live_cdp_override(url)
            return json.loads(bu_cli.browser_exec("print(1)", task_id="t"))["output"].strip()

    assert drive(a, "http://127.0.0.1:9301") == "http://127.0.0.1:9301"
    assert drive(b, "http://127.0.0.1:9302") == "http://127.0.0.1:9302"  # B's own daemon, not A's
    assert drive(a) == "http://127.0.0.1:9301"
    reloads = []
    real_run = bu_cli._run_cli_killing_process_group
    monkeypatch.setattr(bu_cli, "_run_cli_killing_process_group",
                        lambda cmd, code, env, timeout: reloads.append(env.get("BH_RUNTIME_DIR"))
                        or real_run(cmd, code, env, timeout))
    with _profile_runtime_scope(a, hydrate_secrets=False):
        lifecycle.cleanup_profile_browsers(hermes_home_key(a))
        bt_cdp.set_live_cdp_override("http://127.0.0.1:9311")
    assert len(reloads) == 1  # A's reap stops exactly A's daemon
    assert [home for home in bu_cli._driven_daemons.values()] == [hermes_home_key(b)]
    assert drive(b) == "http://127.0.0.1:9302"  # B's daemon survived A's reap
    assert drive(a) == "http://127.0.0.1:9311"  # A respawned on its new endpoint
