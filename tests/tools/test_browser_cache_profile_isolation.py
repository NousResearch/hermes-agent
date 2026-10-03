"""Browser session caches stay isolated by the owning Hermes profile.

Regression for #110032.  The process-global cache must let two homes use the
same task id concurrently; switching A -> B -> A must recover A's original
session rather than hand either profile the other's browser.
"""

from contextlib import contextmanager
import json
from pathlib import Path

import pytest

from hermes_constants import (
    get_hermes_home,
    hermes_home_key,
    reset_hermes_home_override,
    set_hermes_home_override,
)

import tools.browser_tool as bt
from tools import browser_tool_cdp as bt_cdp
from tools import browser_tool_lifecycle as bt_lifecycle
from tools import browser_tool_session as bt_session


@pytest.fixture(autouse=True)
def _reset_browser_caches():
    for cache in (
        bt._active_sessions,
        bt._session_last_activity,
        bt._last_active_session_key,
        bt._suspect_browser_sessions,
        bt._recording_sessions,
        bt._cleanup_failures,
    ):
        cache.clear()
    yield
    for cache in (
        bt._active_sessions,
        bt._session_last_activity,
        bt._last_active_session_key,
        bt._suspect_browser_sessions,
        bt._recording_sessions,
        bt._cleanup_failures,
    ):
        cache.clear()


@contextmanager
def _under(home):
    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def test_browser_session_survives_two_home_a_b_a_round_trip(tmp_path, monkeypatch):
    home_a, home_b = tmp_path / "profile-a", tmp_path / "profile-b"
    home_a.mkdir()
    home_b.mkdir()
    created_for = []

    def create(task_id, force_local):
        owner = str(home_a if len(created_for) == 0 else home_b)
        created_for.append(owner)
        return {"session_name": owner, "bb_session_id": None, "features": {}}

    monkeypatch.setattr(bt_session, "_create_session_for_key", create)
    monkeypatch.setattr(bt_lifecycle, "_start_browser_cleanup_thread", lambda: None)
    monkeypatch.setattr(bt_cdp, "_ensure_cdp_supervisor", lambda _task_id: None)

    with _under(home_a):
        session_a = bt_session._get_session_info("shared")
    with _under(home_b):
        session_b = bt_session._get_session_info("shared")
    with _under(home_a):
        session_a_again = bt_session._get_session_info("shared")

    assert session_a["session_name"] == str(home_a)
    assert session_b["session_name"] == str(home_b)
    assert session_a_again is session_a
    assert len(bt._active_sessions) == 2


def test_all_browser_owner_edges_survive_two_home_a_b_a_round_trip(tmp_path, monkeypatch):
    """Every indirect reader/reaper must resolve the same home-owned entry as creation."""
    from hermes_cli import browser_connect
    from tools import browser_camofox as camofox
    from tools import browser_supervisor as supervisor_mod
    from tools import browser_tool_real_profile as real_profile
    from tools import browser_vault_tool as vault

    home_a, home_b = tmp_path / "profile-a", tmp_path / "profile-b"
    home_a.mkdir()
    home_b.mkdir()
    homes = (home_a, home_b, home_a)
    failures = {}

    sessions = {}
    for home in (home_a, home_b):
        with _under(home):
            session = {
                "owner": home.name,
                "session_name": f"session-{home.name}",
                "bb_session_id": None,
                "cdp_url": None,
                "features": {"local": True},
            }
            sessions[home.name] = session
            bt._active_sessions[bt._home_scoped_key("shared")] = session

    class _Alive:
        def is_alive(self):
            return True

        def is_running(self):
            return True

    class _Supervisor:
        def __init__(self, task_id, cdp_url, **_kwargs):
            self.task_id = task_id
            self.cdp_url = cdp_url
            self._thread = self._loop = _Alive()

        def start(self, timeout=15.0):
            return None

        def stop(self):
            return None

        def evaluate_runtime(self, _expression):
            return {"ok": True, "result": self.cdp_url}

    supervisor_mod.SUPERVISOR_REGISTRY.stop_all()
    monkeypatch.setattr(supervisor_mod, "CDPSupervisor", _Supervisor)
    monkeypatch.setattr(bt_cdp, "_get_dialog_policy_config", lambda: ("must_respond", 30.0))
    monkeypatch.setattr(
        bt_cdp,
        "_get_cdp_override",
        lambda: f"ws://{get_hermes_home().name}",
    )
    for home in (home_a, home_b):
        with _under(home):
            bt_cdp._ensure_cdp_supervisor("shared")

    eval_fences = []

    def fenced(session, fn):
        eval_fences.append(session.get("owner"))
        return fn()

    monkeypatch.setattr(bt_session, "run_fenced", fenced)
    monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(bt._eval_policy, "_eval_ssrf_guard_active", lambda _task_id: False)
    monkeypatch.setattr(bt, "_blocked_private_page_content", lambda _task_id: None)
    monkeypatch.setattr(
        bt_session,
        "_run_browser_command",
        lambda *_args, **_kwargs: {"success": False, "error": "supervisor lookup missed"},
    )
    eval_results = []
    for home in homes:
        with _under(home):
            payload = json.loads(bt.browser_console(expression="location.href", task_id="shared"))
            eval_results.append(payload.get("result"))
    expected_urls = ["ws://profile-a", "ws://profile-b", "ws://profile-a"]
    if eval_results != expected_urls or eval_fences != ["profile-a", "profile-b", "profile-a"]:
        failures["supervisor_and_eval_fence"] = {
            "results": eval_results,
            "fences": eval_fences,
        }

    vault_lookups = []
    monkeypatch.setattr(
        bt_session,
        "_shares_bot_desktop_browser",
        lambda session: vault_lookups.append(session.get("owner")) or bool(session),
    )
    vault_fences = []

    def vault_fenced(session, fn):
        vault_fences.append(session.get("owner"))
        return fn()

    monkeypatch.setattr(bt_session, "run_fenced", vault_fenced)
    for home in homes:
        with _under(home):
            vault._bot_desktop_browser_session("shared")
            assert vault._fenced_page_op("shared", lambda: "ok") == "ok"
    expected_owners = ["profile-a", "profile-b", "profile-a"]
    if vault_lookups != expected_owners or vault_fences != expected_owners:
        failures["vault_lease_fence"] = {
            "lookups": vault_lookups,
            "fences": vault_fences,
        }

    monkeypatch.setattr(bt_session, "_browser_in_sandbox", lambda: True)
    monkeypatch.setattr(bt_session, "_sandbox_close_daemon", lambda _name: None)
    with _under(home_b):
        bt_session._recycle_local_session(
            "shared", sessions["profile-b"], "unused", "test recycle"
        )
    with _under(home_a):
        a_survived = bt._active_sessions.get(bt._home_scoped_key("shared")) is sessions["profile-a"]
    with _under(home_b):
        b_reaped = bt._home_scoped_key("shared") not in bt._active_sessions
        b_flag_reaped = bt._home_scoped_key("shared") not in bt._suspect_browser_sessions
    if not (a_survived and b_reaped and b_flag_reaped):
        failures["sandbox_recycle"] = {
            "a_survived": a_survived,
            "b_reaped": b_reaped,
            "b_flag_reaped": b_flag_reaped,
        }

    camofox._sessions.clear()
    monkeypatch.setattr(camofox, "_get_camofox_config", lambda: {})

    def camofox_post(path, body=None, timeout=None):
        body = body or {}
        if path == "/tabs":
            return {"tabId": f"tab-{body['userId']}"}
        return {"ok": True, "url": body.get("url", "about:blank")}

    monkeypatch.setattr(camofox, "_post", camofox_post)
    camofox_users = []
    for home in homes:
        with _under(home):
            assert json.loads(camofox.camofox_navigate("https://example.test", "shared"))["success"]
            camofox_users.append(camofox._get_session("shared")["user_id"])
    if not (camofox_users[0] != camofox_users[1] and camofox_users[2] == camofox_users[0]):
        failures["camofox_session"] = camofox_users

    bt._real_profile_cdp_cache.clear()
    real_profile_sessions = []
    monkeypatch.setattr(real_profile._cloud, "_use_real_profile", lambda: True)
    monkeypatch.setattr(real_profile._lp, "_using_lightpanda_engine", lambda: False)
    monkeypatch.setattr(real_profile, "_cdp_http_ready", lambda _url: True)
    monkeypatch.setattr(real_profile._session, "_prepare_session_socket_dir", lambda _name: "")
    monkeypatch.setattr(
        real_profile,
        "_agent_browser_get_cdp",
        lambda name: real_profile_sessions.append(name) or None,
    )
    monkeypatch.setattr(real_profile, "_surviving_chrome_cdp", lambda _copy: None)
    monkeypatch.setattr(browser_connect, "detect_default_chromium", lambda: "chrome")
    monkeypatch.setattr(
        browser_connect,
        "real_profile_copy_dir",
        lambda _browser: Path(get_hermes_home()) / "browser-profile-copy",
    )
    monkeypatch.setattr(
        browser_connect,
        "snapshot_real_profile",
        lambda _browser: (Path(get_hermes_home()) / "browser-profile-copy", None),
    )
    monkeypatch.setattr(browser_connect, "chromium_executable", lambda _browser: "/chrome")
    monkeypatch.setattr(
        real_profile,
        "_launch_real_profile_chrome",
        lambda _binary, _copy: (9101 if get_hermes_home().name == "profile-a" else 9102, None),
    )
    monkeypatch.setattr(
        real_profile,
        "_attach_agent_browser_to_real_profile",
        lambda port, _copy: (f"http://127.0.0.1:{port}", None),
    )
    real_profile_urls = []
    for home in homes:
        with _under(home):
            real_profile_urls.append(real_profile._real_profile_cdp()[0])
    expected_real_profile_urls = [
        "http://127.0.0.1:9101",
        "http://127.0.0.1:9102",
        "http://127.0.0.1:9101",
    ]
    if real_profile_urls != expected_real_profile_urls or not (
        len(real_profile_sessions) == 2
        and real_profile_sessions[0] != real_profile_sessions[1]
    ):
        failures["real_profile_session"] = {
            "urls": real_profile_urls,
            "session_names": real_profile_sessions,
        }

    class _Chrome:
        def __init__(self, owner):
            self.owner = owner

    chrome_a, chrome_b = _Chrome("profile-a"), _Chrome("profile-b")
    terminated = []
    monkeypatch.setattr(
        "tools.browser_lightpanda._terminate",
        lambda proc, **_kwargs: terminated.append(proc.owner),
    )
    process_cache = bt._real_profile_chrome_procs
    process_cache.clear()
    key_a, key_b = hermes_home_key(home_a), hermes_home_key(home_b)
    process_cache[key_a], process_cache[key_b] = [chrome_a], [chrome_b]
    real_profile._terminate_real_profile_chrome(key_b)
    after_b = list(terminated)
    real_profile._terminate_real_profile_chrome(key_a)
    if after_b != ["profile-b"] or terminated != ["profile-b", "profile-a"]:
        failures["real_profile_cleanup"] = {
            "after_b": after_b,
            "terminated": terminated,
        }

    supervisor_mod.SUPERVISOR_REGISTRY.stop_all()
    camofox._sessions.clear()
    bt._real_profile_cdp_cache.clear()
    bt._real_profile_chrome_procs.clear()
    assert failures == {}
