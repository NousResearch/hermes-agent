"""GW-23 (#134889): per-profile browser scope edges left after CLI-9.

A multiplexed gateway serves every profile from one process; a live ``/browser connect`` endpoint,
a harness daemon's runtime and tmp dirs, and a browser session belong to ONE profile.
"""
import contextlib
import threading

import pytest

from agent import secret_scope
from gateway.run import _profile_runtime_scope
from hermes_constants import get_hermes_home, hermes_home_key
from tools import browser_tool_cdp as bt_cdp
from tools import browser_harness_dirs as harness_dirs
from tools.browser_harness_dirs import claim_private_dir as _REAL_CLAIM  # conftest stubs it per test


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    monkeypatch.setattr(bt_cdp, "_live_overrides", {})
    return a, b


def _in_bare_thread(fn):
    """Run ``fn`` on a stdlib thread started WITHOUT the caller's copied context."""
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("v", fn()))
    t.start()
    t.join(5)
    return out["v"]


def test_a_bare_thread_on_a_multiplexed_gateway_never_resolves_the_launch_profiles_override(profiles, monkeypatch):
    """No profile scope bound on a multiplexed gateway: the launch profile's live endpoint is NOT the
    answer (``get_secret`` fails closed the same way). A bound scope still sees its own."""
    launch = get_hermes_home()
    with _profile_runtime_scope(launch, hydrate_secrets=False):
        bt_cdp.set_live_cdp_override("http://127.0.0.1:9401")
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    assert _in_bare_thread(bt_cdp.live_cdp_override) == ""
    assert _in_bare_thread(bt_cdp._get_cdp_override_raw) == ""
    with _profile_runtime_scope(launch, hydrate_secrets=False):
        assert bt_cdp.live_cdp_override() == "http://127.0.0.1:9401"


def test_single_profile_bare_thread_keeps_the_process_override(profiles):
    bt_cdp.set_live_cdp_override("http://127.0.0.1:9402")
    assert _in_bare_thread(bt_cdp.live_cdp_override) == "http://127.0.0.1:9402"
    assert hermes_home_key() in bt_cdp._live_overrides


# --- macOS AF_UNIX budget for a served profile's harness sockets -------------------------------------

_LONGEST_NAME = "n" * 64  # browser_harness/_ipc.py _NAME_RE: [A-Za-z0-9_-]{1,64}


def _sock_len(runtime_dir: str) -> int:
    """browser_harness/_ipc.py ``_sock_path``: ``<resolved BH_RUNTIME_DIR>/bu-<NAME>.sock`` when
    ``BH_RUNTIME_DIR_SHARED=1`` (pinned against the installed package by the test below)."""
    return len(f"{runtime_dir}/bu-{_LONGEST_NAME}.sock".encode())


def test_installed_harness_names_a_shared_dir_socket_bu_name_sock(tmp_path):
    """The length rule rests on the installed harness's stem: resolved dir + ``bu-<NAME>.sock``."""
    import os
    import subprocess
    import sys
    rt = tmp_path / "rt"
    env = {**os.environ, "BH_RUNTIME_DIR": str(rt), "BH_RUNTIME_DIR_SHARED": "1",
           "BH_TMP_DIR": str(tmp_path / "tmp"), "BH_HOME": str(tmp_path / "home")}
    out = subprocess.run([sys.executable, "-c",
                          f"from browser_harness import _ipc; print(_ipc._sock_path({_LONGEST_NAME!r}))"],
                         env=env, capture_output=True, text=True, check=True).stdout.strip()
    assert out == f"{rt.resolve()}/bu-{_LONGEST_NAME}.sock"


@pytest.mark.parametrize(("platform", "budget"), [("darwin", 103 - 73), ("linux", 107 - 73), ("win32", None)])
def test_the_socket_dir_budget_is_sun_path_minus_a_64_char_session_socket(platform, budget):
    """macOS ``sun_path`` is 104 bytes (103 + NUL), Linux 108; Windows serves over loopback TCP."""
    assert harness_dirs.socket_dir_budget(platform) == budget


def test_a_default_macos_home_does_not_fit_and_the_short_temp_dir_does():
    """Pure length rule with macOS data (no host platform involved): under
    ``~/.config/browser-harness/runtime/p-<hash>`` a 9-letter account left room for ~31-character
    session names; ``/private/tmp/hbh-<12 hex>`` fits a 64-character one."""
    budget = harness_dirs.socket_dir_budget("darwin")
    home = "/Users/alexander/.config/browser-harness/runtime/p-0123456789ab"
    short = "/private/tmp/hbh-0123456789ab"  # no-tmp: ok — a macOS path as data, nothing is written
    assert not harness_dirs.fits_socket_budget(home, budget)
    assert harness_dirs.fits_socket_budget(short, budget) and _sock_len(short) <= 103
    assert harness_dirs.fits_socket_budget("x" * 500, None)  # no sun_path limit


@pytest.mark.platforms("macos")
def test_on_macos_a_served_profile_binds_a_64_char_session_socket(profiles, tmp_path, monkeypatch):
    """Real branch: the dir is claimed under the real short root and a 64-character name binds."""
    import os
    import socket
    from tools import browser_use_cli as bu_cli
    a, _b = profiles
    monkeypatch.setattr(harness_dirs, "claim_private_dir", _REAL_CLAIM)  # the real claim, real short root
    env = {"HOME": str(tmp_path / ("u" * 40))}
    with _profile_runtime_scope(a, hydrate_secrets=False):
        harness_dirs.namespace_harness_dirs(env, bu_cli._served_profile_tag())
    path = f"{os.path.realpath(env['BH_RUNTIME_DIR'])}/bu-{_LONGEST_NAME}.sock"
    assert len(path.encode()) <= 103
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.bind(path)
    finally:
        s.close()
        with contextlib.suppress(OSError):
            os.unlink(path)


def _kernel_accepts_sun_path(path: str) -> bool:
    """Ask the kernel ABI whether ``path`` fits ``sun_path``: an unbound dir gives ENOENT, an overlong
    path gives "AF_UNIX path too long" before any filesystem access."""
    import socket
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.bind(path)
    except OSError as e:
        return "too long" not in str(e)
    else:
        import os
        os.unlink(path)
        return True
    finally:
        s.close()


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("user_home", ["/home/teknium", "/home/alexander.smith"])
def test_on_linux_a_served_profiles_64_char_session_socket_fits_sun_path(profiles, monkeypatch, user_home):
    """Linux ``sun_path`` is 108 bytes (107 + NUL). ``~/.config/browser-harness/runtime/p-<hash>`` left
    room for 38-character names at /home/teknium and 30 at /home/alexander.smith, while BU_NAME may be
    64. The served profile gets the same private short dir as on macOS (claim stubbed: no temp-root
    write); the kernel judges the length of the exact path the installed harness binds."""
    import os
    from tools import browser_use_cli as bu_cli
    a, _b = profiles
    claimed = []
    monkeypatch.setattr(harness_dirs, "claim_private_dir", lambda path: claimed.append(path) or True)
    env = {"HOME": user_home}
    with _profile_runtime_scope(a, hydrate_secrets=False):
        harness_dirs.namespace_harness_dirs(env, bu_cli._served_profile_tag())
    sock = f"{env['BH_RUNTIME_DIR']}/bu-{_LONGEST_NAME}.sock"
    assert claimed == [env["BH_RUNTIME_DIR"]]
    assert _sock_len(env["BH_RUNTIME_DIR"]) <= 107 and _kernel_accepts_sun_path(sock), sock
    assert os.path.basename(env["BH_RUNTIME_DIR"]).startswith("hbh-")


@pytest.mark.platforms("linux")
def test_on_linux_a_runtime_dir_that_already_fits_is_kept_and_an_unclaimable_one_stays_long(profiles, monkeypatch):
    from tools import browser_use_cli as bu_cli
    a, _b = profiles
    claimed = []
    monkeypatch.setattr(harness_dirs, "claim_private_dir", lambda path: claimed.append(path) or True)
    env = {"BH_HOME": "/b"}
    with _profile_runtime_scope(a, hydrate_secrets=False):
        harness_dirs.namespace_harness_dirs(env, bu_cli._served_profile_tag())
    assert env["BH_RUNTIME_DIR"].startswith("/b/runtime/p-") and claimed == []
    monkeypatch.setattr(harness_dirs, "claim_private_dir", lambda path: False)  # squatted / not ours
    env = {"HOME": "/home/teknium"}
    with _profile_runtime_scope(a, hydrate_secrets=False):
        harness_dirs.namespace_harness_dirs(env, bu_cli._served_profile_tag())
    assert env["BH_RUNTIME_DIR"].startswith("/home/teknium/.config/browser-harness/runtime/p-")


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("via", ["relative", "symlink"])
def test_on_linux_a_short_spelling_of_a_long_runtime_dir_is_measured_resolved(profiles, monkeypatch, tmp_path, via):
    """The harness binds ``Path(BH_RUNTIME_DIR).expanduser().resolve()``: a short relative or symlinked
    spelling of a deep dir still gives an overlong socket, so the length is judged on the resolved path."""
    import os
    from tools import browser_use_cli as bu_cli
    a, _b = profiles
    deep = tmp_path / ("d" * 60)
    deep.mkdir()
    if via == "relative":  # BH_RUNTIME_DIR=rt from a deep working directory
        monkeypatch.chdir(deep)
        base = "rt"
    else:  # a short symlink to a deep dir
        link = tmp_path / "l"
        link.symlink_to(deep)
        monkeypatch.chdir(tmp_path)
        base = "l"
    claimed = []
    monkeypatch.setattr(harness_dirs, "claim_private_dir", lambda path: claimed.append(path) or True)
    env = {"BH_RUNTIME_DIR": base}
    with _profile_runtime_scope(a, hydrate_secrets=False):
        harness_dirs.namespace_harness_dirs(env, bu_cli._served_profile_tag())
    resolved = os.path.realpath(env["BH_RUNTIME_DIR"])
    assert _sock_len(resolved) <= 107, resolved
    assert _kernel_accepts_sun_path(f"{resolved}/bu-{_LONGEST_NAME}.sock")
    assert claimed == [env["BH_RUNTIME_DIR"]]


def test_the_short_socket_dir_is_claimed_only_when_private_and_ours(tmp_path):
    """The short dir sits in a world-writable root: a symlink or another user's/open dir is refused."""
    fresh = tmp_path / "hbh-fresh"
    assert _REAL_CLAIM(str(fresh)) and (fresh.stat().st_mode & 0o777) == 0o700
    assert _REAL_CLAIM(str(fresh))  # ours, 0700: reused
    open_dir = tmp_path / "hbh-open"
    open_dir.mkdir(mode=0o755)
    open_dir.chmod(0o755)
    assert not _REAL_CLAIM(str(open_dir))
    link = tmp_path / "hbh-link"
    link.symlink_to(fresh)
    assert not _REAL_CLAIM(str(link))


# --- per-profile harness tmp dir (screenshots, debug overlays, daemon logs) ---------------------------


def test_each_served_profile_gets_its_own_harness_tmp_dir(profiles, tmp_path):
    """browser_harness/_ipc.py ``_TMP`` (paths.tmp_dir: BH_TMP_DIR, else <home>/tmp) holds ``shot.png``,
    debug-click overlays and ``bu-<NAME>.log``; one shared dir let profile B read A's screenshots."""
    from tools import browser_use_cli as bu_cli
    a, b = profiles
    seen = {}
    for home in (a, b):
        env = {"BH_HOME": str(tmp_path / "bh")}
        with _profile_runtime_scope(home, hydrate_secrets=False):
            harness_dirs.namespace_harness_dirs(env, bu_cli._served_profile_tag())
        assert env.get("BH_TMP_DIR_SHARED") == "1"  # several BU_NAMEs share it: keep bu-<name>.log stems
        seen[home] = env.get("BH_TMP_DIR")
    assert seen[a] and seen[b] and seen[a] != seen[b]
    assert all(str(p).startswith(str(tmp_path / "bh" / "tmp")) for p in seen.values())
    unscoped = {"BH_HOME": str(tmp_path / "bh")}
    harness_dirs.namespace_harness_dirs(unscoped, bu_cli._served_profile_tag())
    assert unscoped == {"BH_HOME": str(tmp_path / "bh")}  # single-profile: unchanged


def test_a_served_profile_under_an_inherited_tmp_dir_nests_inside_it(profiles, tmp_path):
    from tools import browser_use_cli as bu_cli
    a, _b = profiles
    env = {"BH_TMP_DIR": str(tmp_path / "shared-tmp")}
    with _profile_runtime_scope(a, hydrate_secrets=False):
        harness_dirs.namespace_harness_dirs(env, bu_cli._served_profile_tag())
    assert env["BH_TMP_DIR"].startswith(str(tmp_path / "shared-tmp") + "/p-")
    # The runtime dir still derives from the inherited BH_TMP_DIR (harness fallback), not the new one.
    assert env["BH_RUNTIME_DIR"].startswith(str(tmp_path / "shared-tmp") + "/p-")


def test_an_unscoped_shutdown_reload_reaches_the_daemon_with_its_own_tmp_dir(profiles, monkeypatch):
    """The reaper/shutdown runs outside any profile scope; its ``--reload`` must still carry the
    daemon's own tmp dir, or the harness writes that profile's log into the shared one."""
    from tools import browser_use_cli as bu_cli
    a, _b = profiles
    with _profile_runtime_scope(a, hydrate_secrets=False):
        env = {"HOME": "/home/u"}
        harness_dirs.namespace_harness_dirs(env, bu_cli._served_profile_tag())
    key = harness_dirs.daemon_identity({**env, "BU_NAME": "s1"})
    monkeypatch.setattr(bu_cli, "_driven_daemons", {key: hermes_home_key(a)})
    monkeypatch.setattr(bu_cli, "_find_cli", lambda: ["browser-harness"])
    seen = []
    monkeypatch.setattr(bu_cli, "_run_cli_killing_process_group", lambda cmd, code, e, t: seen.append(e))
    bu_cli.stop_harness_daemons()
    assert [(e["BH_TMP_DIR"], e["BH_TMP_DIR_SHARED"], e["BU_NAME"]) for e in seen] == [
        (env["BH_TMP_DIR"], "1", "s1")]


# --- reap/publish race within one profile -----------------------------------------------------------


def test_a_session_that_resolved_the_old_endpoint_during_a_connect_is_closed_not_kept(profiles, monkeypatch):
    """``/browser connect`` reaps, publishes the new endpoint, then reaps again. A tool call that
    resolved the OLD endpoint before the publish but inserted its session after the second reap kept
    driving the old browser. The real ``_browser_connect`` handler runs mid-creation here."""
    from tools import browser_tool, browser_tool_lifecycle as lifecycle, browser_tool_session as bt_session
    from tui_gateway import server  # binds methods_browser helpers onto server globals

    a, _b = profiles
    sessions = {}
    monkeypatch.setattr(browser_tool, "_active_sessions", sessions)
    monkeypatch.setattr(lifecycle, "_start_browser_cleanup_thread", lambda: None)
    monkeypatch.setattr(bt_cdp, "_ensure_cdp_supervisor", lambda key: None)
    monkeypatch.setattr(server, "_cdp_http_reachable", lambda parsed, timeout=2.0: True)
    closed = []  # released by object: the stale session is never in ``sessions``
    monkeypatch.setattr(lifecycle, "_release_session_backend", lambda task_id, info: closed.append(info["cdp_url"]))
    monkeypatch.setattr(bt_session, "_browser_command_preflight", lambda: {"error": "no CLI in this test"})
    created = []

    def create(task_id, local):
        url = bt_cdp._get_cdp_override_raw()  # resolved before the connect publishes
        if not created:
            assert server._browser_connect("r1", {"url": "http://127.0.0.1:9502"})["result"]["connected"]
        created.append(url)
        return {"session_name": f"s{len(created)}", "cdp_url": url}
    monkeypatch.setattr(bt_session, "_create_session_for_key", create)

    with _profile_runtime_scope(a, hydrate_secrets=False):
        bt_cdp.set_live_cdp_override("http://127.0.0.1:9501")
        info = bt_session._get_session_info("task-a")
    assert info["cdp_url"] == "http://127.0.0.1:9502"
    assert sessions["task-a"] is info
    assert closed == ["http://127.0.0.1:9501"]


def test_a_stale_session_is_closed_without_ever_being_handed_to_a_concurrent_caller(profiles, monkeypatch):
    """A session found stale after creation is closed by object, never published first: while its
    backend ``close`` runs, a concurrent ``_get_session_info`` for the same key must not find it in
    ``_active_sessions`` (it would return a browser that is being closed underneath it)."""
    from tools import browser_tool, browser_tool_lifecycle as lifecycle, browser_tool_session as bt_session
    from tui_gateway import server

    a, _b = profiles
    sessions = {}
    monkeypatch.setattr(browser_tool, "_active_sessions", sessions)
    monkeypatch.setattr(lifecycle, "_start_browser_cleanup_thread", lambda: None)
    monkeypatch.setattr(bt_cdp, "_ensure_cdp_supervisor", lambda key: None)
    monkeypatch.setattr(bt_cdp, "_stop_cdp_supervisor", lambda key: None)
    monkeypatch.setattr(server, "_cdp_http_reachable", lambda parsed, timeout=2.0: True)
    monkeypatch.setattr(bt_session, "_browser_command_preflight", lambda: {"browser_cmd": "agent-browser"})
    visible_during_close = []

    def dispatch(task_id, info, browser_cmd, command, args, timeout, engine):
        if command == "close":  # what the existing-session path would hand a concurrent caller now
            visible_during_close.append((info["cdp_url"], (sessions.get(task_id) or {}).get("cdp_url")))
        return "chrome", {"success": True}
    monkeypatch.setattr(bt_session, "_dispatch_browser_command", dispatch)
    released = []
    monkeypatch.setattr(lifecycle, "_release_session_backend", lambda task_id, info: released.append(info["cdp_url"]))
    created = []

    def create(task_id, local):
        url = bt_cdp._get_cdp_override_raw()
        if not created:
            assert server._browser_connect("r1", {"url": "http://127.0.0.1:9512"})["result"]["connected"]
        created.append(url)
        return {"session_name": f"s{len(created)}", "cdp_url": url, "bb_session_id": None}
    monkeypatch.setattr(bt_session, "_create_session_for_key", create)

    with _profile_runtime_scope(a, hydrate_secrets=False):
        bt_cdp.set_live_cdp_override("http://127.0.0.1:9511")
        info = bt_session._get_session_info("task-a")
    assert visible_during_close == [("http://127.0.0.1:9511", None)]
    assert released == ["http://127.0.0.1:9511"]
    assert info["cdp_url"] == "http://127.0.0.1:9512" and sessions["task-a"] is info


def test_another_profiles_connect_does_not_invalidate_this_profiles_new_session(profiles, monkeypatch):
    from tools import browser_tool, browser_tool_lifecycle as lifecycle, browser_tool_session as bt_session
    a, b = profiles
    sessions = {}
    monkeypatch.setattr(browser_tool, "_active_sessions", sessions)
    monkeypatch.setattr(lifecycle, "_start_browser_cleanup_thread", lambda: None)
    monkeypatch.setattr(bt_cdp, "_ensure_cdp_supervisor", lambda key: None)
    closed = []
    monkeypatch.setattr(lifecycle, "_cleanup_single_browser_session", closed.append)

    def create(task_id, local):
        with _profile_runtime_scope(b, hydrate_secrets=False):
            bt_cdp.set_live_cdp_override("http://127.0.0.1:9601")
        return {"session_name": "s", "cdp_url": "x"}
    monkeypatch.setattr(bt_session, "_create_session_for_key", create)
    with _profile_runtime_scope(a, hydrate_secrets=False):
        info = bt_session._get_session_info("task-a")
    assert sessions == {"task-a": info} and closed == []


def test_a_browser_exec_that_routed_to_the_old_endpoint_during_a_connect_reloads_its_daemon(profiles, monkeypatch):
    """Browser Use path (the default mode): ``browser_exec`` resolves the live endpoint into
    ``BU_CDP_URL`` before the daemon is recorded. A ``/browser connect`` landing in between reaped
    nothing, and the harness daemon (which latches ``BU_CDP_*`` at spawn) kept the old browser for every
    later call. The real ``_browser_connect`` handler runs between routing and dispatch here."""
    from tools import browser_use_cli as bu_cli
    from tui_gateway import server

    a, _b = profiles
    monkeypatch.setattr(bu_cli, "_driven_daemons", {})
    monkeypatch.setattr(bu_cli, "_find_cli", lambda: ["browser-harness"])
    monkeypatch.setattr(bu_cli, "_attach_vault_supervisor", lambda env, task_id: None)
    monkeypatch.setattr(bt_cdp, "_resolve_cdp_override", lambda raw: raw)
    monkeypatch.setattr(server, "_cdp_http_reachable", lambda parsed, timeout=2.0: True)
    latched = {}  # like browser-harness: one daemon per identity latches BU_CDP_URL; --reload stops it

    def run(cmd, code, env, timeout):
        import subprocess
        ident = harness_dirs.daemon_identity(env)
        if cmd[-1] == "--reload":
            latched.pop(ident, None)
        else:
            latched.setdefault(ident, env.get("BU_CDP_URL"))
        return subprocess.CompletedProcess(cmd, 0, str(latched.get(ident)), "")
    monkeypatch.setattr(bu_cli, "_run_cli_killing_process_group", run)
    real_route = bu_cli._route_exec
    connects = []

    def route_then_connect(*args, **kw):
        out = real_route(*args, **kw)  # BU_CDP_URL now holds the OLD endpoint
        if not connects:
            connects.append(server._browser_connect("r1", {"url": "http://127.0.0.1:9702"}))
        return out
    monkeypatch.setattr(bu_cli, "_route_exec", route_then_connect)

    import json
    with _profile_runtime_scope(a, hydrate_secrets=False):
        bt_cdp.set_live_cdp_override("http://127.0.0.1:9701")
        first = json.loads(bu_cli.browser_exec("print(1)", task_id="t1"))
        assert connects[0]["result"]["connected"]
        nxt = json.loads(bu_cli.browser_exec("print(1)", task_id="t1"))
    assert first["output"] == nxt["output"] == "http://127.0.0.1:9702"


def test_a_connect_that_lands_while_browser_exec_spawns_its_daemon_reloads_it(profiles, monkeypatch):
    """The publish lands after routing, while the CLI spawns the daemon on the old endpoint: the daemon
    is already recorded (connect's reap reloads it) and the post-dispatch generation check reloads it
    too, so the next call respawns it on the new endpoint."""
    import json
    import subprocess
    from tools import browser_use_cli as bu_cli
    from tui_gateway import server

    a, _b = profiles
    monkeypatch.setattr(bu_cli, "_driven_daemons", {})
    monkeypatch.setattr(bu_cli, "_find_cli", lambda: ["browser-harness"])
    monkeypatch.setattr(bu_cli, "_attach_vault_supervisor", lambda env, task_id: None)
    monkeypatch.setattr(bt_cdp, "_resolve_cdp_override", lambda raw: raw)
    monkeypatch.setattr(server, "_cdp_http_reachable", lambda parsed, timeout=2.0: True)
    latched, connects = {}, []

    def run(cmd, code, env, timeout):
        ident = harness_dirs.daemon_identity(env)
        if cmd[-1] == "--reload":
            latched.pop(ident, None)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        latched.setdefault(ident, env.get("BU_CDP_URL"))
        if not connects:  # mid-spawn: the daemon already latched the old endpoint
            connects.append(server._browser_connect("r1", {"url": "http://127.0.0.1:9712"}))
        return subprocess.CompletedProcess(cmd, 0, str(latched.get(ident)), "")
    monkeypatch.setattr(bu_cli, "_run_cli_killing_process_group", run)
    with _profile_runtime_scope(a, hydrate_secrets=False):
        bt_cdp.set_live_cdp_override("http://127.0.0.1:9711")
        json.loads(bu_cli.browser_exec("print(1)", task_id="t1"))
        assert connects[0]["result"]["connected"]
        nxt = json.loads(bu_cli.browser_exec("print(1)", task_id="t1"))
    assert nxt["output"] == "http://127.0.0.1:9712"
