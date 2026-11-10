"""The dock's Browser and the agent's Chromium resolve to one identity: same executable, same user-data-dir —
and when a human opened that browser first, the agent attaches to it instead of launching a second one
(Chromium's profile singleton would forward the launch and kill it without a DevTools endpoint)."""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.bot_desktop import browser, placement, runtime


def _browser_config(monkeypatch, home: Path, **browser_cfg) -> None:
    """This profile's real config.yaml with a ``browser:`` section."""
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(json.dumps({"browser": browser_cfg}), encoding="utf-8")


def _agent_launch(monkeypatch, display: str = ":37") -> dict:
    """Create the agent's local browser session with the Bot Desktop screen up; returns the launcher call
    (``key`` + kwargs) instead of starting Chromium."""
    from tools import browser_launcher, browser_session, browser_session_cdp, browser_session_cloud
    from tools import browser_session_lifecycle

    for name in ("_active_sessions", "_session_last_activity", "_session_owner_homes", "_cleanup_failures"):
        monkeypatch.setattr(browser_session, name, {})
    monkeypatch.setattr(browser_session_lifecycle, "start_cleanup_thread", lambda: None)
    monkeypatch.setattr(browser_session_cdp, "get_cdp_override", lambda: "")
    monkeypatch.setattr(browser_session_cloud, "get_cloud_provider", lambda: None)
    monkeypatch.setattr("tools.bot_desktop.runtime.tool_placement", lambda: placement.GATEWAY)  # host browser
    monkeypatch.setattr("tools.browser_session_real_profile.real_profile_cdp", lambda: (None, None))
    monkeypatch.setattr("tools.browser_lightpanda.using_lightpanda_engine", lambda: False)
    monkeypatch.setattr(runtime, "published_env", lambda: {"DISPLAY": display})
    call: dict = {}

    def launch(key, **kwargs):
        call.update(kwargs, key=key)
        return SimpleNamespace(cdp_url="ws://127.0.0.1:41234/devtools/browser/abc", pid=4242), None

    monkeypatch.setattr(browser_launcher, "launch_chromium", launch)
    browser_session.get_session_info("agent-task")
    return call


def test_dock_and_agent_share_browser_identity(tmp_path, monkeypatch):
    exe = tmp_path / "chrome"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    exe.chmod(0o755)
    _browser_config(monkeypatch, tmp_path / "home", executable_path=str(exe))
    monkeypatch.setattr(runtime, "state_dir", lambda: tmp_path / "bot-desktop")

    dock_exe, dock_profile = browser.dock_launch()
    agent = _agent_launch(monkeypatch)

    assert dock_exe == agent["executable"] == str(exe)
    assert dock_profile == agent["user_data_dir"] == str(tmp_path / "bot-desktop" / "browser-profile")
    assert agent["ephemeral"] is False, "the shared profile's logins must survive the agent's browser"


def test_user_pinned_profile_wins(tmp_path, monkeypatch):
    _browser_config(monkeypatch, tmp_path / "home", user_data_dir=str(tmp_path / "mine"))
    monkeypatch.setattr(runtime, "state_dir", lambda: tmp_path / "bot-desktop")
    assert browser.profile_dir() == tmp_path / "mine"


def test_pinned_profile_honours_tilde_and_resolves_relative_paths_against_hermes_home(tmp_path, monkeypatch):
    """Regression for #110029: the docs say pinning a user-data-dir pins your own, but only an absolute value
    was honoured — `~/pin` and `pin` silently fell back to the default and the human's dock browser and the
    agent's browser could end up on different jars. A relative path is anchored where the rest of this
    profile's screen state lives (its HERMES_HOME), so two profiles never share one 'pin'."""
    home = tmp_path / "hermes-home"
    monkeypatch.setattr(runtime, "get_hermes_home", lambda: home)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "user")
    monkeypatch.setenv("HOME", str(tmp_path / "user"))

    _browser_config(monkeypatch, home, user_data_dir="~/pin")
    assert browser.profile_dir() == tmp_path / "user" / "pin"
    _browser_config(monkeypatch, home, user_data_dir="pin")
    assert browser.profile_dir() == home / "pin"
    assert _agent_launch(monkeypatch)["user_data_dir"] == str(home / "pin"), "the agent's launch gets the resolved path"


def test_dock_browser_advertises_a_devtools_port():
    """A human-started instance must be attachable, or the agent can never drive it afterwards."""
    assert "--remote-debugging-port=" in browser.dock_argv("/opt/chrome", "/p/dir")[2]


def test_dock_browser_caps_its_disk_cache():
    """The persistent profile lives on the gateway's disk (6 GB on a hosted instance); the HTTP cache
    must not be allowed to grow without bound there."""
    argv = browser.dock_argv("/opt/chrome", "/p/dir")
    cap = next(a for a in argv if a.startswith("--disk-cache-size="))
    assert 0 < int(cap.split("=", 1)[1]) <= 512 * 1024 * 1024


def _fake_running_instance(user_data_dir, pid: int, port: int) -> None:
    (user_data_dir / "DevToolsActivePort").write_text(f"{port}\n/devtools/browser/abc\n", encoding="utf-8")
    os.symlink(f"host-{pid}", user_data_dir / "SingletonLock")


def _devtools_server():
    """A loopback server answering ``/json/version`` like the dock's Chromium (browser id ``abc``)."""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            port = self.server.server_address[1]
            body = json.dumps({"webSocketDebuggerUrl": f"ws://127.0.0.1:{port}/devtools/browser/abc"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_running_instance_port_requires_live_pid_and_open_port(tmp_path):
    from tools import browser_launcher as launcher

    server = _devtools_server()
    port = server.server_address[1]
    try:
        _fake_running_instance(tmp_path, os.getpid(), port)
        assert launcher.running_instance(str(tmp_path)) == (port, "/devtools/browser/abc", os.getpid())

        # Both files outlive a closed Chromium: a dead pid must not be trusted.
        os.unlink(tmp_path / "SingletonLock")
        os.symlink("host-2147483000", tmp_path / "SingletonLock")
        assert launcher.running_instance(str(tmp_path)) is None
    finally:
        server.shutdown()
        server.server_close()
    # Live pid, port no longer accepting: still not attachable.
    os.unlink(tmp_path / "SingletonLock")
    os.symlink(f"host-{os.getpid()}", tmp_path / "SingletonLock")
    assert launcher.running_instance(str(tmp_path)) is None
    assert launcher.running_instance(str(tmp_path / "missing")) is None


def test_agent_attaches_to_human_started_browser(tmp_path, monkeypatch):
    """With a live dock instance on the shared profile the agent's launch ATTACHES to it (never owned, so
    never killed by Hermes); without one it launches its own Chromium on that profile."""
    from tools import browser_launcher as launcher

    monkeypatch.setattr(launcher, "_browsers", {})
    monkeypatch.setattr(launcher, "_procs", {})
    udd = str(tmp_path / "browser-profile")
    spawned: list = []

    def launch_owned(key, executable, user_data_dir, **_kw):
        spawned.append((key, executable, user_data_dir))
        return None, "spawn recorded"

    monkeypatch.setattr(launcher, "_launch_owned", launch_owned)
    monkeypatch.setattr(launcher, "running_instance", lambda d: (41234, "/devtools/browser/abc", os.getpid()) if d == udd else None)
    attached, err = launcher.launch_chromium(f"profile:{udd}", executable="/opt/chrome", user_data_dir=udd, ephemeral=False)
    assert err is None and spawned == []
    assert attached.cdp_url == "ws://127.0.0.1:41234/devtools/browser/abc"
    assert attached.owned is False

    launcher._browsers.clear()
    monkeypatch.setattr(launcher, "running_instance", lambda d: None)
    launcher.launch_chromium(f"profile:{udd}", executable="/opt/chrome", user_data_dir=udd, ephemeral=False)
    assert spawned == [(f"profile:{udd}", "/opt/chrome", udd)]


def _install_browsers(tmp_path, monkeypatch, *, playwright: bool, system: bool):
    """A Playwright build under a private PLAYWRIGHT_BROWSERS_PATH and/or a system chromium on PATH."""
    monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
    monkeypatch.setattr("tools.browser_session_cloud.executable_path", lambda: "")
    roots = tmp_path / "pw"
    roots.mkdir(parents=True)
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(roots))
    pw_exe = roots / "chromium-1200" / "chrome-linux" / "chrome"
    if playwright:
        pw_exe.parent.mkdir(parents=True)
        pw_exe.write_text("#!/bin/sh\n", encoding="utf-8")
        pw_exe.chmod(0o755)
    monkeypatch.setattr(
        "hermes_cli.browser_runtime.chromium_executable",
        lambda *, allow_override=True: str(pw_exe) if playwright else None,
    )
    sys_exe = tmp_path / "bin" / "chromium"
    if system:
        sys_exe.parent.mkdir(parents=True)
        sys_exe.write_text("#!/bin/sh\n", encoding="utf-8")
        sys_exe.chmod(0o755)
    monkeypatch.setattr("shutil.which", lambda name, *a, **k: str(sys_exe) if system and name == "chromium" else None)
    return str(pw_exe), str(sys_exe)


def test_unprivileged_user_under_apparmor_userns_restriction_gets_the_system_browser(tmp_path, monkeypatch):
    """Playwright's bundled Chromium has no setuid chrome_sandbox; with
    kernel.apparmor_restrict_unprivileged_userns=1 it dies 'FATAL: No usable sandbox!' for a non-root user,
    so the dock icon is dead. A distro chromium (which ships the sandbox helper) must win there — and
    the Playwright build stays the answer when it is the only one, started with exactly the flags
    the agent's launch starts it with on that host (one sandbox policy for the human's and the bot's browser)."""
    pw_exe, sys_exe = _install_browsers(tmp_path, monkeypatch, playwright=True, system=True)
    monkeypatch.setattr(browser.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(browser, "_userns_restricted", lambda: True)
    assert browser.executable() == sys_exe

    monkeypatch.setattr(browser, "_userns_restricted", lambda: False)
    assert browser.executable() == pw_exe  # unrestricted host: Playwright's build as before

    _install_browsers(tmp_path / "only-pw", monkeypatch, playwright=True, system=False)
    monkeypatch.setattr(browser, "_userns_restricted", lambda: True)
    exe = browser.executable()
    assert exe and exe.endswith("chrome-linux/chrome")
    agent_flags = _agent_sandbox_flags(exe)
    assert agent_flags <= set(browser.dock_argv(exe, "/p/dir"))
    assert ("--no-sandbox" in browser.dock_argv(exe, "/p/dir")) == ("--no-sandbox" in agent_flags)


def _agent_sandbox_flags(exe: str) -> set:
    """The sandbox-bypass flags the agent's launch of ``exe`` on the shared profile carries."""
    from tools.browser_launcher import CHROMIUM_SANDBOX_BYPASS_ARGS, build_chromium_argv

    argv = build_chromium_argv(exe, "/p/dir", packaged=False, headed=True, sandbox_bypass=True,
                               extra_args=browser.SHARED_PROFILE_ARGS)
    return set(argv) & set(CHROMIUM_SANDBOX_BYPASS_ARGS)


def test_root_dock_browser_starts_with_the_same_sandbox_args_as_the_agents_browser(monkeypatch):
    """Chromium refuses to start as root without --no-sandbox; the agent's launch gets that flag from one
    policy, and the dock icon (same binary, same profile) must get the very same flags or the human's
    click dies while the agent's launch works."""
    from tools import browser_launcher as launcher

    monkeypatch.setattr("tools.browser_session_cloud.chromium_args", list)
    monkeypatch.setattr(launcher.os, "geteuid", lambda: 0)
    agent_flags = _agent_sandbox_flags("/opt/chrome")
    assert agent_flags, "root must inject sandbox flags for the agent's Chromium"
    assert agent_flags <= set(browser.dock_argv("/opt/chrome", "/p/dir"))

    # Same policy for the NON-root container case (the official image runs the gateway as uid 10000 in
    # Docker): the agent's launch bypasses the sandbox there, and the dock icon must too or it dies on click.
    monkeypatch.setattr(launcher.os, "geteuid", lambda: 10000)
    monkeypatch.setattr(launcher, "running_in_docker", lambda: True)
    docker_flags = _agent_sandbox_flags("/opt/chrome")
    assert docker_flags and docker_flags <= set(browser.dock_argv("/opt/chrome", "/p/dir"))
    monkeypatch.setattr(launcher, "running_in_docker", lambda: False)
    monkeypatch.setattr(launcher, "apparmor_restricts_unprivileged_userns", lambda: False)
    assert "--no-sandbox" not in browser.dock_argv("/opt/chrome", "/p/dir")  # seated non-root host: sandboxed


def test_status_reports_the_headed_browser_or_its_absence(monkeypatch):
    """The official image ships only chromium_headless_shell: executable() is None and the dock silently
    has no Browser icon. Status must say so instead of leaving the pane to guess. The subject is the
    browser field, not host policy: status() gates it on is_supported_host(), which is pinned True so the
    test means the same thing on every CI lane."""
    monkeypatch.setattr(runtime, "is_supported_host", lambda: True)
    monkeypatch.setattr(runtime, "_launcher_pid", lambda: None)
    monkeypatch.setattr(runtime, "published_env", dict)
    monkeypatch.setattr(runtime, "geometry", lambda: "1440x900")
    monkeypatch.setattr(browser, "executable", lambda: None)
    assert runtime.status().as_dict()["browser"] is None
    monkeypatch.setattr(browser, "executable", lambda: "/usr/bin/chromium")
    assert runtime.status().browser == "/usr/bin/chromium"


def test_dock_exec_line_survives_spaces_in_the_executable_and_profile_paths():
    """The launcher used to split the shell line on the first space to find the executable, so a
    Chromium under '/opt/Google Chrome/' or a profile under a spaced HERMES_HOME broke the dock icon.
    Exec= follows the Desktop Entry spec: each argument double-quoted, with the reserved characters
    backslash-escaped inside the quotes."""
    exe = "/opt/Google Chrome/chrome"
    profile = '/home/a b/.hermes/browser "x"/profile'
    line = browser.dock_exec_line(exe, profile)
    assert line.startswith('Exec="/opt/Google Chrome/chrome" ')
    assert r'"--user-data-dir=/home/a b/.hermes/browser \\"x\\"/profile"' in line  # spec: \" quoted, then \ string-escaped
    assert "--remote-debugging-port=0" in line


def test_headless_shell_override_is_not_a_headed_browser(tmp_path, monkeypatch):
    """The official Docker image ships only Playwright's chrome-headless-shell and its boot hook exports it
    as the browser override. That binary cannot open a window: taken at face value the dock's
    Browser icon would point at it and status would claim a headed browser exists. Live in the image:
    status.browser named the headless shell while the dock had no working Browser."""
    shell = tmp_path / "shell" / "chromium_headless_shell-1243" / "chrome-headless-shell-linux64" / "chrome-headless-shell"
    shell.parent.mkdir(parents=True)
    shell.write_text("#!/bin/sh\n", encoding="utf-8")
    shell.chmod(0o755)
    _install_browsers(tmp_path, monkeypatch, playwright=False, system=False)
    monkeypatch.setattr("tools.browser_session_cloud.executable_path", lambda: str(shell))
    assert browser.executable() is None

    _, sys_exe = _install_browsers(tmp_path / "with-sys", monkeypatch, playwright=False, system=True)
    monkeypatch.setattr("tools.browser_session_cloud.executable_path", lambda: str(shell))
    assert browser.executable() == sys_exe  # a real headed browser elsewhere still wins over the override


def test_headless_override_does_not_hide_pm_headed_browser(tmp_path, monkeypatch):
    import pm
    from hermes_cli.browser_runtime import chromium_executable

    headless = tmp_path / "chrome-headless-shell"
    headed = tmp_path / "chrome"
    for exe in (headless, headed):
        exe.write_text("#!/bin/sh\n", encoding="utf-8")
        exe.chmod(0o755)
    _browser_config(monkeypatch, tmp_path / "home", executable_path=str(headless))
    monkeypatch.setattr(pm, "installed_package", lambda name: SimpleNamespace(binary=headed))
    monkeypatch.setattr(browser, "_system_executable", lambda: None)
    monkeypatch.setattr(browser, "_userns_restricted", lambda: False)

    assert chromium_executable() == str(headless)  # ordinary headless browsing keeps its override
    dock = browser.dock_launch()
    assert dock is not None and dock[0] == str(headed)
    assert _agent_launch(monkeypatch)["executable"] == str(headed)


@pytest.mark.parametrize("engine, headed, starts", [("chrome", True, 1), ("chrome", False, 0), ("lightpanda", True, 0)])
def test_headed_chromium_spawn_asks_the_screen_to_start_but_the_env_builder_never_does(monkeypatch, engine, headed, starts):
    """Regression for #110050 at the right boundary: creating a local session that launches a headed Chromium
    starts the profile's screen (bot_desktop.auto_start) like computer_use dispatch does; `build_browser_env()`
    itself stays pure — it also serves the harness CLI, the Lightpanda engine and every launch, none of which
    may block on Xvnc+Xfce coming up."""
    from tools import browser_launcher, browser_lightpanda, browser_session, browser_session_cdp
    from tools import browser_session_cloud, browser_session_lifecycle

    calls: list = []
    monkeypatch.setattr(runtime, "ensure_started_for_tool", lambda: calls.append(1))
    monkeypatch.setattr(runtime, "published_env", dict)
    monkeypatch.setattr(browser_session_cloud, "is_headed_mode", lambda: headed)
    browser_session.build_browser_env()
    assert calls == []

    for name in ("_active_sessions", "_session_last_activity", "_session_owner_homes", "_cleanup_failures"):
        monkeypatch.setattr(browser_session, name, {})
    monkeypatch.setattr(browser_session_lifecycle, "start_cleanup_thread", lambda: None)
    monkeypatch.setattr(browser_session_cdp, "get_cdp_override", lambda: "")
    monkeypatch.setattr(browser_session_cloud, "get_cloud_provider", lambda: None)
    monkeypatch.setattr("tools.bot_desktop.runtime.tool_placement", lambda: placement.GATEWAY)  # host browser
    monkeypatch.setattr("tools.browser_session_real_profile.real_profile_cdp", lambda: (None, None))
    monkeypatch.setattr(browser_session_cloud, "get_browser_engine", lambda: engine)
    monkeypatch.setattr(browser_launcher, "launch_chromium", lambda key, **kw: (
        SimpleNamespace(cdp_url="ws://127.0.0.1:41234/devtools/browser/abc", pid=4242), None))
    monkeypatch.setattr(browser_lightpanda, "launch_lightpanda", lambda name, **kw: (
        SimpleNamespace(cdp_url="ws://127.0.0.1:9333", port=9333), None))
    browser_session.get_session_info("t")
    assert len(calls) == starts


def test_janitor_keeps_the_shared_browser_alive_while_a_human_holds_the_lease(monkeypatch):
    """#110064: the agent goes idle BECAUSE the human took over to log in; the janitor must count the human's
    lease as activity for the shared local browser, and reap it again once the lease is handed back."""
    from tools import browser_session as session, browser_session_lifecycle as lifecycle
    from tools.bot_desktop import lease

    monkeypatch.setattr(runtime, "published_env", lambda: {"DISPLAY": ":37"})
    holder = {"human": True}
    monkeypatch.setattr(lease, "human_holds", lambda *a, **k: holder["human"])
    reaped: list = []
    monkeypatch.setattr(lifecycle, "cleanup_single_session", lambda cache_key: reaped.append(cache_key))
    monkeypatch.setattr(session, "BROWSER_SESSION_INACTIVITY_TIMEOUT", 1)
    monkeypatch.setattr(session, "_active_sessions", {
        "bot": {"session_name": "h_bot", "features": {"local": True}},
        "cloud": {"session_name": "c_1", "bb_session_id": "bb", "features": {}},
    })
    monkeypatch.setattr(session, "_session_last_activity", {"bot": 0.0, "cloud": 0.0})
    monkeypatch.setattr(session, "_session_owner_homes", {})
    monkeypatch.setattr(session, "_cleanup_failures", {})

    lifecycle.cleanup_inactive_sessions()
    assert reaped == ["cloud"], "only the browser the human is not typing into may be reaped"
    assert session._session_last_activity["bot"] > 0, "the human's lease refreshed the bot browser's activity"

    holder["human"] = False
    session._session_last_activity["bot"] = 0.0
    lifecycle.cleanup_inactive_sessions()
    assert reaped == ["cloud", "bot"]
    assert session.human_holds_shared_browser({"features": {"local": True}}) is False


def test_a_headless_shell_pin_is_replaced_while_a_screen_is_up(tmp_path, monkeypatch):
    """The boot hook exports a chrome-headless-shell path; leaving it would put the agent and the dock on
    two binaries over one --user-data-dir, where the singleton swallows the dock's launch."""
    shell = tmp_path / "chrome-headless-shell"
    shell.write_text("#!/bin/sh\n", encoding="utf-8")
    shell.chmod(0o755)
    headed = tmp_path / "chrome"
    headed.write_text("#!/bin/sh\n", encoding="utf-8")
    headed.chmod(0o755)
    monkeypatch.setattr(runtime, "state_dir", lambda: tmp_path / "bot-desktop")
    monkeypatch.setattr(browser, "_managed_executable", lambda: str(headed))
    monkeypatch.setattr("tools.browser_session_cloud.executable_path", lambda: str(shell))
    # Unpinned, the ubuntu runner (non-root, userns-restricted) flips executable() to
    # its own /usr/bin/google-chrome; host policy is not the subject here.
    monkeypatch.setattr(browser, "_userns_restricted", lambda: False)

    agent = _agent_launch(monkeypatch)
    dock_exe, _ = browser.dock_launch()
    assert agent["executable"] == dock_exe == str(headed), \
        "the agent and the dock must share one binary once a screen is up"


def test_a_real_user_pin_is_still_honoured(tmp_path, monkeypatch):
    """Only a headless-shell pin is overridden; a human's own headed browser stays put."""
    mine = tmp_path / "my-chrome"
    mine.write_text("#!/bin/sh\n", encoding="utf-8")
    mine.chmod(0o755)
    other = tmp_path / "chrome"
    other.write_text("#!/bin/sh\n", encoding="utf-8")
    other.chmod(0o755)
    monkeypatch.setattr(runtime, "state_dir", lambda: tmp_path / "bot-desktop")
    monkeypatch.setattr(browser, "_managed_executable", lambda: str(other))
    monkeypatch.setattr("tools.browser_session_cloud.executable_path", lambda: str(mine))

    assert _agent_launch(monkeypatch)["executable"] == str(mine)
