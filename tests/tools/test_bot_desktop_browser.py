"""The dock's Browser and agent-browser resolve to one identity: same executable, same user-data-dir —
and when a human opened that browser first, the agent attaches to it instead of launching a second one
(Chromium's profile singleton would forward the launch and kill it without a DevTools endpoint)."""

from __future__ import annotations

import os
import socket

from tools.bot_desktop import browser, runtime


def test_dock_and_agent_share_browser_identity(tmp_path, monkeypatch):
    exe = tmp_path / "chrome"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    exe.chmod(0o755)
    monkeypatch.setenv("AGENT_BROWSER_EXECUTABLE_PATH", str(exe))
    monkeypatch.delenv("AGENT_BROWSER_PROFILE", raising=False)
    monkeypatch.setattr(runtime, "state_dir", lambda: tmp_path / "bot-desktop")

    dock_exe, dock_profile = browser.dock_launch()
    agent_env = browser.env_for_agent({})

    assert dock_exe == agent_env["AGENT_BROWSER_EXECUTABLE_PATH"] == str(exe)
    assert dock_profile == agent_env["AGENT_BROWSER_PROFILE"] == str(tmp_path / "bot-desktop" / "browser-profile")


def test_user_pinned_profile_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_BROWSER_PROFILE", str(tmp_path / "mine"))
    monkeypatch.setattr(runtime, "state_dir", lambda: tmp_path / "bot-desktop")
    assert browser.profile_dir() == tmp_path / "mine"


def test_dock_browser_advertises_a_devtools_port():
    """A human-started instance must be attachable, or the agent can never drive it afterwards."""
    assert "--remote-debugging-port=" in browser.dock_argv("/opt/chrome", "/p/dir")[2]


def _fake_running_instance(user_data_dir, pid: int, port: int) -> None:
    (user_data_dir / "DevToolsActivePort").write_text(f"{port}\n/devtools/browser/abc\n", encoding="utf-8")
    os.symlink(f"host-{pid}", user_data_dir / "SingletonLock")


def test_running_instance_port_requires_live_pid_and_open_port(tmp_path):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        _fake_running_instance(tmp_path, os.getpid(), port)
        assert browser.running_instance_cdp_port(str(tmp_path)) == port

        # Both files outlive a closed Chromium: a dead pid must not be trusted.
        os.unlink(tmp_path / "SingletonLock")
        os.symlink("host-2147483000", tmp_path / "SingletonLock")
        assert browser.running_instance_cdp_port(str(tmp_path)) is None
    finally:
        listener.close()
    # Live pid, port no longer accepting: still not attachable.
    os.unlink(tmp_path / "SingletonLock")
    os.symlink(f"host-{os.getpid()}", tmp_path / "SingletonLock")
    assert browser.running_instance_cdp_port(str(tmp_path)) is None
    assert browser.running_instance_cdp_port(str(tmp_path / "missing")) is None


def test_agent_attaches_to_human_started_browser(monkeypatch):
    """With a live dock instance on the shared profile the local argv carries ``--cdp <port>``; without one
    it stays a plain ``--session`` launch."""
    from tools import browser_tool_session as session

    monkeypatch.setattr(runtime, "published_env", lambda: {"DISPLAY": ":37"})
    monkeypatch.setattr(session._cloud, "_get_browser_engine", lambda: "auto")
    monkeypatch.setattr(session._cloud, "_is_headed_mode", lambda: False)
    monkeypatch.setattr(session, "_agent_browser_argv", lambda cmd: [cmd])
    argvs: list = []

    def spawn(task_id, session_info, cmd_parts, *rest):
        argvs.append(cmd_parts)
        return {"success": True}

    monkeypatch.setattr(session, "_spawn_and_collect", spawn)
    monkeypatch.setattr(session._lp, "_lightpanda_fallback_reason", lambda *a: None)
    info = {"session_name": "h_abc", "cdp_url": None, "features": {"local": True}}

    monkeypatch.setattr(browser, "running_instance_cdp_port", lambda d, **kw: 41234)
    session._run_browser_command_unfenced("t", "open", ["https://x"], 10, None, "agent-browser", info)
    assert argvs[-1][:5] == ["agent-browser", "--session", "h_abc", "--cdp", "41234"]

    monkeypatch.setattr(browser, "running_instance_cdp_port", lambda d, **kw: None)
    session._run_browser_command_unfenced("t", "open", ["https://x"], 10, None, "agent-browser", info)
    assert "--cdp" not in argvs[-1] and argvs[-1][:3] == ["agent-browser", "--session", "h_abc"]


def _install_browsers(tmp_path, monkeypatch, *, playwright: bool, system: bool):
    """A Playwright build under a private PLAYWRIGHT_BROWSERS_PATH and/or a system chromium on PATH."""
    monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
    roots = tmp_path / "pw"
    roots.mkdir(parents=True)
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(roots))
    monkeypatch.setattr("tools.browser_tool_install._chromium_search_roots", lambda: [str(roots)])
    pw_exe = roots / "chromium-1200" / "chrome-linux" / "chrome"
    if playwright:
        pw_exe.parent.mkdir(parents=True)
        pw_exe.write_text("#!/bin/sh\n", encoding="utf-8")
        pw_exe.chmod(0o755)
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
    agent-browser starts it with on that host (one sandbox policy for the human's and the bot's browser)."""
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
    from tools import browser_tool_session as session

    agent_env: dict = {}
    session._apply_chromium_sandbox_args(agent_env)
    agent_flags = set(agent_env.get("AGENT_BROWSER_ARGS", "").split(",")) - {""}
    assert agent_flags <= set(browser.dock_argv(exe, "/p/dir"))
    assert ("--no-sandbox" in browser.dock_argv(exe, "/p/dir")) == ("--no-sandbox" in agent_flags)


def test_root_dock_browser_starts_with_the_same_sandbox_args_as_the_agents_browser(monkeypatch):
    """Chromium refuses to start as root without --no-sandbox; agent-browser gets that flag from one
    policy, and the dock icon (same binary, same profile) must get the very same flags or the human's
    click dies while the agent's launch works."""
    from tools import browser_tool_session as session

    monkeypatch.setattr(session.os, "geteuid", lambda: 0)
    monkeypatch.setattr(browser.os, "geteuid", lambda: 0)
    agent_env: dict = {}
    session._apply_chromium_sandbox_args(agent_env)
    agent_flags = set(agent_env["AGENT_BROWSER_ARGS"].split(","))
    assert agent_flags, "root must inject sandbox flags for agent-browser"
    assert agent_flags <= set(browser.dock_argv("/opt/chrome", "/p/dir"))

    # Same policy for the NON-root container case (the official image runs the gateway as uid 10000 in
    # Docker): agent-browser bypasses the sandbox there, and the dock icon must too or it dies on click.
    monkeypatch.setattr(session.os, "geteuid", lambda: 10000)
    monkeypatch.setattr(browser.os, "geteuid", lambda: 10000)
    monkeypatch.setattr(session._install, "_running_in_docker", lambda: True)
    docker_env: dict = {}
    session._apply_chromium_sandbox_args(docker_env)
    assert set(docker_env["AGENT_BROWSER_ARGS"].split(",")) <= set(browser.dock_argv("/opt/chrome", "/p/dir"))
    monkeypatch.setattr(session._install, "_running_in_docker", lambda: False)
    monkeypatch.setattr(session, "apparmor_restricts_unprivileged_userns", lambda: False)
    assert "--no-sandbox" not in browser.dock_argv("/opt/chrome", "/p/dir")  # seated non-root host: sandboxed


def test_status_reports_the_headed_browser_or_its_absence(monkeypatch):
    """The official image ships only chromium_headless_shell: executable() is None and the dock silently
    has no Browser icon. Status must say so instead of leaving the pane to guess."""
    monkeypatch.setattr(runtime, "_launcher_pid", lambda: None)
    monkeypatch.setattr(runtime, "published_env", lambda: {})
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
    as AGENT_BROWSER_EXECUTABLE_PATH. That binary cannot open a window: taken at face value the dock's
    Browser icon would point at it and status would claim a headed browser exists. Live in the image:
    status.browser named the headless shell while the dock had no working Browser."""
    shell = tmp_path / "shell" / "chromium_headless_shell-1243" / "chrome-headless-shell-linux64" / "chrome-headless-shell"
    shell.parent.mkdir(parents=True)
    shell.write_text("#!/bin/sh\n", encoding="utf-8")
    shell.chmod(0o755)
    _install_browsers(tmp_path, monkeypatch, playwright=False, system=False)
    monkeypatch.setenv("AGENT_BROWSER_EXECUTABLE_PATH", str(shell))
    assert browser.executable() is None

    _, sys_exe = _install_browsers(tmp_path / "with-sys", monkeypatch, playwright=False, system=True)
    monkeypatch.setenv("AGENT_BROWSER_EXECUTABLE_PATH", str(shell))
    assert browser.executable() == sys_exe  # a real headed browser elsewhere still wins over the override
