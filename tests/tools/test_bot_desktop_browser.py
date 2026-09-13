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
    assert "--remote-debugging-port=" in browser.dock_command("/opt/chrome", "/p/dir").split()[2]


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
    info = {"session_name": "h_abc", "cdp_url": None}

    monkeypatch.setattr(browser, "running_instance_cdp_port", lambda d, **kw: 41234)
    session._run_browser_command_unfenced("t", "open", ["https://x"], 10, None, "agent-browser", info)
    assert argvs[-1][:5] == ["agent-browser", "--session", "h_abc", "--cdp", "41234"]

    monkeypatch.setattr(browser, "running_instance_cdp_port", lambda d, **kw: None)
    session._run_browser_command_unfenced("t", "open", ["https://x"], 10, None, "agent-browser", info)
    assert "--cdp" not in argvs[-1] and argvs[-1][:3] == ["agent-browser", "--session", "h_abc"]
