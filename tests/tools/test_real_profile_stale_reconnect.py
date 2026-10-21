"""A replacement Chrome must supersede the resident daemon's old CDP port."""
import subprocess
from unittest.mock import Mock

from tools import browser_tool_real_profile as rp


def test_attach_explicitly_reconnects_stale_daemon_without_navigation(tmp_path, monkeypatch):
    # agent-browser 0.26 auto-launches ordinary commands using the resident
    # daemon's AGENT_BROWSER_CDP; a new CLI --cdp does not update that env.
    # `connect` is an explicit launch action, bypassing that stale auto-launch.
    (tmp_path / "DevToolsActivePort").write_text("44173\n/devtools/browser/new\n")
    monkeypatch.setattr(rp._install, "_find_agent_browser", lambda: "/agent-browser")
    endpoint = {"port": 34605}
    commands = []

    def daemon(argv, **kwargs):
        commands.append(argv)
        if "connect" in argv:
            endpoint["port"] = int(argv[argv.index("connect") + 1])
            return subprocess.CompletedProcess(argv, 0, "Connected", "")
        return subprocess.CompletedProcess(
            argv, 1, "", "Auto-launch failed: All CDP discovery methods failed for 127.0.0.1:34605"
        )

    monkeypatch.setattr(rp, "_capture_agent_browser_cli", daemon)
    monkeypatch.setattr(rp, "_agent_browser_get_cdp", lambda _: f"http://127.0.0.1:{endpoint['port']}")
    close = Mock(side_effect=AssertionError("must not close the live browser"))
    monkeypatch.setattr(rp, "_agent_browser_close_session", close)

    assert rp._attach_agent_browser_to_real_profile(44173, str(tmp_path)) == (
        "http://127.0.0.1:44173", None
    )
    assert all("open" not in argv and "close" not in argv for argv in commands)
    close.assert_not_called()


def test_resolver_reconnects_replacement_without_snapshot_or_launch(tmp_path, monkeypatch):
    import tools.browser_tool as bt
    import hermes_cli.browser_connect as bc

    (tmp_path / "DevToolsActivePort").write_text("44173\n/devtools/browser/new\n")
    monkeypatch.setattr(bt, "_real_profile_cdp_cache", {"cdp": "http://127.0.0.1:34605"})
    monkeypatch.setattr(rp._cloud, "_use_real_profile", lambda: True)
    monkeypatch.setattr(rp._lp, "_using_lightpanda_engine", lambda: False)
    monkeypatch.setattr(bc, "detect_default_chromium", lambda: "chrome")
    monkeypatch.setattr(bc, "real_profile_copy_dir", lambda _: str(tmp_path))
    monkeypatch.setattr(rp, "_cdp_http_ready", lambda _: False)
    monkeypatch.setattr(rp._install, "_find_agent_browser", lambda: "/agent-browser")
    version = Mock()
    version.json.return_value = {"webSocketDebuggerUrl": "ws://127.0.0.1:44173/devtools/browser/new"}
    monkeypatch.setattr("requests.get", lambda *a, **kw: version)
    connected = []

    def daemon(argv, **kwargs):
        if "connect" in argv:
            connected.append(argv[argv.index("connect") + 1])
            return subprocess.CompletedProcess(argv, 0, "Connected", "")
        if connected:
            return subprocess.CompletedProcess(argv, 0, "ws://127.0.0.1:44173/devtools/browser/new", "")
        return subprocess.CompletedProcess(argv, 1, "", "Auto-launch failed: stale CDP")

    monkeypatch.setattr(rp, "_capture_agent_browser_cli", daemon)
    forbidden = Mock(side_effect=AssertionError("must preserve the live profile"))
    monkeypatch.setattr(bc, "snapshot_real_profile", forbidden)
    monkeypatch.setattr(rp, "_launch_real_profile_chrome", forbidden)
    monkeypatch.setattr(rp, "_agent_browser_close_session", forbidden)
    assert rp._real_profile_cdp() == ("http://127.0.0.1:44173", None)
    assert connected == ["44173"]
    assert bt._real_profile_cdp_cache == {"cdp": "http://127.0.0.1:44173"}
    forbidden.assert_not_called()
