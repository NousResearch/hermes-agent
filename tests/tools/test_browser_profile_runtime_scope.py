"""Exercise real snapshots and call-time home routing; fake external browsers only."""
from pathlib import Path
from unittest.mock import Mock

import pytest


def test_real_profile_context_routing_and_cache_validation(tmp_path, monkeypatch):
    import hermes_cli.browser_connect as bc
    import tools.browser_tool as bt
    from tools import browser_tool_real_profile as rp
    from tools import browser_tool_cloud as cloud
    from tools import browser_tool_install as install
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    source = tmp_path / "source"
    (source / "Default").mkdir(parents=True)
    (source / "Local State").write_text("{}")
    (source / "Default" / "Preferences").write_text("{}")
    monkeypatch.setattr(bc, "detect_default_chromium", lambda: "chrome")
    monkeypatch.setattr(bc, "real_profile_data_dir", lambda *a, **kw: str(source))
    monkeypatch.setattr(bc, "chromium_executable", lambda b: "/fake/chrome")
    monkeypatch.setattr(cloud, "_use_real_profile", lambda: True)
    monkeypatch.setattr(cloud, "_is_headed_mode", lambda: False)
    monkeypatch.setattr(install, "_find_agent_browser", lambda: "/fake/agent-browser")
    monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: str(tmp_path / "sockets"))
    monkeypatch.setattr(rp, "_cdp_http_ready", lambda c: True)
    monkeypatch.setattr(rp, "_surviving_chrome_cdp", lambda d: None)
    monkeypatch.setattr(bt, "_real_profile_cdp_cache", {})
    monkeypatch.setattr(bt, "_real_profile_chrome_procs", [])
    daemons = {}
    launches = []

    def popen(argv, **kw):
        directory = Path(next(a.split("=", 1)[1] for a in argv if a.startswith("--user-data-dir=")))
        port = 43001 + len(launches)
        launches.append(directory)
        (directory / "DevToolsActivePort").write_text(f"{port}\n/devtools/browser/test\n")
        return Mock(poll=lambda: None)

    def run(argv, **kw):
        session = argv[argv.index("--session") + 1]
        port = argv[argv.index("--cdp") + 1]
        daemons[session] = f"http://127.0.0.1:{port}"
        return Mock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(rp.subprocess, "Popen", popen)
    monkeypatch.setattr(rp.subprocess, "run", run)
    monkeypatch.setattr(rp, "_agent_browser_get_cdp", lambda s: daemons.get(s))
    monkeypatch.setattr(rp, "_agent_browser_close_session", lambda s: daemons.pop(s, None))
    homes = [tmp_path / "a", tmp_path / "b"]
    endpoints = []
    keys = []
    for home in [homes[0], homes[1], homes[0]]:
        token = set_hermes_home_override(home)
        try:
            endpoints.append(rp._real_profile_cdp())
            keys.append(bt._REAL_PROFILE_CACHE_KEY)
        finally:
            reset_hermes_home_override(token)
    assert all(error is None for _, error in endpoints)
    assert endpoints[0] != endpoints[1]
    assert endpoints[0] == endpoints[2]
    assert keys[0] != keys[1] and keys[0] == keys[2]
    assert len(daemons) == len(launches) == 2
    assert launches == [h / "browser-profile" / "chrome" for h in homes]

    # Poison A with B's live endpoint: the real data-dir check must reject it.
    token = set_hermes_home_override(homes[0])
    try:
        bt._real_profile_cdp_cache[keys[0]] = endpoints[1][0]
        assert rp._real_profile_cdp() == endpoints[0]
        monkeypatch.setattr(cloud, "_use_real_profile", lambda: False)
        assert rp._real_profile_cdp() == (None, None)
        assert keys[0] not in bt._real_profile_cdp_cache
        assert bt._real_profile_cdp_cache[keys[1]] == endpoints[1][0]
        assert not (homes[0] / "browser-profile").exists()
        assert (homes[1] / "browser-profile").exists()
    finally:
        reset_hermes_home_override(token)
