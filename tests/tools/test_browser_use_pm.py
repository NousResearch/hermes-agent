"""Browser Use caller -> real PM worker -> locally built CLI -> actual child."""
import json
import os
from pathlib import Path
import zipfile

import pytest

import pm
from tests.pm._fixtures import _wheel, build_worker, client, isolated_python  # noqa: F401
from tools import browser_use_cli as bu


@pytest.mark.platforms("posix", "windows")
def test_installed_cli_selection_and_child_environment(build_worker, tmp_path, monkeypatch):
    from tools import browser_tool_session, browser_supervisor

    wheel = _wheel(tmp_path, "browser_use_probe")
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("browser_use_probe/cli.py", """import json, os, sys
def main():
    print(json.dumps({'argv': sys.argv, 'stdin': sys.stdin.read(), 'env': dict(os.environ)}))
""")
        archive.writestr("browser_use_probe-1.0.dist-info/entry_points.txt",
                         "[console_scripts]\nbrowser-use = browser_use_probe.cli:main\n")
    monkeypatch.setattr(bu, "_CLI_REQUIREMENTS", (f"browser-use-probe @ {wheel.as_uri()}",))
    monkeypatch.setattr(bu, "_find_cli", bu._find_cli_unpatched)
    monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {"browser": {"backend": "browser-use"}})
    monkeypatch.setattr("tools.browser_tool_cdp._get_cdp_override", lambda: "")
    monkeypatch.setattr("tools.browser_tool_cdp._resolve_cdp_override", lambda url: url)
    monkeypatch.setattr("tools.browser_tool_cloud._get_cloud_provider", lambda: None)
    monkeypatch.setattr("tools.browser_tool_lightpanda_fallback._using_lightpanda_engine", lambda: False)
    def local_browser(task_id, command, args, **kwargs):
        assert (task_id, command, args) == ("bu-named-research", "get", ["cdp-url"])
        return {"success": True, "data": {"cdpUrl": "ws://127.0.0.1:47000/private"}}
    monkeypatch.setattr(browser_tool_session, "_run_browser_command", local_browser)
    attached = []
    monkeypatch.setattr(browser_supervisor.SUPERVISOR_REGISTRY, "get_or_start",
                        lambda task_id, cdp_url, **kw: attached.append((task_id, cdp_url)))
    # A PATH binary must never substitute for missing or corrupted PM selection.
    ambient = tmp_path / "ambient"
    ambient.mkdir()
    (ambient / "browser-use").write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    (ambient / "browser-use").chmod(0o755)
    monkeypatch.setenv("PATH", str(ambient) + os.pathsep + os.environ["PATH"])
    assert bu._find_cli() is None
    assert "hermes tools" in json.loads(bu.browser_exec("print(1)"))["error"]
    ok, message = bu.install_cli(timeout_s=120)
    assert ok, message
    binary = pm.python_tool("browser-use", "browser-use")
    assert binary is not None and bu._find_cli() == [str(binary)]
    assert str(binary) in message
    assert bu.install_cli(timeout_s=120)[0]
    assert pm.python_tool("browser-use", "browser-use") == binary
    monkeypatch.setenv("PYTHONPATH", "/wrong-abi")
    monkeypatch.setenv("PYTHONHOME", "/wrong-python")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("BROWSERBASE_API_KEY", "browser-key")
    monkeypatch.setenv("KEEP_BROWSER_PROBE", "kept")
    monkeypatch.delenv("ANONYMIZED_TELEMETRY", raising=False)
    result = json.loads(bu.browser_exec("print('payload')", session="research", task_id="owner"))
    assert result["success"], result
    child = json.loads(result["output"])
    assert Path(child["argv"][0]) == binary
    assert child["stdin"] == "print('payload')"
    for key in ("PYTHONPATH", "PYTHONHOME", "OPENAI_API_KEY", "_HERMES_BU_PRIVATE_BROWSER"):
        assert key not in child["env"]
    assert child["env"]["BU_NAME"] == "research"
    assert child["env"]["BU_CDP_WS"] == "ws://127.0.0.1:47000/private"
    assert child["env"]["ANONYMIZED_TELEMETRY"] == "false"
    assert child["env"]["BROWSERBASE_API_KEY"] == "browser-key"
    assert child["env"]["KEEP_BROWSER_PROBE"] == "kept"
    assert attached == [("owner", "ws://127.0.0.1:47000/private")]
    # Invalid wheel reaches real worker/uv failure, and leaves the old CLI selected.
    monkeypatch.delenv("PYTHONPATH")
    monkeypatch.delenv("PYTHONHOME")
    monkeypatch.setattr(bu, "_CLI_REQUIREMENTS", (f"browser-use-probe @ {(tmp_path / 'missing.whl').as_uri()}",))
    ok, message = bu.install_cli(timeout_s=120)
    assert not ok and "Could not install browser-use CLI" in message
    assert "missing.whl" in message
    assert bu._find_cli() == [str(binary)]
    binary.unlink()
    assert bu._find_cli() is None
