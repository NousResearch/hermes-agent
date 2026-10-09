"""Browser Use caller -> browser-harness on Hermes's own interpreter -> actual child."""
import importlib.metadata
import json
import subprocess
import sys

import pytest

from tools import browser_use_cli as bu


@pytest.mark.platforms("posix", "windows")
def test_harness_runs_on_an_interpreter_without_its_own_site_packages(monkeypatch):
    """The Desktop bundle boots its store interpreter with no venv to activate (``-S`` emulates
    that); the harness and its daemon must import from the PYTHONPATH the tool env supplies,
    never from what the agent process inherited."""
    monkeypatch.setattr(bu, "_find_cli", bu._find_cli_unpatched)
    monkeypatch.setenv("PYTHONPATH", "/wrong-abi")
    monkeypatch.setenv("PYTHONHOME", "/wrong-python")
    cmd = bu._find_cli()
    assert cmd == [sys.executable, "-m", "browser_harness.run"]
    result = subprocess.run([cmd[0], "-S", *cmd[1:], "--version"], env=bu._base_subprocess_env(),
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == importlib.metadata.version("browser-harness")


@pytest.mark.platforms("posix", "windows")
def test_browser_exec_child_environment(tmp_path, monkeypatch):
    from tools import browser_tool_session, browser_supervisor

    probe = tmp_path / "probe.py"
    probe.write_text("import json, os, sys\n"
                     "print(json.dumps({'argv': sys.argv, 'stdin': sys.stdin.read(), 'env': dict(os.environ)}))\n",
                     encoding="utf-8")
    monkeypatch.setattr(bu, "_find_cli", lambda: [sys.executable, str(probe)])
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
    monkeypatch.setenv("PYTHONPATH", "/wrong-abi")
    monkeypatch.setenv("PYTHONHOME", "/wrong-python")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("BROWSERBASE_API_KEY", "browser-key")
    monkeypatch.setenv("KEEP_BROWSER_PROBE", "kept")
    monkeypatch.delenv("ANONYMIZED_TELEMETRY", raising=False)
    result = json.loads(bu.browser_exec("print('payload')", session="research", task_id="owner"))
    assert result["success"], result
    child = json.loads(result["output"])
    assert child["stdin"] == "print('payload')"
    for key in ("PYTHONHOME", "OPENAI_API_KEY", "_HERMES_BU_PRIVATE_BROWSER"):
        assert key not in child["env"]
    assert child["env"].get("PYTHONPATH") == bu._harness_site_dir()
    assert child["env"]["BU_NAME"] == "research"
    assert child["env"]["BU_CDP_WS"] == "ws://127.0.0.1:47000/private"
    assert child["env"]["ANONYMIZED_TELEMETRY"] == "false"
    assert child["env"]["BROWSERBASE_API_KEY"] == "browser-key"
    assert child["env"]["KEEP_BROWSER_PROBE"] == "kept"
    assert attached == [("owner", "ws://127.0.0.1:47000/private")]
