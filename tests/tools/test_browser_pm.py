"""Browser consumers use real PM facts, not ambient npm caches."""

import json
import os
import sys
from pathlib import Path

import pytest

import pm
from pm import paths
from tools import browser_tool as bt
from tools import browser_tool_install as install
from tools import browser_tool_session as session


@pytest.fixture
def browser_store(tmp_path, monkeypatch):
    home = tmp_path / "home with spaces"
    home.mkdir()
    store = home / "tools"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(bt, "_SANE_PATH_DIRS", ())
    monkeypatch.setattr(install, "_discover_homebrew_node_dirs", lambda: ())
    monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
    monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
    lock = pm.Lockfile(paths.lockfile_path())
    target = pm.current_target()

    def publish(name, content="browser-fixture"):
        package = pm.get_package(name)
        version = lock.version(name)
        assert version is not None
        entry = store / package.store_entry(version, target)
        if name == "chromium":
            binary = entry / "browser" / ("chrome.exe" if os.name == "nt" else "chrome")
        else:
            binary = package.binary(entry, target)
        assert binary is not None
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text(content)
        binary.chmod(0o755)
        pm.Facts(store / "facts.json").record(
            name, version, entry.name, package.env(entry, target), store,
            target=target, artifacts=[a["sha256"] for a in lock.artifacts(name, target)],
        )
        installed = pm.installed_package(name)
        assert installed is not None and installed.binary == binary
        return binary

    return home, store, publish


def test_unrecorded_playwright_cache_is_not_a_pm_browser(browser_store):
    home, _, _ = browser_store
    (home / ".cache" / "ms-playwright" / "chromium-1234").mkdir(parents=True)
    assert install._chromium_installed() is False


def test_doctor_fix_publishes_and_reads_the_pm_browser(browser_store, monkeypatch):
    from hermes_cli import doctor_tools
    import pm.client

    _, _, publish = browser_store
    monkeypatch.setenv("PATH", "")
    requests = []

    def request(operation, payload, **kwargs):
        # Only the worker/download boundary is replaced; selection, admission,
        # facts and environment composition are the actual PM implementation.
        requests.append((operation, payload))
        publish("chromium")
        publish("agent-browser")

    monkeypatch.setattr(pm.client, "_request", request)
    assert doctor_tools._check_agent_browser(False) is False
    assert requests == []
    assert doctor_tools._check_agent_browser(True) is True
    assert requests == [("ensure", {"name": "agent-browser", "explicit": True})]


def test_missing_browser_refuses_lazy_install_even_with_npx(browser_store, monkeypatch, tmp_path):
    _, store, _ = browser_store
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    npx = bin_dir / ("npx.exe" if os.name == "nt" else "npx")
    npx.write_bytes(b"not an agent-browser installation")
    npx.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    with pytest.raises(FileNotFoundError, match="agent-browser CLI not found"):
        install._find_agent_browser(validate=False)
    with pytest.raises(FileNotFoundError, match="[Ll]azy|[Aa]utomatic|disabled"):
        install._find_agent_browser()
    assert not store.exists()


def test_late_pm_install_is_visible_without_cache_reset(browser_store, monkeypatch):
    _, _, publish = browser_store
    monkeypatch.setenv("PATH", "")
    with pytest.raises(FileNotFoundError):
        install._find_agent_browser()
    binary = publish("agent-browser")
    assert install._find_agent_browser() == str(binary)


def test_execution_acquires_missing_browser_through_pm(browser_store, monkeypatch):
    import pm.client

    _, _, publish = browser_store
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "0")
    assert pm.lazy_installs_allowed()
    requests = []

    def request(operation, payload, **kwargs):
        requests.append((operation, payload))
        publish("chromium")
        publish("agent-browser")

    monkeypatch.setattr(pm.client, "_request", request)
    result = install._find_agent_browser()
    installed = pm.installed_package("agent-browser")
    assert installed is not None and result == str(installed.binary)
    assert requests == [("ensure", {"name": "agent-browser", "explicit": False})]
    assert install._chromium_installed()


def test_termux_ownership_policy_never_provisions(browser_store, monkeypatch):
    # Exercise the real environment policy, not an emulated Android binary.
    from hermes_cli import doctor_tools

    monkeypatch.setenv("TERMUX_VERSION", "test")
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "0")
    monkeypatch.setattr(install, "_running_in_docker", lambda: False)
    monkeypatch.setattr(bt, "_chromium_autoinstall_attempted", False)

    def forbidden(*args, **kwargs):
        pytest.fail("Termux browser installation is externally owned")

    monkeypatch.setattr(pm, "ensure", forbidden)
    with pytest.raises(FileNotFoundError, match="npm install -g agent-browser"):
        install._find_agent_browser()
    assert install._maybe_autoinstall_chromium() is False
    assert doctor_tools._check_agent_browser(True) is False


def test_cdp_override_does_not_require_pm_browser(browser_store, monkeypatch):
    monkeypatch.setenv("BROWSER_CDP_URL", "ws://127.0.0.1:9222/devtools/browser/test")
    monkeypatch.setattr(bt, "_is_browser_use_cli_mode", lambda: False)
    monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
    assert install.check_browser_requirements() is True


@pytest.mark.platforms("linux", "darwin")
def test_external_browser_on_path_remains_supported_without_pm(browser_store, monkeypatch, tmp_path):
    _, store, _ = browser_store
    bin_dir = tmp_path / "external browser"
    bin_dir.mkdir()
    binary = bin_dir / "agent-browser"
    binary.write_text(f"#!{sys.executable}\nprint('external-agent-browser')\n")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    assert install._find_agent_browser(validate=False) == str(binary)
    assert install._find_agent_browser() == str(binary)
    assert not store.exists()


@pytest.mark.parametrize("entrypoint", [bt.warm_agent_browser_npx_cache, install.warm_agent_browser_npx_cache])
def test_historical_warmer_is_inert(entrypoint, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("historical browser warmer must do no work")

    monkeypatch.setattr("subprocess.Popen", forbidden)
    monkeypatch.setattr(pm, "ensure", forbidden)
    assert entrypoint(timeout=0.1) is False


def test_external_chromium_override_survives_pm_composition(browser_store, monkeypatch, tmp_path):
    _, _, publish = browser_store
    publish("chromium")
    override = tmp_path / "user browser"
    override.write_bytes(b"external")
    monkeypatch.setenv("AGENT_BROWSER_EXECUTABLE_PATH", str(override))
    assert install._chromium_installed()
    assert session._agent_browser_command_env(str(tmp_path))["AGENT_BROWSER_EXECUTABLE_PATH"] == str(override)


def test_restricted_path_discovers_external_browser_without_execution(browser_store, monkeypatch, tmp_path):
    _, store, _ = browser_store
    external_bin = tmp_path / "external-homebrew" / "bin"
    external_bin.mkdir(parents=True)
    binary = external_bin / ("agent-browser.exe" if os.name == "nt" else "agent-browser")
    binary.write_bytes(b"external browser; presence check must not execute")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    monkeypatch.setattr(bt, "_SANE_PATH_DIRS", (str(external_bin),))

    def forbidden(*args, **kwargs):
        pytest.fail("passive external browser discovery must not execute or install")

    monkeypatch.setattr("subprocess.Popen", forbidden)
    monkeypatch.setattr(pm, "ensure", forbidden)
    assert install._find_agent_browser(validate=False) == str(binary)
    assert not store.exists()


def test_pm_browser_wins_over_legacy_and_ambient_installs(browser_store, monkeypatch):
    home, store, publish = browser_store
    binary = publish("agent-browser")
    publish("chromium")
    legacy = home / "node_modules" / ".bin"
    legacy.mkdir(parents=True)
    external = legacy / ("agent-browser.exe" if os.name == "nt" else "agent-browser")
    external.write_bytes(b"legacy browser")
    external.chmod(0o755)
    monkeypatch.setenv("PATH", str(legacy))
    before = (store / "facts.json").read_bytes()
    assert install._find_agent_browser(validate=False) == str(binary)
    assert (store / "facts.json").read_bytes() == before


@pytest.mark.platforms("linux", "darwin")
def test_exact_pm_child_receives_composed_scrubbed_environment(browser_store, monkeypatch, tmp_path):
    _, store, publish = browser_store
    binary = publish("agent-browser", f"#!{sys.executable}\n" + '''import json, os, sys
print(json.dumps({"argv": sys.argv, "env": dict(os.environ)}))
''')
    chromium = publish("chromium")
    monkeypatch.setenv("PATH", str(tmp_path / "hostile-path"))
    system_bin = tmp_path / "external-system-bin"
    system_bin.mkdir()
    monkeypatch.setattr(bt, "_SANE_PATH_DIRS", (str(system_bin),))
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-browser")
    monkeypatch.setenv("BROWSERBASE_API_KEY", "browser-provider-key")
    before = dict(os.environ)
    socket_dir = tmp_path / "session"
    socket_dir.mkdir()
    command = install._find_agent_browser(validate=False)
    env = session._agent_browser_command_env(str(socket_dir))
    proc = session._popen_agent_browser([command, "--json", "get", "url"], env, str(socket_dir), "probe")
    assert proc.wait(timeout=10) == 0
    output, error = session._read_command_output_files(str(socket_dir / "_stdout_probe"), str(socket_dir / "_stderr_probe"))
    assert error == ""
    child = json.loads(output)
    assert Path(child["argv"][0]) == binary
    assert child["argv"][1:] == ["--json", "get", "url"]
    assert child["env"]["AGENT_BROWSER_EXECUTABLE_PATH"] == str(chromium)
    assert child["env"]["PLAYWRIGHT_BROWSERS_PATH"] == str(store)
    assert child["env"]["PATH"].split(os.pathsep)[0] == str(binary.parent)
    assert str(system_bin) in child["env"]["PATH"].split(os.pathsep)[1:]
    assert "OPENAI_API_KEY" not in child["env"]
    assert child["env"]["BROWSERBASE_API_KEY"] == "browser-provider-key"
    assert child["env"]["AGENT_BROWSER_SOCKET_DIR"] == str(socket_dir)
    assert dict(os.environ) == before


@pytest.mark.platforms("linux", "darwin")
def test_runtime_and_chrome_fallback_launch_the_same_pm_binary(browser_store, monkeypatch, tmp_path):
    from tools import browser_tool_cloud as cloud
    from tools import browser_tool_lightpanda_fallback as fallback

    _, _, publish = browser_store
    binary = publish("agent-browser", f"#!{sys.executable}\n" + '''import json, os, sys
print(json.dumps({"success": True, "data": {
    "url": "https://example.com/", "argv": sys.argv,
    "browser": os.environ.get("AGENT_BROWSER_EXECUTABLE_PATH")
}}))
''')
    chromium = publish("chromium")
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
    monkeypatch.setattr(cloud, "_is_local_mode", lambda: True)
    monkeypatch.setattr(cloud, "_get_browser_engine", lambda: "auto")
    monkeypatch.setattr(cloud, "_is_headed_mode", lambda: False)
    monkeypatch.setattr(session, "_get_session_info", lambda task_id: {"session_name": "fixture", "cdp_url": None})
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

    direct = session._run_browser_command("fixture", "get", ["url"])
    chrome = fallback._run_chrome_fallback_command("fixture", "get", ["url"], timeout=10)
    for result in (direct, chrome):
        assert result["success"] is True
        assert result["data"]["argv"][0] == str(binary)
        assert result["data"]["browser"] == str(chromium)
    assert direct["data"]["argv"][1:3] == ["--session", "fixture"]
    assert chrome["data"]["argv"][1:3] == ["--engine", "chrome"]