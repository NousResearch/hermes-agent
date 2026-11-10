"""Browser consumers use real PM facts, not ambient caches: the launcher's Chromium is PM's pinned
Chromium (or the user's explicit override), acquired through PM at most once per process."""

import json
import os
import sys
from pathlib import Path

import pytest

import pm
from pm import paths
from tests.waiting import wait_until
from tools import browser_launcher as launcher


@pytest.fixture(autouse=True)
def _no_chromium_download(monkeypatch):
    """Overrides the tools/conftest.py pin: these tests exercise the REAL one-shot auto-install, so the
    download boundary itself is the guard. Every test that permits acquisition replaces
    ``pm.client._request`` with a fake publish; any other route to it fails the test."""
    import pm.client

    monkeypatch.setattr(pm.client, "_request", lambda *a, **kw: pytest.fail("unexpected PM acquisition"))
    yield


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
    monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
    monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.setattr(launcher, "_autoinstall_attempted", False)
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
    assert launcher.chromium_ready()[0] is False


def test_late_pm_install_is_visible_without_cache_reset(browser_store, monkeypatch):
    _, _, publish = browser_store
    monkeypatch.setenv("PATH", "")
    assert launcher.chromium_ready()[0] is False
    binary = publish("chromium")
    assert launcher.chromium_ready() == (True, "")
    assert launcher._resolve_executable() == str(binary)


def test_execution_acquires_missing_browser_through_pm(browser_store, monkeypatch):
    import pm.client

    _, _, publish = browser_store
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "0")
    monkeypatch.setattr(launcher, "running_in_docker", lambda: False)
    assert pm.lazy_installs_allowed()
    requests = []

    def request(operation, payload, **kwargs):
        requests.append((operation, payload))
        publish("chromium")

    monkeypatch.setattr(pm.client, "_request", request)
    assert launcher.chromium_ready() == (True, "")
    installed = pm.installed_package("chromium")
    assert installed is not None and launcher._resolve_executable() == str(installed.binary)
    assert requests == [("ensure", {"name": "chromium", "explicit": False})]


def test_doctor_fix_publishes_and_reads_the_pm_browser(browser_store, monkeypatch, capsys):
    """Plain ``hermes doctor`` never acquires Chromium; ``--fix`` acquires it explicitly through PM and reports
    the published binary. Only the worker/download boundary is replaced: selection, admission, facts and
    environment composition are the actual PM implementation."""
    from hermes_cli import doctor_tools
    import pm.client

    _, _, publish = browser_store
    requests, published = [], []

    def request(operation, payload, **kwargs):
        requests.append((operation, payload))
        published.append(publish("chromium"))

    monkeypatch.setattr(pm.client, "_request", request)
    doctor_tools._check_chromium(False)
    assert requests == []
    assert "Chromium not installed" in capsys.readouterr().out
    doctor_tools._check_chromium(True)
    assert requests == [("ensure", {"name": "chromium", "explicit": True})]
    assert str(published[0]) in capsys.readouterr().out


def test_termux_ownership_policy_never_provisions(browser_store, monkeypatch):
    # Exercise the real environment policy, not an emulated Android binary.
    from hermes_cli import doctor_tools

    monkeypatch.setenv("TERMUX_VERSION", "test")
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "0")
    monkeypatch.setattr(launcher, "running_in_docker", lambda: False)

    def forbidden(*args, **kwargs):
        pytest.fail("Termux browser installation is externally owned")

    monkeypatch.setattr(pm, "ensure", forbidden)
    assert launcher._maybe_autoinstall_chromium() is False
    assert launcher.chromium_ready()[0] is False
    doctor_tools._check_chromium(True)  # not even `hermes doctor --fix`


def test_cdp_override_does_not_require_pm_browser(browser_store, monkeypatch):
    from tools import browser_session

    monkeypatch.setenv("BROWSER_CDP_URL", "ws://127.0.0.1:9222/devtools/browser/test")
    monkeypatch.setattr(launcher, "launch_chromium",
                        lambda *a, **kw: pytest.fail("a CDP override must not launch a local Chromium"))
    session = browser_session._create_session_for_key("cdp-task", force_local=False)
    assert session["cdp_url"] == "ws://127.0.0.1:9222/devtools/browser/test"
    assert session["features"] == {"cdp_override": True}
    assert launcher.chromium_ready()[0] is False


def _pin_executable(path) -> None:
    """``browser.executable_path`` in this profile's config.yaml (the operator's browser pin)."""
    import json
    from hermes_constants import get_hermes_home
    (get_hermes_home() / "config.yaml").write_text(json.dumps({"browser": {"executable_path": str(path)}}))


def test_a_missing_pinned_browser_is_named_in_the_hint(browser_store, tmp_path):
    """The pin wins over pm's Chromium, so "install chromium" cannot help: the hint names the pinned path."""
    _, _, publish = browser_store
    publish("chromium")
    missing = tmp_path / "moved away" / "chrome"
    _pin_executable(missing)
    ok, hint = launcher.chromium_ready()
    assert ok is False
    assert "browser.executable_path" in hint and str(missing) in hint
    assert "hermes pm install chromium" not in hint


def test_historical_warmer_is_inert(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("historical browser warmer must do no work")

    monkeypatch.setattr("subprocess.Popen", forbidden)
    monkeypatch.setattr(pm, "ensure", forbidden)
    from tools import browser_tool

    assert browser_tool.warm_agent_browser_npx_cache(timeout=0.1) is False


def test_external_chromium_override_survives_pm_composition(browser_store, monkeypatch, tmp_path):
    _, _, publish = browser_store
    publish("chromium")
    override = tmp_path / "user browser"
    override.write_bytes(b"external")
    _pin_executable(override)
    assert launcher.chromium_ready() == (True, "")
    assert launcher._resolve_executable() == str(override)


_FAKE_PM_CHROMIUM = """#!{python}
import json, os, sys, time
udd = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--user-data-dir="))
with open(os.path.join(udd, "child.json"), "w") as f:
    json.dump({{"argv": sys.argv, "env": dict(os.environ)}}, f)
with open(os.path.join(udd, "DevToolsActivePort"), "w") as f:
    f.write("45679\\n/devtools/browser/pm-fixture\\n")
time.sleep(600)
"""


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass  # the fake PM Chromium is this test's own child
def test_exact_pm_child_receives_composed_scrubbed_environment(browser_store, monkeypatch, tmp_path):
    _, _, publish = browser_store
    chromium = publish("chromium", _FAKE_PM_CHROMIUM.format(python=sys.executable))
    records = tmp_path / "records"
    records.mkdir()
    monkeypatch.setattr(launcher, "_records_dir", lambda _home: records)
    monkeypatch.setattr(launcher, "needs_chromium_sandbox_bypass", lambda: False)
    monkeypatch.setattr("tools.browser_session_cloud.chromium_args", list)
    monkeypatch.setattr("tools.browser_session_cloud.is_headed_mode", lambda: False)
    monkeypatch.setenv("PATH", str(tmp_path / "hostile-path"))
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-browser")
    monkeypatch.setenv("BROWSERBASE_API_KEY", "browser-provider-key")
    before = dict(os.environ)
    browser, err = launcher.launch_chromium("pm-child", timeout=10)
    try:
        assert err is None
        child_file = Path(browser.user_data_dir, "child.json")
        wait_until(child_file.exists, what="the fake Chromium to write child.json")
        child = json.loads(child_file.read_text())
    finally:
        launcher.stop_browser("pm-child")
    assert Path(child["argv"][0]) == chromium
    assert "OPENAI_API_KEY" not in child["env"]
    assert "BROWSERBASE_API_KEY" not in child["env"]
    assert dict(os.environ) == before


@pytest.mark.parametrize("fails", [False, True])
def test_chromium_autoinstall_is_one_shot(browser_store, monkeypatch, fails):
    import pm.client

    _, _, publish = browser_store
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "0")
    monkeypatch.setattr(launcher, "running_in_docker", lambda: False)
    calls = []

    def request(operation, payload, **kwargs):
        calls.append((operation, payload))
        if fails:
            raise pm.InstallError("chromium", "download failed")
        publish("chromium")

    monkeypatch.setattr(pm.client, "_request", request)
    assert launcher.chromium_ready()[0] is (not fails)
    assert launcher.chromium_ready()[0] is (not fails)
    assert calls == [("ensure", {"name": "chromium", "explicit": False})]


@pytest.mark.parametrize("policy", ["docker", "disabled", "termux", "override"])
def test_chromium_acquisition_policy(browser_store, monkeypatch, policy, tmp_path):
    import pm.client

    monkeypatch.setattr(launcher, "running_in_docker", lambda: policy == "docker")
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1" if policy == "disabled" else "0")
    if policy == "termux":
        monkeypatch.setenv("TERMUX_VERSION", "test")
    if policy == "override":
        override = tmp_path / "external-chrome"
        override.touch()
        _pin_executable(override)
        assert launcher._resolve_executable() == str(override)
        override.unlink()
        assert launcher._resolve_executable() is None
    monkeypatch.setattr(pm.client, "_request", lambda *a, **kw: pytest.fail("forbidden acquisition"))
    assert not launcher._maybe_autoinstall_chromium()


def test_browser_readiness_ignores_ambient_chromium(browser_store, monkeypatch):
    home, _, publish = browser_store
    (home / "chromium_headless_shell-1234").mkdir()
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(home))
    ambient = home / ("chromium.exe" if os.name == "nt" else "chromium")
    ambient.touch()
    ambient.chmod(0o755)
    monkeypatch.setenv("PATH", str(home))
    assert launcher.chromium_ready()[0] is False
    publish("chromium")
    assert launcher.chromium_ready()[0] is True
