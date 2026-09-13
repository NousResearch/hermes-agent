"""Local cold starts provision Chromium through the pinned package manager."""
import pytest

from pm.package import InstallError
from tools import browser_tool as bt
from tools import browser_tool_install as bt_install


@pytest.fixture(autouse=True)
def reset_state():
    bt._chromium_autoinstall_attempted = False
    yield
    bt._chromium_autoinstall_attempted = False


def test_install_uses_pm_once_and_preserves_failure(monkeypatch):
    monkeypatch.setattr(bt_install, "_running_in_docker", lambda: False)
    monkeypatch.setattr("pm.lazy_installs_allowed", lambda: True)
    installed = False
    calls = []

    def ensure(name):
        nonlocal installed
        calls.append(name)
        installed = True

    monkeypatch.setattr("pm.ensure", ensure)
    monkeypatch.setattr(bt_install, "_chromium_installed", lambda: installed)
    assert bt_install._maybe_autoinstall_chromium()
    assert bt_install._maybe_autoinstall_chromium()
    assert calls == ["chromium"]

    bt._chromium_autoinstall_attempted = False
    installed = False

    def fail(name):
        raise InstallError(name, "download failed")

    monkeypatch.setattr("pm.ensure", fail)
    assert not bt_install._maybe_autoinstall_chromium()


@pytest.mark.parametrize(("docker", "allowed"), [(True, True), (False, False)])
def test_install_policy_never_provisions(monkeypatch, docker, allowed):
    monkeypatch.setattr(bt_install, "_running_in_docker", lambda: docker)
    monkeypatch.setattr("pm.lazy_installs_allowed", lambda: allowed)

    def forbidden(*args, **kwargs):
        pytest.fail("blocked auto-install reached provisioning")

    monkeypatch.setattr("pm.ensure", forbidden)
    monkeypatch.setattr(bt_install, "_find_agent_browser", forbidden)
    assert not bt_install._maybe_autoinstall_chromium()
