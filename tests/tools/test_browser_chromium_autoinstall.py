"""Tests for gated Chromium-binary auto-install on local cold start."""

import shutil
from types import SimpleNamespace

import pytest

import tools.browser_tool as bt
from tools import browser_tool_install as bt_install


@pytest.fixture(autouse=True)
def _reset_state():
    bt._chromium_autoinstall_attempted = False
    bt._cached_chromium_installed = None
    yield
    bt._chromium_autoinstall_attempted = False
    bt._cached_chromium_installed = None


def _no_subprocess(monkeypatch):
    calls = []
    monkeypatch.setattr(bt.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    return calls


class TestGating:
    def test_disabled_lazy_installs_skips(self, monkeypatch):
        monkeypatch.setattr("tools.browser_tool_install._running_in_docker", lambda: False)
        monkeypatch.setattr("tools.lazy_deps._allow_lazy_installs", lambda: False)
        calls = _no_subprocess(monkeypatch)
        assert bt_install._maybe_autoinstall_chromium() is False
        assert calls == []

    def test_docker_skips(self, monkeypatch):
        monkeypatch.setattr("tools.browser_tool_install._running_in_docker", lambda: True)
        calls = _no_subprocess(monkeypatch)
        assert bt_install._maybe_autoinstall_chromium() is False
        assert calls == []


class TestInstall:
    def test_success_installs_binary_only_and_rechecks(self, monkeypatch):
        monkeypatch.setattr("tools.browser_tool_install._running_in_docker", lambda: False)
        monkeypatch.setattr("tools.lazy_deps._allow_lazy_installs", lambda: True)
        monkeypatch.setattr(bt_install, "_find_agent_browser", lambda: "/x/agent-browser")
        monkeypatch.setattr(bt, "_build_browser_env", lambda: {})
        monkeypatch.setattr("tools.browser_tool_install._chromium_installed", lambda: True)

        captured = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(bt.subprocess, "run", fake_run)

        assert bt_install._maybe_autoinstall_chromium() is True
        assert captured["cmd"] == ["/x/agent-browser", "install"]
        assert "--with-deps" not in captured["cmd"]

    def test_npx_form_is_binary_only(self, monkeypatch):
        monkeypatch.setattr("tools.browser_tool_install._running_in_docker", lambda: False)
        monkeypatch.setattr("tools.lazy_deps._allow_lazy_installs", lambda: True)
        monkeypatch.setattr(bt_install, "_find_agent_browser", lambda: "npx agent-browser")
        monkeypatch.setattr(bt, "_build_browser_env", lambda: {})
        monkeypatch.setattr("tools.browser_tool_install._chromium_installed", lambda: True)
        monkeypatch.setattr(shutil, "which", lambda _, path=None: "/usr/bin/npx")
        monkeypatch.setattr("tools.browser_tool_install.node_tool_runnable", lambda p: True)

        captured = {}
        monkeypatch.setattr(
            bt.subprocess, "run",
            lambda cmd, **kw: captured.update(cmd=cmd) or SimpleNamespace(returncode=0, stdout="", stderr=""),
        )

        assert bt_install._maybe_autoinstall_chromium() is True
        assert captured["cmd"] == [
            "/usr/bin/npx", "--ignore-scripts", "-y", bt.AGENT_BROWSER_NPX_SPEC, "install",
        ]
        assert "--with-deps" not in captured["cmd"]

    def test_nonzero_exit_returns_false(self, monkeypatch):
        monkeypatch.setattr("tools.browser_tool_install._running_in_docker", lambda: False)
        monkeypatch.setattr("tools.lazy_deps._allow_lazy_installs", lambda: True)
        monkeypatch.setattr(bt_install, "_find_agent_browser", lambda: "/x/agent-browser")
        monkeypatch.setattr(bt, "_build_browser_env", lambda: {})
        monkeypatch.setattr(
            bt.subprocess, "run",
            lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
        )
        assert bt_install._maybe_autoinstall_chromium() is False


class TestStaleCacheOnFailure:
    def test_manual_install_after_failed_autoinstall_is_detected(self, monkeypatch):
        """#105746: a failed autoinstall must not freeze the negative cache forever.

        Mirrors the real call order in ``_browser_command_preflight``: ``_chromium_installed()``
        runs first (caching the negative result), then ``_maybe_autoinstall_chromium()``. Once the
        autoinstall attempt has failed, the next preflight round must re-probe disk rather than keep
        reporting the pre-install absence, so a manual ``npx playwright install`` becomes visible
        without a process restart.
        """
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(shutil, "which", lambda name: None)
        chromium_present = {"value": False}
        monkeypatch.setattr(bt_install, "_chromium_search_roots", lambda: ["/fake/ms-playwright"])
        monkeypatch.setattr("os.path.isdir", lambda p: p == "/fake/ms-playwright")
        monkeypatch.setattr(bt_install, "_has_chromium_build", lambda root: chromium_present["value"])

        # Preflight's first check: Chromium absent, negative result cached.
        assert bt_install._chromium_installed() is False

        # Autoinstall attempt fails (e.g. offline, no npx).
        monkeypatch.setattr(bt_install, "_running_in_docker", lambda: False)
        monkeypatch.setattr("tools.lazy_deps._allow_lazy_installs", lambda: True)
        monkeypatch.setattr(bt_install, "_find_agent_browser", lambda: (_ for _ in ()).throw(FileNotFoundError()))
        assert bt_install._maybe_autoinstall_chromium() is False

        # User installs Chromium manually; the on-disk probe now finds it.
        chromium_present["value"] = True

        # Without invalidating the cache on the failure path, this would still return False.
        assert bt_install._maybe_autoinstall_chromium() is True
        assert bt_install._chromium_installed() is True


class TestOneShot:
    def test_second_call_does_not_reinstall(self, monkeypatch):
        monkeypatch.setattr("tools.browser_tool_install._running_in_docker", lambda: False)
        monkeypatch.setattr("tools.lazy_deps._allow_lazy_installs", lambda: True)
        monkeypatch.setattr(bt_install, "_find_agent_browser", lambda: "/x/agent-browser")
        monkeypatch.setattr(bt, "_build_browser_env", lambda: {})
        monkeypatch.setattr("tools.browser_tool_install._chromium_installed", lambda: True)

        runs = []
        monkeypatch.setattr(
            bt.subprocess, "run",
            lambda *a, **k: runs.append(1) or SimpleNamespace(returncode=0, stdout="", stderr=""),
        )

        assert bt_install._maybe_autoinstall_chromium() is True
        assert bt_install._maybe_autoinstall_chromium() is True
        assert len(runs) == 1
