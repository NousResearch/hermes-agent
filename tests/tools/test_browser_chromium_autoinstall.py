"""Tests for gated Chromium-binary auto-install on local cold start."""

from types import SimpleNamespace

import pytest

import tools.browser_tool as bt


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
        monkeypatch.setattr(bt, "_running_in_docker", lambda: False)
        monkeypatch.setattr("tools.lazy_deps._allow_lazy_installs", lambda: False)
        calls = _no_subprocess(monkeypatch)
        assert bt._maybe_autoinstall_chromium() is False
        assert calls == []

    def test_docker_skips(self, monkeypatch):
        monkeypatch.setattr(bt, "_running_in_docker", lambda: True)
        calls = _no_subprocess(monkeypatch)
        assert bt._maybe_autoinstall_chromium() is False
        assert calls == []


class TestInstall:
    def test_success_installs_binary_only_and_rechecks(self, monkeypatch):
        monkeypatch.setattr(bt, "_running_in_docker", lambda: False)
        monkeypatch.setattr("tools.lazy_deps._allow_lazy_installs", lambda: True)
        monkeypatch.setattr(bt, "_find_agent_browser", lambda: "/x/agent-browser")
        monkeypatch.setattr(bt, "_build_browser_env", lambda: {})
        monkeypatch.setattr(bt, "_chromium_installed", lambda: True)

        captured = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(bt.subprocess, "run", fake_run)

        assert bt._maybe_autoinstall_chromium() is True
        assert captured["cmd"] == ["/x/agent-browser", "install"]
        assert "--with-deps" not in captured["cmd"]

    def test_npx_form_is_binary_only(self, monkeypatch):
        monkeypatch.setattr(bt, "_running_in_docker", lambda: False)
        monkeypatch.setattr("tools.lazy_deps._allow_lazy_installs", lambda: True)
        monkeypatch.setattr(bt, "_find_agent_browser", lambda: "npx agent-browser")
        monkeypatch.setattr(bt, "_build_browser_env", lambda: {})
        monkeypatch.setattr(bt, "_chromium_installed", lambda: True)
        monkeypatch.setattr(bt.shutil, "which", lambda _: "/usr/bin/npx")

        captured = {}
        monkeypatch.setattr(
            bt.subprocess, "run",
            lambda cmd, **kw: captured.update(cmd=cmd) or SimpleNamespace(returncode=0, stdout="", stderr=""),
        )

        assert bt._maybe_autoinstall_chromium() is True
        assert captured["cmd"] == ["/usr/bin/npx", "-y", "agent-browser", "install"]
        assert "--with-deps" not in captured["cmd"]

    def test_nonzero_exit_returns_false(self, monkeypatch):
        monkeypatch.setattr(bt, "_running_in_docker", lambda: False)
        monkeypatch.setattr("tools.lazy_deps._allow_lazy_installs", lambda: True)
        monkeypatch.setattr(bt, "_find_agent_browser", lambda: "/x/agent-browser")
        monkeypatch.setattr(bt, "_build_browser_env", lambda: {})
        monkeypatch.setattr(
            bt.subprocess, "run",
            lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
        )
        assert bt._maybe_autoinstall_chromium() is False


class TestOneShot:
    def test_second_call_does_not_reinstall(self, monkeypatch):
        monkeypatch.setattr(bt, "_running_in_docker", lambda: False)
        monkeypatch.setattr("tools.lazy_deps._allow_lazy_installs", lambda: True)
        monkeypatch.setattr(bt, "_find_agent_browser", lambda: "/x/agent-browser")
        monkeypatch.setattr(bt, "_build_browser_env", lambda: {})
        monkeypatch.setattr(bt, "_chromium_installed", lambda: True)

        runs = []
        monkeypatch.setattr(
            bt.subprocess, "run",
            lambda *a, **k: runs.append(1) or SimpleNamespace(returncode=0, stdout="", stderr=""),
        )

        assert bt._maybe_autoinstall_chromium() is True
        assert bt._maybe_autoinstall_chromium() is True
        assert len(runs) == 1
