"""Windows agent-browser npm-shim bypass regression coverage."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import tools.browser_tool as bt


@pytest.fixture(autouse=True)
def _reset_agent_browser_cache():
    bt._cached_agent_browser = None
    bt._agent_browser_resolved = False
    yield
    bt._cached_agent_browser = None
    bt._agent_browser_resolved = False


def _make_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test")
    return path


def test_global_npm_cmd_resolves_to_validated_native_executable(
    tmp_path, monkeypatch
):
    shim = _make_file(tmp_path / "npm" / "agent-browser.CMD")
    native = _make_file(
        tmp_path
        / "npm"
        / "node_modules"
        / "agent-browser"
        / "bin"
        / "agent-browser-win32-x64.exe"
    )
    validated = []

    monkeypatch.setattr(bt.sys, "platform", "win32")
    monkeypatch.setattr(
        bt,
        "agent_browser_runnable",
        lambda path: validated.append(path) or path == str(native),
    )

    assert bt._prefer_windows_native_agent_browser(str(shim)) == str(native)
    assert validated == [str(native)]


def test_project_local_npm_cmd_resolves_to_package_sibling(
    tmp_path, monkeypatch
):
    shim = _make_file(
        tmp_path / "repo" / "node_modules" / ".bin" / "agent-browser.cmd"
    )
    native = _make_file(
        tmp_path
        / "repo"
        / "node_modules"
        / "agent-browser"
        / "bin"
        / "agent-browser-win32-arm64.exe"
    )

    monkeypatch.setattr(bt.sys, "platform", "win32")
    monkeypatch.setattr(bt, "agent_browser_runnable", lambda path: path == str(native))

    assert bt._prefer_windows_native_agent_browser(str(shim)) == str(native)


@pytest.mark.parametrize("failure", ["missing", "unrunnable"])
def test_native_resolution_failure_keeps_discovered_shim(
    tmp_path, monkeypatch, failure
):
    shim = _make_file(tmp_path / "npm" / "agent-browser.cmd")
    native = (
        tmp_path
        / "npm"
        / "node_modules"
        / "agent-browser"
        / "bin"
        / "agent-browser-win32-x64.exe"
    )
    if failure == "unrunnable":
        _make_file(native)

    monkeypatch.setattr(bt.sys, "platform", "win32")
    monkeypatch.setattr(bt, "agent_browser_runnable", lambda _path: False)

    assert bt._prefer_windows_native_agent_browser(str(shim)) == str(shim)


def test_non_windows_candidate_is_unchanged(tmp_path, monkeypatch):
    shim = _make_file(tmp_path / "npm" / "agent-browser.cmd")

    def runnable(_path):
        pytest.fail("non-Windows path must not be validated")

    monkeypatch.setattr(bt.sys, "platform", "linux")
    monkeypatch.setattr(bt, "agent_browser_runnable", runnable)

    assert bt._prefer_windows_native_agent_browser(str(shim)) == str(shim)


def test_find_agent_browser_caches_native_instead_of_global_cmd(
    tmp_path, monkeypatch
):
    shim = _make_file(tmp_path / "npm" / "agent-browser.cmd")
    native = _make_file(
        tmp_path
        / "npm"
        / "node_modules"
        / "agent-browser"
        / "bin"
        / "agent-browser-win32-x64.exe"
    )
    validated = []

    monkeypatch.setattr(bt.sys, "platform", "win32")
    monkeypatch.setattr(bt.shutil, "which", lambda *_args, **_kwargs: str(shim))
    monkeypatch.setattr(
        bt,
        "agent_browser_runnable",
        lambda path: validated.append(path) or path in {str(shim), str(native)},
    )

    assert bt._find_agent_browser() == str(native)
    assert bt._find_agent_browser() == str(native)
    assert bt._cached_agent_browser == str(native)
    assert validated == [str(native)]


def test_validated_candidate_falls_back_to_runnable_cmd(tmp_path, monkeypatch):
    shim = _make_file(tmp_path / "npm" / "agent-browser.cmd")
    validated = []

    monkeypatch.setattr(bt.sys, "platform", "win32")
    monkeypatch.setattr(
        bt,
        "agent_browser_runnable",
        lambda path: validated.append(path) or path == str(shim),
    )

    assert bt._validated_agent_browser_candidate(str(shim)) == str(shim)
    assert validated == [str(shim)]


def test_find_without_validation_preserves_cmd_and_skips_native_probe(
    tmp_path, monkeypatch
):
    shim = _make_file(tmp_path / "npm" / "agent-browser.cmd")
    _make_file(
        tmp_path
        / "npm"
        / "node_modules"
        / "agent-browser"
        / "bin"
        / "agent-browser-win32-x64.exe"
    )

    monkeypatch.setattr(bt.sys, "platform", "win32")
    monkeypatch.setattr(bt.shutil, "which", lambda *_args, **_kwargs: str(shim))

    def runnable(_path):
        pytest.fail("validate=False must not probe either executable")

    monkeypatch.setattr(bt, "agent_browser_runnable", runnable)

    assert bt._find_agent_browser(validate=False) == str(shim)
    assert bt._cached_agent_browser is None
    assert bt._agent_browser_resolved is False


@pytest.mark.skipif(sys.platform != "win32", reason="real npm layout is Windows-only")
def test_real_windows_npm_install_launches_native_cli():
    """E2E: the installed npm shim resolves to a native CLI that runs."""
    discovered = shutil.which("agent-browser")
    if not discovered or Path(discovered).suffix.lower() != ".cmd":
        pytest.skip("no Windows agent-browser npm command shim installed")

    resolved = bt._find_agent_browser()

    assert Path(resolved).suffix.lower() == ".exe"
    completed = subprocess.run(
        [resolved, "--version"],
        capture_output=True,
        timeout=10,
        creationflags=bt.windows_hide_flags(),
    )
    assert completed.returncode == 0
