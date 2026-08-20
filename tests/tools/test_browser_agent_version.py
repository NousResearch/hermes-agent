"""Executable-level capability tests for agent-browser CDP tab pinning."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools import browser_tool as bt


pytestmark = pytest.mark.skipif(
    os.name == "nt",
    reason="fake POSIX executables use /bin/sh",
)


@pytest.fixture(autouse=True)
def _reset_agent_browser_caches():
    original = (
        bt._cached_agent_browser,
        bt._agent_browser_resolved,
        bt._cached_pin_tab_agent_browser,
        bt._pin_tab_agent_browser_resolved,
        bt._pin_tab_failure_cache,
    )
    bt._cached_agent_browser = None
    bt._agent_browser_resolved = False
    bt._cached_pin_tab_agent_browser = None
    bt._pin_tab_agent_browser_resolved = False
    bt._pin_tab_failure_cache = None
    yield
    (
        bt._cached_agent_browser,
        bt._agent_browser_resolved,
        bt._cached_pin_tab_agent_browser,
        bt._pin_tab_agent_browser_resolved,
        bt._pin_tab_failure_cache,
    ) = original


def _write_executable(path: Path, body: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return str(path)


def _fake_agent_browser(root: Path, version: str, node_major: int) -> str:
    cli = _write_executable(
        root / "agent-browser",
        (
            'if [ "$1" = "--version" ]; then\n'
            f"  printf '%s\\n' 'agent-browser {version}'\n"
            "  exit 0\n"
            "fi\n"
            "exit 91"
        ),
    )
    _write_executable(
        root / "node",
        f"printf '%s\\n' 'v{node_major}.0.0'",
    )
    return cli


def _fake_npx(root: Path, node_major: int) -> str:
    npx = _write_executable(root / "npx", "printf '%s\\n' '11.19.0'")
    _write_executable(
        root / "node",
        f"printf '%s\\n' 'v{node_major}.0.0'",
    )
    return npx


def _configure_resolution(
    monkeypatch,
    *,
    current_agent_browser: str | None = None,
    extended_agent_browser: str | None = None,
    npx: str | None = None,
) -> None:
    extra_dirs = []
    for executable in (extended_agent_browser, npx):
        if executable:
            parent = str(Path(executable).parent)
            if parent not in extra_dirs:
                extra_dirs.append(parent)

    def merge_path(_existing: str) -> str:
        return os.pathsep.join(extra_dirs)

    def which(command: str, path: str | None = None) -> str | None:
        search_dirs = [] if path is None else path.split(os.pathsep)
        if command == "agent-browser":
            if path is None:
                return current_agent_browser
            if (
                extended_agent_browser
                and str(Path(extended_agent_browser).parent) in search_dirs
            ):
                return extended_agent_browser
            return None
        if command == "npx":
            if npx and (path is None or str(Path(npx).parent) in search_dirs):
                return npx
            return None
        if command == "node":
            for directory in search_dirs:
                candidate = Path(directory) / "node"
                if candidate.is_file():
                    return str(candidate)
            return None
        return None

    monkeypatch.setattr(bt, "_build_browser_env", lambda: {"PATH": ""})
    monkeypatch.setattr(bt, "_merge_browser_path", merge_path)
    monkeypatch.setattr(bt.shutil, "which", which)


def test_runnable_033_remains_generic_but_is_rejected_for_pin(monkeypatch, tmp_path):
    old_cli = _fake_agent_browser(tmp_path / "old", "0.33.1", 24)
    _configure_resolution(monkeypatch, current_agent_browser=old_cli)
    clock = [100.0]
    monkeypatch.setattr(bt.time, "monotonic", lambda: clock[0])

    assert bt._find_agent_browser() == old_cli
    with pytest.raises(bt.AgentBrowserCapabilityError) as exc_info:
        bt._find_agent_browser(require_pin_tab=True)

    message = str(exc_info.value)
    assert "agent-browser >=0.34.0" in message
    assert "Node.js >=24" in message
    assert "ordinary non-CDP browser commands remain supported" in message
    assert bt._find_agent_browser() == old_cli

    # The negative cache suppresses repeated subprocess probes, but expires
    # quickly enough that an in-place upgrade works without restarting Hermes.
    _fake_agent_browser(tmp_path / "old", "0.34.0", 24)
    real_probe = bt._probe_agent_browser_version
    monkeypatch.setattr(
        bt,
        "_probe_agent_browser_version",
        lambda _path: pytest.fail("negative cache should suppress a second probe"),
    )
    with pytest.raises(bt.AgentBrowserCapabilityError, match="agent-browser >=0.34.0"):
        bt._find_agent_browser(require_pin_tab=True)
    clock[0] += bt.AGENT_BROWSER_PIN_TAB_FAILURE_TTL_SECONDS + 0.01
    monkeypatch.setattr(bt, "_probe_agent_browser_version", real_probe)
    assert bt._find_agent_browser(require_pin_tab=True) == old_cli


def test_generic_install_guidance_keeps_node22_package_line(monkeypatch):
    monkeypatch.setattr(bt, "_is_termux_environment", lambda: False)

    assert bt._browser_install_hint() == (
        "npm install -g 'agent-browser@0.26.0' && "
        "agent-browser install --with-deps"
    )


def test_pin_lookup_falls_through_033_to_034_and_caches(monkeypatch, tmp_path):
    old_cli = _fake_agent_browser(tmp_path / "old", "0.33.1", 24)
    new_cli = _fake_agent_browser(tmp_path / "new", "0.34.0", 24)
    _configure_resolution(
        monkeypatch,
        current_agent_browser=old_cli,
        extended_agent_browser=new_cli,
    )

    assert bt._find_agent_browser() == old_cli
    assert bt._find_agent_browser(require_pin_tab=True) == new_cli

    monkeypatch.setattr(
        bt,
        "_probe_agent_browser_version",
        lambda _path: pytest.fail("pin-capability cache should avoid a second probe"),
    )
    assert bt._find_agent_browser(require_pin_tab=True) == new_cli
    assert bt._find_agent_browser() == old_cli


def test_agent_browser_034_on_node22_is_not_pin_capable(monkeypatch, tmp_path):
    cli = _fake_agent_browser(tmp_path / "node22", "0.34.0", 22)
    _configure_resolution(monkeypatch, current_agent_browser=cli)

    with pytest.raises(bt.AgentBrowserCapabilityError, match=r"Detected Node\.js 22"):
        bt._find_agent_browser(require_pin_tab=True)


@pytest.mark.parametrize(
    ("node_major", "pin_available"),
    [(22, False), (24, True)],
)
def test_npx_pin_capability_respects_node_floor(
    monkeypatch, tmp_path, node_major, pin_available
):
    npx = _fake_npx(tmp_path / f"node-{node_major}", node_major)
    _configure_resolution(monkeypatch, npx=npx)

    if pin_available:
        assert (
            bt._find_agent_browser(require_pin_tab=True)
            == bt.NPX_AGENT_BROWSER_SENTINEL
        )
    else:
        with pytest.raises(
            bt.AgentBrowserCapabilityError, match=r"Detected Node\.js 22"
        ):
            bt._find_agent_browser(require_pin_tab=True)
        # The capability failure is scoped to shared CDP. The same runnable
        # npx remains available for the Node-22-compatible ordinary package.
        assert bt._find_agent_browser() == bt.NPX_AGENT_BROWSER_SENTINEL


def test_cdp_command_does_not_spawn_runnable_old_cli(monkeypatch, tmp_path):
    old_cli = _fake_agent_browser(tmp_path / "old", "0.33.1", 24)
    _configure_resolution(monkeypatch, current_agent_browser=old_cli)
    real_popen = bt.subprocess.Popen
    command_spawns = []

    def guarded_popen(command, *args, **kwargs):
        if command == [old_cli, "--version"]:
            return real_popen(command, *args, **kwargs)
        command_spawns.append(command)
        raise AssertionError("old CLI must not be invoked for the CDP command")

    monkeypatch.setattr(bt, "_requires_real_termux_browser_install", lambda _cmd: False)
    monkeypatch.setattr(bt, "_is_local_mode", lambda: False)
    monkeypatch.setattr(
        bt,
        "_get_session_info",
        lambda _task: {
            "session_name": "cdp_task",
            "cdp_url": "ws://shared",
        },
    )
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
    monkeypatch.setattr(bt.subprocess, "Popen", guarded_popen)

    result = bt._run_browser_command("task", "snapshot", ["-c"])

    assert result["success"] is False
    assert result["code"] == "pin_tab_unavailable"
    assert result["data"] == {
        "required_agent_browser": ">=0.34.0",
        "required_node": ">=24",
        "non_cdp_available": True,
    }
    assert "agent-browser >=0.34.0" in result["error"]
    assert command_spawns == []
