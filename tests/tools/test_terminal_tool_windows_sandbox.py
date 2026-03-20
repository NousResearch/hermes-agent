"""Focused terminal_tool tests for the windows-sandbox backend."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


terminal_tool_module = importlib.import_module("tools.terminal_tool")


def _windows_sandbox_config(tmp_path: Path) -> dict:
    return {
        "env_type": "windows-sandbox",
        "cwd": str(tmp_path),
        "timeout": 180,
        "lifetime_seconds": 300,
        "windows_sandbox_mode": "workspace-write",
        "windows_sandbox_setup": "explicit",
        "windows_sandbox_network": False,
        "windows_sandbox_bin_dir": str(tmp_path),
        "windows_sandbox_writable_roots": [],
    }


def test_windows_sandbox_background_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(
        terminal_tool_module,
        "_get_env_config",
        lambda: _windows_sandbox_config(tmp_path),
    )

    result = json.loads(terminal_tool_module.terminal_tool("echo hi", background=True))
    assert result["status"] == "error"
    assert "does not support background execution" in result["error"]


def test_windows_sandbox_pty_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(
        terminal_tool_module,
        "_get_env_config",
        lambda: _windows_sandbox_config(tmp_path),
    )

    result = json.loads(terminal_tool_module.terminal_tool("echo hi", pty=True))
    assert result["status"] == "error"
    assert "does not support PTY mode" in result["error"]


def test_windows_sandbox_requirements_check_uses_wrapper(monkeypatch, tmp_path):
    wrapper = tmp_path / "hermes-windows-sandbox-wrapper.exe"
    wrapper.write_text("stub", encoding="utf-8")
    helper = tmp_path / "codex-windows-sandbox-setup.exe"
    helper.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(
        terminal_tool_module,
        "_get_env_config",
        lambda: _windows_sandbox_config(tmp_path),
    )
    monkeypatch.setattr(terminal_tool_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(terminal_tool_module.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(
        terminal_tool_module,
        "_find_windows_sandbox_wrapper",
        lambda _bin_dir="": wrapper,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_find_windows_sandbox_setup_helper",
        lambda _bin_dir="", wrapper_path=None: helper,
    )
    monkeypatch.setattr(
        terminal_tool_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )

    assert terminal_tool_module.check_terminal_requirements() is True


def test_windows_sandbox_requirements_check_uses_hermes_bin_by_default(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    hermes_bin = hermes_home / "bin"
    hermes_bin.mkdir(parents=True)
    wrapper = hermes_bin / "hermes-windows-sandbox-wrapper.exe"
    helper = hermes_bin / "codex-windows-sandbox-setup.exe"
    wrapper.write_text("stub", encoding="utf-8")
    helper.write_text("stub", encoding="utf-8")

    config = _windows_sandbox_config(tmp_path)
    config["windows_sandbox_bin_dir"] = ""

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(
        terminal_tool_module,
        "_get_env_config",
        lambda: config,
    )
    monkeypatch.setattr(terminal_tool_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(terminal_tool_module.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(
        terminal_tool_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )

    assert terminal_tool_module.check_terminal_requirements() is True


def test_windows_sandbox_requirements_fail_when_wrapper_missing(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(
        terminal_tool_module,
        "_get_env_config",
        lambda: _windows_sandbox_config(tmp_path),
    )
    monkeypatch.setattr(terminal_tool_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(terminal_tool_module.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(
        terminal_tool_module,
        "_find_windows_sandbox_wrapper",
        lambda _bin_dir="": None,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_find_windows_sandbox_setup_helper",
        lambda _bin_dir="", wrapper_path=None: None,
    )

    with caplog.at_level("ERROR"):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert "wrapper executable not found" in caplog.text


def test_windows_sandbox_requirements_fail_when_setup_helper_missing(monkeypatch, tmp_path, caplog):
    wrapper = tmp_path / "hermes-windows-sandbox-wrapper.exe"
    wrapper.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(
        terminal_tool_module,
        "_get_env_config",
        lambda: _windows_sandbox_config(tmp_path),
    )
    monkeypatch.setattr(terminal_tool_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(terminal_tool_module.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(
        terminal_tool_module,
        "_find_windows_sandbox_wrapper",
        lambda _bin_dir="": wrapper,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_find_windows_sandbox_setup_helper",
        lambda _bin_dir="", wrapper_path=None: None,
    )

    with caplog.at_level("ERROR"):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert "setup helper executable not found" in caplog.text


def test_windows_sandbox_terminal_tool_does_not_fallback_to_local(monkeypatch, tmp_path):
    monkeypatch.setattr(
        terminal_tool_module,
        "_get_env_config",
        lambda: _windows_sandbox_config(tmp_path),
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_WindowsSandboxEnvironment",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("windows-sandbox wrapper executable not found")
        ),
    )

    result = json.loads(
        terminal_tool_module.terminal_tool(
            "Write-Output 'hello from local fallback'",
            task_id="windows-sandbox-no-fallback",
        )
    )

    assert result["exit_code"] == -1
    assert "windows-sandbox wrapper executable not found" in result["error"]
    assert result["status"] == "error"


def test_windows_sandbox_requirements_fail_on_arm64(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(
        terminal_tool_module,
        "_get_env_config",
        lambda: _windows_sandbox_config(tmp_path),
    )
    monkeypatch.setattr(terminal_tool_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(terminal_tool_module.platform, "machine", lambda: "ARM64")

    with caplog.at_level("ERROR"):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert "only supported on x64 Windows hosts" in caplog.text
