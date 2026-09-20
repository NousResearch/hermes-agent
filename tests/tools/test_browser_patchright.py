"""Regression tests for native Patchright browser routing."""

from __future__ import annotations

from unittest.mock import Mock

from tools import browser_patchright as patchright
from tools import browser_tool


def test_patchright_mode_from_config(monkeypatch):
    monkeypatch.delenv("HERMES_BROWSER_BACKEND", raising=False)
    monkeypatch.setattr(
        patchright,
        "read_raw_config",
        lambda: {"browser": {"cloud_provider": "patchright"}},
    )
    assert patchright.is_patchright_mode() is True


def test_patchright_mode_explicit_env_override(monkeypatch):
    monkeypatch.setenv("HERMES_BROWSER_BACKEND", "patchright")
    monkeypatch.setattr(
        patchright,
        "read_raw_config",
        lambda: {"browser": {"cloud_provider": "camofox"}},
    )
    assert patchright.is_patchright_mode() is True


def test_camofox_is_suppressed_when_patchright_is_selected(monkeypatch):
    monkeypatch.setattr(browser_tool, "_is_patchright_mode", lambda: True)
    monkeypatch.setattr(browser_tool, "_raw_is_camofox_mode", lambda: True)
    assert browser_tool._is_camofox_mode() is False


def test_cloud_provider_bypass_returns_local_backend(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"browser": {"cloud_provider": "patchright"}},
    )
    monkeypatch.setattr(browser_tool, "_cloud_provider_resolved", False)
    monkeypatch.setattr(browser_tool, "_cached_cloud_provider", Mock())
    assert browser_tool._get_cloud_provider() is None


def test_patchright_requirements_do_not_need_agent_browser(monkeypatch):
    monkeypatch.setattr(browser_tool, "_is_patchright_mode", lambda: True)
    monkeypatch.setattr(patchright, "check_patchright_available", lambda: True)
    assert browser_tool.check_browser_requirements() is True


def test_run_command_dispatches_before_agent_browser(monkeypatch):
    expected = {"success": True, "data": {"snapshot": "native"}}
    monkeypatch.setattr(browser_tool, "_is_patchright_mode", lambda: True)
    monkeypatch.setattr(
        patchright,
        "run_browser_command",
        lambda task_id, command, args, timeout=None, _engine_override=None: expected,
    )
    monkeypatch.setattr(
        browser_tool,
        "_find_agent_browser",
        lambda: (_ for _ in ()).throw(AssertionError("agent-browser was consulted")),
    )
    assert browser_tool._run_browser_command("task", "snapshot", []) == expected
