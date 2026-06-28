"""Regression coverage for lazy agent-browser probing."""

from __future__ import annotations

import pytest

import tools.browser_tool as browser_tool


@pytest.fixture(autouse=True)
def _reset_agent_browser_cache(monkeypatch):
    monkeypatch.setattr(browser_tool, "_cached_agent_browser", None)
    monkeypatch.setattr(browser_tool, "_cached_agent_browser_validated", False)
    monkeypatch.setattr(browser_tool, "_agent_browser_resolved", False)


def test_check_browser_requirements_does_not_execute_agent_browser(monkeypatch):
    """Tool-list assembly must not run agent-browser --version as a side effect."""

    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: "")
    monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: None)
    monkeypatch.setattr(browser_tool, "_requires_real_termux_browser_install", lambda _cmd: False)
    monkeypatch.setattr(browser_tool, "_using_lightpanda_engine", lambda: True)
    monkeypatch.setattr(
        browser_tool.shutil,
        "which",
        lambda name, path=None: "C:/repo/node_modules/.bin/agent-browser.CMD"
        if name == "agent-browser"
        else None,
    )

    def fail_if_spawned(_path):  # pragma: no cover - only reached on regression
        raise AssertionError("agent_browser_runnable should not run during availability checks")

    monkeypatch.setattr(browser_tool, "agent_browser_runnable", fail_if_spawned)

    assert browser_tool.check_browser_requirements() is True


def test_validating_lookup_rechecks_candidate_cached_by_probe(monkeypatch):
    """A lightweight probe cache must not bypass validation on real browser use."""

    calls: list[str] = []
    candidate = "C:/repo/node_modules/.bin/agent-browser.CMD"

    monkeypatch.setattr(
        browser_tool.shutil,
        "which",
        lambda name, path=None: candidate if name == "agent-browser" else None,
    )

    def runnable(path):
        calls.append(path)
        return False

    monkeypatch.setattr(browser_tool, "agent_browser_runnable", runnable)

    assert browser_tool._find_agent_browser(validate=False) == candidate
    assert calls == []

    with pytest.raises(FileNotFoundError):
        browser_tool._find_agent_browser(validate=True)

    assert calls
    assert all(path == candidate for path in calls)
