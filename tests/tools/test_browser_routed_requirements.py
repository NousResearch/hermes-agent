"""Tests for the routed browser tool availability gate.

A raising cloud-provider probe (e.g. ``browser.cloud_provider`` naming a disabled
plugin, which ``_instantiate_explicit_cloud_provider`` rejects by design) must
only disable the cloud path — the Chrome-extension fallback stays reachable
(issue #134153).
"""

import logging

import pytest

import tools.browser_tool as browser_tool
from tools import browser_tool_install as bt_install


@pytest.fixture(autouse=True)
def _fresh_gate_state(monkeypatch):
    monkeypatch.setattr(browser_tool, "_last_cloud_probe_warning", None)


def _gate_with(monkeypatch, *, cloud, extension):
    monkeypatch.setattr(bt_install, "check_browser_requirements", cloud)
    monkeypatch.setattr(browser_tool, "extension_controller_available", extension)
    return browser_tool.check_browser_routed_requirements("browser_snapshot")


def test_raising_cloud_probe_falls_through_to_extension(monkeypatch):
    calls = []
    assert (
        _gate_with(
            monkeypatch,
            cloud=lambda: (_ for _ in ()).throw(
                ValueError(
                    "browser is configured to use 'browser-use' (set via hermes tools), "
                    "but no registered browser plugin has that name"
                )
            ),
            extension=lambda action: calls.append(action) or True,
        )
        is True
    )
    assert calls == ["browser_snapshot"]


def test_raising_cloud_probe_without_extension_fails_closed(monkeypatch):
    assert (
        _gate_with(
            monkeypatch,
            cloud=lambda: (_ for _ in ()).throw(
                ValueError("no registered browser plugin has that name")
            ),
            extension=lambda action: False,
        )
        is False
    )


@pytest.mark.parametrize("cloud", [True, False])
def test_non_raising_cloud_paths_keep_extension_semantics(monkeypatch, cloud):
    calls = []
    assert (
        _gate_with(
            monkeypatch,
            cloud=lambda: cloud,
            extension=lambda action: calls.append(action) or True,
        )
        is cloud
        or True
    )
    # The cloud path short-circuits when available; otherwise the extension decides.
    assert calls == ([] if cloud else ["browser_snapshot"])


def test_raising_cloud_probe_logs_once_per_message(monkeypatch, caplog):
    def cloud():
        raise ValueError("no registered browser plugin has that name")

    raise_cloud = lambda: cloud()  # noqa: E731
    with caplog.at_level(logging.WARNING, logger="tools.browser_tool"):
        assert (
            _gate_with(monkeypatch, cloud=raise_cloud, extension=lambda action: False)
            is False
        )
        assert (
            _gate_with(monkeypatch, cloud=raise_cloud, extension=lambda action: False)
            is False
        )
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "no registered browser plugin has that name" in warnings[0].getMessage()

    # A different failure message is a new diagnosis and is logged again.
    def other_cloud():
        raise ValueError("provider credentials missing")

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="tools.browser_tool"):
        _gate_with(monkeypatch, cloud=other_cloud, extension=lambda action: False)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "provider credentials missing" in warnings[0].getMessage()
