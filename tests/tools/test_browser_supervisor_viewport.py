"""Unit tests for the CDP supervisor viewport override (browser.viewport).

Covers:
  * ``CDPSupervisor._apply_viewport_override`` — builds the right
    ``Emulation.setDeviceMetricsOverride`` payload from config, falls back to
    built-in defaults for missing/invalid values, and never raises.
  * ``CDPSupervisor._attach_initial_page`` — applies the override on the page
    session during attach.
  * ``browser_tool._get_viewport_config`` — config reading/normalization.
  * ``browser_tool._ensure_cdp_supervisor`` — forwards the viewport to the
    supervisor registry.

These are pure unit tests (mocked ``_cdp``, no real Chrome).
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest

from tools.browser_supervisor import (
    CDPSupervisor,
    DEFAULT_VIEWPORT_DEVICE_SCALE_FACTOR,
    DEFAULT_VIEWPORT_HEIGHT,
    DEFAULT_VIEWPORT_MOBILE,
    DEFAULT_VIEWPORT_WIDTH,
)


def _make_supervisor(viewport):
    """Build a CDPSupervisor without a real WS connection.

    ``_cdp`` is replaced by a recording fake; ``_page_session_id`` is set to a
    fixed value so ``_apply_viewport_override`` exercises the send path.
    """
    sup = object.__new__(CDPSupervisor)
    sup._state_lock = threading.Lock()
    sup._active = True
    sup._page_session_id = "sess-1"
    sup.viewport = dict(viewport or {})

    loop = asyncio.new_event_loop()

    def _runner():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()

    calls = []

    async def _fake_cdp(method, params=None, *, session_id=None, timeout=10.0):
        calls.append((method, params, session_id))
        return {"result": {}}

    sup._cdp = _fake_cdp  # type: ignore[method-assign]
    sup._loop = loop
    sup._thread = thread
    return sup, calls


def _stop_supervisor(sup):
    sup._loop.call_soon_threadsafe(sup._loop.stop)
    sup._thread.join(timeout=2)


def _run_async(coro):
    return asyncio.run(coro)


class TestApplyViewportOverride:
    def test_sends_custom_viewport(self):
        sup, calls = _make_supervisor(
            {"width": 1920, "height": 1080, "device_scale_factor": 1, "mobile": False}
        )
        try:
            _run_async(sup._apply_viewport_override())
        finally:
            _stop_supervisor(sup)

        assert calls == [
            (
                "Emulation.setDeviceMetricsOverride",
                {"width": 1920, "height": 1080, "deviceScaleFactor": 1, "mobile": False},
                "sess-1",
            )
        ]

    def test_mobile_viewport_passes_mobile_true(self):
        sup, calls = _make_supervisor(
            {"width": 390, "height": 844, "device_scale_factor": 3, "mobile": True}
        )
        try:
            _run_async(sup._apply_viewport_override())
        finally:
            _stop_supervisor(sup)

        assert calls == [
            (
                "Emulation.setDeviceMetricsOverride",
                {"width": 390, "height": 844, "deviceScaleFactor": 3, "mobile": True},
                "sess-1",
            )
        ]

    def test_empty_viewport_uses_defaults(self):
        sup, calls = _make_supervisor({})
        try:
            _run_async(sup._apply_viewport_override())
        finally:
            _stop_supervisor(sup)

        assert calls == [
            (
                "Emulation.setDeviceMetricsOverride",
                {
                    "width": DEFAULT_VIEWPORT_WIDTH,
                    "height": DEFAULT_VIEWPORT_HEIGHT,
                    "deviceScaleFactor": DEFAULT_VIEWPORT_DEVICE_SCALE_FACTOR,
                    "mobile": DEFAULT_VIEWPORT_MOBILE,
                },
                "sess-1",
            )
        ]

    def test_partial_viewport_fills_missing_keys_from_defaults(self):
        sup, calls = _make_supervisor({"width": 1440})
        try:
            _run_async(sup._apply_viewport_override())
        finally:
            _stop_supervisor(sup)

        assert calls == [
            (
                "Emulation.setDeviceMetricsOverride",
                {
                    "width": 1440,
                    "height": DEFAULT_VIEWPORT_HEIGHT,
                    "deviceScaleFactor": DEFAULT_VIEWPORT_DEVICE_SCALE_FACTOR,
                    "mobile": DEFAULT_VIEWPORT_MOBILE,
                },
                "sess-1",
            )
        ]

    def test_invalid_values_fall_back_to_defaults(self):
        sup, calls = _make_supervisor(
            {"width": "wide", "height": -5, "device_scale_factor": 0, "mobile": "yes"}
        )
        try:
            _run_async(sup._apply_viewport_override())
        finally:
            _stop_supervisor(sup)

        assert calls == [
            (
                "Emulation.setDeviceMetricsOverride",
                {
                    "width": DEFAULT_VIEWPORT_WIDTH,
                    "height": DEFAULT_VIEWPORT_HEIGHT,
                    "deviceScaleFactor": DEFAULT_VIEWPORT_DEVICE_SCALE_FACTOR,
                    "mobile": DEFAULT_VIEWPORT_MOBILE,
                },
                "sess-1",
            )
        ]

    def test_cdp_failure_is_swallowed(self):
        sup, _ = _make_supervisor({"width": 1280, "height": 720})

        async def _boom(method, params=None, *, session_id=None, timeout=10.0):
            raise RuntimeError("websocket closed")

        sup._cdp = _boom  # type: ignore[method-assign]
        try:
            # Must not raise — the supervisor is a non-fatal enhancement.
            _run_async(sup._apply_viewport_override())
        finally:
            _stop_supervisor(sup)

    def test_no_page_session_is_noop(self):
        sup, calls = _make_supervisor({"width": 1280, "height": 720})
        sup._page_session_id = None
        try:
            _run_async(sup._apply_viewport_override())
        finally:
            _stop_supervisor(sup)

        assert calls == []


class TestAttachInitialPageAppliesViewport:
    def test_emulation_override_sent_on_page_session(self):
        sup, calls = _make_supervisor({"width": 1920, "height": 1080})
        # _attach_initial_page finds its own page target and attach result.
        sup._page_session_id = None

        async def _fake_cdp(method, params=None, *, session_id=None, timeout=10.0):
            calls.append((method, params, session_id))
            if method == "Target.getTargets":
                return {
                    "result": {
                        "targetInfos": [{"type": "page", "targetId": "tgt-1"}]
                    }
                }
            if method == "Target.attachToTarget":
                return {"result": {"sessionId": "sess-1"}}
            return {"result": {}}

        sup._cdp = _fake_cdp  # type: ignore[method-assign]
        try:
            _run_async(sup._attach_initial_page())
        finally:
            _stop_supervisor(sup)

        emulation_calls = [
            c for c in calls if c[0] == "Emulation.setDeviceMetricsOverride"
        ]
        assert len(emulation_calls) == 1
        method, params, session_id = emulation_calls[0]
        assert session_id == "sess-1"
        assert params["width"] == 1920
        assert params["height"] == 1080
        assert params["deviceScaleFactor"] == 1
        assert params["mobile"] is False


class TestGetViewportConfig:
    def _import_module(self):
        import tools.browser_tool as browser_tool

        return browser_tool

    def test_returns_configured_viewport(self):
        browser_tool = self._import_module()
        with patch(
            "hermes_cli.config.read_raw_config",
            return_value={
                "browser": {
                    "viewport": {
                        "width": 1920,
                        "height": 1080,
                        "device_scale_factor": 1,
                        "mobile": False,
                    }
                }
            },
        ):
            assert browser_tool._get_viewport_config() == {
                "width": 1920,
                "height": 1080,
                "device_scale_factor": 1,
                "mobile": False,
            }

    def test_missing_viewport_returns_empty(self):
        browser_tool = self._import_module()
        with patch(
            "hermes_cli.config.read_raw_config",
            return_value={"browser": {"dialog_policy": "must_respond"}},
        ):
            assert browser_tool._get_viewport_config() == {}

    def test_no_browser_section_returns_empty(self):
        browser_tool = self._import_module()
        with patch("hermes_cli.config.read_raw_config", return_value={}):
            assert browser_tool._get_viewport_config() == {}

    def test_non_dict_viewport_returns_empty(self):
        browser_tool = self._import_module()
        with patch(
            "hermes_cli.config.read_raw_config",
            return_value={"browser": {"viewport": "1920x1080"}},
        ):
            assert browser_tool._get_viewport_config() == {}

    def test_config_read_failure_returns_empty(self):
        browser_tool = self._import_module()
        with patch(
            "hermes_cli.config.read_raw_config",
            side_effect=RuntimeError("boom"),
        ):
            assert browser_tool._get_viewport_config() == {}


class TestEnsureSupervisorForwardsViewport:
    def test_viewport_passed_to_registry(self):
        import tools.browser_tool as browser_tool
        import tools.browser_supervisor as browser_supervisor

        fake_registry = MagicMock()
        fake_registry.get_or_start.return_value = object()
        viewport = {"width": 1920, "height": 1080}

        with patch.object(
            browser_tool, "_get_cdp_override", return_value="ws://127.0.0.1:9222/devtools"
        ), patch.object(
            browser_tool, "_get_viewport_config", return_value=viewport
        ), patch.object(
            browser_supervisor, "SUPERVISOR_REGISTRY", fake_registry
        ):
            browser_tool._ensure_cdp_supervisor(task_id="test-viewport-task")

        fake_registry.get_or_start.assert_called_once()
        kwargs = fake_registry.get_or_start.call_args.kwargs
        assert kwargs["viewport"] == viewport
        assert kwargs["task_id"] == "test-viewport-task"
        assert kwargs["cdp_url"] == "ws://127.0.0.1:9222/devtools"

    def test_no_cdp_url_skips_registry(self):
        import tools.browser_tool as browser_tool
        import tools.browser_supervisor as browser_supervisor

        fake_registry = MagicMock()
        with patch.object(
            browser_tool, "_get_cdp_override", return_value=""
        ), patch.object(
            browser_supervisor, "SUPERVISOR_REGISTRY", fake_registry
        ):
            browser_tool._ensure_cdp_supervisor(task_id="test-viewport-task")

        fake_registry.get_or_start.assert_not_called()
