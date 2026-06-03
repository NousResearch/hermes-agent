"""Tests for the explicit wait knobs on browser_navigate / browser_snapshot.

Pattern provenance: derived from Lightpanda source commit
b57798e4a34d385a34da3c37ffab925760255e03 (AGPL-3.0-only).  This is a
pattern-only adoption — no Lightpanda code is imported here; the tests
exercise Hermes' own mapping of the wait surface to the existing
``agent-browser wait`` subcommand.
"""

import json
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class TestNormalizeWaitKnobs:
    def test_all_none_returns_no_errors(self):
        from tools.browser_tool import _normalize_wait_knobs
        wu, ws, wsc, tm, errs = _normalize_wait_knobs(None, None, None, None)
        assert wu is None and ws is None and wsc is None and tm is None
        assert errs == []

    def test_wait_until_lowercased(self):
        from tools.browser_tool import _normalize_wait_knobs
        wu, _, _, _, errs = _normalize_wait_knobs("NetworkIdle", None, None, None)
        assert wu == "networkidle"
        assert errs == []

    def test_wait_until_invalid_collected_as_error(self):
        from tools.browser_tool import _normalize_wait_knobs
        wu, _, _, _, errs = _normalize_wait_knobs("bogus", None, None, None)
        assert wu is None
        assert any("wait_until" in e for e in errs)

    def test_wait_until_empty_treated_as_none(self):
        from tools.browser_tool import _normalize_wait_knobs
        wu, _, _, _, errs = _normalize_wait_knobs("   ", None, None, None)
        assert wu is None
        assert errs == []

    def test_blank_selector_and_script_become_none(self):
        from tools.browser_tool import _normalize_wait_knobs
        _, ws, wsc, _, errs = _normalize_wait_knobs(None, "  ", "", None)
        assert ws is None and wsc is None
        assert errs == []

    def test_timeout_ms_clamped_to_ceiling(self):
        from tools.browser_tool import _normalize_wait_knobs, _WAIT_TIMEOUT_MS_CEILING
        _, _, _, tm, errs = _normalize_wait_knobs(None, None, None, 10**9)
        assert tm == _WAIT_TIMEOUT_MS_CEILING
        assert errs == []

    def test_timeout_ms_clamped_to_floor(self):
        from tools.browser_tool import _normalize_wait_knobs, _WAIT_TIMEOUT_MS_FLOOR
        _, _, _, tm, errs = _normalize_wait_knobs(None, None, None, 1)
        assert tm == _WAIT_TIMEOUT_MS_FLOOR
        assert errs == []

    def test_timeout_ms_zero_rejected(self):
        from tools.browser_tool import _normalize_wait_knobs
        _, _, _, tm, errs = _normalize_wait_knobs(None, None, None, 0)
        assert tm is None
        assert any("timeout_ms" in e for e in errs)

    def test_timeout_ms_negative_rejected(self):
        from tools.browser_tool import _normalize_wait_knobs
        _, _, _, tm, errs = _normalize_wait_knobs(None, None, None, -5)
        assert tm is None
        assert any("timeout_ms" in e for e in errs)

    def test_timeout_ms_non_integer_rejected(self):
        from tools.browser_tool import _normalize_wait_knobs
        _, _, _, tm, errs = _normalize_wait_knobs(None, None, None, "abc")
        assert tm is None
        assert any("timeout_ms" in e for e in errs)


class TestWaitTimeoutSeconds:
    def test_none_passthrough(self):
        from tools.browser_tool import _wait_timeout_seconds
        assert _wait_timeout_seconds(None) is None

    def test_sub_second_rounds_up_to_one(self):
        from tools.browser_tool import _wait_timeout_seconds
        assert _wait_timeout_seconds(500) == 1

    def test_exact_second(self):
        from tools.browser_tool import _wait_timeout_seconds
        assert _wait_timeout_seconds(1000) == 1

    def test_ceil_division(self):
        from tools.browser_tool import _wait_timeout_seconds
        # 1500ms must not silently truncate to 1s.
        assert _wait_timeout_seconds(1500) == 2


# ---------------------------------------------------------------------------
# _apply_wait_knobs — CLI argument mapping
# ---------------------------------------------------------------------------


class TestApplyWaitKnobs:
    def test_no_request_returns_ok_empty(self):
        from tools.browser_tool import _apply_wait_knobs
        out = _apply_wait_knobs("t")
        assert out == {"ok": True, "applied": []}

    def test_wait_until_maps_to_load_flag(self):
        from tools.browser_tool import _apply_wait_knobs
        calls = []

        def fake(task_id, command, args, timeout=None):
            calls.append((task_id, command, list(args), timeout))
            return {"success": True, "data": {}}

        with patch("tools.browser_tool._run_browser_command", side_effect=fake):
            out = _apply_wait_knobs("task1", wait_until="networkidle", timeout_seconds=3)

        assert out["ok"] is True
        assert calls == [("task1", "wait", ["--load", "networkidle"], 3)]
        assert out["applied"] == [{"kind": "wait_until", "args": ["--load", "networkidle"]}]

    def test_wait_selector_maps_to_bare_selector(self):
        from tools.browser_tool import _apply_wait_knobs
        calls = []

        def fake(task_id, command, args, timeout=None):
            calls.append((command, list(args)))
            return {"success": True, "data": {}}

        with patch("tools.browser_tool._run_browser_command", side_effect=fake):
            out = _apply_wait_knobs("t", wait_selector="#results")
        assert out["ok"] is True
        assert calls == [("wait", ["#results"])]

    def test_wait_script_maps_to_fn_flag(self):
        from tools.browser_tool import _apply_wait_knobs
        calls = []

        def fake(task_id, command, args, timeout=None):
            calls.append((command, list(args)))
            return {"success": True, "data": {}}

        with patch("tools.browser_tool._run_browser_command", side_effect=fake):
            out = _apply_wait_knobs("t", wait_script="window.ready === true")
        assert out["ok"] is True
        assert calls == [("wait", ["--fn", "window.ready === true"])]

    def test_multiple_knobs_run_in_lifecycle_order(self):
        from tools.browser_tool import _apply_wait_knobs
        order = []

        def fake(task_id, command, args, timeout=None):
            order.append(tuple(args[:2]))
            return {"success": True, "data": {}}

        with patch("tools.browser_tool._run_browser_command", side_effect=fake):
            out = _apply_wait_knobs(
                "t",
                wait_until="domcontentloaded",
                wait_selector="#main",
                wait_script="document.title.length > 0",
            )
        assert out["ok"] is True
        assert order == [
            ("--load", "domcontentloaded"),
            ("#main", ),  # wait_selector — single arg, tuple of length 1
            ("--fn", "document.title.length > 0"),
        ]

    def test_first_failure_stops_subsequent_waits(self):
        from tools.browser_tool import _apply_wait_knobs
        invocations = []

        def fake(task_id, command, args, timeout=None):
            invocations.append(list(args))
            if args[:1] == ["--load"]:
                return {"success": False, "error": "load timed out"}
            return {"success": True, "data": {}}

        with patch("tools.browser_tool._run_browser_command", side_effect=fake):
            out = _apply_wait_knobs(
                "t",
                wait_until="load",
                wait_selector="#never-reached",
            )
        assert out["ok"] is False
        assert out["failed"] == "wait_until"
        assert "load timed out" in out["error"]
        # The selector wait must NOT have been invoked.
        assert invocations == [["--load", "load"]]


# ---------------------------------------------------------------------------
# Schema surface
# ---------------------------------------------------------------------------


class TestSchema:
    def _schema(self, name):
        from tools.browser_tool import BROWSER_TOOL_SCHEMAS
        return next(s for s in BROWSER_TOOL_SCHEMAS if s["name"] == name)

    def test_navigate_schema_exposes_wait_until_enum(self):
        s = self._schema("browser_navigate")
        props = s["parameters"]["properties"]
        assert "wait_until" in props
        assert set(props["wait_until"]["enum"]) == {"load", "domcontentloaded", "networkidle"}

    def test_navigate_schema_exposes_all_wait_knobs(self):
        s = self._schema("browser_navigate")
        props = s["parameters"]["properties"]
        for key in ("wait_until", "wait_selector", "wait_script", "timeout_ms"):
            assert key in props, f"missing {key} on browser_navigate"
        # url stays the only required parameter.
        assert s["parameters"]["required"] == ["url"]

    def test_snapshot_schema_exposes_all_wait_knobs(self):
        s = self._schema("browser_snapshot")
        props = s["parameters"]["properties"]
        for key in ("wait_until", "wait_selector", "wait_script", "timeout_ms"):
            assert key in props, f"missing {key} on browser_snapshot"
        # Snapshot has no required parameters.
        assert s["parameters"]["required"] == []

    def test_timeout_ms_is_integer(self):
        s = self._schema("browser_navigate")
        assert s["parameters"]["properties"]["timeout_ms"]["type"] == "integer"


# ---------------------------------------------------------------------------
# browser_navigate integration
# ---------------------------------------------------------------------------


class TestBrowserNavigateWaitKnobs:
    def _common_patches(self):
        # Bypass the (heavyweight) safety checks so we can exercise the wait
        # dispatch deterministically.
        return [
            patch("tools.browser_tool._is_camofox_mode", return_value=False),
            patch("tools.browser_tool._is_local_backend", return_value=True),
            patch("tools.browser_tool._get_cloud_provider", return_value=None),
            patch("tools.browser_tool.check_website_access", return_value=None),
            patch("tools.browser_tool._is_safe_url", return_value=True),
            patch("tools.browser_tool._is_always_blocked_url", return_value=False),
            patch("tools.browser_tool._get_session_info", return_value={
                "session_name": "sess",
                "_first_nav": False,
                "features": {"local": True, "proxies": True},
            }),
        ]

    def test_navigate_with_wait_until_invokes_wait_load(self):
        import tools.browser_tool as bt

        calls = []

        def fake_run(task_id, command, args=None, timeout=None, **kw):
            calls.append((command, list(args or [])))
            if command == "open":
                return {"success": True, "data": {"title": "T", "url": "https://example.com/"}}
            if command == "wait":
                return {"success": True, "data": {}}
            if command == "snapshot":
                return {"success": True, "data": {"snapshot": "- heading [ref=e1]", "refs": {"e1": {}}}}
            return {"success": True, "data": {}}

        ctxs = self._common_patches()
        for c in ctxs:
            c.start()
        try:
            with patch("tools.browser_tool._run_browser_command", side_effect=fake_run):
                resp = json.loads(bt.browser_navigate(
                    "https://example.com",
                    task_id="wk-1",
                    wait_until="networkidle",
                ))
        finally:
            for c in ctxs:
                c.stop()
            bt._last_active_session_key.pop("wk-1", None)

        assert resp["success"] is True
        # open → wait → snapshot
        commands = [c[0] for c in calls]
        assert commands.index("open") < commands.index("wait") < commands.index("snapshot")
        wait_args = next(c[1] for c in calls if c[0] == "wait")
        assert wait_args == ["--load", "networkidle"]
        assert resp.get("wait_applied") == [{"kind": "wait_until", "args": ["--load", "networkidle"]}]

    def test_navigate_with_timeout_ms_propagates_to_open_and_snapshot(self):
        import tools.browser_tool as bt
        captured = {}

        def fake_run(task_id, command, args=None, timeout=None, **kw):
            captured.setdefault(command, []).append(timeout)
            if command == "open":
                return {"success": True, "data": {"title": "T", "url": "https://example.com/"}}
            if command == "snapshot":
                return {"success": True, "data": {"snapshot": "- heading [ref=e1]", "refs": {"e1": {}}}}
            return {"success": True, "data": {}}

        ctxs = self._common_patches()
        for c in ctxs:
            c.start()
        try:
            with patch("tools.browser_tool._run_browser_command", side_effect=fake_run):
                resp = json.loads(bt.browser_navigate(
                    "https://example.com",
                    task_id="wk-tm",
                    timeout_ms=4500,
                ))
        finally:
            for c in ctxs:
                c.stop()
            bt._last_active_session_key.pop("wk-tm", None)

        assert resp["success"] is True
        # 4500ms → ceil to 5 seconds.
        assert captured["open"][0] == 5
        # Regression: auto-snapshot after navigate must also honor the explicit
        # timeout_ms contract advertised in the schema/docstring.
        assert captured["snapshot"][0] == 5

    def test_navigate_invalid_wait_until_surfaces_validation_error(self):
        import tools.browser_tool as bt

        def fake_run(task_id, command, args=None, timeout=None, **kw):
            if command == "open":
                return {"success": True, "data": {"title": "T", "url": "https://example.com/"}}
            if command == "snapshot":
                return {"success": True, "data": {"snapshot": "- ok [ref=e1]", "refs": {"e1": {}}}}
            return {"success": True, "data": {}}

        ctxs = self._common_patches()
        for c in ctxs:
            c.start()
        try:
            with patch("tools.browser_tool._run_browser_command", side_effect=fake_run):
                resp = json.loads(bt.browser_navigate(
                    "https://example.com",
                    task_id="wk-bad",
                    wait_until="forever",
                ))
        finally:
            for c in ctxs:
                c.stop()
            bt._last_active_session_key.pop("wk-bad", None)

        assert resp["success"] is True
        assert "wait_validation_errors" in resp
        assert any("wait_until" in e for e in resp["wait_validation_errors"])
        # Navigation must not have queued a wait call when the only requested
        # wait was the invalid one.
        assert "wait_applied" not in resp

    def test_navigate_wait_failure_surfaces_warning_but_navigation_succeeds(self):
        import tools.browser_tool as bt

        def fake_run(task_id, command, args=None, timeout=None, **kw):
            if command == "open":
                return {"success": True, "data": {"title": "T", "url": "https://example.com/"}}
            if command == "wait":
                return {"success": False, "error": "selector #missing not found"}
            if command == "snapshot":
                return {"success": True, "data": {"snapshot": "- heading [ref=e1]", "refs": {"e1": {}}}}
            return {"success": True, "data": {}}

        ctxs = self._common_patches()
        for c in ctxs:
            c.start()
        try:
            with patch("tools.browser_tool._run_browser_command", side_effect=fake_run):
                resp = json.loads(bt.browser_navigate(
                    "https://example.com",
                    task_id="wk-fail",
                    wait_selector="#missing",
                ))
        finally:
            for c in ctxs:
                c.stop()
            bt._last_active_session_key.pop("wk-fail", None)

        assert resp["success"] is True
        assert "wait_warning" in resp
        assert "selector #missing not found" in resp["wait_warning"]


# ---------------------------------------------------------------------------
# browser_snapshot integration
# ---------------------------------------------------------------------------


class TestBrowserSnapshotWaitKnobs:
    def test_snapshot_runs_wait_before_snapshot(self):
        import tools.browser_tool as bt
        order = []

        def fake_run(task_id, command, args=None, timeout=None, **kw):
            order.append(command)
            if command == "wait":
                return {"success": True, "data": {}}
            if command == "snapshot":
                return {"success": True, "data": {"snapshot": "- ok [ref=e1]", "refs": {"e1": {}}}}
            return {"success": True, "data": {}}

        with patch("tools.browser_tool._is_camofox_mode", return_value=False), \
             patch("tools.browser_tool._run_browser_command", side_effect=fake_run):
            resp = json.loads(bt.browser_snapshot(
                task_id="snap-wk",
                wait_selector="#ready",
            ))

        assert resp["success"] is True
        assert order == ["wait", "snapshot"]
        assert resp.get("wait_applied") == [{"kind": "wait_selector", "args": ["#ready"]}]

    def test_snapshot_passes_timeout_to_snapshot_command(self):
        import tools.browser_tool as bt
        captured = {}

        def fake_run(task_id, command, args=None, timeout=None, **kw):
            captured[command] = timeout
            if command == "snapshot":
                return {"success": True, "data": {"snapshot": "- ok [ref=e1]", "refs": {"e1": {}}}}
            return {"success": True, "data": {}}

        with patch("tools.browser_tool._is_camofox_mode", return_value=False), \
             patch("tools.browser_tool._run_browser_command", side_effect=fake_run):
            resp = json.loads(bt.browser_snapshot(task_id="snap-tm", timeout_ms=2000))

        assert resp["success"] is True
        assert captured["snapshot"] == 2

    def test_snapshot_wait_failure_attached_to_success_response(self):
        import tools.browser_tool as bt

        def fake_run(task_id, command, args=None, timeout=None, **kw):
            if command == "wait":
                return {"success": False, "error": "timed out waiting for #ready"}
            if command == "snapshot":
                return {"success": True, "data": {"snapshot": "- ok [ref=e1]", "refs": {"e1": {}}}}
            return {"success": True, "data": {}}

        with patch("tools.browser_tool._is_camofox_mode", return_value=False), \
             patch("tools.browser_tool._run_browser_command", side_effect=fake_run):
            resp = json.loads(bt.browser_snapshot(task_id="snap-fail", wait_selector="#ready"))

        # Snapshot itself succeeded, but the wait warning is preserved so the
        # agent knows the snapshot may pre-date the expected page state.
        assert resp["success"] is True
        assert "wait_warning" in resp
        assert "timed out waiting for #ready" in resp["wait_warning"]

    def test_snapshot_validation_errors_propagate_on_failure(self):
        import tools.browser_tool as bt

        def fake_run(task_id, command, args=None, timeout=None, **kw):
            if command == "snapshot":
                return {"success": False, "error": "no session"}
            return {"success": True, "data": {}}

        with patch("tools.browser_tool._is_camofox_mode", return_value=False), \
             patch("tools.browser_tool._run_browser_command", side_effect=fake_run):
            resp = json.loads(bt.browser_snapshot(task_id="snap-vbad", wait_until="bogus"))

        assert resp["success"] is False
        assert resp["error"] == "no session"
        assert "wait_validation_errors" in resp


# ---------------------------------------------------------------------------
# Camofox no-op guard
# ---------------------------------------------------------------------------


class TestCamofoxNoOpGuard:
    def test_navigate_camofox_with_wait_attaches_unsupported_warning(self):
        import tools.browser_tool as bt

        # Bypass URL safety so we reach the Camofox branch.
        with patch("tools.browser_tool._is_camofox_mode", return_value=True), \
             patch("tools.browser_tool._is_local_backend", return_value=True), \
             patch("tools.browser_tool.check_website_access", return_value=None), \
             patch("tools.browser_tool._is_safe_url", return_value=True), \
             patch("tools.browser_tool._is_always_blocked_url", return_value=False), \
             patch("tools.browser_camofox.camofox_navigate",
                   return_value=json.dumps({"success": True, "url": "https://example.com/"})):
            resp = json.loads(bt.browser_navigate(
                "https://example.com",
                task_id="camo",
                wait_until="networkidle",
                timeout_ms=1000,
            ))

        assert resp["success"] is True
        assert "wait_unsupported" in resp
        assert "Camofox" in resp["wait_unsupported"]

    def test_snapshot_camofox_with_wait_attaches_unsupported_warning(self):
        import tools.browser_tool as bt

        with patch("tools.browser_tool._is_camofox_mode", return_value=True), \
             patch("tools.browser_camofox.camofox_snapshot",
                   return_value=json.dumps({"success": True, "snapshot": "...", "element_count": 0})):
            resp = json.loads(bt.browser_snapshot(
                task_id="camo-snap",
                wait_selector="#ready",
            ))

        assert resp["success"] is True
        assert "wait_unsupported" in resp


# ---------------------------------------------------------------------------
# Registry handler forwards new args
# ---------------------------------------------------------------------------


class TestRegistryHandler:
    def test_navigate_handler_forwards_wait_args(self):
        import tools.browser_tool as bt
        from tools.registry import registry

        captured = {}

        def fake_navigate(url, task_id=None, **kwargs):
            captured.update({"url": url, "task_id": task_id, **kwargs})
            return json.dumps({"success": True})

        spec = registry.get_entry("browser_navigate")
        assert spec is not None
        # Patch the function the lambda closes over.
        with patch.object(bt, "browser_navigate", side_effect=fake_navigate):
            spec.handler(
                {
                    "url": "https://example.com",
                    "wait_until": "networkidle",
                    "wait_selector": "#main",
                    "wait_script": "window.ready",
                    "timeout_ms": 3000,
                },
                task_id="reg-1",
            )

        assert captured["url"] == "https://example.com"
        assert captured["task_id"] == "reg-1"
        assert captured["wait_until"] == "networkidle"
        assert captured["wait_selector"] == "#main"
        assert captured["wait_script"] == "window.ready"
        assert captured["timeout_ms"] == 3000

    def test_snapshot_handler_forwards_wait_args(self):
        import tools.browser_tool as bt
        from tools.registry import registry

        captured = {}

        def fake_snapshot(full=False, task_id=None, user_task=None, **kwargs):
            captured.update({"full": full, "task_id": task_id, "user_task": user_task, **kwargs})
            return json.dumps({"success": True})

        spec = registry.get_entry("browser_snapshot")
        assert spec is not None
        with patch.object(bt, "browser_snapshot", side_effect=fake_snapshot):
            spec.handler(
                {
                    "full": True,
                    "wait_until": "load",
                    "timeout_ms": 1500,
                },
                task_id="reg-2",
                user_task="check page",
            )

        assert captured["full"] is True
        assert captured["task_id"] == "reg-2"
        assert captured["user_task"] == "check page"
        assert captured["wait_until"] == "load"
        assert captured["timeout_ms"] == 1500
