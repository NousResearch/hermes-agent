"""Tests for the browser_wait tool."""

import json
import os
import sys
import time
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _snapshot_json(text: str, success: bool = True) -> str:
    payload = {"success": success, "snapshot": text}
    if not success:
        payload["error"] = text
    return json.dumps(payload)


class TestBrowserWaitSleepMode:
    """Without 'text', browser_wait is a bounded sleep."""

    def test_sleeps_for_the_requested_duration(self):
        from tools.browser_tool import browser_wait

        started = time.monotonic()
        result = json.loads(browser_wait(seconds=0.2))
        elapsed = time.monotonic() - started

        assert result["success"] is True
        assert elapsed >= 0.2
        assert "found" not in result

    def test_invalid_seconds_falls_back_to_default_budget(self):
        from tools.browser_tool import browser_wait

        # Text mode that matches immediately, so the fallback budget is
        # never actually slept through — this only exercises coercion.
        with patch(
            "tools.browser_tool.browser_snapshot",
            return_value=_snapshot_json("Results ready"),
        ):
            result = json.loads(browser_wait(seconds="not-a-number", text="ready"))

        assert result["success"] is True
        assert result["found"] is True


class TestBrowserWaitTextMode:
    """With 'text', browser_wait polls snapshots until the text appears."""

    def test_returns_immediately_when_text_already_present(self):
        from tools.browser_tool import browser_wait

        with patch(
            "tools.browser_tool.browser_snapshot",
            return_value=_snapshot_json("Flight results: $743 round trip"),
        ) as mock_snapshot:
            result = json.loads(browser_wait(seconds=10, text="flight results"))

        assert result["success"] is True
        assert result["found"] is True
        assert mock_snapshot.call_count == 1
        assert result["waited_seconds"] < 1

    def test_polls_until_text_appears(self):
        from tools.browser_tool import browser_wait

        with patch(
            "tools.browser_tool.browser_snapshot",
            side_effect=[
                _snapshot_json("Loading…"),
                _snapshot_json("Loading…"),
                _snapshot_json("Cheapest flights from HYD"),
            ],
        ) as mock_snapshot:
            result = json.loads(browser_wait(seconds=10, text="cheapest flights"))

        assert result["found"] is True
        assert mock_snapshot.call_count == 3

    def test_reports_not_found_on_timeout_with_hint(self):
        from tools.browser_tool import browser_wait

        with patch(
            "tools.browser_tool.browser_snapshot",
            return_value=_snapshot_json("Loading…"),
        ):
            result = json.loads(browser_wait(seconds=0.3, text="never appears"))

        assert result["found"] is False
        assert result["success"] is True
        assert "hint" in result

    def test_propagates_snapshot_error_when_text_never_found(self):
        from tools.browser_tool import browser_wait

        with patch(
            "tools.browser_tool.browser_snapshot",
            return_value=_snapshot_json("No browser session for task", success=False),
        ):
            result = json.loads(browser_wait(seconds=0.3, text="anything"))

        assert result["success"] is False
        assert result["found"] is False
        assert "No browser session" in result["error"]

    def test_envelope_keys_do_not_count_as_page_text(self):
        from tools.browser_tool import browser_wait

        # Every snapshot reply contains '"success": true' in its JSON
        # envelope. Waiting for a common word like "success" must match the
        # parsed page content only — never the envelope — or the wait returns
        # found=true on any page whatsoever.
        with patch(
            "tools.browser_tool.browser_snapshot",
            return_value=_snapshot_json("Loading…"),
        ):
            result = json.loads(browser_wait(seconds=0.3, text="success"))

        assert result["found"] is False

        with patch(
            "tools.browser_tool.browser_snapshot",
            return_value=_snapshot_json("Payment success — order confirmed"),
        ):
            result = json.loads(browser_wait(seconds=0.3, text="success"))

        assert result["found"] is True

    def test_failed_snapshot_containing_needle_is_not_a_match(self):
        from tools.browser_tool import browser_wait

        # The error payload mentions the needle, but a failed snapshot must
        # not count as the text being visible on the page.
        with patch(
            "tools.browser_tool.browser_snapshot",
            return_value=_snapshot_json("error while loading results", success=False),
        ):
            result = json.loads(browser_wait(seconds=0.3, text="results"))

        assert result["found"] is False


class TestBrowserWaitRegistration:
    def test_schema_registered(self):
        from tools.browser_tool import BROWSER_TOOL_SCHEMAS

        schema = next(s for s in BROWSER_TOOL_SCHEMAS if s["name"] == "browser_wait")
        assert "seconds" in schema["parameters"]["properties"]
        assert "text" in schema["parameters"]["properties"]
        assert schema["parameters"]["required"] == []

    def test_registry_handler_wired(self):
        import tools.browser_tool  # noqa: F401  # ensure registrations ran
        from tools.registry import registry

        entry = registry.get_entry("browser_wait")
        assert entry is not None
        assert entry.toolset == "browser"
