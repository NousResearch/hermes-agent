"""browser_vision must foreground the attached tab before capturing (#103280).

Chrome produces no compositor frame for a hidden tab (document.visibilityState === 'hidden'),
so Page.captureScreenshot never returns inside browser.command_timeout while the frame-free
commands (navigate/snapshot/console) keep working on the same tab. The capture path now
switches agent-browser to the active tab first (tab list -> tab <id>, which sends
Page.bringToFront); the pre-step is best-effort so a screenshot never dies on it.
"""

import json
from unittest.mock import patch

import tools.browser_tool as bt


def _fake_runner(seen, *, tab_list=None, tab_switch=None, screenshot=None, raise_on_tab_list=False):
    """Record (command, args) pairs; the screenshot answers from a real file."""
    def _run(task_id, command, args=None, timeout=None, _engine_override=None):
        seen.append((command, tuple(args or [])))
        if command == "tab" and (args or ["list"])[0] == "list":
            if raise_on_tab_list:
                raise RuntimeError("daemon gone")
            return tab_list if tab_list is not None else {"success": True, "data": {"tabs": []}}
        if command == "tab":
            return tab_switch if tab_switch is not None else {"success": True, "data": {}}
        return screenshot
    return _run


class TestVisionForegroundsTabBeforeScreenshot:
    """The foreground pre-step must run on the plain local/CDP capture path."""

    def test_active_tab_brought_to_front_before_screenshot(self, tmp_path):
        """tab list -> tab <active id> runs immediately before the screenshot command."""
        shot = tmp_path / "shot.png"
        shot.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
        seen: list = []
        runner = _fake_runner(
            seen,
            tab_list={"success": True, "data": {"tabs": [
                {"active": False, "tabId": "t1", "type": "page", "url": "https://other.example/"},
                {"active": True, "tabId": "t2", "type": "page", "url": "https://example.com/"},
            ]}},
            screenshot={"success": True, "data": {"path": str(shot)}},
        )
        with (
            patch("tools.browser_tool_session._run_browser_command", side_effect=runner),
            patch("hermes_constants.get_hermes_dir", return_value=tmp_path),
            patch("tools.browser_tool_lifecycle._cleanup_old_screenshots"),
            patch("tools.browser_tool._get_vision_model", return_value="test-model"),
            patch("hermes_cli.config.load_config", return_value={"model": {}}),
            patch("agent.auxiliary_client.call_llm") as mock_llm,
        ):
            response = json.loads(bt.browser_vision("what is on the page?", task_id="test"))

        assert response["success"] is True
        # The pre-step (tab list, tab t2) precedes the screenshot command.
        commands = [c for c, _ in seen]
        assert commands[:3] == ["tab", "tab", "screenshot"]
        assert seen[1][1] == ("t2",)
        assert mock_llm.called

    def test_foreground_pre_step_failure_still_runs_screenshot(self, tmp_path):
        """A pre-step that raises must not stop the capture: the screenshot still runs."""
        shot = tmp_path / "shot.png"
        shot.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
        seen: list = []
        runner = _fake_runner(
            seen,
            screenshot={"success": True, "data": {"path": str(shot)}},
            raise_on_tab_list=True,
        )
        with (
            patch("tools.browser_tool_session._run_browser_command", side_effect=runner),
            patch("hermes_constants.get_hermes_dir", return_value=tmp_path),
            patch("tools.browser_tool_lifecycle._cleanup_old_screenshots"),
            patch("tools.browser_tool._get_vision_model", return_value="test-model"),
            patch("hermes_cli.config.load_config", return_value={"model": {}}),
            patch("agent.auxiliary_client.call_llm"),
        ):
            response = json.loads(bt.browser_vision("what is on the page?", task_id="test"))

        assert response["success"] is True
        assert [c for c, _ in seen] == ["tab", "screenshot"]

    def test_no_active_tab_in_list_still_screenshots(self, tmp_path):
        """A tab list without an active entry (empty/odd daemon answer) skips the switch."""
        shot = tmp_path / "shot.png"
        shot.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
        seen: list = []
        runner = _fake_runner(
            seen,
            tab_list={"success": True, "data": {"tabs": []}},
            screenshot={"success": True, "data": {"path": str(shot)}},
        )
        with (
            patch("tools.browser_tool_session._run_browser_command", side_effect=runner),
            patch("hermes_constants.get_hermes_dir", return_value=tmp_path),
            patch("tools.browser_tool_lifecycle._cleanup_old_screenshots"),
            patch("tools.browser_tool._get_vision_model", return_value="test-model"),
            patch("hermes_cli.config.load_config", return_value={"model": {}}),
            patch("agent.auxiliary_client.call_llm"),
        ):
            response = json.loads(bt.browser_vision("what is on the page?", task_id="test"))

        assert response["success"] is True
        assert [c for c, _ in seen] == ["tab", "screenshot"]


class TestLightpandaPrerouteSkipsForeground:
    """The Lightpanda pre-route captures via the Chrome fallback before the normal path,
    so the pre-step must not run (lp_prerouted=True answers the screenshot directly)."""

    def test_prerouted_capture_runs_no_tab_commands(self, tmp_path):
        """The pre-routed path (lp_prerouted=True) answers without any browser command."""
        shot = tmp_path / "prerouted.png"
        shot.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
        seen: list = []
        with patch("tools.browser_tool_session._run_browser_command",
                   side_effect=_fake_runner(seen, screenshot={"success": True, "data": {"path": str(shot)}})):
            from tools.browser_tool import _capture_vision_screenshot
            result, _path, error = _capture_vision_screenshot("test", False, shot, lp_prerouted=True)
        assert error is None
        assert result["success"] is True
        assert seen == []  # no tab list, no tab switch, no screenshot command
