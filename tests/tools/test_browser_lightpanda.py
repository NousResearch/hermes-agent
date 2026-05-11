"""Tests for Lightpanda engine support in browser_tool.py."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_engine_cache():
    """Reset the module-level engine cache so tests start clean."""
    import tools.browser_tool as bt
    bt._cached_browser_engine = None
    bt._browser_engine_resolved = False


@pytest.fixture(autouse=True)
def _clean_engine_cache():
    """Reset engine cache before and after each test."""
    _reset_engine_cache()
    yield
    _reset_engine_cache()


# ---------------------------------------------------------------------------
# _get_browser_engine
# ---------------------------------------------------------------------------

class TestGetBrowserEngine:
    """Test engine resolution from config and env vars."""

    def test_default_is_auto(self):
        """With no config or env var, engine defaults to 'auto'."""
        from tools.browser_tool import _get_browser_engine
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AGENT_BROWSER_ENGINE", None)
            with patch("hermes_cli.config.read_raw_config", return_value={}):
                assert _get_browser_engine() == "auto"

    def test_config_lightpanda(self):
        """Config browser.engine = 'lightpanda' is respected."""
        from tools.browser_tool import _get_browser_engine
        cfg = {"browser": {"engine": "lightpanda"}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert _get_browser_engine() == "lightpanda"

    def test_config_chrome(self):
        """Config browser.engine = 'chrome' is respected."""
        from tools.browser_tool import _get_browser_engine
        cfg = {"browser": {"engine": "chrome"}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert _get_browser_engine() == "chrome"

    def test_env_var_fallback(self):
        """AGENT_BROWSER_ENGINE env var is used when config has no engine key."""
        from tools.browser_tool import _get_browser_engine
        with patch.dict(os.environ, {"AGENT_BROWSER_ENGINE": "lightpanda"}):
            with patch("hermes_cli.config.read_raw_config", return_value={}):
                assert _get_browser_engine() == "lightpanda"

    def test_config_takes_priority_over_env(self):
        """Config value wins over env var."""
        from tools.browser_tool import _get_browser_engine
        cfg = {"browser": {"engine": "chrome"}}
        with patch.dict(os.environ, {"AGENT_BROWSER_ENGINE": "lightpanda"}):
            with patch("hermes_cli.config.read_raw_config", return_value=cfg):
                assert _get_browser_engine() == "chrome"

    def test_value_is_lowercased(self):
        """Engine value is normalized to lowercase."""
        from tools.browser_tool import _get_browser_engine
        cfg = {"browser": {"engine": "Lightpanda"}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert _get_browser_engine() == "lightpanda"

    def test_invalid_engine_falls_back_to_auto(self):
        """Unknown engine values are rejected and fall back to 'auto'."""
        from tools.browser_tool import _get_browser_engine
        cfg = {"browser": {"engine": "firefox"}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert _get_browser_engine() == "auto"

    def test_caching(self):
        """Result is cached — second call doesn't re-read config."""
        from tools.browser_tool import _get_browser_engine
        mock_read = MagicMock(return_value={"browser": {"engine": "lightpanda"}})
        with patch("hermes_cli.config.read_raw_config", mock_read):
            assert _get_browser_engine() == "lightpanda"
            assert _get_browser_engine() == "lightpanda"
            mock_read.assert_called_once()


# ---------------------------------------------------------------------------
# _should_inject_engine
# ---------------------------------------------------------------------------

class TestShouldInjectEngine:
    """Test whether --engine flag is injected based on mode."""

    def test_auto_never_injects(self):
        from tools.browser_tool import _should_inject_engine
        assert _should_inject_engine("auto") is False

    def test_lightpanda_injects_in_local_mode(self):
        from tools.browser_tool import _should_inject_engine
        with patch("tools.browser_tool._is_camofox_mode", return_value=False), \
             patch("tools.browser_tool._get_cdp_override", return_value=""), \
             patch("tools.browser_tool._get_cloud_provider", return_value=None):
            assert _should_inject_engine("lightpanda") is True

    def test_chrome_injects_in_local_mode(self):
        from tools.browser_tool import _should_inject_engine
        with patch("tools.browser_tool._is_camofox_mode", return_value=False), \
             patch("tools.browser_tool._get_cdp_override", return_value=""), \
             patch("tools.browser_tool._get_cloud_provider", return_value=None):
            assert _should_inject_engine("chrome") is True

    def test_no_inject_in_camofox_mode(self):
        from tools.browser_tool import _should_inject_engine
        with patch("tools.browser_tool._is_camofox_mode", return_value=True):
            assert _should_inject_engine("lightpanda") is False

    def test_no_inject_with_cdp_override(self):
        from tools.browser_tool import _should_inject_engine
        with patch("tools.browser_tool._is_camofox_mode", return_value=False), \
             patch("tools.browser_tool._get_cdp_override", return_value="ws://localhost:9222"):
            assert _should_inject_engine("lightpanda") is False

    def test_no_inject_with_cloud_provider(self):
        from tools.browser_tool import _should_inject_engine
        mock_provider = MagicMock()
        with patch("tools.browser_tool._is_camofox_mode", return_value=False), \
             patch("tools.browser_tool._get_cdp_override", return_value=""), \
             patch("tools.browser_tool._get_cloud_provider", return_value=mock_provider):
            assert _should_inject_engine("lightpanda") is False


# ---------------------------------------------------------------------------
# _needs_lightpanda_fallback
# ---------------------------------------------------------------------------

class TestNeedsLightpandaFallback:
    """Test fallback detection for Lightpanda results."""

    def test_non_lightpanda_never_falls_back(self):
        from tools.browser_tool import _needs_lightpanda_fallback
        result = {"success": False, "error": "timeout"}
        assert _needs_lightpanda_fallback("chrome", "open", result) is False
        assert _needs_lightpanda_fallback("auto", "open", result) is False

    def test_failed_command_triggers_fallback(self):
        from tools.browser_tool import _needs_lightpanda_fallback
        result = {"success": False, "error": "page.goto: Timeout"}
        assert _needs_lightpanda_fallback("lightpanda", "open", result) is True

    def test_failed_command_reason_is_user_visible(self):
        from tools.browser_tool import _lightpanda_fallback_reason
        result = {"success": False, "error": "page.goto: Timeout"}
        reason = _lightpanda_fallback_reason("lightpanda", "open", result)
        assert reason is not None
        assert "page.goto: Timeout" in reason
        assert "retried with Chrome" in reason

    def test_empty_snapshot_triggers_fallback(self):
        from tools.browser_tool import _needs_lightpanda_fallback
        result = {"success": True, "data": {"snapshot": ""}}
        assert _needs_lightpanda_fallback("lightpanda", "snapshot", result) is True

    def test_short_snapshot_triggers_fallback(self):
        from tools.browser_tool import _needs_lightpanda_fallback
        result = {"success": True, "data": {"snapshot": "- none"}}
        assert _needs_lightpanda_fallback("lightpanda", "snapshot", result) is True

    def test_normal_snapshot_does_not_trigger(self):
        from tools.browser_tool import _needs_lightpanda_fallback
        result = {"success": True, "data": {
            "snapshot": '- heading "Example Domain" [ref=e1]\n- link "Learn more" [ref=e2]'
        }}
        assert _needs_lightpanda_fallback("lightpanda", "snapshot", result) is False

    def test_small_screenshot_triggers_fallback(self, tmp_path):
        from tools.browser_tool import _needs_lightpanda_fallback
        # Create a tiny file simulating the Lightpanda placeholder PNG
        placeholder = tmp_path / "placeholder.png"
        placeholder.write_bytes(b"\x89PNG" + b"\x00" * 2000)  # ~2KB
        result = {"success": True, "data": {"path": str(placeholder)}}
        assert _needs_lightpanda_fallback("lightpanda", "screenshot", result) is True

    def test_actual_placeholder_size_triggers_fallback(self, tmp_path):
        from tools.browser_tool import _needs_lightpanda_fallback
        # Lightpanda PR #1766 resized the placeholder to 1920x1080 (~17 KB)
        placeholder = tmp_path / "placeholder_1920.png"
        placeholder.write_bytes(b"\x89PNG" + b"\x00" * 16693)  # actual measured: 16697 bytes
        result = {"success": True, "data": {"path": str(placeholder)}}
        assert _needs_lightpanda_fallback("lightpanda", "screenshot", result) is True

    def test_normal_screenshot_does_not_trigger(self, tmp_path):
        from tools.browser_tool import _needs_lightpanda_fallback
        # Create a larger file simulating a real Chrome screenshot
        real_screenshot = tmp_path / "real.png"
        real_screenshot.write_bytes(b"\x89PNG" + b"\x00" * 50_000)  # ~50KB
        result = {"success": True, "data": {"path": str(real_screenshot)}}
        assert _needs_lightpanda_fallback("lightpanda", "screenshot", result) is False

    def test_successful_open_does_not_trigger(self):
        from tools.browser_tool import _needs_lightpanda_fallback
        result = {"success": True, "data": {"title": "Example", "url": "https://example.com"}}
        assert _needs_lightpanda_fallback("lightpanda", "open", result) is False

    def test_close_command_never_triggers_fallback(self):
        """Session-management commands like 'close' are not fallback-eligible."""
        from tools.browser_tool import _needs_lightpanda_fallback
        result = {"success": False, "error": "session closed"}
        assert _needs_lightpanda_fallback("lightpanda", "close", result) is False

    def test_record_command_never_triggers_fallback(self):
        """The 'record' command is tied to the engine daemon — not retryable."""
        from tools.browser_tool import _needs_lightpanda_fallback
        result = {"success": False, "error": "recording failed"}
        assert _needs_lightpanda_fallback("lightpanda", "record", result) is False

    def test_unknown_command_does_not_trigger_fallback(self):
        """Commands not in the whitelist should not trigger fallback."""
        from tools.browser_tool import _needs_lightpanda_fallback
        result = {"success": False, "error": "nope"}
        assert _needs_lightpanda_fallback("lightpanda", "some_future_cmd", result) is False


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    """Verify engine config is in DEFAULT_CONFIG."""

    def test_engine_in_default_config(self):
        from hermes_cli.config import DEFAULT_CONFIG
        assert "engine" in DEFAULT_CONFIG["browser"]
        assert DEFAULT_CONFIG["browser"]["engine"] == "auto"

    def test_env_var_registered(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS
        assert "AGENT_BROWSER_ENGINE" in OPTIONAL_ENV_VARS
        entry = OPTIONAL_ENV_VARS["AGENT_BROWSER_ENGINE"]
        assert entry["category"] == "tool"
        assert entry["advanced"] is True




class TestLightpandaRequirements:
    """Lightpanda should expose browser tools without local Chromium."""

    def test_lightpanda_local_mode_does_not_require_chromium(self):
        import tools.browser_tool as bt

        with patch("tools.browser_tool._is_camofox_mode", return_value=False), \
             patch("tools.browser_tool._get_cdp_override", return_value=""), \
             patch("tools.browser_tool._find_agent_browser", return_value="/usr/bin/agent-browser"), \
             patch("tools.browser_tool._requires_real_termux_browser_install", return_value=False), \
             patch("tools.browser_tool._get_cloud_provider", return_value=None), \
             patch("tools.browser_tool._get_browser_engine", return_value="lightpanda"), \
             patch("tools.browser_tool._chromium_installed", return_value=False):
            assert bt.check_browser_requirements() is True

    def test_chrome_local_mode_still_requires_chromium(self):
        import tools.browser_tool as bt

        with patch("tools.browser_tool._is_camofox_mode", return_value=False), \
             patch("tools.browser_tool._get_cdp_override", return_value=""), \
             patch("tools.browser_tool._find_agent_browser", return_value="/usr/bin/agent-browser"), \
             patch("tools.browser_tool._requires_real_termux_browser_install", return_value=False), \
             patch("tools.browser_tool._get_cloud_provider", return_value=None), \
             patch("tools.browser_tool._get_browser_engine", return_value="auto"), \
             patch("tools.browser_tool._chromium_installed", return_value=False):
            assert bt.check_browser_requirements() is False


# ---------------------------------------------------------------------------
# cleanup_all_browsers resets engine cache
# ---------------------------------------------------------------------------

class TestCleanupResetsEngineCache:
    """Verify cleanup_all_browsers resets engine-related globals."""

    def test_engine_cache_reset(self):
        import tools.browser_tool as bt
        # Seed the cache
        bt._cached_browser_engine = "lightpanda"
        bt._browser_engine_resolved = True
        # cleanup should reset them
        bt.cleanup_all_browsers()
        assert bt._cached_browser_engine is None
        assert bt._browser_engine_resolved is False




# ---------------------------------------------------------------------------
# fallback warning annotation
# ---------------------------------------------------------------------------

class TestLightpandaFallbackWarning:
    """Verify Chrome fallback results are annotated for users."""

    def test_fallback_result_gets_user_visible_warning(self):
        from tools.browser_tool import _annotate_lightpanda_fallback

        result = {"success": True, "data": {"snapshot": "- heading \"Hello\" [ref=e1]"}}
        annotated = _annotate_lightpanda_fallback(
            result,
            "Lightpanda returned an empty/too-short snapshot; retried with Chrome.",
        )

        assert annotated["browser_engine"] == "chrome"
        assert "Lightpanda fallback" in annotated["fallback_warning"]
        assert annotated["browser_engine_fallback"] == {
            "from": "lightpanda",
            "to": "chrome",
            "reason": "Lightpanda returned an empty/too-short snapshot; retried with Chrome.",
        }
        assert annotated["data"]["fallback_warning"] == annotated["fallback_warning"]
        assert annotated["data"]["browser_engine"] == "chrome"


    def test_browser_navigate_surfaces_fallback_warning(self):
        import json
        import tools.browser_tool as bt

        result = bt._annotate_lightpanda_fallback(
            {"success": True, "data": {"title": "Fallback OK", "url": "https://example.com/"}},
            "synthetic Lightpanda failure; retried with Chrome.",
        )

        with patch("tools.browser_tool._is_local_backend", return_value=True), \
             patch("tools.browser_tool._get_cloud_provider", return_value=None), \
             patch("tools.browser_tool._get_session_info", return_value={
                 "session_name": "test", "_first_nav": False, "features": {"local": True, "proxies": True}
             }), \
             patch("tools.browser_tool._run_browser_command", side_effect=[
                 result,
                 {"success": True, "data": {"snapshot": "- heading \"Fallback OK\" [ref=e1]", "refs": {"e1": {}}}},
             ]):
            response = json.loads(bt.browser_navigate("https://example.com", task_id="warn-test"))

        assert response["success"] is True
        assert response["browser_engine"] == "chrome"
        assert "Lightpanda fallback" in response["fallback_warning"]
        assert response["browser_engine_fallback"]["from"] == "lightpanda"
        assert response["browser_engine_fallback"]["to"] == "chrome"
        bt._last_active_session_key.pop("warn-test", None)

    def test_browser_navigate_surfaces_auto_snapshot_fallback_warning(self):
        import json
        import tools.browser_tool as bt

        snapshot_result = bt._annotate_lightpanda_fallback(
            {"success": True, "data": {"snapshot": "- heading \"Fallback OK\" [ref=e1]", "refs": {"e1": {}}}},
            "Lightpanda returned an empty/too-short snapshot; retried with Chrome.",
        )

        with patch("tools.browser_tool._is_local_backend", return_value=True), \
             patch("tools.browser_tool._get_cloud_provider", return_value=None), \
             patch("tools.browser_tool._get_session_info", return_value={
                 "session_name": "test", "_first_nav": False, "features": {"local": True, "proxies": True}
             }), \
             patch("tools.browser_tool._run_browser_command", side_effect=[
                 {"success": True, "data": {"title": "Fallback OK", "url": "https://example.com/"}},
                 snapshot_result,
             ]):
            response = json.loads(bt.browser_navigate("https://example.com", task_id="warn-test2"))

        assert response["success"] is True
        assert response["browser_engine"] == "chrome"
        assert "Lightpanda fallback" in response["fallback_warning"]
        assert response["element_count"] == 1
        bt._last_active_session_key.pop("warn-test2", None)

    def test_failed_fallback_warning_is_preserved_on_click_error(self):
        import json
        import tools.browser_tool as bt

        result = bt._annotate_lightpanda_fallback(
            {"success": False, "error": "Chrome fallback failed"},
            "Lightpanda 'click' failed (timeout); retried with Chrome.",
        )
        bt._last_active_session_key["warn-test3"] = "warn-test3"
        with patch("tools.browser_tool._run_browser_command", return_value=result):
            response = json.loads(bt.browser_click("@e1", task_id="warn-test3"))

        assert response["success"] is False
        assert "Lightpanda fallback" in response["fallback_warning"]
        assert response["browser_engine"] == "chrome"
        bt._last_active_session_key.pop("warn-test3", None)


    def test_browser_vision_lightpanda_uses_chrome_capture_and_normal_call_llm_shape(self, tmp_path):
        import json
        import tools.browser_tool as bt

        chrome_shot = tmp_path / "chrome.png"
        chrome_shot.write_bytes(b"\x89PNG" + b"0" * 128)

        class _Msg:
            content = "Example Domain screenshot"

        class _Choice:
            message = _Msg()

        class _Response:
            choices = [_Choice()]

        captured_kwargs = {}

        def fake_call_llm(**kwargs):
            captured_kwargs.update(kwargs)
            return _Response()

        with patch("tools.browser_tool._get_browser_engine", return_value="lightpanda"), \
             patch("tools.browser_tool._should_inject_engine", return_value=True), \
             patch("tools.browser_tool._chrome_fallback_screenshot", return_value={
                 "success": True, "data": {"path": str(chrome_shot)}
             }), \
             patch("hermes_constants.get_hermes_dir", return_value=tmp_path), \
             patch("tools.browser_tool.call_llm", side_effect=fake_call_llm):
            response = json.loads(bt.browser_vision("what is this?", task_id="vision-test"))

        assert response["success"] is True
        assert response["analysis"] == "Example Domain screenshot"
        assert response["browser_engine"] == "chrome"
        assert "Lightpanda fallback" in response["fallback_warning"]
        assert "messages" in captured_kwargs
        assert "images" not in captured_kwargs
        assert captured_kwargs["task"] == "vision"


    def test_browser_get_images_preserves_fallback_warning(self):
        import json
        import tools.browser_tool as bt

        result = bt._annotate_lightpanda_fallback(
            {"success": True, "data": {"result": "[]"}},
            "Lightpanda 'eval' failed (timeout); retried with Chrome.",
        )
        bt._last_active_session_key["warn-images"] = "warn-images"
        with patch("tools.browser_tool._run_browser_command", return_value=result):
            response = json.loads(bt.browser_get_images(task_id="warn-images"))

        assert response["success"] is True
        assert response["browser_engine"] == "chrome"
        assert "Lightpanda fallback" in response["fallback_warning"]
        bt._last_active_session_key.pop("warn-images", None)

    def test_browser_vision_lightpanda_response_has_structured_fallback(self, tmp_path):
        import json
        import tools.browser_tool as bt

        chrome_shot = tmp_path / "chrome-structured.png"
        chrome_shot.write_bytes(b"\x89PNG" + b"0" * 128)

        class _Msg:
            content = "Example Domain screenshot"

        class _Choice:
            message = _Msg()

        class _Response:
            choices = [_Choice()]

        with patch("tools.browser_tool._get_browser_engine", return_value="lightpanda"), \
             patch("tools.browser_tool._should_inject_engine", return_value=True), \
             patch("tools.browser_tool._chrome_fallback_screenshot", return_value={
                 "success": True, "data": {"path": str(chrome_shot)}
             }), \
             patch("hermes_constants.get_hermes_dir", return_value=tmp_path), \
             patch("tools.browser_tool.call_llm", return_value=_Response()):
            response = json.loads(bt.browser_vision("what is this?", task_id="vision-structured"))

        assert response["success"] is True
        assert response["browser_engine"] == "chrome"
        assert response["browser_engine_fallback"] == {
            "from": "lightpanda",
            "to": "chrome",
            "reason": "Lightpanda has no graphical renderer for screenshots; used Chrome for vision capture.",
        }

# ---------------------------------------------------------------------------
# _engine_override parameter
# ---------------------------------------------------------------------------

class TestEngineOverride:
    """Verify _engine_override bypasses the cached engine."""

    @patch("tools.browser_tool._get_session_info")
    @patch("tools.browser_tool._find_agent_browser", return_value="/usr/bin/agent-browser")
    @patch("tools.browser_tool._is_local_mode", return_value=True)
    @patch("tools.browser_tool._chromium_installed", return_value=True)
    @patch("tools.browser_tool._get_cloud_provider", return_value=None)
    @patch("tools.browser_tool._get_cdp_override", return_value="")
    @patch("tools.browser_tool._is_camofox_mode", return_value=False)
    def test_override_prevents_engine_injection(
        self, _camofox, _cdp, _cloud, _chromium, _local, _find, _session
    ):
        """When _engine_override='auto', --engine flag is NOT injected."""
        import tools.browser_tool as bt

        # Set the global cache to lightpanda
        bt._cached_browser_engine = "lightpanda"
        bt._browser_engine_resolved = True

        _session.return_value = {"session_name": "test-sess"}

        # Track the cmd_parts that Popen receives
        captured_cmds = []
        mock_proc = MagicMock()
        mock_proc.wait.return_value = None
        mock_proc.returncode = 0

        def capture_popen(cmd, **kwargs):
            captured_cmds.append(cmd)
            return mock_proc

        # We need to mock the file operations too
        with patch("subprocess.Popen", side_effect=capture_popen), \
             patch("os.open", return_value=99), \
             patch("os.close"), \
             patch("os.unlink"), \
             patch("os.makedirs"), \
             patch("builtins.open", MagicMock(return_value=MagicMock(
                 __enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value='{"success": true, "data": {}}'))),
                 __exit__=MagicMock(return_value=False),
             ))), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("tools.browser_tool._write_owner_pid"):
            bt._run_browser_command("task1", "snapshot", [], _engine_override="auto")

        # Should NOT contain "--engine" since override is "auto"
        assert len(captured_cmds) == 1
        assert "--engine" not in captured_cmds[0]

    @patch("tools.browser_tool._get_session_info")
    @patch("tools.browser_tool._find_agent_browser", return_value="/usr/bin/agent-browser")
    @patch("tools.browser_tool._is_local_mode", return_value=True)
    @patch("tools.browser_tool._chromium_installed", return_value=True)
    @patch("tools.browser_tool._get_cloud_provider", return_value=None)
    @patch("tools.browser_tool._get_cdp_override", return_value="")
    @patch("tools.browser_tool._is_camofox_mode", return_value=False)
    def test_no_override_uses_cached_engine(
        self, _camofox, _cdp, _cloud, _chromium, _local, _find, _session
    ):
        """Without _engine_override, the cached engine is used."""
        import tools.browser_tool as bt

        bt._cached_browser_engine = "lightpanda"
        bt._browser_engine_resolved = True

        _session.return_value = {"session_name": "test-sess"}

        captured_cmds = []
        mock_proc = MagicMock()
        mock_proc.wait.return_value = None
        mock_proc.returncode = 0

        def capture_popen(cmd, **kwargs):
            captured_cmds.append(cmd)
            return mock_proc

        # Return a substantive snapshot so the LP fallback does NOT trigger.
        mock_stdout = '{"success": true, "data": {"snapshot": "- heading \\"Hello\\" [ref=e1]", "refs": {"e1": {}}}}'
        with patch("subprocess.Popen", side_effect=capture_popen), \
             patch("os.open", return_value=99), \
             patch("os.close"), \
             patch("os.unlink"), \
             patch("os.makedirs"), \
             patch("builtins.open", MagicMock(return_value=MagicMock(
                 __enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value=mock_stdout))),
                 __exit__=MagicMock(return_value=False),
             ))), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("tools.browser_tool._write_owner_pid"):
            bt._run_browser_command("task1", "snapshot", [])

        # SHOULD contain "--engine lightpanda"
        assert len(captured_cmds) == 1
        assert "--engine" in captured_cmds[0]
        engine_idx = captured_cmds[0].index("--engine")
        assert captured_cmds[0][engine_idx + 1] == "lightpanda"

    def test_hybrid_local_sidecar_injects_engine_even_with_cloud_provider(self):
        """A task::local sidecar is local even when global cloud config exists."""
        import tools.browser_tool as bt

        bt._cached_browser_engine = "lightpanda"
        bt._browser_engine_resolved = True
        captured_cmds = []
        mock_provider = MagicMock()

        mock_proc = MagicMock()
        mock_proc.wait.return_value = None
        mock_proc.returncode = 0

        def capture_popen(cmd, **kwargs):
            captured_cmds.append(cmd)
            return mock_proc

        mock_stdout = json.dumps({
            "success": True,
            "data": {"snapshot": '- heading "Hello" [ref=e1]', "refs": {"e1": {}}},
        })
        with patch("tools.browser_tool._get_session_info", return_value={"session_name": "local-sidecar"}), \
             patch("tools.browser_tool._find_agent_browser", return_value="/usr/bin/agent-browser"), \
             patch("tools.browser_tool._is_local_mode", return_value=False), \
             patch("tools.browser_tool._chromium_installed", return_value=True), \
             patch("tools.browser_tool._get_cloud_provider", return_value=mock_provider), \
             patch("tools.browser_tool._get_cdp_override", return_value=""), \
             patch("tools.browser_tool._is_camofox_mode", return_value=False), \
             patch("subprocess.Popen", side_effect=capture_popen), \
             patch("os.open", return_value=99), \
             patch("os.close"), \
             patch("os.unlink"), \
             patch("os.makedirs"), \
             patch("builtins.open", MagicMock(return_value=MagicMock(
                 __enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value=mock_stdout))),
                 __exit__=MagicMock(return_value=False),
             ))), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("tools.browser_tool._write_owner_pid"):
            bt._run_browser_command("task::local", "snapshot", [])

        assert len(captured_cmds) == 1
        assert "--engine" in captured_cmds[0]
        assert captured_cmds[0][captured_cmds[0].index("--engine") + 1] == "lightpanda"
