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


def _public_command_result(result):
    """Strip internal dispatcher markers from a low-level command result."""
    public = dict(result)
    public.pop("_browser_session_key", None)
    return public


@pytest.fixture(autouse=True)
def _isolate_browser_module_state():
    """Keep browser ownership/session/config caches isolated between tests."""
    import tools.browser_tool as bt

    mapping_names = (
        "_last_active_session_key",
        "_last_active_session_generation",
        "_last_browser_url_by_session_key",
        "_active_sessions",
        "_session_last_activity",
    )
    set_names = (
        "_recording_sessions",
        "_cdp_fallback_local_session_keys",
    )
    scalar_names = (
        "_cached_browser_engine",
        "_browser_engine_resolved",
        "_cached_cdp_fallback_to_local",
        "_cdp_fallback_to_local_resolved",
        "_cached_cloud_provider",
        "_cloud_provider_resolved",
        "_cached_allow_private_urls",
        "_allow_private_urls_resolved",
        "_cached_auto_local_for_private_urls",
        "_auto_local_for_private_urls_resolved",
        "_cached_agent_browser",
        "_agent_browser_resolved",
        "_cached_chromium_installed",
        "_chromium_autoinstall_attempted",
    )
    mapping_snapshots = {
        name: dict(getattr(bt, name))
        for name in mapping_names
        if hasattr(bt, name)
    }
    set_snapshots = {
        name: set(getattr(bt, name))
        for name in set_names
        if hasattr(bt, name)
    }
    scalar_snapshots = {
        name: getattr(bt, name)
        for name in scalar_names
        if hasattr(bt, name)
    }

    def reset_for_test():
        with bt._cleanup_lock:
            for name in mapping_names:
                if hasattr(bt, name):
                    getattr(bt, name).clear()
            for name in set_names:
                if hasattr(bt, name):
                    getattr(bt, name).clear()
        _reset_engine_cache()
        bt._cached_cdp_fallback_to_local = None
        bt._cdp_fallback_to_local_resolved = False
        bt._cached_cloud_provider = None
        bt._cloud_provider_resolved = False
        bt._cached_allow_private_urls = None
        bt._allow_private_urls_resolved = False
        bt._cached_auto_local_for_private_urls = True
        bt._auto_local_for_private_urls_resolved = False
        bt._cached_agent_browser = None
        bt._agent_browser_resolved = False
        bt._cached_chromium_installed = None
        bt._chromium_autoinstall_attempted = False

    reset_for_test()
    try:
        yield
    finally:
        with bt._cleanup_lock:
            for name, snapshot in mapping_snapshots.items():
                target = getattr(bt, name)
                target.clear()
                target.update(snapshot)
            for name, snapshot in set_snapshots.items():
                target = getattr(bt, name)
                target.clear()
                target.update(snapshot)
        for name, value in scalar_snapshots.items():
            setattr(bt, name, value)


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
        assert DEFAULT_CONFIG["browser"]["cdp_fallback_to_local"] is False

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


# ---------------------------------------------------------------------------
# CDP override fallback to local engine chain
# ---------------------------------------------------------------------------

class TestCdpFallbackToLocalEngine:
    """Verify external CDP failures can fall through to local engines."""

    @staticmethod
    def _popen_writer(outputs, captured_cmds):
        import os

        class FakeProc:
            def __init__(self, stdout_fd, stderr_fd, output, returncode=0):
                self._stdout_fd = stdout_fd
                self._stderr_fd = stderr_fd
                self._output = output
                self.returncode = returncode

            def wait(self, timeout=None):
                if self._output:
                    os.write(self._stdout_fd, self._output.encode("utf-8"))
                os.close(self._stdout_fd)
                os.close(self._stderr_fd)
                return self.returncode

            def kill(self):
                self.returncode = -9

        def fake_popen(cmd, stdout, stderr, **kwargs):
            captured_cmds.append(cmd)
            output = outputs(cmd) if callable(outputs) else outputs.pop(0)
            # subprocess.Popen gives the child its own descriptor. Duplicate here
            # because _run_browser_command closes the parent fd immediately after
            # Popen returns.
            return FakeProc(os.dup(stdout), os.dup(stderr), output)

        return fake_popen

    def test_cdp_failure_does_not_trigger_lightpanda_chrome_fallback_when_disabled(self, tmp_path):
        """A CDP backend failure is not misclassified as a Lightpanda failure."""
        import tools.browser_tool as bt

        captured_cmds = []
        outputs = [json.dumps({"success": False, "error": "CDP socket closed"})]

        with patch("tools.browser_tool._get_session_info", return_value={
                 "session_name": "cdp-sess",
                 "cdp_url": "ws://127.0.0.1:9223/devtools/browser",
                 "features": {"cdp_override": True},
             }), \
             patch("tools.browser_tool._find_agent_browser", return_value="/usr/bin/agent-browser"), \
             patch("tools.browser_tool._requires_real_termux_browser_install", return_value=False), \
             patch("tools.browser_tool._is_local_mode", return_value=False), \
             patch("tools.browser_tool._get_browser_engine", return_value="lightpanda"), \
             patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)), \
             patch("tools.browser_tool._write_owner_pid"), \
             patch("tools.browser_tool._run_chrome_fallback_command", side_effect=AssertionError("wrong fallback")), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("subprocess.Popen", side_effect=self._popen_writer(outputs, captured_cmds)):
            result = bt._run_browser_command("cdp-task", "snapshot", ["-c"])

        assert _public_command_result(result) == {"success": False, "error": "CDP socket closed"}
        assert len(captured_cmds) == 1
        assert "--cdp" in captured_cmds[0]
        assert "--engine" not in captured_cmds[0]

    def test_cdp_semantic_failure_does_not_retry_locally(self, tmp_path):
        """Invalid refs/app-level command errors should not switch browsers."""
        import tools.browser_tool as bt

        captured_cmds = []
        outputs = [json.dumps({"success": False, "error": "Element reference @e999 not found"})]

        with patch("tools.browser_tool._get_session_info", return_value={
                 "session_name": "cdp-sess",
                 "cdp_url": "ws://127.0.0.1:9223/devtools/browser",
                 "features": {"cdp_override": True},
             }), \
             patch("tools.browser_tool._find_agent_browser", return_value="/usr/bin/agent-browser"), \
             patch("tools.browser_tool._requires_real_termux_browser_install", return_value=False), \
             patch("tools.browser_tool._is_local_mode", return_value=False), \
             patch("tools.browser_tool._get_browser_engine", return_value="lightpanda"), \
             patch("tools.browser_tool._cdp_fallback_to_local", return_value=True), \
             patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)), \
             patch("tools.browser_tool._write_owner_pid"), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("subprocess.Popen", side_effect=self._popen_writer(outputs, captured_cmds)):
            result = bt._run_browser_command("cdp-task", "click", ["@e999"])

        assert _public_command_result(result) == {
            "success": False,
            "error": "Element reference @e999 not found",
        }
        assert len(captured_cmds) == 1
        assert "--cdp" in captured_cmds[0]

    @pytest.mark.parametrize(
        "semantic_error",
        [
            "JavaScript exception: application WebSocket is unavailable",
            "Element text assertion failed: connection refused by upstream API",
            "Evaluation returned app status: io error",
            "Page reported: command timed out while importing data",
            "DOM assertion failed: operation timed out after 5 seconds",
        ],
    )
    def test_cdp_semantic_failure_containing_transport_words_does_not_retry_locally(
        self, tmp_path, semantic_error
    ):
        """Transport-like words inside page semantics are not CDP failures."""
        import tools.browser_tool as bt

        captured_cmds = []
        outputs = [json.dumps({"success": False, "error": semantic_error})]

        with patch("tools.browser_tool._get_session_info", return_value={
                 "session_name": "cdp-sess",
                 "cdp_url": "ws://127.0.0.1:9223/devtools/browser",
                 "features": {"cdp_override": True},
             }), \
             patch("tools.browser_tool._find_agent_browser", return_value="/usr/bin/agent-browser"), \
             patch("tools.browser_tool._requires_real_termux_browser_install", return_value=False), \
             patch("tools.browser_tool._is_local_mode", return_value=False), \
             patch("tools.browser_tool._get_browser_engine", return_value="lightpanda"), \
             patch("tools.browser_tool._cdp_fallback_to_local", return_value=True), \
             patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)), \
             patch("tools.browser_tool._write_owner_pid"), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("subprocess.Popen", side_effect=self._popen_writer(outputs, captured_cmds)):
            result = bt._run_browser_command("cdp-task", "eval", ["document.title"])

        assert _public_command_result(result) == {"success": False, "error": semantic_error}
        assert len(captured_cmds) == 1
        assert "--cdp" in captured_cmds[0]

    @pytest.mark.parametrize("snapshot", ["", "- none"])
    def test_cdp_successful_empty_or_short_snapshot_does_not_retry_locally(
        self, snapshot
    ):
        """Snapshot content length is page semantics, not backend health."""
        import tools.browser_tool as bt

        with patch("tools.browser_tool._cdp_fallback_to_local", return_value=True):
            reason = bt._cdp_fallback_reason(
                {
                    "cdp_url": "ws://127.0.0.1:9223/devtools/browser",
                    "features": {"cdp_override": True},
                },
                "snapshot",
                {"success": True, "data": {"snapshot": snapshot, "refs": {}}},
            )

        assert reason is None

    def test_cdp_failure_can_retry_on_local_lightpanda_session(self, tmp_path):
        """Opt-in CDP fallback retries the command through local Lightpanda."""
        import tools.browser_tool as bt

        captured_cmds = []
        outputs = [
            json.dumps({"success": False, "error": "CDP socket closed"}),
            json.dumps({
                "success": True,
                "data": {"snapshot": '- heading "Fallback OK" [ref=e1]', "refs": {"e1": {}}},
            }),
        ]

        def fake_session_info(session_key):
            if session_key.endswith("::local"):
                return {"session_name": "local-sess", "cdp_url": None, "features": {"local": True}}
            return {
                "session_name": "cdp-sess",
                "cdp_url": "ws://127.0.0.1:9223/devtools/browser",
                "features": {"cdp_override": True},
            }

        with patch("tools.browser_tool._get_session_info", side_effect=fake_session_info), \
             patch("tools.browser_tool._find_agent_browser", return_value="/usr/bin/agent-browser"), \
             patch("tools.browser_tool._requires_real_termux_browser_install", return_value=False), \
             patch("tools.browser_tool._is_local_mode", return_value=False), \
             patch("tools.browser_tool._get_browser_engine", return_value="lightpanda"), \
             patch("tools.browser_tool._cdp_fallback_to_local", return_value=True), \
             patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)), \
             patch("tools.browser_tool._write_owner_pid"), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("subprocess.Popen", side_effect=self._popen_writer(outputs, captured_cmds)):
            result = bt._run_browser_command("cdp-task", "snapshot", ["-c"])

        assert result["success"] is True
        assert result["browser_session_key"] == "cdp-task::local"
        assert "cdp-task::local" in bt._cdp_fallback_local_session_keys
        assert result["browser_backend_fallback"] == {
            "from": "cdp",
            "to": "local",
            "reason": "CDP backend 'snapshot' failed (CDP socket closed); retried with local browser.",
        }
        assert "CDP fallback" in result["fallback_warning"]
        assert len(captured_cmds) == 2
        assert "--cdp" in captured_cmds[0]
        assert "--session" in captured_cmds[1]
        assert "--engine" in captured_cmds[1]
        assert captured_cmds[1][captured_cmds[1].index("--engine") + 1] == "lightpanda"

    def test_cdp_snapshot_fallback_warms_local_session_with_last_url(self, tmp_path):
        """A later CDP snapshot fallback opens the last successful URL locally first."""
        import tools.browser_tool as bt

        captured_cmds = []
        outputs = [
            json.dumps({"success": False, "error": "CDP socket closed"}),
            json.dumps({"success": True, "data": {"title": "Fallback OK", "url": "https://example.com/"}}),
            json.dumps({"success": True, "data": {"result": "https://example.com/"}}),
            json.dumps({
                "success": True,
                "data": {"snapshot": '- heading "Fallback OK" [ref=e1]', "refs": {"e1": {}}},
            }),
        ]
        bt._last_browser_url_by_session_key["cdp-task"] = "https://example.com/"

        def fake_session_info(session_key):
            if session_key.endswith("::local"):
                return {"session_name": "local-sess", "cdp_url": None, "features": {"local": True}}
            return {
                "session_name": "cdp-sess",
                "cdp_url": "ws://127.0.0.1:9223/devtools/browser",
                "features": {"cdp_override": True},
            }

        try:
            with patch("tools.browser_tool._get_session_info", side_effect=fake_session_info), \
                 patch("tools.browser_tool._find_agent_browser", return_value="/usr/bin/agent-browser"), \
                 patch("tools.browser_tool._requires_real_termux_browser_install", return_value=False), \
                 patch("tools.browser_tool._is_local_mode", return_value=False), \
                 patch("tools.browser_tool._get_browser_engine", return_value="lightpanda"), \
                 patch("tools.browser_tool._cdp_fallback_to_local", return_value=True), \
                 patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)), \
                 patch("tools.browser_tool._write_owner_pid"), \
                 patch("tools.interrupt.is_interrupted", return_value=False), \
                 patch("subprocess.Popen", side_effect=self._popen_writer(outputs, captured_cmds)):
                result = bt._run_browser_command("cdp-task", "snapshot", ["-c"])
        finally:
            bt._last_browser_url_by_session_key.pop("cdp-task", None)
            bt._last_active_session_key.pop("cdp-task", None)

        assert result["success"] is True
        assert len(captured_cmds) == 4
        assert captured_cmds[0][-2:] == ["snapshot", "-c"]
        assert captured_cmds[1][-2:] == ["open", "https://example.com/"]
        assert captured_cmds[2][-2:] == ["eval", "window.location.href"]
        assert captured_cmds[3][-2:] == ["snapshot", "-c"]

    def test_cdp_snapshot_fallback_blocks_unsafe_cached_warmup_url(self, tmp_path):
        """Never run the follow-up command after a cached metadata/private URL."""
        import tools.browser_tool as bt

        captured_cmds = []
        outputs = [
            json.dumps({"success": False, "error": "CDP socket closed"}),
            json.dumps({
                "success": True,
                "data": {"snapshot": '- heading "Fallback OK" [ref=e1]', "refs": {"e1": {}}},
            }),
        ]
        bt._last_browser_url_by_session_key["cdp-task"] = "http://169.254.169.254/latest/meta-data"

        def fake_session_info(session_key):
            if session_key.endswith("::local"):
                return {"session_name": "local-sess", "cdp_url": None, "features": {"local": True}}
            return {
                "session_name": "cdp-sess",
                "cdp_url": "ws://127.0.0.1:9223/devtools/browser",
                "features": {"cdp_override": True},
            }

        try:
            with patch("tools.browser_tool._get_session_info", side_effect=fake_session_info), \
                 patch("tools.browser_tool._find_agent_browser", return_value="/usr/bin/agent-browser"), \
                 patch("tools.browser_tool._requires_real_termux_browser_install", return_value=False), \
                 patch("tools.browser_tool._is_local_mode", return_value=False), \
                 patch("tools.browser_tool._get_browser_engine", return_value="lightpanda"), \
                 patch("tools.browser_tool._cdp_fallback_to_local", return_value=True), \
                 patch("tools.browser_tool._allow_private_urls", return_value=False), \
                 patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)), \
                 patch("tools.browser_tool._write_owner_pid"), \
                 patch("tools.interrupt.is_interrupted", return_value=False), \
                 patch("subprocess.Popen", side_effect=self._popen_writer(outputs, captured_cmds)):
                result = bt._run_browser_command("cdp-task", "snapshot", ["-c"])
        finally:
            bt._last_browser_url_by_session_key.pop("cdp-task", None)
            bt._last_active_session_key.pop("cdp-task", None)

        assert result["success"] is False
        assert len(captured_cmds) == 2
        assert captured_cmds[0][-2:] == ["snapshot", "-c"]
        assert captured_cmds[1][-2:] == ["open", "about:blank"]
        assert "cdp-task" not in bt._last_browser_url_by_session_key

    def test_browser_navigate_uses_fallback_session_for_auto_snapshot(self):
        """After CDP→local fallback, follow-up snapshot uses the local session key."""
        import tools.browser_tool as bt

        bt._last_active_session_key.pop("nav-cdp", None)
        nav_result = {
            "success": True,
            "data": {"title": "Fallback OK", "url": "https://example.com/"},
            "browser_session_key": "nav-cdp::local",
            "fallback_warning": "⚠ CDP fallback: local browser was used.",
            "browser_backend_fallback": {
                "from": "cdp",
                "to": "local",
                "reason": "CDP backend 'open' failed (CDP socket closed); retried with local browser.",
            },
        }
        snapshot_result = {
            "success": True,
            "data": {"snapshot": '- heading "Fallback OK" [ref=e1]', "refs": {"e1": {}}},
        }

        with patch("tools.browser_tool._is_local_backend", return_value=True), \
             patch("tools.browser_tool._get_session_info", return_value={
                 "session_name": "cdp-sess",
                 "cdp_url": "ws://127.0.0.1:9223/devtools/browser",
                 "features": {"cdp_override": True},
                 "_first_nav": False,
             }), \
             patch(
                 "tools.browser_tool._run_browser_command",
                 side_effect=[
                     nav_result,
                     {"success": True, "data": {"result": "https://example.com/"}},
                     snapshot_result,
                     {"success": True, "data": {"result": "https://example.com/"}},
                 ],
             ) as run_cmd:
            response = json.loads(bt.browser_navigate("https://example.com", task_id="nav-cdp"))

        assert response["success"] is True
        assert response["browser_backend_fallback"]["from"] == "cdp"
        assert response["element_count"] == 1
        assert bt._last_active_session_key["nav-cdp"] == "nav-cdp::local"
        assert run_cmd.call_args_list[2].args[0] == "nav-cdp::local"
        assert run_cmd.call_args_list[2].args[1] == "snapshot"
        bt._last_active_session_key.pop("nav-cdp", None)

    @pytest.mark.parametrize(
        ("redirect_url", "expected_error"),
        [
            (
                "http://169.254.169.254/latest/meta-data",
                "Blocked: page URL targets a cloud metadata endpoint "
                "(http://169.254.169.254/latest/meta-data).",
            ),
            (
                "http://127.0.0.1/admin",
                "Blocked: page URL targets a private or internal address "
                "(http://127.0.0.1/admin).",
            ),
        ],
        ids=["metadata", "private"],
    )
    def test_browser_navigate_blocked_redirect_is_not_cached_for_warmup(
        self, redirect_url, expected_error
    ):
        """Blocked fallback redirects are blanked on the serving session and not cached."""
        import tools.browser_tool as bt

        bt._last_active_session_key["nav-cdp"] = "nav-cdp::local"
        bt._last_browser_url_by_session_key["nav-cdp"] = "https://previous.example/"
        nav_result = {
            "success": True,
            "data": {
                "title": "blocked redirect",
                "url": redirect_url,
            },
            "browser_session_key": "nav-cdp::local",
        }
        probe_result = {"success": True, "data": {"result": redirect_url}}
        blank_result = {"success": True, "data": {}}

        try:
            with patch("tools.browser_tool._is_local_backend", return_value=False), \
                 patch("tools.browser_tool._get_session_info", return_value={
                     "session_name": "cdp-sess",
                     "cdp_url": "ws://127.0.0.1:9223/devtools/browser",
                     "features": {"cdp_override": True},
                     "_first_nav": False,
                 }), \
                 patch(
                     "tools.browser_tool._run_browser_command",
                     side_effect=[nav_result, probe_result, blank_result],
                 ) as run_cmd:
                response = json.loads(bt.browser_navigate("https://example.com", task_id="nav-cdp"))
                binding_after_block = bt._last_active_session_key.get("nav-cdp")
        finally:
            bt._last_active_session_key.pop("nav-cdp", None)
            bt._last_browser_url_by_session_key.pop("nav-cdp", None)

        assert response == {
            "success": False,
            "error": expected_error,
        }
        assert "nav-cdp" not in bt._last_browser_url_by_session_key
        assert binding_after_block is None
        assert run_cmd.call_args_list[2].args == ("nav-cdp::local", "open", ["about:blank"])

    def test_browser_navigate_blocked_redirect_cleanup_failure_drops_fallback_binding(self):
        """A failed blanking attempt cannot leave the unsafe fallback session active."""
        import tools.browser_tool as bt

        local_session_key = "nav-cdp::local"
        bt._last_active_session_key["nav-cdp"] = local_session_key
        bt._active_sessions[local_session_key] = {
            "session_key": local_session_key,
            "owner_task_id": "nav-cdp",
            "session_name": "local-sess",
        }
        bt._last_browser_url_by_session_key["nav-cdp"] = "https://previous.example/"
        bt._last_browser_url_by_session_key[local_session_key] = (
            "http://169.254.169.254/latest/meta-data"
        )
        nav_result = {
            "success": True,
            "data": {
                "title": "metadata",
                "url": "http://169.254.169.254/latest/meta-data",
            },
            "browser_session_key": local_session_key,
        }
        probe_result = {
            "success": True,
            "data": {"result": "http://169.254.169.254/latest/meta-data"},
        }
        blank_failure = {"success": False, "error": "local sidecar unavailable"}
        close_result = {"success": True, "data": {}}
        snapshot_result = {
            "success": True,
            "data": {"snapshot": '- heading "Recovered CDP" [ref=e1]', "refs": {"e1": {}}},
        }

        try:
            with patch("tools.browser_tool._is_local_backend", return_value=True), \
                 patch("tools.browser_tool._get_session_info", return_value={
                     "session_name": "cdp-sess",
                     "cdp_url": "ws://127.0.0.1:9223/devtools/browser",
                     "features": {"cdp_override": True},
                     "_first_nav": False,
                 }), \
                 patch(
                     "tools.browser_tool._run_browser_command",
                     side_effect=[
                         nav_result,
                         probe_result,
                         blank_failure,
                         close_result,
                         snapshot_result,
                     ],
                 ) as run_cmd:
                nav_response = json.loads(
                    bt.browser_navigate("https://example.com", task_id="nav-cdp")
                )
                binding_after_nav = bt._last_active_session_key.get("nav-cdp")
                snapshot_response = json.loads(bt.browser_snapshot(task_id="nav-cdp"))
        finally:
            bt._last_active_session_key.pop("nav-cdp", None)
            bt._active_sessions.pop(local_session_key, None)
            bt._last_browser_url_by_session_key.pop("nav-cdp", None)
            bt._last_browser_url_by_session_key.pop(local_session_key, None)

        assert nav_response == {
            "success": False,
            "error": (
                "Blocked: page URL targets a cloud metadata endpoint "
                "(http://169.254.169.254/latest/meta-data)."
            ),
        }
        assert binding_after_nav is None
        assert snapshot_response["success"] is True
        assert run_cmd.call_args_list[2].args == (
            local_session_key,
            "open",
            ["about:blank"],
        )
        assert run_cmd.call_args_list[3].args == (
            local_session_key,
            "close",
            [],
        )
        assert run_cmd.call_args_list[4].args[0] == "nav-cdp"
        assert local_session_key not in bt._active_sessions

    def test_blocked_redirect_cleanup_does_not_mask_failure_with_temporary_chrome(
        self, tmp_path
    ):
        """Blanking stays on the serving sidecar and destroys it on failure."""
        import tools.browser_tool as bt

        task_id = "cleanup-mask"
        local_session_key = f"{task_id}::local"
        bt._active_sessions.update({
            task_id: {
                "session_key": task_id,
                "owner_task_id": task_id,
                "session_name": "cdp-sess",
                "cdp_url": "ws://127.0.0.1:9223/devtools/browser",
                "features": {"cdp_override": True},
                "_first_nav": False,
            },
            local_session_key: {
                "session_key": local_session_key,
                "owner_task_id": task_id,
                "session_name": "unsafe-local-sess",
                "features": {"local": True},
            },
        })
        captured_cmds = []
        outputs = [
            json.dumps({"success": False, "error": "CDP socket closed"}),
            json.dumps({
                "success": True,
                "data": {
                    "title": "metadata",
                    "url": "http://169.254.169.254/latest/meta-data",
                },
            }),
            json.dumps({
                "success": True,
                "data": {"result": "http://169.254.169.254/latest/meta-data"},
            }),
            json.dumps({"success": False, "error": "Lightpanda open failed"}),
            json.dumps({"success": True, "data": {}}),
        ]

        with patch("tools.browser_tool._find_agent_browser", return_value="/usr/bin/agent-browser"), \
             patch("tools.browser_tool._requires_real_termux_browser_install", return_value=False), \
             patch("tools.browser_tool._is_local_mode", return_value=False), \
             patch("tools.browser_tool._is_local_backend", return_value=False), \
             patch("tools.browser_tool._get_browser_engine", return_value="lightpanda"), \
             patch("tools.browser_tool._cdp_fallback_to_local", return_value=True), \
             patch("tools.browser_tool._navigation_session_key", return_value=task_id), \
             patch("tools.browser_tool._allow_private_urls", return_value=False), \
             patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)), \
             patch("tools.browser_tool._write_owner_pid"), \
             patch("tools.browser_tool._start_browser_cleanup_thread"), \
             patch("tools.browser_tool._stop_cdp_supervisor"), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("subprocess.Popen", side_effect=self._popen_writer(outputs, captured_cmds)), \
             patch(
                 "tools.browser_tool._run_chrome_fallback_command",
                 return_value={"success": True, "data": {"url": "about:blank"}},
             ) as chrome_fallback:
            response = json.loads(bt.browser_navigate("https://example.com", task_id=task_id))

        assert response == {
            "success": False,
            "error": (
                "Blocked: page URL targets a cloud metadata endpoint "
                "(http://169.254.169.254/latest/meta-data)."
            ),
        }
        chrome_fallback.assert_not_called()
        assert local_session_key not in bt._active_sessions
        assert task_id not in bt._last_active_session_key
        assert [cmd[-2:] for cmd in captured_cmds] == [
            ["open", "https://example.com"],
            ["open", "https://example.com"],
            ["eval", "window.location.href"],
            ["open", "about:blank"],
            ["--json", "close"],
        ]

    def test_failed_cdp_fallback_does_not_change_last_active_owner(self):
        """A failed local retry cannot retarget later tools to the local sidecar."""
        import tools.browser_tool as bt

        bt._last_active_session_key["owner-task"] = "owner-task"
        with patch(
            "tools.browser_tool._run_browser_command",
            return_value={"success": False, "error": "local snapshot failed"},
        ):
            result = bt._run_cdp_local_fallback_command(
                "owner-task",
                "snapshot",
                ["-c"],
                30,
                "CDP socket closed",
            )

        assert result["success"] is False
        assert bt._last_active_session_key["owner-task"] == "owner-task"

    def test_failed_cdp_fallback_warmup_aborts_followup_command(self):
        """A failed local warm-up must not run a command against stale page state."""
        import tools.browser_tool as bt

        bt._last_active_session_key["warmup-task"] = "warmup-task"
        bt._last_browser_url_by_session_key["warmup-task"] = "https://example.com/"
        with patch(
            "tools.browser_tool._run_browser_command",
            side_effect=[
                {"success": False, "error": "local open failed"},
                {
                    "success": True,
                    "data": {"snapshot": "METADATA_SECRET_CONTENT"},
                },
            ],
        ) as run_cmd:
            result = bt._run_cdp_local_fallback_command(
                "warmup-task",
                "snapshot",
                ["-c"],
                30,
                "CDP socket closed",
            )

        assert result["success"] is False
        assert "local open failed" in result["error"]
        assert run_cmd.call_count == 2
        assert run_cmd.call_args_list[1].args[:3] == (
            "warmup-task::local",
            "open",
            ["about:blank"],
        )
        assert all(call.args[1] != "snapshot" for call in run_cmd.call_args_list)
        assert run_cmd.call_args_list[0].kwargs["_allow_fallback"] is False
        assert bt._last_active_session_key["warmup-task"] == "warmup-task"

    @pytest.mark.parametrize(
        "probe_result",
        [
            {"success": False, "error": "probe failed"},
            {"success": True, "data": {"result": ""}},
            {
                "success": True,
                "data": {"result": "https://example.com/"},
                "browser_session_key": "wrong-session",
            },
            {
                "success": True,
                "data": {"result": "https://example.com/"},
                "browser_engine_fallback": {"from": "lightpanda", "to": "chrome"},
            },
            {"success": True, "data": {"result": "http://127.0.0.1/admin"}},
            {
                "success": True,
                "data": {"result": "http://169.254.169.254/latest/meta-data"},
            },
        ],
        ids=["failed", "empty", "session-mismatch", "fallback", "private", "imds"],
    )
    def test_cdp_warmup_probes_exact_serving_url_before_followup_and_quarantines(
        self, probe_result
    ):
        """Warm-up proof failure withholds the command and discards all sidecar state."""
        import tools.browser_tool as bt

        task_id = "warmup-proof"
        local_session_key = f"{task_id}::local"
        bt._last_browser_url_by_session_key[task_id] = "https://example.com/"
        bt._last_browser_url_by_session_key[local_session_key] = "https://stale.example/"
        bt._active_sessions[local_session_key] = {
            "session_key": local_session_key,
            "owner_task_id": task_id,
            "session_name": "warmup-local",
            "bb_session_id": None,
        }
        bt._session_last_activity[local_session_key] = 123.0
        bt._recording_sessions.add(local_session_key)
        bt._cdp_fallback_local_session_keys.add(local_session_key)
        bt._set_last_active_session_key(task_id, local_session_key)
        commands = []

        def run_command(session_key, command, args=None, timeout=None, **kwargs):
            commands.append((session_key, command, args, kwargs))
            assert session_key == local_session_key
            if command == "open" and args == ["https://example.com/"]:
                return {
                    "success": True,
                    "data": {"title": "redirected", "url": "https://example.com/"},
                }
            if command == "eval":
                return probe_result
            if command == "open" and args == ["about:blank"]:
                return {"success": True, "data": {"url": "about:blank"}}
            if command == "record":
                return {"success": True, "data": {}}
            if command == "close":
                return {"success": True, "data": {}}
            raise AssertionError((session_key, command, args, kwargs))

        with patch("tools.browser_tool._run_browser_command", side_effect=run_command), \
             patch("tools.browser_tool._stop_cdp_supervisor"), \
             patch("tools.browser_tool._is_camofox_mode", return_value=False), \
             patch("tools.browser_tool._allow_private_urls", return_value=False):
            result = bt._run_cdp_local_fallback_command(
                task_id,
                "snapshot",
                ["-c"],
                30,
                "CDP socket closed",
            )

        assert result["success"] is False
        assert all(command != "snapshot" for _, command, _, _ in commands)
        assert commands[1][1] == "eval"
        assert commands[1][3]["_allow_fallback"] is False
        assert local_session_key not in bt._active_sessions
        assert local_session_key not in bt._session_last_activity
        assert local_session_key not in bt._recording_sessions
        assert local_session_key not in bt._cdp_fallback_local_session_keys
        assert local_session_key not in bt._last_browser_url_by_session_key
        assert task_id not in bt._last_active_session_key

    @pytest.mark.parametrize("failed_phase", ["warmup", "followup"])
    def test_failed_cdp_local_phase_quarantines_every_sidecar_tracker(
        self, failed_phase
    ):
        """A failed warm-up or substantive retry never leaves a reusable daemon."""
        import tools.browser_tool as bt

        task_id = f"failed-{failed_phase}"
        local_session_key = f"{task_id}::local"
        if failed_phase == "warmup":
            bt._last_browser_url_by_session_key[task_id] = "https://example.com/"
        bt._last_browser_url_by_session_key[local_session_key] = "https://stale.example/"
        bt._active_sessions[local_session_key] = {
            "session_key": local_session_key,
            "owner_task_id": task_id,
            "session_name": "failed-local",
            "bb_session_id": None,
        }
        bt._session_last_activity[local_session_key] = 123.0
        bt._recording_sessions.add(local_session_key)
        bt._cdp_fallback_local_session_keys.add(local_session_key)
        bt._set_last_active_session_key(task_id, local_session_key)
        commands = []

        def run_command(session_key, command, args=None, timeout=None, **kwargs):
            commands.append((session_key, command, args, kwargs))
            assert session_key == local_session_key
            if command in {"snapshot", "record", "close"}:
                if command == "snapshot":
                    return {"success": False, "error": "fallback command failed"}
                return {"success": True, "data": {}}
            if command == "open" and args == ["https://example.com/"]:
                return {"success": False, "error": "warm-up open failed"}
            if command == "open" and args == ["about:blank"]:
                return {"success": True, "data": {"url": "about:blank"}}
            raise AssertionError((session_key, command, args, kwargs))

        with patch("tools.browser_tool._run_browser_command", side_effect=run_command), \
             patch("tools.browser_tool._stop_cdp_supervisor"), \
             patch("tools.browser_tool._is_camofox_mode", return_value=False):
            result = bt._run_cdp_local_fallback_command(
                task_id,
                "snapshot",
                ["-c"],
                30,
                "CDP socket closed",
            )

        assert result["success"] is False
        assert local_session_key not in bt._active_sessions
        assert local_session_key not in bt._session_last_activity
        assert local_session_key not in bt._recording_sessions
        assert local_session_key not in bt._cdp_fallback_local_session_keys
        assert local_session_key not in bt._last_browser_url_by_session_key
        assert task_id not in bt._last_active_session_key
        assert [command for _, command, _, _ in commands][-1] == "close"

    def test_slow_old_fallback_cannot_steal_owner_from_new_navigation(self):
        """A fallback uses CAS so a newer successful navigation remains owner."""
        import threading
        import tools.browser_tool as bt

        task_id = "stale-owner"
        fallback_started = threading.Event()
        release_fallback = threading.Event()
        fallback_results = []

        def fake_run(session_key, command, args=None, timeout=None, **kwargs):
            if session_key.endswith("::local") and command == "snapshot":
                fallback_started.set()
                assert release_fallback.wait(timeout=5)
                return {
                    "success": True,
                    "data": {
                        "snapshot": '- heading "Old fallback" [ref=e1]',
                        "refs": {"e1": {}},
                    },
                }
            if session_key == task_id and command == "open":
                return {
                    "success": True,
                    "data": {
                        "title": "New navigation",
                        "url": "https://new.example/",
                    },
                }
            if session_key == task_id and command == "snapshot":
                return {
                    "success": True,
                    "data": {
                        "snapshot": '- heading "New navigation" [ref=e2]',
                        "refs": {"e2": {}},
                    },
                }
            raise AssertionError((session_key, command, args, timeout, kwargs))

        def run_old_fallback():
            fallback_results.append(bt._run_cdp_local_fallback_command(
                task_id,
                "snapshot",
                ["-c"],
                30,
                "old CDP failure",
            ))

        with patch("tools.browser_tool._run_browser_command", side_effect=fake_run), \
             patch("tools.browser_tool._navigation_session_key", return_value=task_id), \
             patch("tools.browser_tool._is_local_backend", return_value=True), \
             patch("tools.browser_tool._get_session_info", return_value={
                 "session_name": "primary",
                 "_first_nav": False,
                 "features": {"local": True},
             }):
            thread = threading.Thread(target=run_old_fallback)
            thread.start()
            assert fallback_started.wait(timeout=5)
            navigation = json.loads(
                bt.browser_navigate("https://new.example/", task_id=task_id)
            )
            release_fallback.set()
            thread.join(timeout=5)

        assert not thread.is_alive()
        assert navigation["success"] is True
        assert fallback_results[0]["success"] is True
        assert bt._last_active_session_key[task_id] == task_id

    def test_original_cdp_command_start_generation_blocks_stale_fallback_publish(
        self, tmp_path
    ):
        """The CAS token predates the original CDP command, not only its retry."""
        import threading
        import tools.browser_tool as bt

        task_id = "original-command-race"
        original_cdp_started = threading.Event()
        release_original_cdp = threading.Event()
        captured_cmds = []
        results = []

        def output_for(cmd):
            if "--cdp" in cmd:
                original_cdp_started.set()
                assert release_original_cdp.wait(timeout=5)
                return json.dumps({"success": False, "error": "CDP socket closed"})
            return json.dumps({
                "success": True,
                "data": {
                    "snapshot": '- heading "stale fallback" [ref=e1]',
                    "refs": {"e1": {}},
                },
            })

        def session_info(session_key):
            if session_key.endswith("::local"):
                return {
                    "session_name": "fallback-local",
                    "cdp_url": None,
                    "features": {"local": True},
                }
            return {
                "session_name": "external-cdp",
                "cdp_url": "ws://127.0.0.1:9223/devtools/browser",
                "features": {"cdp_override": True},
            }

        bt._set_last_active_session_key(task_id, task_id)
        with patch("tools.browser_tool._get_session_info", side_effect=session_info), \
             patch("tools.browser_tool._find_agent_browser", return_value="/usr/bin/agent-browser"), \
             patch("tools.browser_tool._requires_real_termux_browser_install", return_value=False), \
             patch("tools.browser_tool._is_local_mode", return_value=False), \
             patch("tools.browser_tool._get_browser_engine", return_value="lightpanda"), \
             patch("tools.browser_tool._cdp_fallback_to_local", return_value=True), \
             patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)), \
             patch("tools.browser_tool._write_owner_pid"), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("subprocess.Popen", side_effect=self._popen_writer(output_for, captured_cmds)):
            worker = threading.Thread(target=lambda: results.append(
                bt._run_browser_command(task_id, "snapshot", ["-c"])
            ))
            worker.start()
            assert original_cdp_started.wait(timeout=5)
            bt._set_last_active_session_key(task_id, task_id)
            release_original_cdp.set()
            worker.join(timeout=5)

        assert not worker.is_alive()
        assert results[0]["success"] is True
        assert bt._last_active_session_key[task_id] == task_id
        assert f"{task_id}::local" not in bt._cdp_fallback_local_session_keys

    def test_cdp_fallback_owner_and_provenance_publish_share_one_cas(self):
        """A failed CAS changes neither owner nor fallback provenance."""
        import tools.browser_tool as bt

        task_id = "atomic-fallback-owner"
        local_session_key = f"{task_id}::local"
        generation = bt._set_last_active_session_key(task_id, task_id)
        bt._cdp_fallback_local_session_keys.add("unrelated::local")

        published = bt._compare_publish_cdp_fallback_owner(
            task_id,
            generation - 1,
            local_session_key,
        )

        assert published is False
        assert bt._last_active_session_key[task_id] == task_id
        assert bt._last_active_session_generation[task_id] == generation
        assert bt._cdp_fallback_local_session_keys == {"unrelated::local"}

    def test_cleanup_compare_clear_preserves_new_same_binding_generation(self):
        """Cleanup cannot clear a new navigation that reused the same session key."""
        import tools.browser_tool as bt

        task_id = "cleanup-same-key-race"
        first_generation = bt._set_last_active_session_key(task_id, task_id)

        def cleanup_then_new_navigation(session_key):
            assert session_key == task_id
            bt._set_last_active_session_key(task_id, task_id)

        with patch(
            "tools.browser_tool._cleanup_single_browser_session",
            side_effect=cleanup_then_new_navigation,
        ):
            bt.cleanup_browser(task_id)

        assert bt._last_active_session_key[task_id] == task_id
        assert bt._last_active_session_generation[task_id] == first_generation + 1

    def test_neutralize_compare_clear_preserves_new_same_binding_generation(self):
        """A stale neutralizer cannot clear a same-key navigation completed meanwhile."""
        import tools.browser_tool as bt

        task_id = "neutralize-same-key-race"
        first_generation = bt._set_last_active_session_key(task_id, task_id)

        def blank_then_new_navigation(*args, **kwargs):
            bt._set_last_active_session_key(task_id, task_id)
            return {"success": True, "data": {"url": "about:blank"}}

        with patch(
            "tools.browser_tool._run_browser_command",
            side_effect=blank_then_new_navigation,
        ):
            bt._neutralize_blocked_redirect(task_id, task_id, task_id)

        assert bt._last_active_session_key[task_id] == task_id
        assert bt._last_active_session_generation[task_id] == first_generation + 1

    def test_cleanup_compare_clear_preserves_new_different_binding(self):
        """All owner writes share the cleanup lock, making compare-clear atomic."""
        import threading
        import tools.browser_tool as bt

        task_id = "owner-race"
        stale_session_key = f"{task_id}::local"
        owner_read = threading.Event()
        new_owner_written = threading.Event()
        original_bindings = bt._last_active_session_key
        cleanup_thread_id = None

        class InterleavingBindings(dict):
            def get(self, key, default=None):
                value = super().get(key, default)
                if key == task_id and threading.get_ident() == cleanup_thread_id:
                    owner_read.set()
                    new_owner_written.wait(timeout=0.5)
                return value

            def __setitem__(self, key, value):
                super().__setitem__(key, value)
                if key == task_id and value == task_id:
                    new_owner_written.set()

        bindings = InterleavingBindings({task_id: stale_session_key})
        bt._last_active_session_key = bindings

        def fake_run(session_key, command, args=None, timeout=None, **kwargs):
            if session_key == stale_session_key and command == "open":
                assert new_owner_written.wait(timeout=5)
                return {"success": True, "data": {"url": "about:blank"}}
            if session_key == task_id and command == "open":
                return {
                    "success": True,
                    "data": {"title": "new", "url": "https://new.example/"},
                }
            if session_key == task_id and command == "snapshot":
                return {
                    "success": True,
                    "data": {
                        "snapshot": '- heading "new" [ref=e1]',
                        "refs": {"e1": {}},
                    },
                }
            raise AssertionError((session_key, command, args, timeout, kwargs))

        try:
            with patch("tools.browser_tool._run_browser_command", side_effect=fake_run), \
                 patch("tools.browser_tool._navigation_session_key", return_value=task_id), \
                 patch("tools.browser_tool._is_local_backend", return_value=True), \
                 patch("tools.browser_tool._get_session_info", return_value={
                     "session_name": "primary",
                     "_first_nav": False,
                     "features": {"local": True},
                 }):
                def run_cleanup():
                    nonlocal cleanup_thread_id
                    cleanup_thread_id = threading.get_ident()
                    bt._neutralize_blocked_redirect(
                        task_id,
                        task_id,
                        stale_session_key,
                    )

                cleanup = threading.Thread(target=run_cleanup)
                cleanup.start()
                assert owner_read.wait(timeout=5)

                navigation = {}

                def run_navigation():
                    navigation.update(json.loads(
                        bt.browser_navigate("https://new.example/", task_id=task_id)
                    ))

                writer = threading.Thread(target=run_navigation)
                writer.start()
                cleanup.join(timeout=5)
                writer.join(timeout=5)

            assert not cleanup.is_alive()
            assert not writer.is_alive()
            assert navigation["success"] is True
            assert bindings.get(task_id) == task_id
        finally:
            bt._last_active_session_key = original_bindings

    @pytest.mark.parametrize(
        ("unsafe_url", "expected_error"),
        [
            (
                "http://169.254.169.254/latest/meta-data",
                "cloud metadata endpoint",
            ),
            (
                "http://127.0.0.1/admin",
                "private or internal address",
            ),
        ],
        ids=["metadata", "private"],
    )
    def test_snapshot_rechecks_actual_fallback_session_and_recreates_after_quarantine(
        self, tmp_path, unsafe_url, expected_error
    ):
        """Never return an unsafe local snapshot based on a recovered public CDP probe."""
        import tools.browser_tool as bt

        task_id = "snapshot-guard"
        local_session_key = f"{task_id}::local"
        secret = "METADATA_SECRET_CONTENT"
        bt._active_sessions.update({
            task_id: {
                "session_key": task_id,
                "owner_task_id": task_id,
                "session_name": "cdp-sess",
                "cdp_url": "ws://127.0.0.1:9223/devtools/browser",
                "features": {"cdp_override": True},
            },
            local_session_key: {
                "session_key": local_session_key,
                "owner_task_id": task_id,
                "session_name": "unsafe-old-local",
                "features": {"local": True},
            },
        })
        captured_cmds = []

        def output_for(cmd):
            command_index = cmd.index("--json") + 1
            command = cmd[command_index]
            command_args = cmd[command_index + 1:]
            if "--cdp" in cmd:
                if command == "snapshot":
                    return json.dumps({
                        "success": False,
                        "error": "CDP socket closed",
                    })
                if command == "eval":
                    # The old implementation probes the recovered CDP session
                    # and incorrectly treats this public URL as proof that the
                    # local snapshot is safe.
                    return json.dumps({
                        "success": True,
                        "data": {"result": "https://public.example/"},
                    })
            session_name = cmd[cmd.index("--session") + 1]
            if session_name == "unsafe-old-local":
                if command == "snapshot":
                    return json.dumps({
                        "success": True,
                        "data": {
                            "snapshot": f'- heading "{secret}" [ref=e1]',
                            "refs": {"e1": {}},
                        },
                    })
                if command == "eval":
                    return json.dumps({
                        "success": True,
                        "data": {"result": unsafe_url},
                    })
                if command == "open" and command_args == ["about:blank"]:
                    return json.dumps({
                        "success": False,
                        "error": "unsafe sidecar cannot blank",
                    })
                if command == "close":
                    return json.dumps({"success": True, "data": {}})
            if session_name == "fresh-local":
                if command == "snapshot":
                    return json.dumps({
                        "success": True,
                        "data": {
                            "snapshot": '- heading "Public recovery" [ref=e2]',
                            "refs": {"e2": {}},
                        },
                    })
                if command == "eval":
                    return json.dumps({
                        "success": True,
                        "data": {"result": "https://public.example/"},
                    })
            raise AssertionError(cmd)

        with patch("tools.browser_tool._find_agent_browser", return_value="/usr/bin/agent-browser"), \
             patch("tools.browser_tool._requires_real_termux_browser_install", return_value=False), \
             patch("tools.browser_tool._is_local_mode", return_value=False), \
             patch("tools.browser_tool._is_local_backend", return_value=False), \
             patch("tools.browser_tool._get_browser_engine", return_value="lightpanda"), \
             patch("tools.browser_tool._cdp_fallback_to_local", return_value=True), \
             patch("tools.browser_tool._allow_private_urls", return_value=False), \
             patch(
                 "tools.browser_tool._is_safe_url",
                 side_effect=lambda url: not (
                     url.startswith("http://127.0.0.1")
                     or url.startswith("http://169.254.169.254")
                 ),
             ), \
             patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)), \
             patch("tools.browser_tool._write_owner_pid"), \
             patch("tools.browser_tool._start_browser_cleanup_thread"), \
             patch("tools.browser_tool._stop_cdp_supervisor"), \
             patch("tools.browser_tool._create_local_session", return_value={
                 "session_name": "fresh-local",
                 "features": {"local": True},
             }) as create_local, \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("subprocess.Popen", side_effect=self._popen_writer(output_for, captured_cmds)), \
             patch(
                 "tools.browser_tool._run_chrome_fallback_command",
                 side_effect=AssertionError("safety cleanup must not fallback"),
             ):
            blocked = json.loads(bt.browser_snapshot(task_id=task_id))
            recovered = json.loads(bt.browser_snapshot(task_id=task_id))

        assert blocked["success"] is False
        assert expected_error in blocked["error"]
        assert secret not in json.dumps(blocked)
        assert recovered["success"] is True
        assert "Public recovery" in recovered["snapshot"]
        assert secret not in json.dumps(recovered)
        assert create_local.call_count == 1
        assert bt._active_sessions[local_session_key]["session_name"] == "fresh-local"

        local_commands = [
            cmd[cmd.index("--json") + 1]
            for cmd in captured_cmds
            if "--session" in cmd
        ]
        assert local_commands == [
            "snapshot",
            "eval",
            "open",
            "close",
            "snapshot",
            "eval",
        ]

    def test_snapshot_url_probe_failure_is_fail_closed(self):
        """A failed serving-session URL probe withholds snapshot content."""
        import tools.browser_tool as bt

        task_id = "snapshot-probe-failure"
        bt._active_sessions[task_id] = {
            "session_key": task_id,
            "owner_task_id": task_id,
            "session_name": "cdp-sess",
            "cdp_url": "ws://127.0.0.1:9223/devtools/browser",
            "features": {"cdp_override": True},
        }
        with patch("tools.browser_tool._is_local_backend", return_value=False), \
             patch("tools.browser_tool._allow_private_urls", return_value=False), \
             patch(
                 "tools.browser_tool._run_browser_command",
                 side_effect=[
                     {
                         "success": True,
                         "data": {
                             "snapshot": '- heading "METADATA_SECRET_CONTENT" [ref=e1]',
                             "refs": {"e1": {}},
                         },
                     },
                     {"success": False, "error": None},
                     {"success": True, "data": {"url": "about:blank"}},
                 ],
             ) as run_cmd:
            response = json.loads(bt.browser_snapshot(task_id=task_id))

        assert response["success"] is False
        assert "unable to verify" in response["error"]
        assert "METADATA_SECRET_CONTENT" not in json.dumps(response)
        assert run_cmd.call_args_list[1].kwargs["_allow_fallback"] is False
        assert run_cmd.call_args_list[2].kwargs["_allow_fallback"] is False

    @pytest.mark.parametrize(
        "tool_name",
        ["snapshot", "eval", "get_images", "vision"],
    )
    def test_quarantined_sidecar_is_not_reused_by_followup_tools(
        self, tmp_path, tool_name
    ):
        """Every non-navigation tool resolves away from a discarded unsafe sidecar."""
        import tools.browser_tool as bt

        task_id = "quarantine-followup"
        local_session_key = f"{task_id}::local"
        bt._active_sessions.update({
            task_id: {
                "session_key": task_id,
                "owner_task_id": task_id,
                "session_name": "primary",
            },
            local_session_key: {
                "session_key": local_session_key,
                "owner_task_id": task_id,
                "session_name": "unsafe-local",
            },
        })
        bt._last_active_session_key[task_id] = local_session_key
        bt._cdp_fallback_local_session_keys.add(local_session_key)

        with patch(
            "tools.browser_tool._run_browser_command",
            return_value={"success": False, "error": "cannot blank"},
        ), patch("tools.browser_tool._cleanup_single_browser_session"):
            bt._neutralize_blocked_redirect(
                task_id,
                task_id,
                local_session_key,
            )

        served_sessions = []

        def followup_run(session_key, command, args=None, timeout=None, **kwargs):
            served_sessions.append(session_key)
            if command == "snapshot":
                return {
                    "success": True,
                    "data": {
                        "snapshot": '- heading "safe primary" [ref=e1]',
                        "refs": {"e1": {}},
                    },
                }
            if command == "eval":
                if tool_name == "get_images":
                    return {"success": True, "data": {"result": "[]"}}
                return {"success": True, "data": {"result": "1"}}
            if command == "screenshot":
                return {"success": False, "error": "stop after routing check"}
            raise AssertionError((session_key, command, args, timeout, kwargs))

        with patch("tools.browser_tool._run_browser_command", side_effect=followup_run), \
             patch("tools.browser_tool._is_local_backend", return_value=True), \
             patch("tools.browser_tool._is_camofox_mode", return_value=False), \
             patch("tools.browser_tool._get_browser_engine", return_value="auto"), \
             patch("tools.browser_tool._get_cloud_provider", return_value=None), \
             patch("hermes_constants.get_hermes_dir", return_value=tmp_path):
            if tool_name == "snapshot":
                bt.browser_snapshot(task_id=task_id)
            elif tool_name == "eval":
                bt.browser_console(expression="1", task_id=task_id)
            elif tool_name == "get_images":
                bt.browser_get_images(task_id=task_id)
            else:
                bt.browser_vision("routing only", task_id=task_id)

        assert local_session_key not in bt._active_sessions
        assert local_session_key not in bt._cdp_fallback_local_session_keys
        assert task_id not in bt._last_active_session_key
        assert served_sessions
        assert set(served_sessions) == {task_id}

    def test_later_snapshot_from_cdp_fallback_sidecar_remains_ssrf_guarded(
        self, monkeypatch
    ):
        """A persisted CDP fallback sidecar never inherits hybrid-private trust."""
        import tools.browser_tool as bt

        task_id = "persistent-fallback-guard"
        local_session_key = f"{task_id}::local"
        bt._active_sessions[local_session_key] = {
            "session_key": local_session_key,
            "owner_task_id": task_id,
            "session_name": "fallback-local",
            "features": {"local": True},
        }
        bt._last_active_session_key[task_id] = local_session_key
        monkeypatch.setattr(
            bt,
            "_cdp_fallback_local_session_keys",
            {local_session_key},
            raising=False,
        )
        secret = "METADATA_SECRET_CONTENT"

        def run_command(session_key, command, args=None, timeout=None, **kwargs):
            assert session_key == local_session_key
            if command == "snapshot":
                return {
                    "success": True,
                    "data": {
                        "snapshot": f'- heading "{secret}" [ref=e1]',
                        "refs": {"e1": {}},
                    },
                }
            if command == "eval":
                return {
                    "success": True,
                    "data": {
                        "result": "http://169.254.169.254/latest/meta-data"
                    },
                }
            if command == "open":
                return {"success": True, "data": {"url": "about:blank"}}
            raise AssertionError((session_key, command, args, kwargs))

        with patch("tools.browser_tool._run_browser_command", side_effect=run_command), \
             patch("tools.browser_tool._is_local_backend", return_value=False), \
             patch("tools.browser_tool._allow_private_urls", return_value=False):
            response = json.loads(bt.browser_snapshot(task_id=task_id))

        assert response["success"] is False
        assert "metadata endpoint" in response["error"]
        assert secret not in json.dumps(response)

    def test_failed_explicit_hybrid_navigation_retains_fallback_provenance(self):
        """Repurpose is committed only after the intentional hybrid open succeeds."""
        import tools.browser_tool as bt

        task_id = "hybrid-repurpose-failure"
        local_session_key = f"{task_id}::local"
        private_url = "http://127.0.0.1/admin"
        bt._cdp_fallback_local_session_keys.add(local_session_key)
        bt._set_last_active_session_key(task_id, task_id)

        with patch(
            "tools.browser_tool._navigation_session_key",
            return_value=local_session_key,
        ), patch("tools.browser_tool._is_local_backend", return_value=False), patch(
            "tools.browser_tool._allow_private_urls", return_value=False
        ), patch(
            "tools.browser_tool.check_website_access", return_value=None
        ), patch(
            "tools.browser_tool._is_camofox_mode", return_value=False
        ), patch(
            "tools.browser_tool._get_cloud_provider", return_value=None
        ), patch(
            "tools.browser_tool._get_session_info",
            return_value={
                "session_key": local_session_key,
                "owner_task_id": task_id,
                "session_name": "hybrid-local",
                "features": {"local": True},
                "_first_nav": False,
            },
        ), patch(
            "tools.browser_tool._run_browser_command",
            return_value={"success": False, "error": "local open failed"},
        ):
            response = json.loads(bt.browser_navigate(private_url, task_id=task_id))

        assert response["success"] is False
        assert local_session_key in bt._cdp_fallback_local_session_keys
        assert bt._last_active_session_key[task_id] == task_id

    def test_explicit_hybrid_private_navigation_drops_fallback_provenance(self):
        """Intentional hybrid-private routing may repurpose a fallback sidecar."""
        import tools.browser_tool as bt

        task_id = "hybrid-repurpose"
        local_session_key = f"{task_id}::local"
        private_url = "http://127.0.0.1/admin"
        bt._cdp_fallback_local_session_keys.add(local_session_key)
        nav_result = {
            "success": True,
            "data": {"title": "local admin", "url": private_url},
        }
        snapshot_result = {
            "success": True,
            "data": {"snapshot": '- heading "local admin" [ref=e1]', "refs": {"e1": {}}},
        }

        with patch(
            "tools.browser_tool._navigation_session_key",
            return_value=local_session_key,
        ), patch("tools.browser_tool._is_local_backend", return_value=False), patch(
            "tools.browser_tool._allow_private_urls", return_value=False
        ), patch(
            "tools.browser_tool.check_website_access", return_value=None
        ), patch(
            "tools.browser_tool._is_camofox_mode", return_value=False
        ), patch(
            "tools.browser_tool._get_cloud_provider", return_value=None
        ), patch(
            "tools.browser_tool._get_session_info",
            return_value={
                "session_key": local_session_key,
                "owner_task_id": task_id,
                "session_name": "hybrid-local",
                "features": {"local": True},
                "_first_nav": False,
            },
        ), patch(
            "tools.browser_tool._run_browser_command",
            side_effect=[nav_result, snapshot_result],
        ):
            response = json.loads(bt.browser_navigate(private_url, task_id=task_id))

        assert response["success"] is True
        assert local_session_key not in bt._cdp_fallback_local_session_keys

    def test_fallback_sidecar_retains_shared_eval_guard(self):
        """Eval/images/vision guard fallback sidecars but exempt hybrid sidecars."""
        import tools.browser_tool as bt

        fallback_key = "guard-provenance::local"
        hybrid_key = "hybrid-only::local"
        bt._cdp_fallback_local_session_keys.add(fallback_key)

        with patch("tools.browser_tool._is_local_backend", return_value=True), patch(
            "tools.browser_tool._allow_private_urls", return_value=False
        ):
            assert bt._eval_ssrf_guard_active(fallback_key) is True
            assert bt._eval_ssrf_guard_active(hybrid_key) is False
