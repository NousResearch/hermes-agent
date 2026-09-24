"""Tests for Lightpanda engine support in browser_tool.py."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from tools import browser_tool_lifecycle as bt_lifecycle
from tools import browser_tool_lightpanda_fallback as bt_lightpanda_fallback
from tools import browser_tool_session as bt_session
from tools import browser_tool_install as bt_install
from tools import browser_tool_cloud as bt_cloud
from tools import browser_tool_cdp as bt_cdp


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
        from tools.browser_tool_cloud import _get_browser_engine
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AGENT_BROWSER_ENGINE", None)
            with patch("hermes_cli.config.read_raw_config", return_value={}):
                assert _get_browser_engine() == "auto"

    def test_config_lightpanda(self):
        """Config browser.engine = 'lightpanda' is respected."""
        from tools.browser_tool_cloud import _get_browser_engine
        cfg = {"browser": {"engine": "lightpanda"}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert _get_browser_engine() == "lightpanda"


    def test_caching(self):
        """Result is cached — second call doesn't re-read config."""
        from tools.browser_tool_cloud import _get_browser_engine
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
        from tools.browser_tool_cloud import _should_inject_engine
        assert _should_inject_engine("auto") is False

    def test_lightpanda_injects_in_local_mode(self):
        from tools.browser_tool_cloud import _should_inject_engine
        with patch("tools.browser_tool._is_camofox_mode", return_value=False), \
             patch("tools.browser_tool_cdp._get_cdp_override", return_value=""), \
             patch("tools.browser_tool_cloud._get_cloud_provider", return_value=None):
            assert _should_inject_engine("lightpanda") is True

    def test_chrome_injects_in_local_mode(self):
        from tools.browser_tool_cloud import _should_inject_engine
        with patch("tools.browser_tool._is_camofox_mode", return_value=False), \
             patch("tools.browser_tool_cdp._get_cdp_override", return_value=""), \
             patch("tools.browser_tool_cloud._get_cloud_provider", return_value=None):
            assert _should_inject_engine("chrome") is True

    def test_no_inject_in_camofox_mode(self):
        from tools.browser_tool_cloud import _should_inject_engine
        with patch("tools.browser_tool._is_camofox_mode", return_value=True):
            assert _should_inject_engine("lightpanda") is False

    def test_no_inject_with_cdp_override(self):
        from tools.browser_tool_cloud import _should_inject_engine
        with patch("tools.browser_tool._is_camofox_mode", return_value=False), \
             patch("tools.browser_tool_cdp._get_cdp_override_raw", return_value="ws://localhost:9222"):
            assert _should_inject_engine("lightpanda") is False


# ---------------------------------------------------------------------------
# _needs_lightpanda_fallback
# ---------------------------------------------------------------------------

class TestNeedsLightpandaFallback:
    """Test fallback detection for Lightpanda results."""

    def test_non_lightpanda_never_falls_back(self):
        from tools.browser_tool_lightpanda_fallback import _needs_lightpanda_fallback
        result = {"success": False, "error": "timeout"}
        assert _needs_lightpanda_fallback("chrome", "open", result) is False
        assert _needs_lightpanda_fallback("auto", "open", result) is False

    def test_failed_command_triggers_fallback(self):
        from tools.browser_tool_lightpanda_fallback import _needs_lightpanda_fallback
        result = {"success": False, "error": "page.goto: Timeout"}
        assert _needs_lightpanda_fallback("lightpanda", "open", result) is True


    def test_empty_snapshot_triggers_fallback(self):
        from tools.browser_tool_lightpanda_fallback import _needs_lightpanda_fallback
        result = {"success": True, "data": {"snapshot": ""}}
        assert _needs_lightpanda_fallback("lightpanda", "snapshot", result) is True


    def test_unknown_command_does_not_trigger_fallback(self):
        """Commands not in the whitelist should not trigger fallback."""
        from tools.browser_tool_lightpanda_fallback import _needs_lightpanda_fallback
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

        with patch("tools.browser_tool._is_camofox_mode", return_value=False), \
             patch("tools.browser_tool_cdp._get_cdp_override", return_value=""), \
             patch("tools.browser_tool_install._find_agent_browser", return_value="/usr/bin/agent-browser"), \
             patch("tools.browser_tool_install._requires_real_termux_browser_install", return_value=False), \
             patch("tools.browser_tool_cloud._get_cloud_provider", return_value=None), \
             patch("tools.browser_tool_cloud._get_browser_engine", return_value="lightpanda"), \
             patch("tools.browser_tool_install._chromium_installed", return_value=False):
            assert bt_install.check_browser_requirements() is True

    def test_chrome_local_mode_still_requires_chromium(self):

        with patch("tools.browser_tool._is_camofox_mode", return_value=False), \
             patch("tools.browser_tool_cdp._get_cdp_override", return_value=""), \
             patch("tools.browser_tool_install._find_agent_browser", return_value="/usr/bin/agent-browser"), \
             patch("tools.browser_tool_install._requires_real_termux_browser_install", return_value=False), \
             patch("tools.browser_tool_cloud._get_cloud_provider", return_value=None), \
             patch("tools.browser_tool_cloud._get_browser_engine", return_value="auto"), \
             patch("tools.browser_tool_install._chromium_installed", return_value=False):
            assert bt_install.check_browser_requirements() is False


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
        bt_lifecycle.cleanup_all_browsers()
        assert bt._cached_browser_engine is None
        assert bt._browser_engine_resolved is False


# ---------------------------------------------------------------------------
# Chrome fallback behavior
# ---------------------------------------------------------------------------

class TestChromeFallback:
    """Chrome fallback must hand off from Lightpanda without leaking engine policy."""

    def test_uses_non_recursive_lightpanda_get_url(self):

        with patch("tools.browser_tool_session._run_browser_command", return_value={
                 "success": True, "data": {"url": "https://example.com/"}
             }) as run_command, \
             patch("tools.browser_tool_install._find_agent_browser", side_effect=FileNotFoundError("stop")):
            result = bt_lightpanda_fallback._run_chrome_fallback_command(
                "task1", "screenshot", [], timeout=30
            )

        run_command.assert_called_once_with(
            "task1", "get", ["url"], timeout=10, _engine_override="lightpanda"
        )
        assert result == {"success": False, "error": "stop"}

    def test_chrome_fallback_injects_required_sandbox_args(self, tmp_path):

        captured_envs = []
        mock_proc = MagicMock()
        mock_proc.wait.return_value = None
        mock_proc.returncode = 1

        def capture_popen(_cmd, **kwargs):
            captured_envs.append(kwargs["env"])
            return mock_proc

        # Keep the fallback's socket dir under this test's private tmp_path.
        # Using the real shared tmpdir raced concurrent orphan reapers from
        # sibling pytest processes (atexit _emergency_cleanup_all_sessions),
        # which rmtree'd the fresh pidless dir mid-command — the CI flake
        # this test kept hitting before the reaper grace fix.
        with patch("tools.browser_tool_session._run_browser_command", return_value={
                 "success": True, "data": {"url": "https://example.com/"}
             }), \
             patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)), \
             patch("tools.browser_tool_install._find_agent_browser", return_value="/usr/bin/agent-browser"), \
             patch("tools.browser_tool_install._chromium_installed", return_value=True), \
             patch("tools.browser_tool_session._needs_chromium_sandbox_bypass", return_value=True), \
             patch("subprocess.Popen", side_effect=capture_popen):
            result = bt_lightpanda_fallback._run_chrome_fallback_command(
                "task1", "screenshot", [], timeout=30
            )

        assert result["success"] is False
        assert captured_envs
        assert all(
            env.get("AGENT_BROWSER_ARGS") == "--no-sandbox,--disable-dev-shm-usage"
            for env in captured_envs
        )


# ---------------------------------------------------------------------------
# fallback warning annotation
# ---------------------------------------------------------------------------

class TestLightpandaFallbackWarning:
    """Verify Chrome fallback results are annotated for users."""

    def test_fallback_result_gets_user_visible_warning(self):
        from tools.browser_tool_lightpanda_fallback import _annotate_lightpanda_fallback

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

        result = bt_lightpanda_fallback._annotate_lightpanda_fallback(
            {"success": True, "data": {"title": "Fallback OK", "url": "https://example.com/"}},
            "synthetic Lightpanda failure; retried with Chrome.",
        )

        with patch("tools.browser_tool_cloud._is_local_backend", return_value=True), \
             patch("tools.browser_tool_cloud._get_cloud_provider", return_value=None), \
             patch("tools.browser_tool_session._get_session_info", return_value={
                 "session_name": "test", "_first_nav": False, "features": {"local": True, "proxies": True}
             }), \
             patch("tools.browser_tool_session._run_browser_command", side_effect=[
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

        with patch("tools.browser_tool_cloud._get_browser_engine", return_value="lightpanda"), \
             patch("tools.browser_tool_cloud._should_inject_engine", return_value=True), \
             patch("tools.browser_tool_lightpanda_fallback._chrome_fallback_screenshot", return_value={
                 "success": True, "data": {"path": str(chrome_shot)}
             }), \
             patch("hermes_constants.get_hermes_dir", return_value=tmp_path), \
             patch("agent.auxiliary_client.call_llm", return_value=_Response()):
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

    @patch("tools.browser_tool_session._get_session_info")
    @patch("tools.browser_tool_install._find_agent_browser", return_value="/usr/bin/agent-browser")
    @patch("tools.browser_tool_cloud._is_local_mode", return_value=True)
    @patch("tools.browser_tool_install._chromium_installed", return_value=True)
    @patch("tools.browser_tool_cloud._get_cloud_provider", return_value=None)
    @patch("tools.browser_tool_cdp._get_cdp_override", return_value="")
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
             patch("tools.browser_tool_lifecycle._write_owner_pid"):
            bt_session._run_browser_command("task1", "snapshot", [], _engine_override="auto")

        # Should NOT contain "--engine" since override is "auto"
        assert len(captured_cmds) == 1
        assert "--engine" not in captured_cmds[0]

    @patch("tools.browser_tool_session._get_session_info")
    @patch("tools.browser_tool_install._find_agent_browser", return_value="/usr/bin/agent-browser")
    @patch("tools.browser_tool_cloud._is_local_mode", return_value=True)
    @patch("tools.browser_tool_install._chromium_installed", return_value=True)
    @patch("tools.browser_tool_cloud._get_cloud_provider", return_value=None)
    @patch("tools.browser_tool_cdp._get_cdp_override", return_value="")
    @patch("tools.browser_tool._is_camofox_mode", return_value=False)
    def test_no_override_uses_cached_engine(
        self, _camofox, _cdp, _cloud, _chromium, _local, _find, _session
    ):
        """Lightpanda gets neither auto-injected nor inherited Chrome arguments."""
        import tools.browser_tool as bt

        bt._cached_browser_engine = "lightpanda"
        bt._browser_engine_resolved = True

        _session.return_value = {"session_name": "test-sess"}

        captured_cmds = []
        captured_envs = []
        mock_proc = MagicMock()
        mock_proc.wait.return_value = None
        mock_proc.returncode = 0

        def capture_popen(cmd, **kwargs):
            captured_cmds.append(cmd)
            captured_envs.append(kwargs["env"])
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
             patch("tools.browser_tool_session._needs_chromium_sandbox_bypass", return_value=True), \
             patch("tools.browser_tool_lifecycle._write_owner_pid"), \
             patch.dict(os.environ, {}, clear=True):
            # AppArmor/root detection would normally auto-inject Chromium args.
            bt_session._run_browser_command("task1", "snapshot", [])

            # User-supplied current and legacy Chromium knobs must also be removed.
            with patch.dict(os.environ, {
                "AGENT_BROWSER_ARGS": "--no-sandbox",
                "AGENT_BROWSER_CHROME_FLAGS": "--disable-dev-shm-usage",
            }):
                bt_session._run_browser_command("task1", "snapshot", [])

        assert len(captured_cmds) == 2
        for command, environment in zip(captured_cmds, captured_envs):
            assert "--engine" in command
            engine_idx = command.index("--engine")
            assert command[engine_idx + 1] == "lightpanda"
            assert "AGENT_BROWSER_ARGS" not in environment
            assert "AGENT_BROWSER_CHROME_FLAGS" not in environment

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
        with patch("tools.browser_tool_session._get_session_info", return_value={"session_name": "local-sidecar"}), \
             patch("tools.browser_tool_install._find_agent_browser", return_value="/usr/bin/agent-browser"), \
             patch("tools.browser_tool_cloud._is_local_mode", return_value=False), \
             patch("tools.browser_tool_install._chromium_installed", return_value=True), \
             patch("tools.browser_tool_cloud._get_cloud_provider", return_value=mock_provider), \
             patch("tools.browser_tool_cdp._get_cdp_override", return_value=""), \
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
             patch("tools.browser_tool_lifecycle._write_owner_pid"):
            bt_session._run_browser_command("task::local", "snapshot", [])

        assert len(captured_cmds) == 1
        assert "--engine" in captured_cmds[0]
        assert captured_cmds[0][captured_cmds[0].index("--engine") + 1] == "lightpanda"


# ---------------------------------------------------------------------------
# lightpanda_engine_status — is the engine in effect, or shadowed?
# ---------------------------------------------------------------------------

class TestLightpandaEngineStatus:
    def _gates(self, monkeypatch, **overrides):
        import tools.browser_tool as bt

        gates = dict(
            _using_lightpanda_engine=lambda: True,
            _get_cdp_override_raw=lambda: "",
            _is_camofox_mode=lambda: False,
            _get_cloud_provider=lambda: None,
            _is_browser_use_cli_mode=lambda: True,
            _use_real_profile=lambda: False,
        )
        gates.update(overrides)
        homes = {
            "_using_lightpanda_engine": bt_lightpanda_fallback, "_get_cdp_override_raw": bt_cdp,
            "_get_cloud_provider": bt_cloud, "_use_real_profile": bt_cloud,
        }
        for name, fn in gates.items():
            monkeypatch.setattr(homes.get(name, bt), name, fn)
        monkeypatch.setattr(
            "tools.browser_use_cli.is_legacy_browser_use_cloud_config", lambda cfg: False
        )
        return bt

    def test_not_lightpanda(self, monkeypatch):
        bt = self._gates(monkeypatch, _using_lightpanda_engine=lambda: False)
        assert bt_lightpanda_fallback.lightpanda_engine_status() == (False, "")

    def test_used_in_browser_use_mode(self, monkeypatch):
        bt = self._gates(monkeypatch)
        used, reason = bt_lightpanda_fallback.lightpanda_engine_status()
        assert used is True
        assert "lightpanda serve" in reason

    def test_used_with_builtin_tools(self, monkeypatch):
        bt = self._gates(monkeypatch, _is_browser_use_cli_mode=lambda: False)
        used, reason = bt_lightpanda_fallback.lightpanda_engine_status()
        assert used is True
        assert "--engine lightpanda" in reason

    def test_shadowed_by_cdp_override(self, monkeypatch):
        bt = self._gates(monkeypatch, _get_cdp_override_raw=lambda: "ws://x")
        used, reason = bt_lightpanda_fallback.lightpanda_engine_status()
        assert used is False and "CDP override" in reason

    def test_shadowed_by_camofox(self, monkeypatch):
        bt = self._gates(monkeypatch, _is_camofox_mode=lambda: True)
        used, reason = bt_lightpanda_fallback.lightpanda_engine_status()
        assert used is False and "Camofox" in reason

    def test_shadowed_by_cloud_provider(self, monkeypatch):
        provider = MagicMock()
        provider.display_name = "Browserbase"
        bt = self._gates(monkeypatch, _get_cloud_provider=lambda: provider)
        used, reason = bt_lightpanda_fallback.lightpanda_engine_status()
        assert used is False and "Browserbase" in reason

    def test_shadowed_by_legacy_browser_use_cloud(self, monkeypatch):
        bt = self._gates(monkeypatch)
        monkeypatch.setattr(
            "tools.browser_use_cli.is_legacy_browser_use_cloud_config", lambda cfg: True
        )
        used, reason = bt_lightpanda_fallback.lightpanda_engine_status()
        assert used is False and "Browser Use cloud" in reason

    def test_shadowed_by_real_profile(self, monkeypatch):
        bt = self._gates(monkeypatch, _use_real_profile=lambda: True)
        used, reason = bt_lightpanda_fallback.lightpanda_engine_status()
        assert used is False and "use_real_profile" in reason

    def test_real_profile_wins_over_cloud_provider(self, monkeypatch):
        """browser_exec resolves real-profile before the backend, so with
        both set the real-profile toggle is the actual shadow."""
        provider = MagicMock()
        provider.display_name = "Browserbase"
        bt = self._gates(
            monkeypatch,
            _use_real_profile=lambda: True,
            _get_cloud_provider=lambda: provider,
        )
        used, reason = bt_lightpanda_fallback.lightpanda_engine_status()
        assert used is False and "use_real_profile" in reason


# ---------------------------------------------------------------------------
# Browser Use mode session lifecycle
# ---------------------------------------------------------------------------

class _FakeServer:
    def __init__(self, port=4321, alive=True):
        self.port = port
        self.cdp_url = f"http://127.0.0.1:{port}"
        self._alive = alive

    def is_alive(self):
        return self._alive


class TestLightpandaSessionCreation:
    def _common(self, monkeypatch, *, bu_mode=True, local_backend=True, launch=None):
        import tools.browser_tool as bt

        calls = []

        def fake_launch(session_name, *, block_private_networks=False):
            calls.append((session_name, block_private_networks))
            if launch is not None:
                return launch
            return _FakeServer(), None

        monkeypatch.setattr("tools.browser_tool_real_profile._real_profile_cdp", lambda: (None, None))
        monkeypatch.setattr(bt, "_is_browser_use_cli_mode", lambda: bu_mode)
        monkeypatch.setattr("tools.browser_tool_lightpanda_fallback._using_lightpanda_engine", lambda: True)
        monkeypatch.setattr(bt_cloud, "_is_local_backend", lambda: local_backend)
        monkeypatch.setattr("tools.browser_lightpanda.launch_lightpanda", fake_launch)
        return bt, calls

    def test_spawns_lightpanda_in_browser_use_mode(self, monkeypatch):
        bt, calls = self._common(monkeypatch)
        info = bt_session._create_local_session("task-1")
        assert info["session_name"].startswith("lp_")
        assert info["cdp_url"] == "http://127.0.0.1:4321"
        assert info["features"] == {"local": True, "lightpanda": True}
        assert info["bb_session_id"] is None
        assert calls == [(info["session_name"], False)]

    def test_blocks_private_networks_for_containerised_terminal(self, monkeypatch):
        bt, calls = self._common(monkeypatch, local_backend=False)
        bt_session._create_local_session("task-1")
        assert calls[0][1] is True

    def test_ignores_engine_outside_browser_use_mode(self, monkeypatch):
        bt, calls = self._common(monkeypatch, bu_mode=False)
        info = bt_session._create_local_session("task-1")
        assert info["features"] == {"local": True}
        assert info["cdp_url"] is None
        assert calls == []

    def test_launch_failure_raises(self, monkeypatch):
        bt, _ = self._common(monkeypatch, launch=(None, "no lightpanda binary was found"))
        with pytest.raises(RuntimeError, match="no lightpanda binary"):
            bt_session._create_local_session("task-1")


class TestLightpandaSessionLifecycle:
    def setup_method(self):
        import tools.browser_tool as bt

        self.bt = bt
        self.orig_sessions = bt._active_sessions.copy()
        self.orig_activity = bt._session_last_activity.copy()
        self.orig_cleanup_done = bt._cleanup_done
        bt._active_sessions.clear()
        bt._session_last_activity.clear()

    def teardown_method(self):
        bt = self.bt
        bt._active_sessions.clear()
        bt._active_sessions.update(self.orig_sessions)
        bt._session_last_activity.clear()
        bt._session_last_activity.update(self.orig_activity)
        bt._cleanup_done = self.orig_cleanup_done

    def _seed(self, key="task-1", name="lp_dead"):
        info = {
            "session_name": name,
            "bb_session_id": None,
            "cdp_url": "http://127.0.0.1:1",
            "features": {"local": True, "lightpanda": True},
        }
        self.bt._active_sessions[key] = info
        self.bt._session_last_activity[key] = 1.0
        return info

    def test_dead_process_is_detected(self, monkeypatch):
        info = self._seed()
        monkeypatch.setattr("tools.browser_lightpanda.get_server", lambda name: None)
        assert bt_session._local_backend_process_dead(info) is True
        monkeypatch.setattr(
            "tools.browser_lightpanda.get_server", lambda name: _FakeServer(alive=False)
        )
        assert bt_session._local_backend_process_dead(info) is True
        monkeypatch.setattr("tools.browser_lightpanda.get_server", lambda name: _FakeServer())
        assert bt_session._local_backend_process_dead(info) is False
        assert bt_session._local_backend_process_dead({"features": {"local": True}}) is False

    def test_get_session_info_respawns_dead_lightpanda(self, monkeypatch):
        bt = self.bt
        stale = self._seed()
        fresh = {
            "session_name": "lp_fresh",
            "bb_session_id": None,
            "cdp_url": "http://127.0.0.1:2",
            "features": {"local": True, "lightpanda": True},
        }
        cleaned = []

        def fake_cleanup(key):
            cleaned.append(key)
            bt._active_sessions.pop(key, None)

        monkeypatch.setattr("tools.browser_tool_lifecycle._start_browser_cleanup_thread", lambda: None)
        monkeypatch.setattr(
            bt, "_browser_session_backend",
            lambda key: MagicMock(ensure_healthy=lambda: True),
        )
        monkeypatch.setattr("tools.browser_lightpanda.get_server", lambda name: None)
        monkeypatch.setattr(bt_lifecycle, "_cleanup_single_browser_session", fake_cleanup)
        monkeypatch.setattr("tools.browser_tool_cdp._get_cdp_override", lambda: "")
        monkeypatch.setattr(bt_cloud, "_get_cloud_provider", lambda: None)
        monkeypatch.setattr("tools.browser_tool_session._create_local_session", lambda *a, **k: fresh)
        supervised = []
        monkeypatch.setattr("tools.browser_tool_cdp._ensure_cdp_supervisor", supervised.append)

        info = bt_session._get_session_info("task-1")
        assert cleaned == ["task-1"]
        assert info["session_name"] == "lp_fresh"
        assert bt._active_sessions["task-1"]["session_name"] == "lp_fresh"
        assert info["session_name"] != stale["session_name"]
        # Browser Use mode hides the browser_* tools that read supervisor
        # state; a Lightpanda session never attaches one.
        assert supervised == []

    def test_cleanup_stops_lightpanda_without_agent_browser_close(self, monkeypatch):
        bt = self.bt
        self._seed()
        stopped = []
        monkeypatch.setattr("tools.browser_lightpanda.stop_lightpanda", stopped.append)
        with patch("tools.browser_tool._maybe_stop_recording"), \
             patch("tools.browser_tool_session._run_browser_command") as run, \
             patch("tools.browser_tool.os.path.exists", return_value=False):
            bt_lifecycle.cleanup_browser("task-1")
        run.assert_not_called()
        assert stopped == ["lp_dead"]
        assert "task-1" not in bt._active_sessions
        assert "task-1" not in bt._session_last_activity

    def test_emergency_cleanup_stops_all_lightpanda(self, monkeypatch):
        bt = self.bt
        bt._cleanup_done = False
        with patch("tools.browser_lightpanda.stop_all_lightpanda") as stop_all, \
             patch("tools.browser_tool_real_profile._terminate_real_profile_chrome"), \
             patch("tools.browser_tool_lifecycle.cleanup_all_browsers"), \
             patch("tools.browser_tool_lifecycle._reap_orphaned_browser_sessions"):
            bt_lifecycle._emergency_cleanup_all_sessions()
        stop_all.assert_called_once()

    def test_orphan_reaper_sweeps_lightpanda_records(self, tmp_path):
        with patch("tools.browser_lightpanda.reap_orphaned_lightpanda") as reap, \
             patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)):
            bt_lifecycle._reap_orphaned_browser_sessions()
        reap.assert_called_once()
