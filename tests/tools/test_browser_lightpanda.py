"""Tests for the LightpandaProvider browser backend.

Covers:
  - is_configured(): binary discovery via PATH / env var / node_modules
  - create_session(): happy path, startup timeout, missing binary
  - close_session(): graceful terminate, SIGKILL fallback, unknown ID
  - emergency_cleanup(): no-raise guarantee
  - provider registry: "lightpanda" key wired up in browser_tool
  - auto-detection: Lightpanda chosen when binary present and no cloud creds
  - _find_free_port(): returns a usable port number
  - _wait_for_cdp(): success and timeout paths
"""

import subprocess
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_popen_mock():
    proc = MagicMock(spec=subprocess.Popen)
    proc.kill.return_value = None
    proc.terminate.return_value = None
    proc.wait.return_value = 0
    return proc


# ---------------------------------------------------------------------------
# is_configured
# ---------------------------------------------------------------------------

class TestIsConfigured:
    def test_true_when_cdp_url_env_set(self, monkeypatch):
        from tools.browser_providers.lightpanda import LightpandaProvider

        monkeypatch.setenv("LIGHTPANDA_CDP_URL", "ws://127.0.0.1:9222/")
        assert LightpandaProvider().is_configured() is True

    def test_true_when_binary_in_path(self):
        from tools.browser_providers.lightpanda import LightpandaProvider

        with patch("tools.browser_providers.lightpanda.shutil.which", return_value="/usr/bin/lightpanda"):
            assert LightpandaProvider().is_configured() is True

    def test_false_when_nothing_found(self, tmp_path, monkeypatch):
        from tools.browser_providers.lightpanda import LightpandaProvider

        monkeypatch.delenv("LIGHTPANDA_PATH", raising=False)
        with patch("tools.browser_providers.lightpanda.shutil.which", return_value=None), \
             patch.object(Path, "exists", return_value=False):
            assert LightpandaProvider().is_configured() is False

    def test_true_when_lightpanda_path_env_points_to_executable(self, tmp_path, monkeypatch):
        from tools.browser_providers.lightpanda import LightpandaProvider

        binary = tmp_path / "lightpanda"
        binary.touch()
        binary.chmod(0o755)
        monkeypatch.setenv("LIGHTPANDA_PATH", str(binary))
        with patch("tools.browser_providers.lightpanda.shutil.which", return_value=None):
            assert LightpandaProvider().is_configured() is True

    def test_false_when_lightpanda_path_env_points_to_nonexistent_file(self, monkeypatch):
        from tools.browser_providers.lightpanda import LightpandaProvider

        monkeypatch.setenv("LIGHTPANDA_PATH", "/nonexistent/lightpanda")
        with patch("tools.browser_providers.lightpanda.shutil.which", return_value=None), \
             patch.object(Path, "exists", return_value=False):
            assert LightpandaProvider().is_configured() is False

    def test_true_when_found_in_local_node_modules(self, tmp_path, monkeypatch):
        from tools.browser_providers.lightpanda import LightpandaProvider

        # Build a fake node_modules/.bin/lightpanda alongside the repo root
        fake_bin = tmp_path / "node_modules" / ".bin" / "lightpanda"
        fake_bin.parent.mkdir(parents=True)
        fake_bin.touch()
        fake_bin.chmod(0o755)

        monkeypatch.delenv("LIGHTPANDA_PATH", raising=False)
        with patch("tools.browser_providers.lightpanda.shutil.which", return_value=None), \
             patch.object(
                 Path, "resolve",
                 return_value=tmp_path / "tools" / "browser_providers" / "lightpanda.py",
             ):
            assert LightpandaProvider().is_configured() is True


# ---------------------------------------------------------------------------
# create_session
# ---------------------------------------------------------------------------

class TestCreateSession:
    def test_success_returns_correct_dict(self):
        from tools.browser_providers.lightpanda import LightpandaProvider

        provider = LightpandaProvider()
        mock_proc = _make_popen_mock()

        with patch.object(provider, "_find_binary", return_value="/usr/bin/lightpanda"), \
             patch.object(provider, "_pick_port", return_value=19222), \
             patch("tools.browser_providers.lightpanda.subprocess.Popen", return_value=mock_proc), \
             patch.object(LightpandaProvider, "_wait_for_cdp"):
            result = provider.create_session("task_abc")

        assert result["cdp_url"] == "ws://127.0.0.1:19222"
        assert result["session_name"].startswith("lightpanda_task_abc_")
        assert result["bb_session_id"] is not None
        assert result["features"] == {"lightpanda": True}

    def test_session_is_tracked_in_processes(self):
        from tools.browser_providers.lightpanda import LightpandaProvider

        provider = LightpandaProvider()
        mock_proc = _make_popen_mock()

        with patch.object(provider, "_find_binary", return_value="/usr/bin/lightpanda"), \
             patch.object(provider, "_pick_port", return_value=19223), \
             patch("tools.browser_providers.lightpanda.subprocess.Popen", return_value=mock_proc), \
             patch.object(LightpandaProvider, "_wait_for_cdp"):
            result = provider.create_session("task_track")

        assert result["bb_session_id"] in provider._processes
        assert provider._processes[result["bb_session_id"]] is mock_proc

    def test_kills_proc_and_reraises_on_startup_timeout(self):
        from tools.browser_providers.lightpanda import LightpandaProvider

        provider = LightpandaProvider()
        mock_proc = _make_popen_mock()

        with patch.object(provider, "_find_binary", return_value="/usr/bin/lightpanda"), \
             patch.object(provider, "_pick_port", return_value=19224), \
             patch("tools.browser_providers.lightpanda.subprocess.Popen", return_value=mock_proc), \
             patch.object(LightpandaProvider, "_wait_for_cdp",
                          side_effect=RuntimeError("CDP never ready")):
            with pytest.raises(RuntimeError, match="CDP never ready"):
                provider.create_session("task_timeout")

        mock_proc.kill.assert_called_once()
        mock_proc.wait.assert_called_once()
        assert len(provider._processes) == 0

    def test_raises_runtime_error_when_binary_missing(self):
        from tools.browser_providers.lightpanda import LightpandaProvider

        provider = LightpandaProvider()
        with patch.object(provider, "_find_binary", return_value=None):
            with pytest.raises(RuntimeError, match="lightpanda binary not found"):
                provider.create_session("task_no_bin")

    def test_custom_cdp_host_from_env(self, monkeypatch):
        from tools.browser_providers.lightpanda import LightpandaProvider

        provider = LightpandaProvider()
        mock_proc = _make_popen_mock()
        monkeypatch.setenv("LIGHTPANDA_CDP_HOST", "0.0.0.0")

        with patch.object(provider, "_find_binary", return_value="/usr/bin/lightpanda"), \
             patch.object(provider, "_pick_port", return_value=19225), \
             patch("tools.browser_providers.lightpanda.subprocess.Popen", return_value=mock_proc) as mock_popen, \
             patch.object(LightpandaProvider, "_wait_for_cdp"):
            result = provider.create_session("task_host")

        assert result["cdp_url"] == "ws://0.0.0.0:19225"
        argv = mock_popen.call_args[0][0]
        assert "--host" in argv
        assert "0.0.0.0" in argv

    def test_popen_called_with_serve_subcommand(self):
        from tools.browser_providers.lightpanda import LightpandaProvider

        provider = LightpandaProvider()
        mock_proc = _make_popen_mock()

        with patch.object(provider, "_find_binary", return_value="/usr/bin/lightpanda"), \
             patch.object(provider, "_pick_port", return_value=19226), \
             patch("tools.browser_providers.lightpanda.subprocess.Popen", return_value=mock_proc) as mock_popen, \
             patch.object(LightpandaProvider, "_wait_for_cdp"):
            provider.create_session("task_cmd")

        argv = mock_popen.call_args[0][0]
        assert argv[0] == "/usr/bin/lightpanda"
        assert "serve" in argv
        assert "--port" in argv
        assert "19226" in argv

    def test_external_cdp_url_skips_binary_spawn(self, monkeypatch):
        from tools.browser_providers.lightpanda import LightpandaProvider

        monkeypatch.setenv("LIGHTPANDA_CDP_URL", "ws://10.0.0.5:9222/")
        provider = LightpandaProvider()
        result = provider.create_session("task_ext")

        assert result["cdp_url"] == "ws://10.0.0.5:9222/"
        assert result["features"] == {"lightpanda": True, "external": True}
        assert result["session_name"].startswith("lightpanda_task_ext_")
        # No process should be tracked for external sessions
        assert len(provider._processes) == 0

    def test_external_cdp_url_close_session_returns_true(self, monkeypatch):
        from tools.browser_providers.lightpanda import LightpandaProvider

        monkeypatch.setenv("LIGHTPANDA_CDP_URL", "ws://10.0.0.5:9222/")
        provider = LightpandaProvider()
        result = provider.create_session("task_ext_close")
        assert provider.close_session(result["bb_session_id"]) is True


# ---------------------------------------------------------------------------
# _build_serve_cmd
# ---------------------------------------------------------------------------

class TestBuildServeCmd:
    def test_returns_expected_argv(self):
        from tools.browser_providers.lightpanda import LightpandaProvider

        cmd = LightpandaProvider._build_serve_cmd("/usr/bin/lightpanda", "127.0.0.1", 9222)
        assert cmd == ["/usr/bin/lightpanda", "serve", "--host", "127.0.0.1", "--port", "9222"]


# ---------------------------------------------------------------------------
# close_session
# ---------------------------------------------------------------------------

class TestCloseSession:
    def test_success_terminates_process(self):
        from tools.browser_providers.lightpanda import LightpandaProvider

        provider = LightpandaProvider()
        mock_proc = _make_popen_mock()
        provider._processes["sess_001"] = mock_proc

        result = provider.close_session("sess_001")

        assert result is True
        mock_proc.terminate.assert_called_once()
        assert "sess_001" not in provider._processes

    def test_unknown_session_id_returns_false(self):
        from tools.browser_providers.lightpanda import LightpandaProvider

        provider = LightpandaProvider()
        assert provider.close_session("does_not_exist") is False

    def test_falls_back_to_kill_on_terminate_timeout(self):
        from tools.browser_providers.lightpanda import LightpandaProvider

        provider = LightpandaProvider()
        mock_proc = _make_popen_mock()
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="lightpanda", timeout=5),
            0,  # second wait() after kill()
        ]
        provider._processes["sess_kill"] = mock_proc

        result = provider.close_session("sess_kill")

        assert result is True
        mock_proc.kill.assert_called_once()

    def test_process_removed_from_tracking_after_close(self):
        from tools.browser_providers.lightpanda import LightpandaProvider

        provider = LightpandaProvider()
        mock_proc = _make_popen_mock()
        provider._processes["sess_rm"] = mock_proc

        provider.close_session("sess_rm")

        assert "sess_rm" not in provider._processes


# ---------------------------------------------------------------------------
# emergency_cleanup
# ---------------------------------------------------------------------------

class TestEmergencyCleanup:
    def test_kills_tracked_process(self):
        from tools.browser_providers.lightpanda import LightpandaProvider

        provider = LightpandaProvider()
        mock_proc = _make_popen_mock()
        provider._processes["sess_em"] = mock_proc

        provider.emergency_cleanup("sess_em")

        mock_proc.kill.assert_called_once()
        assert "sess_em" not in provider._processes

    def test_does_not_raise_when_kill_fails(self):
        from tools.browser_providers.lightpanda import LightpandaProvider

        provider = LightpandaProvider()
        mock_proc = _make_popen_mock()
        mock_proc.kill.side_effect = OSError("no such process")
        provider._processes["sess_oserr"] = mock_proc

        provider.emergency_cleanup("sess_oserr")  # must not raise

    def test_does_not_raise_for_unknown_session(self):
        from tools.browser_providers.lightpanda import LightpandaProvider

        provider = LightpandaProvider()
        provider.emergency_cleanup("nonexistent")  # must not raise


# ---------------------------------------------------------------------------
# is_session_alive
# ---------------------------------------------------------------------------

class TestIsSessionAlive:
    def test_returns_true_for_running_process(self):
        from tools.browser_providers.lightpanda import LightpandaProvider

        provider = LightpandaProvider()
        mock_proc = _make_popen_mock()
        mock_proc.poll.return_value = None  # still running
        provider._processes["sess_alive"] = mock_proc

        assert provider.is_session_alive("sess_alive") is True

    def test_returns_false_for_dead_process(self):
        from tools.browser_providers.lightpanda import LightpandaProvider

        provider = LightpandaProvider()
        mock_proc = _make_popen_mock()
        mock_proc.poll.return_value = -11  # SIGSEGV
        provider._processes["sess_dead"] = mock_proc

        assert provider.is_session_alive("sess_dead") is False

    def test_returns_true_for_unknown_session(self):
        from tools.browser_providers.lightpanda import LightpandaProvider

        provider = LightpandaProvider()
        # Unknown session IDs return True (may be external-URL mode)
        assert provider.is_session_alive("unknown_id") is True


# ---------------------------------------------------------------------------
# provider_name
# ---------------------------------------------------------------------------

class TestProviderName:
    def test_returns_lightpanda(self):
        from tools.browser_providers.lightpanda import LightpandaProvider

        assert LightpandaProvider().provider_name() == "Lightpanda"


# ---------------------------------------------------------------------------
# Provider registry wiring
# ---------------------------------------------------------------------------

class TestProviderRegistry:
    def test_lightpanda_key_present(self):
        import tools.browser_tool as bt

        assert "lightpanda" in bt._PROVIDER_REGISTRY

    def test_registry_maps_to_lightpanda_provider_class(self):
        import tools.browser_tool as bt
        from tools.browser_providers.lightpanda import LightpandaProvider

        assert bt._PROVIDER_REGISTRY["lightpanda"] is LightpandaProvider

    def test_explicit_config_selects_lightpanda(self, monkeypatch):
        """Setting cloud_provider: lightpanda in config returns a LightpandaProvider."""
        import tools.browser_tool as bt
        from tools.browser_providers.lightpanda import LightpandaProvider

        monkeypatch.setattr(bt, "_cloud_provider_resolved", False)
        monkeypatch.setattr(bt, "_cached_cloud_provider", None)

        fake_cfg = {"browser": {"cloud_provider": "lightpanda"}}
        with patch("hermes_cli.config.read_raw_config", return_value=fake_cfg):
            provider = bt._get_cloud_provider()

        assert isinstance(provider, LightpandaProvider)

        # Reset cache for other tests
        monkeypatch.setattr(bt, "_cloud_provider_resolved", False)
        monkeypatch.setattr(bt, "_cached_cloud_provider", None)


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

class TestAutoDetection:
    def _clear_cache(self, monkeypatch):
        import tools.browser_tool as bt
        monkeypatch.setattr(bt, "_cloud_provider_resolved", False)
        monkeypatch.setattr(bt, "_cached_cloud_provider", None)

    def test_lightpanda_chosen_when_binary_present_and_no_cloud_creds(self, monkeypatch):
        import tools.browser_tool as bt
        from tools.browser_providers.lightpanda import LightpandaProvider

        self._clear_cache(monkeypatch)
        monkeypatch.delenv("BROWSER_USE_API_KEY", raising=False)
        monkeypatch.delenv("HERMES_ENABLE_NOUS_MANAGED_TOOLS", raising=False)
        monkeypatch.delenv("BROWSERBASE_API_KEY", raising=False)
        monkeypatch.delenv("BROWSERBASE_PROJECT_ID", raising=False)

        with patch("tools.browser_providers.lightpanda.shutil.which",
                   return_value="/usr/bin/lightpanda"), \
             patch("hermes_cli.config.read_raw_config", return_value={}):
            provider = bt._get_cloud_provider()

        assert isinstance(provider, LightpandaProvider)
        self._clear_cache(monkeypatch)

    def test_lightpanda_not_chosen_when_binary_absent(self, monkeypatch):
        import tools.browser_tool as bt

        self._clear_cache(monkeypatch)
        monkeypatch.delenv("BROWSER_USE_API_KEY", raising=False)
        monkeypatch.delenv("HERMES_ENABLE_NOUS_MANAGED_TOOLS", raising=False)
        monkeypatch.delenv("BROWSERBASE_API_KEY", raising=False)
        monkeypatch.delenv("BROWSERBASE_PROJECT_ID", raising=False)
        monkeypatch.delenv("LIGHTPANDA_PATH", raising=False)

        with patch("tools.browser_providers.lightpanda.shutil.which", return_value=None), \
             patch.object(Path, "exists", return_value=False), \
             patch("hermes_cli.config.read_raw_config", return_value={}):
            provider = bt._get_cloud_provider()

        assert provider is None
        self._clear_cache(monkeypatch)

    def test_lightpanda_not_chosen_when_browseruse_configured(self, monkeypatch):
        """BrowserUse takes priority over Lightpanda in auto-detection."""
        import tools.browser_tool as bt
        from tools.browser_providers.browser_use import BrowserUseProvider

        self._clear_cache(monkeypatch)
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test_key")

        with patch("tools.browser_providers.lightpanda.shutil.which",
                   return_value="/usr/bin/lightpanda"), \
             patch("hermes_cli.config.read_raw_config", return_value={}):
            provider = bt._get_cloud_provider()

        assert isinstance(provider, BrowserUseProvider)
        self._clear_cache(monkeypatch)


# ---------------------------------------------------------------------------
# _find_free_port
# ---------------------------------------------------------------------------

class TestFindFreePort:
    def test_returns_integer_in_valid_range(self):
        from tools.browser_providers.lightpanda import _find_free_port

        port = _find_free_port()
        assert isinstance(port, int)
        assert 1024 <= port <= 65535

    def test_port_is_not_bound_after_call(self):
        """The socket should be released so the caller can bind to the port."""
        import socket
        from tools.browser_providers.lightpanda import _find_free_port

        port = _find_free_port()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Should not raise — port should be free
            s.bind(("127.0.0.1", port))


# ---------------------------------------------------------------------------
# _wait_for_cdp
# ---------------------------------------------------------------------------

class TestWaitForCdp:
    def test_returns_immediately_on_first_success(self):
        from tools.browser_providers.lightpanda import LightpandaProvider

        with patch("urllib.request.urlopen"):
            LightpandaProvider._wait_for_cdp("127.0.0.1", 9222, timeout=5.0)

    def test_retries_and_eventually_succeeds(self):
        from tools.browser_providers.lightpanda import LightpandaProvider

        attempts = [0]

        def urlopen_side_effect(url, timeout):
            attempts[0] += 1
            if attempts[0] < 3:
                raise urllib.error.URLError("not ready yet")

        with patch("urllib.request.urlopen", side_effect=urlopen_side_effect), \
             patch("tools.browser_providers.lightpanda.time.sleep"):
            LightpandaProvider._wait_for_cdp("127.0.0.1", 9222, timeout=5.0)

        assert attempts[0] == 3

    def test_raises_runtime_error_on_timeout(self):
        from tools.browser_providers.lightpanda import LightpandaProvider

        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("connection refused")):
            with pytest.raises(RuntimeError, match="did not become ready"):
                # timeout=0 means deadline is immediately in the past after first failure
                LightpandaProvider._wait_for_cdp("127.0.0.1", 9222, timeout=0)

    def test_error_message_includes_host_and_port(self):
        from tools.browser_providers.lightpanda import LightpandaProvider

        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("refused")):
            with pytest.raises(RuntimeError) as exc_info:
                LightpandaProvider._wait_for_cdp("192.168.1.5", 8765, timeout=0)

        assert "192.168.1.5" in str(exc_info.value)
        assert "8765" in str(exc_info.value)
