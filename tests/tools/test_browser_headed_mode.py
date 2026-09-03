"""Tests for headed browser mode: config/env resolution, --headed injection,
and the per-turn cleanup skip that keeps headed sessions alive between turns.

Salvaged from PR #24064 (fixes #11020 lead bug).
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _reset_headed_cache():
    """Reset headed-mode and active-runtime caches so tests start clean."""
    import tools.browser_tool as bt
    bt._cached_headed_mode = None
    bt._headed_mode_resolved = False
    bt._real_profile_cdp_cache.clear()


@pytest.fixture(autouse=True)
def _clean_headed_cache():
    _reset_headed_cache()
    yield
    _reset_headed_cache()


# ---------------------------------------------------------------------------
# _is_headed_mode resolution
# ---------------------------------------------------------------------------

class TestIsHeadedMode:
    def test_default_is_false(self):
        from tools.browser_tool import _is_headed_mode
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AGENT_BROWSER_HEADED", None)
            with patch("hermes_cli.config.read_raw_config", return_value={}):
                assert _is_headed_mode() is False

    def test_config_true(self):
        from tools.browser_tool import _is_headed_mode
        cfg = {"browser": {"headed": True}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert _is_headed_mode() is True


    def test_caching(self):
        from tools.browser_tool import _is_headed_mode
        cfg = {"browser": {"headed": True}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg) as mock_read:
            assert _is_headed_mode() is True
            assert _is_headed_mode() is True
            assert mock_read.call_count == 1


# ---------------------------------------------------------------------------
# Per-turn cleanup skip (agent/chat_completion_helpers.cleanup_task_resources)
# ---------------------------------------------------------------------------

def _make_agent(verbose=False):
    return SimpleNamespace(verbose_logging=verbose)


class TestCleanupTaskResourcesHeadedSkip:
    def test_headless_still_cleans_browser(self):
        from agent.chat_completion_helpers import cleanup_task_resources
        with (
            patch("tools.browser_tool._preserve_browser_between_turns", return_value=False),
            patch("run_agent.cleanup_vm"),
            patch("run_agent.cleanup_browser") as mock_cb,
            patch(
                "agent.chat_completion_helpers.is_persistent_env",
                return_value=False,
            ),
        ):
            cleanup_task_resources(_make_agent(), "task-x")
            mock_cb.assert_called_once_with("task-x")


    def test_headed_does_not_skip_vm_cleanup(self):
        """Headed mode only affects the browser; VM teardown is untouched."""
        from agent.chat_completion_helpers import cleanup_task_resources
        with (
            patch("tools.browser_tool._preserve_browser_between_turns", return_value=True),
            patch("run_agent.cleanup_vm") as mock_vm,
            patch("run_agent.cleanup_browser"),
            patch(
                "agent.chat_completion_helpers.is_persistent_env",
                return_value=False,
            ),
        ):
            cleanup_task_resources(_make_agent(), "task-x")
            mock_vm.assert_called_once_with("task-x")


class TestEffectiveHeadedPersistence:
    def test_explicit_headed_runtime_overrides_headless_config(self):
        import tools.browser_tool as bt
        bt._real_profile_cdp_cache.update(cdp="http://127.0.0.1:41000", headed=True)
        with patch.object(bt, "_cdp_http_ready", return_value=True), \
             patch.object(bt, "_is_headed_mode", return_value=False):
            assert bt._preserve_browser_between_turns() is True

    def test_explicit_headless_runtime_overrides_headed_config(self):
        import tools.browser_tool as bt
        bt._real_profile_cdp_cache.update(cdp="http://127.0.0.1:41000", headed=False)
        with patch.object(bt, "_cdp_http_ready", return_value=True), \
             patch.object(bt, "_is_headed_mode", return_value=True):
            assert bt._preserve_browser_between_turns() is False

    def test_recovers_headed_marker_after_process_restart(self, tmp_path):
        import tools.browser_tool as bt

        (tmp_path / ".hermes-browser-mode").write_text("headed", encoding="utf-8")
        with (
            patch.object(bt, "_use_real_profile", return_value=True),
            patch.object(bt, "_using_lightpanda_engine", return_value=False),
            patch.object(
                bt,
                "_agent_browser_get_cdp",
                return_value="http://127.0.0.1:41000",
            ),
            patch.object(bt, "_cdp_http_ready", return_value=True),
            patch.object(bt, "_cdp_on_data_dir", return_value=True),
            patch.object(bt, "_is_headed_mode", return_value=False),
            patch(
                "hermes_cli.browser_connect.detect_default_chromium",
                return_value="chrome",
            ),
            patch(
                "hermes_cli.browser_connect.real_profile_copy_dir",
                return_value=str(tmp_path),
            ),
        ):
            assert bt._preserve_browser_between_turns() is True
        assert bt._real_profile_cdp_cache == {
            "cdp": "http://127.0.0.1:41000",
            "headed": True,
        }

    def test_invalid_mode_marker_does_not_claim_headless(self, tmp_path):
        import tools.browser_tool as bt

        (tmp_path / ".hermes-browser-mode").write_text("invalid", encoding="utf-8")
        assert bt._read_real_profile_headed_mode(str(tmp_path)) is None
        with (
            patch.object(bt, "_use_real_profile", return_value=True),
            patch.object(bt, "_using_lightpanda_engine", return_value=False),
            patch.object(
                bt,
                "_agent_browser_get_cdp",
                return_value="http://127.0.0.1:41000",
            ),
            patch.object(bt, "_cdp_http_ready", return_value=True),
            patch.object(bt, "_cdp_on_data_dir", return_value=True),
            patch.object(bt, "_is_headed_mode", return_value=False),
            patch(
                "hermes_cli.browser_connect.detect_default_chromium",
                return_value="chrome",
            ),
            patch(
                "hermes_cli.browser_connect.real_profile_copy_dir",
                return_value=str(tmp_path),
            ),
        ):
            assert bt._preserve_browser_between_turns() is True
        assert bt._real_profile_cdp_cache == {}

    def test_recovery_error_preserves_unknown_runtime(self):
        import tools.browser_tool as bt

        with (
            patch.object(bt, "_use_real_profile", return_value=True),
            patch.object(bt, "_using_lightpanda_engine", return_value=False),
            patch.object(bt, "_is_headed_mode", return_value=False),
            patch(
                "hermes_cli.browser_connect.detect_default_chromium",
                side_effect=RuntimeError("unavailable"),
            ),
        ):
            assert bt._preserve_browser_between_turns() is True


# ---------------------------------------------------------------------------
# --headed flag injection in local mode
# ---------------------------------------------------------------------------

class TestHeadedFlagInjection:
    def _run_and_capture(self, bt):
        """Run a snapshot command with Popen mocked; return captured argv."""
        captured_cmds = []
        mock_proc = MagicMock()
        mock_proc.wait.return_value = None
        mock_proc.returncode = 0

        def capture_popen(cmd, **kwargs):
            captured_cmds.append(cmd)
            return mock_proc

        mock_stdout = (
            '{"success": true, "data": {"snapshot": '
            '"- heading \\"Hi\\" [ref=e1]", "refs": {"e1": {}}}}'
        )
        with patch("subprocess.Popen", side_effect=capture_popen), \
             patch("os.open", return_value=99), \
             patch("os.close"), \
             patch("os.unlink"), \
             patch("os.makedirs"), \
             patch("builtins.open", MagicMock(return_value=MagicMock(
                 __enter__=MagicMock(return_value=MagicMock(
                     read=MagicMock(return_value=mock_stdout))),
                 __exit__=MagicMock(return_value=False),
             ))), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("tools.browser_tool._write_owner_pid"):
            bt._run_browser_command("task1", "snapshot", [], _engine_override="auto")
        return captured_cmds

    @patch("tools.browser_tool._get_session_info")
    @patch("tools.browser_tool._find_agent_browser", return_value="/usr/bin/agent-browser")
    @patch("tools.browser_tool._is_local_mode", return_value=True)
    @patch("tools.browser_tool._chromium_installed", return_value=True)
    @patch("tools.browser_tool._get_cloud_provider", return_value=None)
    @patch("tools.browser_tool._get_cdp_override", return_value="")
    @patch("tools.browser_tool._is_camofox_mode", return_value=False)
    def test_headed_flag_added_in_local_mode(
        self, _camofox, _cdp, _cloud, _chromium, _local, _find, _session
    ):
        import tools.browser_tool as bt
        bt._cached_headed_mode = True
        bt._headed_mode_resolved = True
        _session.return_value = {"session_name": "test-sess"}

        captured = self._run_and_capture(bt)
        assert len(captured) == 1
        assert "--headed" in captured[0]


    @patch("tools.browser_tool._get_session_info")
    @patch("tools.browser_tool._find_agent_browser", return_value="/usr/bin/agent-browser")
    @patch("tools.browser_tool._is_local_mode", return_value=True)
    @patch("tools.browser_tool._chromium_installed", return_value=True)
    @patch("tools.browser_tool._get_cloud_provider", return_value=None)
    @patch("tools.browser_tool._get_cdp_override", return_value="")
    @patch("tools.browser_tool._is_camofox_mode", return_value=False)
    def test_headed_flag_not_added_in_cloud_mode(
        self, _camofox, _cdp, _cloud, _chromium, _local, _find, _session
    ):
        """Cloud (CDP) sessions never get --headed — it's a local-only flag."""
        import tools.browser_tool as bt
        bt._cached_headed_mode = True
        bt._headed_mode_resolved = True
        _session.return_value = {
            "session_name": "test-sess",
            "cdp_url": "wss://example.invalid/cdp",
        }

        captured = self._run_and_capture(bt)
        assert len(captured) == 1
        assert "--headed" not in captured[0]
        assert "--cdp" in captured[0]
