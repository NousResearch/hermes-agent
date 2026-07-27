"""Tests for headed browser mode: config/env resolution, --headed injection,
and the per-turn cleanup skip that keeps headed sessions alive between turns.

Salvaged from PR #24064 (fixes #11020 lead bug).
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _reset_headed_cache():
    """Reset the module-level headed-mode cache so tests start clean."""
    import tools.browser_tool as bt
    bt._cached_headed_mode = None
    bt._headed_mode_resolved = False


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

    def test_config_string_true(self):
        from tools.browser_tool import _is_headed_mode
        cfg = {"browser": {"headed": "true"}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert _is_headed_mode() is True

    def test_config_false_beats_missing_env(self):
        from tools.browser_tool import _is_headed_mode
        cfg = {"browser": {"headed": False}}
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AGENT_BROWSER_HEADED", None)
            with patch("hermes_cli.config.read_raw_config", return_value=cfg):
                assert _is_headed_mode() is False

    def test_env_var_fallback(self):
        from tools.browser_tool import _is_headed_mode
        with patch.dict(os.environ, {"AGENT_BROWSER_HEADED": "1"}):
            with patch("hermes_cli.config.read_raw_config", return_value={}):
                assert _is_headed_mode() is True

    def test_env_var_garbage_is_false(self):
        from tools.browser_tool import _is_headed_mode
        with patch.dict(os.environ, {"AGENT_BROWSER_HEADED": "banana"}):
            with patch("hermes_cli.config.read_raw_config", return_value={}):
                assert _is_headed_mode() is False

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
            patch("tools.browser_tool._is_headed_mode", return_value=False),
            patch("run_agent.cleanup_vm"),
            patch("run_agent.cleanup_browser") as mock_cb,
            patch(
                "agent.chat_completion_helpers.is_persistent_env",
                return_value=False,
            ),
        ):
            cleanup_task_resources(_make_agent(), "task-x")
            mock_cb.assert_called_once_with("task-x")

    def test_headed_skips_browser_cleanup(self):
        from agent.chat_completion_helpers import cleanup_task_resources
        with (
            patch("tools.browser_tool._is_headed_mode", return_value=True),
            patch("run_agent.cleanup_vm"),
            patch("run_agent.cleanup_browser") as mock_cb,
            patch(
                "agent.chat_completion_helpers.is_persistent_env",
                return_value=False,
            ),
        ):
            cleanup_task_resources(_make_agent(), "task-x")
            mock_cb.assert_not_called()

    def test_headed_env_var_fallback_when_import_fails(self):
        """If browser_tool import blows up, the env var still gates the skip."""
        from agent.chat_completion_helpers import cleanup_task_resources
        with (
            patch(
                "tools.browser_tool._is_headed_mode",
                side_effect=RuntimeError("boom"),
            ),
            patch.dict(os.environ, {"AGENT_BROWSER_HEADED": "1"}),
            patch("run_agent.cleanup_vm"),
            patch("run_agent.cleanup_browser") as mock_cb,
            patch(
                "agent.chat_completion_helpers.is_persistent_env",
                return_value=False,
            ),
        ):
            cleanup_task_resources(_make_agent(), "task-x")
            mock_cb.assert_not_called()

    def test_headed_does_not_skip_vm_cleanup(self):
        """Headed mode only affects the browser; VM teardown is untouched."""
        from agent.chat_completion_helpers import cleanup_task_resources
        with (
            patch("tools.browser_tool._is_headed_mode", return_value=True),
            patch("run_agent.cleanup_vm") as mock_vm,
            patch("run_agent.cleanup_browser"),
            patch(
                "agent.chat_completion_helpers.is_persistent_env",
                return_value=False,
            ),
        ):
            cleanup_task_resources(_make_agent(), "task-x")
            mock_vm.assert_called_once_with("task-x")


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
    def test_headed_flag_not_added_when_headless(
        self, _camofox, _cdp, _cloud, _chromium, _local, _find, _session
    ):
        import tools.browser_tool as bt
        bt._cached_headed_mode = False
        bt._headed_mode_resolved = True
        _session.return_value = {"session_name": "test-sess"}

        captured = self._run_and_capture(bt)
        assert len(captured) == 1
        assert "--headed" not in captured[0]

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
