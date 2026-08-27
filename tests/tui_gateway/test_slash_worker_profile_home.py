"""Tests for TUI gateway slash_worker profile_home propagation (#40677)."""

from pathlib import Path
from unittest.mock import MagicMock, patch


def test_slash_worker_accepts_profile_home():
    """_SlashWorker.__init__ accepts profile_home parameter."""
    # hermes_state evaluates get_hermes_home() / "state.db" at import time, so
    # the mock must return a Path (a bare str raises TypeError under per-file
    # subprocess isolation).
    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(
            get_hermes_home=MagicMock(return_value=Path("/tmp/hermes_test")),
        ),
    }):
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value.stdout = MagicMock()
            mock_popen.return_value.stderr = MagicMock()

            from tui_gateway.server import _SlashWorker

            # Test initialization with profile_home
            worker = _SlashWorker(
                session_key="test_key",
                model="test-model",
                profile_home="/home/luke/.hermes/profiles/work"
            )

            worker_calls = [
                call
                for call in mock_popen.call_args_list
                if call.args and "tui_gateway.slash_worker" in call.args[0]
            ]
            assert len(worker_calls) == 1

            # Check the worker process rather than unrelated subprocess probes.
            call_kwargs = worker_calls[0].kwargs
            assert "env" in call_kwargs
            assert call_kwargs["env"]["HERMES_HOME"] == "/home/luke/.hermes/profiles/work"

