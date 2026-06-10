"""Tests for tui_gateway/entry.py shell-hook registration (issue #43823).

The TUI gateway must register declarative shell hooks from config.yaml at
startup so that ``pre_tool_call`` and other hooks fire in Desktop/TUI
sessions, matching CLI and gateway behaviour.
"""

from unittest.mock import patch


def test_main_registers_shell_hooks():
    """``main()`` must call ``register_from_config`` at startup."""
    mock_config = {"hooks": {}}

    with (
        patch("tui_gateway.entry._install_sidecar_publisher"),
        patch("tui_gateway.entry.write_json", return_value=True),
        patch("tui_gateway.entry.resolve_skin", return_value="default"),
        patch("hermes_cli.config.read_raw_config", return_value=mock_config),
        patch("hermes_cli.config.load_config", return_value=mock_config),
        patch(
            "agent.shell_hooks.register_from_config"
        ) as mock_register,
        patch("sys.stdin", iter([])),  # EOF immediately → clean exit
        patch("tui_gateway.entry._log_exit"),
    ):
        from tui_gateway.entry import main

        try:
            main()
        except (SystemExit, StopIteration):
            pass

        mock_register.assert_called_once_with(mock_config, accept_hooks=False)


def test_main_continues_on_shell_hook_failure():
    """``main()`` must not abort if shell-hook registration fails."""
    with (
        patch("tui_gateway.entry._install_sidecar_publisher"),
        patch("tui_gateway.entry.write_json", return_value=True),
        patch("tui_gateway.entry.resolve_skin", return_value="default"),
        patch("hermes_cli.config.read_raw_config", return_value=None),
        patch(
            "agent.shell_hooks.register_from_config",
            side_effect=RuntimeError("config parse error"),
        ),
        patch("sys.stdin", iter([])),
        patch("tui_gateway.entry._log_exit") as mock_log_exit,
    ):
        from tui_gateway.entry import main

        try:
            main()
        except (SystemExit, StopIteration):
            pass

        # Should reach normal exit, not crash on hook registration failure
        mock_log_exit.assert_called_once_with(
            "stdin EOF (TUI closed the command pipe)"
        )
