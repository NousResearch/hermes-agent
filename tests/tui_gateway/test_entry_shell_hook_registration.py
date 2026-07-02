"""Regression test: shell hooks and plugins must be registered when the TUI
gateway is spawned directly (``python -m tui_gateway.entry``).

``ui-tui``'s ``gatewayClient.ts`` spawns this module as a raw subprocess
(``spawn(python, ['-m', 'tui_gateway.entry'], ...)``), bypassing
``hermes_cli/main.py``'s ``_prepare_agent_startup`` entirely. That function is
the only place ``discover_plugins()`` and ``register_from_config()`` normally
run for CLI-driven agent turns; the gateway process (``gateway/run.py``) has
its own equivalent calls, but ``tui_gateway/entry.py`` had neither, so shell
hooks configured under ``hooks:`` in ``cli-config.yaml`` silently never
registered for anyone using the TUI (desktop app or ``hermes --tui``).
"""

from unittest.mock import patch

import tui_gateway.entry as entry


def test_main_discovers_plugins_and_registers_shell_hooks():
    """entry.main() must call discover_plugins() and register_from_config()
    at startup, mirroring gateway/run.py's equivalent startup calls."""
    with patch.object(entry, "_install_sidecar_publisher"), \
         patch("hermes_cli.config.read_raw_config", return_value={}), \
         patch("hermes_cli.config.load_config", return_value={}) as mock_load_config, \
         patch("hermes_cli.plugins.discover_plugins") as mock_discover_plugins, \
         patch("agent.shell_hooks.register_from_config") as mock_register_hooks, \
         patch.object(entry, "write_json", return_value=True), \
         patch("sys.stdin", iter([])):
        entry.main()

    mock_discover_plugins.assert_called_once()
    mock_register_hooks.assert_called_once()
    (cfg_arg,), kwargs = mock_register_hooks.call_args
    assert cfg_arg == mock_load_config.return_value
    assert kwargs.get("accept_hooks") is False


def test_shell_hook_registration_failure_does_not_block_startup():
    """A registration failure must be swallowed, not crash the TUI gateway
    (mirrors the try/except around the equivalent gateway/run.py call)."""
    with patch.object(entry, "_install_sidecar_publisher"), \
         patch("hermes_cli.config.read_raw_config", return_value={}), \
         patch("hermes_cli.config.load_config", return_value={}), \
         patch("hermes_cli.plugins.discover_plugins"), \
         patch(
             "agent.shell_hooks.register_from_config",
             side_effect=RuntimeError("boom"),
         ), \
         patch.object(entry, "write_json", return_value=True), \
         patch("sys.stdin", iter([])):
        entry.main()  # must not raise
