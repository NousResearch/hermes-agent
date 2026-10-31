"""terminal.* config -> TERMINAL_* env bridging has one source of truth.

``terminal_tool`` reads every setting from ``TERMINAL_*`` env vars, so config.yaml values are
bridged at startup by the CLI, the gateway and the standalone bridge. All of them derive from
``hermes_cli.config.TERMINAL_CONFIG_ENV_MAP``; a key missing from it (``home_mode`` once was)
is silently ignored by every bridge that skips it.
"""

import os
from unittest.mock import patch

from hermes_cli.config import TERMINAL_CONFIG_ENV_MAP, apply_terminal_config_to_env

# Absolute, so the cwd placeholder skip never fires.
_PROBE = "/hermes-bridge-probe"


def _bridged_by_gateway(key: str) -> None:
    from gateway.run import _bridge_terminal_config_to_env

    _bridge_terminal_config_to_env({key: _PROBE})


def _bridged_by_cli(key: str) -> None:
    from cli import _mirror_config_to_env

    # A non-local backend keeps an explicit cwd instead of replacing it with os.getcwd().
    _mirror_config_to_env({"terminal": {"env_type": "docker", key: _PROBE}}, True)


def test_cli_and_gateway_bridge_every_terminal_key_to_its_env_var():
    not_bridged = []
    for bridge in (_bridged_by_cli, _bridged_by_gateway):
        for key, env_var in TERMINAL_CONFIG_ENV_MAP.items():
            with patch.dict(os.environ):  # restores the process env on exit
                for var in [v for v in os.environ if v.startswith("TERMINAL_")]:
                    del os.environ[var]
                bridge(key)
                if os.environ.get(env_var) != _PROBE:
                    not_bridged.append((bridge.__name__, key, env_var))
    assert not not_bridged, f"terminal.* keys a bridge left out of {sorted(not_bridged)}"


def test_home_mode_in_config_yaml_reaches_the_standalone_bridge(tmp_path, monkeypatch):
    """``hermes serve``/dashboard/TUI launchers bridge through ``apply_terminal_config_to_env``."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("terminal:\n  home_mode: profile\n")

    env: dict[str, str] = {}
    apply_terminal_config_to_env(env=env)

    assert env["TERMINAL_HOME_MODE"] == "profile"
