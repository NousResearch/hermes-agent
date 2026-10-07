"""Regression tests for terminal config -> env-var bridging.

terminal_tool._get_env_config() reads ALL terminal settings from os.environ
(TERMINAL_*).  config.yaml values therefore have to be bridged into env vars
at startup, by THREE separate code paths:

  1. cli.py            -> ``env_mappings`` dict (CLI / TUI startup)
  2. gateway/run.py    -> ``_terminal_env_map`` dict (gateway / messaging
                          platforms)
  3. hermes_cli/config.py:set_config_value
                       -> bridges via the canonical ``TERMINAL_CONFIG_ENV_MAP``
                          (one-shot when the user runs ``hermes config set …``)

If any one of these is missing a key, the corresponding config.yaml setting
silently does nothing for that entry-point.  This bug already shipped once
for ``docker_run_as_host_user`` (gateway and CLI maps) and once for
``docker_mount_cwd_to_workspace`` (gateway map).

This test guards against future drift by driving the gateway bridge with
every known ``terminal.*`` key and comparing the env var it writes against
cli.py's map; the config-set path is checked for key coverage only.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _mirror_cwd_only(monkeypatch, defaults):
    """Drive the cwd bridge without importing the installed CLI bootstrap."""
    fake_cli = SimpleNamespace(
        _AUXILIARY_TASK_ENV={},
        _CWD_PLACEHOLDERS=(".", "auto", "cwd"),
        _TERMINAL_ENV_MAPPINGS={"cwd": "TERMINAL_CWD"},
    )
    monkeypatch.setitem(sys.modules, "cli", fake_cli)
    from hermes_cli.cli_config_load import _mirror_config_to_env

    _mirror_config_to_env(defaults, True)


def _cli_env_map() -> dict[str, str]:
    """terminal config key -> env var bridged by cli.load_cli_config() (via _mirror_config_to_env)."""
    import cli
    return dict(cli._TERMINAL_ENV_MAPPINGS)


def _gateway_env_map() -> dict[str, str]:
    """terminal config key -> env var actually written by the gateway bridge."""
    from gateway.run import _bridge_terminal_config_to_env
    from hermes_cli import config as hc_config

    class _KeyRecorder(dict):
        """Empty config that records every key the bridge looks up."""

        def __init__(self):
            super().__init__()
            self.seen: set[str] = set()

        def __contains__(self, key):
            self.seen.add(key)
            return super().__contains__(key)

        def get(self, key, default=None):
            self.seen.add(key)
            return super().get(key, default)

        def __getitem__(self, key):
            self.seen.add(key)
            return super().__getitem__(key)

    recorder = _KeyRecorder()
    _bridge_terminal_config_to_env(recorder)  # empty: writes nothing
    # A bridge that stopped consulting its config would make the probe below vacuous.
    assert len(recorder.seen) > 1, "gateway bridge looked up no terminal keys"
    probe = "/hermes-bridge-probe"  # absolute, so the cwd placeholder skip never fires
    candidates = set(_cli_env_map()) | set(hc_config.TERMINAL_CONFIG_ENV_MAP) | recorder.seen
    bridged: dict[str, str] = {}
    with patch.dict(os.environ):  # restores the process env on exit
        for key in sorted(candidates):
            for var in [v for v in os.environ if v.startswith("TERMINAL_")]:
                del os.environ[var]
            _bridge_terminal_config_to_env({key: probe})
            written = [v for v, val in os.environ.items() if v.startswith("TERMINAL_") and val == probe]
            if written:
                (bridged[key],) = written
    return bridged


def _save_config_env_sync_keys() -> set[str]:
    """terminal config keys bridged by ``hermes config set foo bar``.

    ``set_config_value`` no longer carries its own ``_config_to_env_sync``
    dict — it bridges through the canonical ``TERMINAL_CONFIG_ENV_MAP`` via
    ``terminal_config_env_var_for_key()`` (config.py), excluding ``cwd``
    (handled separately).  Read the live map so this test tracks the actual
    source of truth that the config-set path uses, rather than a string
    literal that the consolidation removed.
    """
    from hermes_cli import config as hc_config
    # set_config_value bridges every TERMINAL_CONFIG_ENV_MAP key except
    # terminal.cwd (see the ``key != "terminal.cwd"`` guard in
    # set_config_value); mirror that exclusion here.
    return {k for k in hc_config.TERMINAL_CONFIG_ENV_MAP if k != "cwd"}


# Keys present in cli.py env_mappings but intentionally absent from
# gateway/run.py or set_config_value.  Each entry must be justified.
_CLI_ONLY_OK = frozenset({
    # `env_type` is a legacy YAML key alias for `backend` that cli.py
    # accepts for backwards-compat with older cli-config.yaml.  The
    # gateway path normalizes on the canonical `backend` key, which is
    # also in the map and handles the same bridging.  See cli.py ~line 515.
    "env_type",
    # sudo_password is not a terminal-backend option — it's a credential
    # used across backends, bridged to $SUDO_PASSWORD (not TERMINAL_*).
    # Treating it as terminal-only would be misleading.
    "sudo_password",
})


def test_cli_and_gateway_env_maps_agree():
    """cli.py and gateway/run.py must bridge each terminal key to the same env var.

    Both feed the same downstream consumer (terminal_tool).  Drift between
    them means a config.yaml setting that "works in CLI mode but not gateway
    mode" (or vice-versa) — the bug class that shipped twice already.
    """
    cli_map = {k: v for k, v in _cli_env_map().items() if k not in _CLI_ONLY_OK}
    gw_map = _gateway_env_map()
    # cli.py copies the canonical `backend` key onto the legacy `env_type`
    # alias before bridging, so the gateway's `backend` is cli's `env_type`.
    gw_map.pop("backend", None)

    missing_in_gateway = sorted(set(cli_map) - set(gw_map))
    missing_in_cli = sorted(set(gw_map) - set(cli_map))
    mismatched = {k: (cli_map[k], gw_map[k]) for k in set(cli_map) & set(gw_map) if cli_map[k] != gw_map[k]}

    assert not missing_in_gateway, (
        f"Keys the CLI bridges but gateway/run.py _bridge_terminal_config_to_env "
        f"ignores: {missing_in_gateway}.  Add them to both maps (same bug class "
        f"as docker_run_as_host_user shipping wired in cli but not gateway)."
    )
    assert not missing_in_cli, (
        f"Keys the gateway bridges but cli.py env_mappings ignores: "
        f"{missing_in_cli}.  Add them to both maps."
    )
    assert not mismatched, f"Same key bridged to different env vars (cli, gateway): {mismatched}"


def test_save_config_set_bridges_every_cli_terminal_key():
    """``hermes config set terminal.X`` must propagate every key the CLI
    startup path bridges, so a config-set value takes effect without restart.
    """
    save_keys = _save_config_env_sync_keys()
    # cwd is bridged separately by set_config_value; home_mode is CLI-only.
    exempt = _CLI_ONLY_OK | {"cwd", "home_mode"}
    missing = (set(_cli_env_map()) - exempt) - save_keys
    assert not missing, (
        f"`hermes config set terminal.X` doesn't sync these keys to .env: "
        f"{sorted(missing)}.  Add them to TERMINAL_CONFIG_ENV_MAP in "
        f"hermes_cli/config.py (set_config_value bridges through it)."
    )


def test_cli_preserves_exact_dispatcher_pinned_kanban_cwd(tmp_path, monkeypatch):
    """A worker's resolved task worktree survives profile-config mirroring."""
    workspace = tmp_path / "task-worktree"
    workspace.mkdir()
    monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
    monkeypatch.setenv("TERMINAL_CWD", str(workspace))

    defaults = {"terminal": {"backend": "docker", "cwd": "/configured-static-cwd"}}
    _mirror_cwd_only(monkeypatch, defaults)

    assert os.environ["TERMINAL_CWD"] == str(workspace.resolve())


@pytest.mark.parametrize(
    "missing",
    ["HERMES_KANBAN_TASK", "HERMES_KANBAN_WORKSPACE", "TERMINAL_CWD"],
)
def test_cli_does_not_preserve_incomplete_kanban_cwd_pin(tmp_path, monkeypatch, missing):
    """Partial launch markers cannot override the profile's configured cwd."""
    workspace = tmp_path / "task-worktree"
    workspace.mkdir()
    monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
    monkeypatch.setenv("TERMINAL_CWD", str(workspace))
    monkeypatch.delenv(missing, raising=False)

    defaults = {"terminal": {"backend": "docker", "cwd": "/configured-static-cwd"}}
    _mirror_cwd_only(monkeypatch, defaults)

    assert os.environ["TERMINAL_CWD"] == "/configured-static-cwd"


def test_cli_rejects_mismatched_dispatcher_kanban_cwd_pin(tmp_path, monkeypatch):
    """A stale/mismatched process cwd cannot become a Kanban worker mount."""
    workspace = tmp_path / "task-worktree"
    other = tmp_path / "other-worktree"
    workspace.mkdir()
    other.mkdir()
    monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
    monkeypatch.setenv("TERMINAL_CWD", str(other))

    defaults = {"terminal": {"backend": "docker", "cwd": "/configured-static-cwd"}}
    _mirror_cwd_only(monkeypatch, defaults)

    assert os.environ["TERMINAL_CWD"] == "/configured-static-cwd"
