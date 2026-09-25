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

This test guards against future drift by extracting all three maps via source
inspection and asserting they all bridge the same set of writable
``terminal.*`` keys.  Source inspection (rather than importing the live
dicts) keeps the test independent of the user's ~/.hermes/config.yaml and
mirrors the pattern used in tests/hermes_cli/test_config_drift.py.
"""

import ast
import inspect


def _extract_dict_keys(source: str, dict_name: str) -> set[str]:
    """Return the set of *key* strings in `dict_name = { "KEY": "v", ... }`."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t for t in node.targets if isinstance(t, ast.Name)]
        if not any(t.id == dict_name for t in targets):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        out: set[str] = set()
        for k in node.value.keys:
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                out.add(k.value)
        return out
    raise AssertionError(f"Could not find `{dict_name} = {{...}}` literal in source")


def _cli_env_map_keys() -> set[str]:
    """terminal config keys bridged by cli.load_cli_config() (via _mirror_config_to_env)."""
    import cli
    return set(cli._TERMINAL_ENV_MAPPINGS.keys())


def _gateway_env_map_keys() -> set[str]:
    """terminal config keys bridged by gateway/run.py at module load."""
    # gateway/run.py builds the dict at module top-level (not inside a
    # function), so inspect the whole module source.
    import gateway.run as gr
    source = inspect.getsource(gr)
    return _extract_dict_keys(source, "_terminal_env_map")


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
    """cli.py and gateway/run.py must bridge the same set of terminal keys.

    Both feed the same downstream consumer (terminal_tool).  Drift between
    them means a config.yaml setting that "works in CLI mode but not gateway
    mode" (or vice-versa) — the bug class that shipped twice already.
    """
    cli_keys = _cli_env_map_keys() - _CLI_ONLY_OK
    gw_keys = _gateway_env_map_keys()

    # Normalize the legacy `env_type` alias: cli.py accepts both `env_type`
    # and `backend` as source keys for TERMINAL_ENV; gateway only accepts
    # `backend`.  Since cli.py copies `backend` → `env_type` before the
    # lookup, they're equivalent.  Remove `backend` from the gateway side
    # to avoid a spurious "backend missing from cli" failure.
    gw_keys = gw_keys - {"backend"}

    missing_in_gateway = cli_keys - gw_keys
    missing_in_cli = gw_keys - cli_keys

    assert not missing_in_gateway, (
        f"Keys in cli.py env_mappings but missing from gateway/run.py "
        f"_terminal_env_map: {sorted(missing_in_gateway)}.  Add them to "
        f"both maps (same bug class as docker_run_as_host_user shipping "
        f"wired in cli but not gateway in April 2026)."
    )
    assert not missing_in_cli, (
        f"Keys in gateway/run.py _terminal_env_map but missing from cli.py "
        f"env_mappings: {sorted(missing_in_cli)}.  Add them to both maps."
    )


def test_save_config_set_bridges_every_cli_terminal_key():
    """``hermes config set terminal.X`` must propagate every key the CLI
    startup path bridges, so a config-set value takes effect without restart.
    """
    save_keys = _save_config_env_sync_keys()
    # cwd is bridged separately by set_config_value; home_mode is CLI-only.
    exempt = _CLI_ONLY_OK | {"cwd", "home_mode"}
    missing = (_cli_env_map_keys() - exempt) - save_keys
    assert not missing, (
        f"`hermes config set terminal.X` doesn't sync these keys to .env: "
        f"{sorted(missing)}.  Add them to TERMINAL_CONFIG_ENV_MAP in "
        f"hermes_cli/config.py (set_config_value bridges through it)."
    )
