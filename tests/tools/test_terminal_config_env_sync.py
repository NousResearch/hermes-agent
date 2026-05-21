"""Regression tests for terminal config -> env-var bridging.

terminal_tool._get_env_config() reads terminal settings from os.environ
(TERMINAL_*).  config.yaml values therefore have to be bridged into env vars
at startup, by the config source-of-truth and by adjacent entry-point code paths:

  1. hermes_cli.terminal_config.TERMINAL_ENV_MAPPINGS
                       -> shared CLI / TUI source-of-truth used by cli.py
  2. gateway/run.py    -> shared terminal_config helpers (gateway / messaging
                          platforms)
  3. hermes_cli/config.py:save_config_value
                       -> ``_config_to_env_sync`` dict (one-shot when the
                          user runs ``hermes config set …``)

If any one of these is missing a key, the corresponding config.yaml setting
silently does nothing for that entry-point.  This bug already shipped once
for ``docker_run_as_host_user`` (gateway and CLI maps) and once for
``docker_mount_cwd_to_workspace`` (gateway map).

This test guards against future drift by treating the shared terminal_config
mapping as the CLI source-of-truth, source-inspecting the remaining inline maps,
and asserting the load-bearing terminal env vars stay aligned.  Source
inspection for the remaining config map (rather than executing its bridge path)
keeps the test independent of the user's ~/.hermes/config.yaml and mirrors the
pattern used in tests/hermes_cli/test_config_drift.py.
"""

import ast
import inspect


def _extract_dict_values(source: str, dict_name: str) -> set[str]:
    """Return the set of *value* strings in `dict_name = { "k": "VALUE", ... }`.

    We parse the source with ast (so multi-line dicts and comments are
    handled) instead of regex.  The first matching assignment wins.
    """
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
        for k, v in zip(node.value.keys, node.value.values):
            if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                if isinstance(v.value, str):
                    out.add(v.value)
        return out
    raise AssertionError(f"Could not find `{dict_name} = {{...}}` literal in source")


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
    """terminal config keys bridged by the shared CLI/source-of-truth map."""
    from hermes_cli.terminal_config import SENSITIVE_TERMINAL_ENV_MAPPINGS, TERMINAL_ENV_MAPPINGS

    # Include the intentionally sensitive SUDO_PASSWORD mapping by key name only
    # so the CLI bridge surface stays represented without exposing secret values.
    return set(TERMINAL_ENV_MAPPINGS) | set(SENSITIVE_TERMINAL_ENV_MAPPINGS)


def _gateway_env_map_keys() -> set[str]:
    """Terminal config keys bridged by gateway/run.py via shared helpers."""
    from hermes_cli.terminal_config import TERMINAL_ENV_MAPPINGS

    # gateway/run.py additionally accepts the canonical `backend` key before
    # terminal_env_values serializes it to TERMINAL_ENV.  The parity helper
    # normalizes this alias away before comparing against the CLI map.
    return set(TERMINAL_ENV_MAPPINGS) | {"backend"}


def _gateway_env_map_values() -> set[str]:
    """TERMINAL_* env vars bridged by gateway/run.py via shared helpers."""
    from hermes_cli.terminal_config import TERMINAL_ENV_MAPPINGS

    return set(TERMINAL_ENV_MAPPINGS.values())


def _save_config_env_sync_keys() -> set[str]:
    """terminal config keys bridged by ``hermes config set foo bar``."""
    from hermes_cli import config as hc_config
    source = inspect.getsource(hc_config.set_config_value)
    keys = _extract_dict_keys(source, "_config_to_env_sync")
    # set_config_value uses fully-qualified ``terminal.foo`` keys; strip the
    # prefix so we can compare against the other two maps which use bare
    # leaf keys.
    return {k.split(".", 1)[1] for k in keys if k.startswith("terminal.")}


# Keys present in the shared CLI/source-of-truth mapping but intentionally
# not expected in gateway/run.py's terminal bridge.  Each entry must be
# justified as an alias or non-TERMINAL_* credential, not a terminal_tool
# setting that gateway merely has not wired yet.
_GATEWAY_BRIDGE_EXEMPT_KEYS = frozenset({
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


# Task 3 wires the gateway bridge through the shared terminal config helpers;
# this set must remain empty so newly shared terminal keys cannot be hidden as
# temporary gateway-only omissions.
TASK3_PENDING_GATEWAY_BRIDGE_KEYS = frozenset()


def _terminal_tool_env_var_names() -> set[str]:
    """All TERMINAL_* env vars actually consumed by terminal_tool."""
    import tools.terminal_tool as tt
    source = inspect.getsource(tt)
    # Naive scan: every os.getenv("TERMINAL_X", ...) and _parse_env_var("TERMINAL_X", ...).
    import re
    pat = re.compile(r'["\'](TERMINAL_[A-Z0-9_]+)["\']')
    return set(pat.findall(source))


def _normalized_gateway_env_map_keys() -> set[str]:
    """Gateway terminal bridge keys normalized to the shared CLI key surface."""
    # Normalize the legacy/backend aliases: `env_type` is a legacy YAML key
    # alias and `backend` is the canonical gateway spelling for TERMINAL_ENV.
    # Both serialize to the same env var, so exclude them from key-drift parity.
    return _gateway_env_map_keys() - {"backend", "env_type"}


def _shared_terminal_keys_missing_from_gateway() -> set[str]:
    """Shared terminal keys that are neither gateway-bridged nor exempt aliases."""
    cli_terminal_keys = _cli_env_map_keys() - _GATEWAY_BRIDGE_EXEMPT_KEYS
    return cli_terminal_keys - _normalized_gateway_env_map_keys()


def test_task3_pending_gateway_bridge_gaps_are_exact():
    """Task 3 consumed all known gateway bridge omissions."""
    assert (
        _shared_terminal_keys_missing_from_gateway()
        == TASK3_PENDING_GATEWAY_BRIDGE_KEYS
    )


def test_gateway_bridge_uses_shared_terminal_config_helpers():
    """gateway/run.py should not carry a drift-prone inline terminal map."""
    import gateway.run as gr

    source = inspect.getsource(gr)
    assert "normalize_terminal_config" in source
    assert "resolve_gateway_terminal_cwd" in source
    assert "terminal_env_values" in source
    assert "_terminal_env_map" not in source


def test_cli_and_gateway_env_maps_agree_except_task3_pending_gaps():
    """Shared CLI config and gateway/run.py must bridge the same terminal keys.

    Both feed the same downstream consumer (terminal_tool).  Drift between
    them means a config.yaml setting that "works in CLI mode but not gateway
    mode" (or vice-versa) — the bug class that shipped twice already.
    """
    cli_keys = (
        _cli_env_map_keys()
        - _GATEWAY_BRIDGE_EXEMPT_KEYS
        - TASK3_PENDING_GATEWAY_BRIDGE_KEYS
    )
    gw_keys = _normalized_gateway_env_map_keys()

    missing_in_gateway = cli_keys - gw_keys
    missing_in_cli = gw_keys - cli_keys

    assert not missing_in_gateway, (
        f"Keys in shared TERMINAL_ENV_MAPPINGS but missing from gateway/run.py "
        f"shared terminal bridge: {sorted(missing_in_gateway)}. Keep gateway "
        f"wired through terminal_env_values so shared terminal config keys do "
        f"not drift by entry point."
    )
    assert not missing_in_cli, (
        f"Keys in the gateway terminal bridge but missing from shared "
        f"TERMINAL_ENV_MAPPINGS or still listed in "
        f"TASK3_PENDING_GATEWAY_BRIDGE_KEYS: {sorted(missing_in_cli)}. Add them "
        f"to the shared map, or remove resolved Task 3 pending keys from the test."
    )


def test_shared_terminal_env_map_points_at_terminal_tool_consumers():
    """Every shared terminal env var should be consumed by terminal_tool."""
    from hermes_cli.terminal_config import TERMINAL_ENV_MAPPINGS

    missing_consumers = set(TERMINAL_ENV_MAPPINGS.values()) - _terminal_tool_env_var_names()

    assert not missing_consumers, (
        "Shared TERMINAL_ENV_MAPPINGS contains env vars terminal_tool does not "
        f"consume: {sorted(missing_consumers)}. Remove dead mappings or add the "
        "terminal_tool consumer."
    )


def test_gateway_terminal_env_map_points_at_terminal_tool_consumers():
    """Every gateway terminal env var should be consumed by terminal_tool."""
    missing_consumers = _gateway_env_map_values() - _terminal_tool_env_var_names()

    assert not missing_consumers, (
        "gateway/run.py shared terminal bridge contains env vars terminal_tool does "
        f"not consume: {sorted(missing_consumers)}. Remove dead mappings or add "
        "the terminal_tool consumer."
    )


def test_save_config_set_supports_critical_bridged_keys():
    """``hermes config set terminal.X true`` must propagate to .env for
    known-critical keys.  This used to be an all-keys invariant but several
    pre-existing terminal keys (ssh_*, docker_forward_env, docker_volumes)
    aren't in _config_to_env_sync and are instead handled via the separate
    api_keys TERMINAL_SSH_* fallback path or user-edits-yaml-directly.

    Until those gaps are audited and fixed, pin the specific keys that are
    load-bearing for the docker backend's ownership flag so the bug we just
    fixed cannot silently regress.
    """
    save_keys = _save_config_env_sync_keys()
    required = {
        "docker_run_as_host_user",
        "docker_mount_cwd_to_workspace",
        "backend",
        "docker_image",
        "container_cpu",
        "container_memory",
        "container_disk",
        "container_persistent",
    }
    missing = required - save_keys
    assert not missing, (
        f"`hermes config set terminal.X` doesn't sync these load-bearing "
        f"keys to .env: {sorted(missing)}.  Add them to _config_to_env_sync "
        f"in hermes_cli/config.py:set_config_value."
    )


def test_docker_run_as_host_user_is_bridged_everywhere():
    """Explicit pin for the bug we just fixed.

    docker_run_as_host_user was added to terminal_tool._get_env_config and
    DockerEnvironment but NOT to the then-inline CLI/gateway bridge maps, so
    ``terminal.docker_run_as_host_user: true`` in config.yaml had no effect at
    runtime.  This guard makes the regression impossible to reintroduce silently.
    """
    assert "docker_run_as_host_user" in _cli_env_map_keys()
    assert "docker_run_as_host_user" in _gateway_env_map_keys()
    assert "docker_run_as_host_user" in _save_config_env_sync_keys()
    assert "TERMINAL_DOCKER_RUN_AS_HOST_USER" in _terminal_tool_env_var_names()


def test_docker_mount_cwd_to_workspace_is_bridged_everywhere():
    """Same regression class — docker_mount_cwd_to_workspace was missing from
    gateway/run.py's legacy bridge until the docker_run_as_host_user audit
    caught it.
    """
    assert "docker_mount_cwd_to_workspace" in _cli_env_map_keys()
    assert "docker_mount_cwd_to_workspace" in _gateway_env_map_keys()
    assert "docker_mount_cwd_to_workspace" in _save_config_env_sync_keys()
    assert "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE" in _terminal_tool_env_var_names()


def test_docker_env_is_bridged_everywhere():
    """Regression pin for docker_env config key being silently ignored.

    ``terminal.docker_env`` in config.yaml specifies extra env vars to inject
    into the Docker container at runtime.  The key was present in
    _create_environment's container_config consumer (line ~1130) but never
    bridged from config.yaml to TERMINAL_DOCKER_ENV, so the dict was always
    empty regardless of what the user set.  Guard all four bridging points so
    this cannot regress.
    """
    assert "docker_env" in _cli_env_map_keys()
    assert "docker_env" in _gateway_env_map_keys()
    assert "docker_env" in _save_config_env_sync_keys()
    assert "TERMINAL_DOCKER_ENV" in _terminal_tool_env_var_names()
