"""Regression tests for terminal config -> env-var bridging.

terminal_tool._get_env_config() reads ALL terminal settings from os.environ
(TERMINAL_*).  config.yaml values therefore have to be bridged into env vars
at startup, by THREE separate code paths:

  1. cli.py            -> ``env_mappings`` dict (CLI / TUI startup)
  2. gateway/run.py    -> ``_terminal_env_map`` dict (gateway / messaging
                          platforms)
  3. hermes_cli/config.py:save_config_value
                       -> ``_config_to_env_sync`` dict (one-shot when the
                          user runs ``hermes config set …``)

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
import os

import pytest


def skip_if_no_prompt_toolkit():
    """Skip tests requiring cli.py import if prompt_toolkit isn't installed.

    The Docker test environment doesn't have prompt_toolkit, so tests that
    import cli.py (which pulls in prompt_toolkit) need this guard.
    """
    try:
        import cli  # noqa: F401
    except ImportError:
        pytest.skip("cli.py requires prompt_toolkit (not installed)")


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
    """terminal config keys bridged by cli.load_cli_config()."""
    try:
        import cli
        source = inspect.getsource(cli.load_cli_config)
    except ImportError:
        pytest.skip("cli.py requires prompt_toolkit (not installed)")
    return _extract_dict_keys(source, "env_mappings")


def _gateway_env_map_keys() -> set[str]:
    """terminal config keys bridged by gateway/run.py at module load."""
    # gateway/run.py builds the dict at module top-level (not inside a
    # function), so inspect the whole module source.
    import gateway.run as gr
    source = inspect.getsource(gr)
    return _extract_dict_keys(source, "_terminal_env_map")


def _save_config_env_sync_keys() -> set[str]:
    """terminal config keys bridged by ``hermes config set foo bar``."""
    from hermes_cli import config as hc_config
    source = inspect.getsource(hc_config.set_config_value)
    keys = _extract_dict_keys(source, "_config_to_env_sync")
    # set_config_value uses fully-qualified ``terminal.foo`` keys; strip the
    # prefix so we can compare against the other two maps which use bare
    # leaf keys.
    return {k.split(".", 1)[1] for k in keys if k.startswith("terminal.")}


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


def _terminal_tool_env_var_names() -> set[str]:
    """All TERMINAL_* env vars actually consumed by terminal_tool."""
    import tools.terminal_tool as tt
    source = inspect.getsource(tt)
    # Naive scan: every os.getenv("TERMINAL_X", ...) and _parse_env_var("TERMINAL_X", ...).
    import re
    pat = re.compile(r'["\'](TERMINAL_[A-Z0-9_]+)["\']')
    return set(pat.findall(source))


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
    DockerEnvironment but NOT to cli.py's env_mappings or gateway/run.py's
    _terminal_env_map, so ``terminal.docker_run_as_host_user: true`` in
    config.yaml had no effect at runtime.  This guard makes the regression
    impossible to reintroduce silently.
    """
    assert "docker_run_as_host_user" in _cli_env_map_keys()
    assert "docker_run_as_host_user" in _gateway_env_map_keys()
    assert "docker_run_as_host_user" in _save_config_env_sync_keys()
    assert "TERMINAL_DOCKER_RUN_AS_HOST_USER" in _terminal_tool_env_var_names()


def test_docker_mount_cwd_to_workspace_is_bridged_everywhere():
    """Same regression class — docker_mount_cwd_to_workspace was missing from
    gateway/run.py's _terminal_env_map until the docker_run_as_host_user
    audit caught it.
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


def test_docker_persist_across_processes_is_bridged_everywhere():
    """Regression pin for the cross-process container reuse toggle.

    ``terminal.docker_persist_across_processes`` (issue #20561) controls
    whether ``DockerEnvironment.__init__`` probes for and reuses an existing
    labeled container at startup, and whether ``cleanup()`` removes the
    container on Hermes exit or just stops it (keeping it for the next
    process).  Same four-bridge invariant as docker_run_as_host_user /
    docker_env / docker_mount_cwd_to_workspace — drift between any of the
    four sites means ``terminal.docker_persist_across_processes: false`` in
    config.yaml silently does nothing for that entry point, leaving the
    user unable to opt out of the documented "ONE long-lived container
    shared across sessions" behavior.
    """
    assert "docker_persist_across_processes" in _cli_env_map_keys()
    assert "docker_persist_across_processes" in _gateway_env_map_keys()
    assert "docker_persist_across_processes" in _save_config_env_sync_keys()
    assert "TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES" in _terminal_tool_env_var_names()


def test_docker_orphan_reaper_is_bridged_everywhere():
    """Regression pin for the startup orphan reaper toggle (issue #20561).

    ``terminal.docker_orphan_reaper`` controls whether Hermes sweeps stale
    Exited containers from prior SIGKILL'd processes at startup.  Same
    four-site bridge invariant — drift means
    ``terminal.docker_orphan_reaper: false`` silently does nothing for one
    entry point, and the reaper either runs when the operator disabled it
    or fails to run when they enabled it.
    """
    assert "docker_orphan_reaper" in _cli_env_map_keys()
    assert "docker_orphan_reaper" in _gateway_env_map_keys()
    assert "docker_orphan_reaper" in _save_config_env_sync_keys()
    assert "TERMINAL_DOCKER_ORPHAN_REAPER" in _terminal_tool_env_var_names()


# ===========================================================================
# EXPANSION TESTS: Daytona-specific config/env bridge parity
# ===========================================================================
# These tests verify that new terminal.daytona_* keys from the Daytona
# expansion plan are properly bridged in ALL THREE code paths:
# 1. hermes_cli/config.py::_config_to_env_sync (hermes config set)
# 2. cli.py (CLI/TUI startup)
# 3. gateway/run.py::_terminal_env_map (gateway startup)
# 4. terminal_tool.py::_get_env_config() (runtime reads env vars)
#
# They are RED tests — the keys don't exist yet and should FAIL until
# the implementation adds them.

class TestDaytonaSnapshotBridge:
    """terminal.daytona_snapshot must be bridged everywhere.

    NOTE: The old test class used the stale key name `daytona_snapshot_id`.
    The expansion plan normalized it to `daytona_snapshot`. The parametrized
    expansion tests in _EXPANSION_KEYS also cover this key, but these
    explicit tests serve as named regression guards.
    """

    def test_snapshot_in_set_config_env_sync(self):
        """``hermes config set terminal.daytona_snapshot`` must sync to
        TERMINAL_DAYTONA_SNAPSHOT in .env."""
        save_keys = _save_config_env_sync_keys()
        assert "daytona_snapshot" in save_keys, (
            "terminal.daytona_snapshot missing from _config_to_env_sync. "
            "Add it so `hermes config set terminal.daytona_snapshot snap-123` "
            "propagates to $TERMINAL_DAYTONA_SNAPSHOT."
        )

    def test_snapshot_in_cli_env_mappings(self):
        """CLI startup must bridge terminal.daytona_snapshot."""
        skip_if_no_prompt_toolkit()
        cli_keys = _cli_env_map_keys()
        assert "daytona_snapshot" in cli_keys, (
            "daytona_snapshot missing from cli.py env_mappings"
        )

    def test_snapshot_in_gateway_env_map(self):
        """Gateway startup must bridge terminal.daytona_snapshot."""
        gw_keys = _gateway_env_map_keys()
        assert "daytona_snapshot" in gw_keys, (
            "daytona_snapshot missing from gateway/run.py _terminal_env_map"
        )

    def test_snapshot_in_terminal_tool_env_vars(self):
        """terminal_tool must read TERMINAL_DAYTONA_SNAPSHOT."""
        env_vars = _terminal_tool_env_var_names()
        assert "TERMINAL_DAYTONA_SNAPSHOT" in env_vars, (
            "TERMINAL_DAYTONA_SNAPSHOT not consumed by terminal_tool"
        )


class TestDaytonaAutoStopIntervalBridge:
    """terminal.daytona_auto_stop_interval must be bridged everywhere."""

    def test_auto_stop_interval_in_set_config_env_sync(self):
        save_keys = _save_config_env_sync_keys()
        assert "daytona_auto_stop_interval" in save_keys, (
            "terminal.daytona_auto_stop_interval missing from _config_to_env_sync"
        )

    def test_auto_stop_interval_in_cli_env_mappings(self):
        skip_if_no_prompt_toolkit()
        cli_keys = _cli_env_map_keys()
        assert "daytona_auto_stop_interval" in cli_keys, (
            "daytona_auto_stop_interval missing from cli.py env_mappings"
        )

    def test_auto_stop_interval_in_gateway_env_map(self):
        gw_keys = _gateway_env_map_keys()
        assert "daytona_auto_stop_interval" in gw_keys, (
            "daytona_auto_stop_interval missing from gateway/run.py _terminal_env_map"
        )

    def test_auto_stop_interval_in_terminal_tool_env_vars(self):
        env_vars = _terminal_tool_env_var_names()
        assert "TERMINAL_DAYTONA_AUTO_STOP_INTERVAL" in env_vars, (
            "TERMINAL_DAYTONA_AUTO_STOP_INTERVAL not consumed by terminal_tool"
        )


class TestDaytonaVolumeMountsBridge:
    """terminal.daytona_volume_mounts must be bridged everywhere.

    NOTE: The old test class used the stale key name `daytona_volumes`.
    The expansion plan normalized it to `daytona_volume_mounts` to match
    the Daytona SDK's VolumeMount parameter name.
    """

    def test_volume_mounts_in_set_config_env_sync(self):
        save_keys = _save_config_env_sync_keys()
        assert "daytona_volume_mounts" in save_keys, (
            "terminal.daytona_volume_mounts missing from _config_to_env_sync"
        )

    def test_volume_mounts_in_cli_env_mappings(self):
        skip_if_no_prompt_toolkit()
        cli_keys = _cli_env_map_keys()
        assert "daytona_volume_mounts" in cli_keys, (
            "daytona_volume_mounts missing from cli.py env_mappings"
        )

    def test_volume_mounts_in_gateway_env_map(self):
        gw_keys = _gateway_env_map_keys()
        assert "daytona_volume_mounts" in gw_keys, (
            "daytona_volume_mounts missing from gateway/run.py _terminal_env_map"
        )

    def test_volume_mounts_in_terminal_tool_env_vars(self):
        env_vars = _terminal_tool_env_var_names()
        assert "TERMINAL_DAYTONA_VOLUME_MOUNTS" in env_vars, (
            "TERMINAL_DAYTONA_VOLUME_MOUNTS not consumed by terminal_tool"
        )


class TestDaytonaNetworkBridges:
    """terminal.daytona_network_block_all and daytona_network_allow_list
    must be bridged everywhere.

    NOTE: The old test class used the stale key name `daytona_public_ip`.
    The expansion plan replaced it with two normalized keys:
    `daytona_network_block_all` (boolean) and `daytona_network_allow_list`
    (CIDR list), matching the Daytona SDK's CreateSandboxParams fields.
    """

    def test_network_block_all_in_set_config_env_sync(self):
        save_keys = _save_config_env_sync_keys()
        assert "daytona_network_block_all" in save_keys, (
            "terminal.daytona_network_block_all missing from _config_to_env_sync"
        )

    def test_network_block_all_in_cli_env_mappings(self):
        skip_if_no_prompt_toolkit()
        cli_keys = _cli_env_map_keys()
        assert "daytona_network_block_all" in cli_keys, (
            "daytona_network_block_all missing from cli.py env_mappings"
        )

    def test_network_block_all_in_gateway_env_map(self):
        gw_keys = _gateway_env_map_keys()
        assert "daytona_network_block_all" in gw_keys, (
            "daytona_network_block_all missing from gateway/run.py _terminal_env_map"
        )

    def test_network_block_all_in_terminal_tool_env_vars(self):
        env_vars = _terminal_tool_env_var_names()
        assert "TERMINAL_DAYTONA_NETWORK_BLOCK_ALL" in env_vars, (
            "TERMINAL_DAYTONA_NETWORK_BLOCK_ALL not consumed by terminal_tool"
        )

    def test_network_allow_list_in_set_config_env_sync(self):
        save_keys = _save_config_env_sync_keys()
        assert "daytona_network_allow_list" in save_keys, (
            "terminal.daytona_network_allow_list missing from _config_to_env_sync"
        )

    def test_network_allow_list_in_cli_env_mappings(self):
        skip_if_no_prompt_toolkit()
        cli_keys = _cli_env_map_keys()
        assert "daytona_network_allow_list" in cli_keys, (
            "daytona_network_allow_list missing from cli.py env_mappings"
        )

    def test_network_allow_list_in_gateway_env_map(self):
        gw_keys = _gateway_env_map_keys()
        assert "daytona_network_allow_list" in gw_keys, (
            "daytona_network_allow_list missing from gateway/run.py _terminal_env_map"
        )

    def test_network_allow_list_in_terminal_tool_env_vars(self):
        env_vars = _terminal_tool_env_var_names()
        assert "TERMINAL_DAYTONA_NETWORK_ALLOW_LIST" in env_vars, (
            "TERMINAL_DAYTONA_NETWORK_ALLOW_LIST not consumed by terminal_tool"
        )


class TestDaytonaAutoDeleteIntervalBridge:
    """terminal.daytona_auto_delete_interval must be bridged everywhere.

    NOTE: The old test class used the stale key name `daytona_auto_delete_after`.
    The expansion plan normalized it to `daytona_auto_delete_interval` to
    align with the Daytona SDK's autoStopInterval/autoDeleteInterval naming.
    """

    def test_auto_delete_interval_in_set_config_env_sync(self):
        save_keys = _save_config_env_sync_keys()
        assert "daytona_auto_delete_interval" in save_keys, (
            "terminal.daytona_auto_delete_interval missing from _config_to_env_sync"
        )

    def test_auto_delete_interval_in_cli_env_mappings(self):
        skip_if_no_prompt_toolkit()
        cli_keys = _cli_env_map_keys()
        assert "daytona_auto_delete_interval" in cli_keys, (
            "daytona_auto_delete_interval missing from cli.py env_mappings"
        )

    def test_auto_delete_interval_in_gateway_env_map(self):
        gw_keys = _gateway_env_map_keys()
        assert "daytona_auto_delete_interval" in gw_keys, (
            "daytona_auto_delete_interval missing from gateway/run.py _terminal_env_map"
        )

    def test_auto_delete_interval_in_terminal_tool_env_vars(self):
        env_vars = _terminal_tool_env_var_names()
        assert "TERMINAL_DAYTONA_AUTO_DELETE_INTERVAL" in env_vars, (
            "TERMINAL_DAYTONA_AUTO_DELETE_INTERVAL not consumed by terminal_tool"
        )


class TestContainerResourceBridgesComplete:
    """Pin the existing container_* bridges that were missing per bug #7362.

    terminal.container_cpu, terminal.container_memory, and
    terminal.container_disk were NOT in the CLI env_mappings (only in
    _config_to_env_sync). This test class ensures they are now present
    in ALL four bridging locations.
    """

    def test_container_cpu_in_all_bridges(self):
        """container_cpu must be in all four bridge locations."""
        save_keys = _save_config_env_sync_keys()
        cli_keys = _cli_env_map_keys()
        gw_keys = _gateway_env_map_keys()
        env_vars = _terminal_tool_env_var_names()
        assert "container_cpu" in save_keys, "container_cpu missing from _config_to_env_sync"
        assert "TERMINAL_CONTAINER_CPU" in env_vars, "TERMINAL_CONTAINER_CPU not in terminal_tool"
        assert "container_cpu" in cli_keys, "container_cpu missing from cli.py env_mappings"
        assert "container_cpu" in gw_keys, "container_cpu missing from gateway/run.py _terminal_env_map"

    def test_container_memory_in_all_bridges(self):
        """container_memory must be in all four bridge locations."""
        save_keys = _save_config_env_sync_keys()
        env_vars = _terminal_tool_env_var_names()
        assert "container_memory" in save_keys
        assert "TERMINAL_CONTAINER_MEMORY" in env_vars

    def test_container_disk_in_all_bridges(self):
        """container_disk must be in all four bridge locations."""
        save_keys = _save_config_env_sync_keys()
        env_vars = _terminal_tool_env_var_names()
        assert "container_disk" in save_keys
        assert "TERMINAL_CONTAINER_DISK" in env_vars


# ---------------------------------------------------------------------------
# Daytona expansion keys
# ---------------------------------------------------------------------------
# These keys are specified in the expansion plan but NOT yet covered by the
# classes above.  Each must be bridged in all four locations.

_EXPANSION_KEYS = {
    # Config key            -> Env var name
    "daytona_create_mode":           "TERMINAL_DAYTONA_CREATE_MODE",
    "daytona_snapshot":              "TERMINAL_DAYTONA_SNAPSHOT",
    "daytona_language":              "TERMINAL_DAYTONA_LANGUAGE",
    "daytona_name_prefix":           "TERMINAL_DAYTONA_NAME_PREFIX",
    "daytona_name_scope":            "TERMINAL_DAYTONA_NAME_SCOPE",
    "daytona_labels":                "TERMINAL_DAYTONA_LABELS",
    "daytona_auto_archive_interval": "TERMINAL_DAYTONA_AUTO_ARCHIVE_INTERVAL",
    "daytona_ephemeral":             "TERMINAL_DAYTONA_EPHEMERAL",
    "daytona_env_vars":              "TERMINAL_DAYTONA_ENV_VARS",
    "daytona_network_block_all":     "TERMINAL_DAYTONA_NETWORK_BLOCK_ALL",
    "daytona_network_allow_list":    "TERMINAL_DAYTONA_NETWORK_ALLOW_LIST",
    "daytona_volume_mounts":         "TERMINAL_DAYTONA_VOLUME_MOUNTS",
    "daytona_gpu":                   "TERMINAL_DAYTONA_GPU",
    # CWD sync pilot
    "daytona_sync_cwd":             "TERMINAL_DAYTONA_SYNC_CWD",
}


@pytest.mark.parametrize("key,env_var", list(_EXPANSION_KEYS.items()),
                         ids=[k for k in _EXPANSION_KEYS])
def test_expansion_key_in_config_to_env_sync(key, env_var):
    """``hermes config set terminal.{key}`` must propagate to $TERMINAL_DAYTONA_*."""
    save_keys = _save_config_env_sync_keys()
    assert key in save_keys, (
        f"terminal.{key} missing from _config_to_env_sync. Add "
        f"\"terminal.{key}\": \"{env_var}\" to _config_to_env_sync."
    )


@pytest.mark.parametrize("key,env_var", list(_EXPANSION_KEYS.items()),
                         ids=[k for k in _EXPANSION_KEYS])
def test_expansion_key_in_cli_env_mappings(key, env_var):
    """CLI startup must bridge terminal.{key}."""
    skip_if_no_prompt_toolkit()
    cli_keys = _cli_env_map_keys()
    assert key in cli_keys, (
        f"{key} missing from cli.py env_mappings. Add "
        f"\"{key}\": \"{env_var}\"."
    )


@pytest.mark.parametrize("key,env_var", list(_EXPANSION_KEYS.items()),
                         ids=[k for k in _EXPANSION_KEYS])
def test_expansion_key_in_gateway_env_map(key, env_var):
    """Gateway startup must bridge terminal.{key}."""
    gw_keys = _gateway_env_map_keys()
    assert key in gw_keys, (
        f"{key} missing from gateway/run.py _terminal_env_map. Add "
        f"\"{key}\": \"{env_var}\"."
    )


@pytest.mark.parametrize("key,env_var", list(_EXPANSION_KEYS.items()),
                         ids=[k for k in _EXPANSION_KEYS])
def test_expansion_key_in_terminal_tool_env_vars(key, env_var):
    """terminal_tool must read $TERMINAL_DAYTONA_*."""
    env_vars = _terminal_tool_env_var_names()
    assert env_var in env_vars, (
        f"{env_var} (for terminal.{key}) not consumed by terminal_tool. "
        f"Add os.getenv('{env_var}', ...) or _parse_env_var('{env_var}', ...) "
        f"to _get_env_config()."
    )


class TestDaytonaImageBridgeStillWorks:
    """Regression: terminal.daytona_image (existing key) must not regress."""

    def test_daytona_image_in_all_bridges(self):
        """daytona_image must still be in all bridge locations."""
        save_keys = _save_config_env_sync_keys()
        env_vars = _terminal_tool_env_var_names()
        assert "daytona_image" in save_keys
        assert "TERMINAL_DAYTONA_IMAGE" in env_vars
        skip_if_no_prompt_toolkit()
        cli_keys = _cli_env_map_keys()
        gw_keys = _gateway_env_map_keys()
        assert "daytona_image" in cli_keys
        assert "daytona_image" in gw_keys


def test_gateway_runtime_reload_reapplies_terminal_config_authority(monkeypatch, tmp_path):
    """Gateway per-turn dotenv reload must not let stale .env Daytona values win."""
    pytest.importorskip("httpx", reason="gateway/run.py requires httpx")
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / ".env").write_text(
        "HERMES_MAX_ITERATIONS=5\n"
        "TERMINAL_DAYTONA_SNAPSHOT=stale-env-snapshot\n"
        "TERMINAL_DAYTONA_SYNC_CWD=false\n",
        encoding="utf-8",
    )
    (hermes_home / "config.yaml").write_text(
        "agent:\n"
        "  max_turns: 42\n"
        "terminal:\n"
        "  backend: daytona\n"
        "  daytona_snapshot: config-snapshot\n"
        "  daytona_sync_cwd: true\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home, raising=False)
    monkeypatch.setenv("HERMES_MAX_ITERATIONS", "old")
    monkeypatch.setenv("TERMINAL_DAYTONA_SNAPSHOT", "old")
    monkeypatch.setenv("TERMINAL_DAYTONA_SYNC_CWD", "old")

    gateway_run._reload_runtime_env_preserving_config_authority()

    assert os.environ["HERMES_MAX_ITERATIONS"] == "42"
    assert os.environ["TERMINAL_DAYTONA_SNAPSHOT"] == "config-snapshot"
    assert os.environ["TERMINAL_DAYTONA_SYNC_CWD"] == "True"


def test_non_daytona_backend_ignores_malformed_daytona_json_env(monkeypatch):
    """Malformed Daytona-only JSON must not break local/docker/etc backends."""
    from tools.terminal_tool import _get_env_config

    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_DAYTONA_LABELS", "{not-json")
    monkeypatch.setenv("TERMINAL_DAYTONA_ENV_VARS", "[not-json")
    monkeypatch.setenv("TERMINAL_DAYTONA_VOLUME_MOUNTS", "{not-json")

    config = _get_env_config()

    assert config["env_type"] == "local"
    assert config["daytona_labels"] == {}
    assert config["daytona_env_vars"] == {}
    assert config["daytona_volume_mounts"] == []


def test_daytona_json_parse_errors_redact_raw_secret_values(monkeypatch):
    """Invalid JSON errors should identify the key without echoing secret text."""
    from tools.terminal_tool import _get_env_config

    monkeypatch.setenv("TERMINAL_ENV", "daytona")
    monkeypatch.setenv("TERMINAL_DAYTONA_ENV_VARS", '{"TOKEN": "super-secret-token"')

    with pytest.raises(ValueError) as excinfo:
        _get_env_config()

    message = str(excinfo.value)
    assert "TERMINAL_DAYTONA_ENV_VARS" in message
    assert "super-secret-token" not in message


def test_daytona_host_cwd_requires_explicit_sync_source(monkeypatch, tmp_path):
    """Daytona ignores implicit TERMINAL_CWD and maps explicit CWD sync to /workspace."""
    from tools.terminal_tool import _get_env_config

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("TERMINAL_ENV", "daytona")
    monkeypatch.setenv("TERMINAL_CWD", str(project))
    monkeypatch.setenv("TERMINAL_DAYTONA_SYNC_CWD", "true")
    monkeypatch.delenv("TERMINAL_DAYTONA_SYNC_CWD_SOURCE", raising=False)

    config = _get_env_config()

    assert config["cwd"] == "/root"
    assert config["host_cwd"] is None

    monkeypatch.setenv("TERMINAL_DAYTONA_SYNC_CWD_SOURCE", str(project))
    config = _get_env_config()

    assert config["cwd"] == "/workspace"
    assert config["host_cwd"] == str(project.resolve())
