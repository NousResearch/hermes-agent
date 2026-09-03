"""Tests for typed child-process authority contracts."""

from __future__ import annotations

import json
import os
import subprocess
from types import SimpleNamespace

import pytest

from tools.child_process_authority import (
    BWS_VAULT_GRANTS,
    CONTAINER_REGISTRY_AUTH_KEYS,
    KANBAN_WORKER_GRANT_PREFIXES,
    KANBAN_WORKER_GRANTS,
    ChildStdinPolicy,
    build_child_process_env,
    build_spawn_receipt,
    bws_vault_spec,
    checkpoint_git_spec,
    container_image_build_spec,
    interactive_hermes_pty_spec,
    model_driver_spec,
    op_vault_spec,
    probe_spec,
    secret_helper_spec,
    stdin_for_spec,
    trusted_hermes_child_spec,
)


@pytest.fixture(autouse=True)
def _clear_multiplex_state():
    from agent.secret_scope import (
        reset_secret_scope,
        set_multiplex_active,
        set_secret_scope,
    )

    set_multiplex_active(False)
    token = set_secret_scope(None)
    try:
        yield
    finally:
        reset_secret_scope(token)
        set_multiplex_active(False)


def _plant(monkeypatch):
    values = {
        "PATH": os.environ.get("PATH", "/usr/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "HTTPS_PROXY": "http://proxy.invalid:8080",
        "OPENAI_API_KEY": "provider-secret",
        "ACME_LOGIN": "arbitrary-profile-secret",
        "GH_TOKEN": "github-tool-secret",
        "DISCORD_BOT_TOKEN": "gateway-transport-secret",
        "BWS_ACCESS_TOKEN": "vault-bootstrap-secret",
        "BWS_SERVER_URL": "https://vault.example.invalid",
        "GATEWAY_RELAY_SECRET": "relay-secret",
        "AUXILIARY_VISION_API_KEY": "aux-secret",
        "_HERMES_GATEWAY": "1",
        "HERMES_KANBAN_TASK": "task-a",
        "HERMES_KANBAN_RUN_ID": "run-a",
        "HERMES_SESSION_ID": "session-a",
        "HERMES_PROFILE": "profile-a",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    return values


def test_model_driver_gets_model_auth_not_ambient_tool_or_role_authority(monkeypatch):
    _plant(monkeypatch)

    env = build_child_process_env(model_driver_spec(source="test:model-driver"))

    assert env["OPENAI_API_KEY"] == "provider-secret"
    assert env["PATH"]
    assert "ACME_LOGIN" not in env
    assert "GH_TOKEN" not in env
    assert "DISCORD_BOT_TOKEN" not in env
    assert "BWS_ACCESS_TOKEN" not in env
    assert "GATEWAY_RELAY_SECRET" not in env
    assert "_HERMES_GATEWAY" not in env
    assert "HERMES_KANBAN_TASK" not in env
    assert "HERMES_SESSION_ID" not in env


def test_kanban_authority_crosses_only_the_declared_model_driver_edge(monkeypatch):
    _plant(monkeypatch)
    spec = model_driver_spec(
        source="test:kanban-model-driver",
        grants=KANBAN_WORKER_GRANTS,
        grant_prefixes=KANBAN_WORKER_GRANT_PREFIXES,
    )

    env = build_child_process_env(spec)

    assert env["HERMES_KANBAN_TASK"] == "task-a"
    assert env["HERMES_KANBAN_RUN_ID"] == "run-a"
    assert env["HERMES_SESSION_ID"] == "session-a"
    assert env["HERMES_PROFILE"] == "profile-a"
    assert "_HERMES_GATEWAY" not in env
    assert "DISCORD_BOT_TOKEN" not in env


def test_trusted_hermes_child_gets_profile_tools_not_gateway_transport(monkeypatch):
    _plant(monkeypatch)
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "op-token")
    monkeypatch.setenv("OP_SESSION_acme", "session-token")

    env = build_child_process_env(
        trusted_hermes_child_spec(source="test:trusted-runtime")
    )

    assert env["OPENAI_API_KEY"] == "provider-secret"
    assert env["ACME_LOGIN"] == "arbitrary-profile-secret"
    assert env["GH_TOKEN"] == "github-tool-secret"
    assert env["AUXILIARY_VISION_API_KEY"] == "aux-secret"
    assert env["BWS_ACCESS_TOKEN"] == "vault-bootstrap-secret"
    assert env["OP_SERVICE_ACCOUNT_TOKEN"] == "op-token"
    assert env["OP_SESSION_acme"] == "session-token"
    assert "DISCORD_BOT_TOKEN" not in env
    assert "GATEWAY_RELAY_SECRET" not in env
    assert "_HERMES_GATEWAY" not in env
    assert "HERMES_KANBAN_TASK" not in env


def test_explicit_profile_home_override_wins_context_home(monkeypatch):
    from tools.environments import local

    _plant(monkeypatch)
    monkeypatch.setattr(
        local,
        "_inject_context_hermes_home",
        lambda env: env.__setitem__("HERMES_HOME", "/ambient-profile"),
    )

    env = build_child_process_env(
        trusted_hermes_child_spec(source="test:target-profile"),
        overrides={"HERMES_HOME": "/target-profile"},
    )

    assert env["HERMES_HOME"] == "/target-profile"


def test_overrides_require_positive_policy_and_block_execution_authority(monkeypatch):
    _plant(monkeypatch)
    spec = trusted_hermes_child_spec(source="test:override")

    env = build_child_process_env(
        spec,
        overrides={
            "SAFE_OVERRIDE": "no-longer-ambiently-safe",
            "LD_PRELOAD": "/tmp/evil.so",
            "DYLD_INSERT_LIBRARIES": "/tmp/evil.dylib",
            "PYTHONPATH": "/tmp/evil-python",
            "PYTHONHOME": "/tmp/evil-home",
            "NODE_OPTIONS": "--require=/tmp/evil.js",
            "BASH_ENV": "/tmp/evil-bashrc",
            "ENV": "/tmp/evil-shrc",
            "APPTAINERENV_LD_PRELOAD": "/tmp/wrapped-evil.so",
            "SINGULARITYENV_NODE_OPTIONS": "--require=/tmp/wrapped-evil.js",
            "DISCORD_BOT_TOKEN": "smuggled",
            "APPTAINERENV_GATEWAY_RELAY_SECRET": "wrapped-smuggle",
            "_HERMES_GATEWAY": "1",
            "HERMES_HOME": "/target-profile",
        },
    )

    assert env["HERMES_HOME"] == "/target-profile"
    for denied in (
        "SAFE_OVERRIDE",
        "LD_PRELOAD",
        "DYLD_INSERT_LIBRARIES",
        "PYTHONPATH",
        "PYTHONHOME",
        "NODE_OPTIONS",
        "BASH_ENV",
        "ENV",
        "APPTAINERENV_LD_PRELOAD",
        "SINGULARITYENV_NODE_OPTIONS",
        "DISCORD_BOT_TOKEN",
        "APPTAINERENV_GATEWAY_RELAY_SECRET",
        "_HERMES_GATEWAY",
    ):
        assert denied not in env


def test_safe_baseline_is_inherited_but_not_override_authority(monkeypatch):
    import hermes_constants
    from tools.environments import local

    monkeypatch.setattr(local, "_inject_context_hermes_home", lambda _env: None)
    monkeypatch.setattr(hermes_constants, "apply_subprocess_home_env", lambda _env: None)
    source = {
        "PATH": "/ambient/bin",
        "HOME": "/ambient/home",
        "TMP": "/ambient/tmp",
        "XDG_CONFIG_HOME": "/ambient/xdg",
    }
    overrides = {
        "PATH": "/attacker/bin",
        "HOME": "/attacker/home",
        "TMP": "/attacker/tmp",
        "XDG_CONFIG_HOME": "/attacker/xdg",
    }

    for spec in (
        probe_spec(source="test:probe-baseline"),
        checkpoint_git_spec(source="test:checkpoint-baseline"),
    ):
        env = build_child_process_env(spec, source_env=source, overrides=overrides)
        assert env["PATH"] == "/ambient/bin"
        assert env["HOME"] == "/ambient/home"
        assert env["TMP"] == "/ambient/tmp"
        assert env["XDG_CONFIG_HOME"] == "/ambient/xdg"


def test_bws_vault_edge_is_minimal_and_keeps_network_controls(monkeypatch):
    _plant(monkeypatch)
    spec = bws_vault_spec(source="test:bws")

    env = build_child_process_env(
        spec,
        overrides={
            "BWS_ACCESS_TOKEN": "edge-token",
            "NO_COLOR": "1",
        },
    )

    assert set(BWS_VAULT_GRANTS) <= set(env)
    assert env["BWS_ACCESS_TOKEN"] == "edge-token"
    assert env["BWS_SERVER_URL"] == "https://vault.example.invalid"
    assert env["HTTPS_PROXY"] == "http://proxy.invalid:8080"
    assert env["NO_COLOR"] == "1"
    assert "OPENAI_API_KEY" not in env
    assert "ACME_LOGIN" not in env
    assert "GH_TOKEN" not in env
    assert "HERMES_KANBAN_TASK" not in env


def test_op_vault_edge_admits_only_op_auth_family(monkeypatch):
    _plant(monkeypatch)
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "op-token")
    monkeypatch.setenv("OP_SESSION_acme", "session-token")
    monkeypatch.setenv("OP_LOAD_DESKTOP_APP_SETTINGS", "false")
    monkeypatch.setenv("OP_CACHE", "false")

    env = build_child_process_env(op_vault_spec(source="test:op"))

    assert env["OP_SERVICE_ACCOUNT_TOKEN"] == "op-token"
    assert env["OP_SESSION_acme"] == "session-token"
    assert env["OP_LOAD_DESKTOP_APP_SETTINGS"] == "false"
    assert env["OP_CACHE"] == "false"
    assert "OPENAI_API_KEY" not in env
    assert "ACME_LOGIN" not in env
    assert "GH_TOKEN" not in env


def test_secret_helper_keeps_profile_data_but_not_parent_role(monkeypatch):
    _plant(monkeypatch)

    env = build_child_process_env(
        secret_helper_spec(source="test:helper"),
        overrides={"HERMES_SECRET_KEY": "WANTED"},
    )

    assert env["ACME_LOGIN"] == "arbitrary-profile-secret"
    assert env["OPENAI_API_KEY"] == "provider-secret"
    assert env["GH_TOKEN"] == "github-tool-secret"
    assert env["HERMES_SECRET_KEY"] == "WANTED"
    assert "DISCORD_BOT_TOKEN" not in env
    assert "_HERMES_GATEWAY" not in env
    assert "HERMES_KANBAN_TASK" not in env


def test_checkpoint_git_does_not_inherit_or_override_git_injection(monkeypatch):
    _plant(monkeypatch)
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "credential.helper")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "!evil")
    monkeypatch.setenv("GIT_ASKPASS", "/tmp/askpass")
    spec = checkpoint_git_spec(source="test:checkpoint")

    env = build_child_process_env(
        spec,
        overrides={
            "GIT_DIR": "/tmp/store",
            "GIT_WORK_TREE": "/tmp/work",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_COUNT": "2",
            "GIT_SSH_COMMAND": "ssh -oProxyCommand=evil",
            "APPTAINERENV_GIT_SSH_COMMAND": "wrapped-evil",
            "NODE_OPTIONS": "--require=/tmp/evil.js",
        },
    )

    assert env["GIT_DIR"] == "/tmp/store"
    assert env["GIT_WORK_TREE"] == "/tmp/work"
    assert "GIT_CONFIG_COUNT" not in env
    assert "GIT_CONFIG_KEY_0" not in env
    assert "GIT_CONFIG_VALUE_0" not in env
    assert "GIT_ASKPASS" not in env
    assert "GIT_SSH_COMMAND" not in env
    assert "APPTAINERENV_GIT_SSH_COMMAND" not in env
    assert "NODE_OPTIONS" not in env
    assert "HTTPS_PROXY" not in env
    assert "OPENAI_API_KEY" not in env


def test_container_image_build_gets_only_registry_auth(monkeypatch):
    _plant(monkeypatch)
    for index, key in enumerate(CONTAINER_REGISTRY_AUTH_KEYS):
        monkeypatch.setenv(key, f"registry-{index}")

    env = build_child_process_env(
        container_image_build_spec(source="test:image-build"),
        overrides={
            "APPTAINER_TMPDIR": "/tmp/apptainer",
            "APPTAINER_CACHEDIR": "/tmp/cache",
        },
    )

    for index, key in enumerate(CONTAINER_REGISTRY_AUTH_KEYS):
        assert env[key] == f"registry-{index}"
    assert env["APPTAINER_TMPDIR"] == "/tmp/apptainer"
    assert env["APPTAINER_CACHEDIR"] == "/tmp/cache"
    assert "OPENAI_API_KEY" not in env
    assert "BWS_ACCESS_TOKEN" not in env
    assert "DISCORD_BOT_TOKEN" not in env


def test_interactive_pty_is_typed_and_stdin_is_not_synthesized(monkeypatch):
    _plant(monkeypatch)
    spec = interactive_hermes_pty_spec(source="test:pty")
    env = build_child_process_env(spec)

    assert spec.stdin is ChildStdinPolicy.PTY
    assert stdin_for_spec(spec) is None
    assert env["OPENAI_API_KEY"] == "provider-secret"
    assert "DISCORD_BOT_TOKEN" not in env


def test_closed_probe_stdin_and_environment(monkeypatch):
    _plant(monkeypatch)
    monkeypatch.setenv("TMP", "/tmp/ambient")
    spec = probe_spec(source="test:probe")
    env = build_child_process_env(
        spec,
        overrides={
            "TMP": "/tmp/probe",
            "LD_PRELOAD": "/tmp/evil.so",
            "PYTHONPATH": "/tmp/evil-python",
            "NODE_OPTIONS": "--require=/tmp/evil.js",
            "GIT_CONFIG_COUNT": "1",
            "SINGULARITYENV_LD_PRELOAD": "/tmp/wrapped-evil.so",
        },
    )

    assert stdin_for_spec(spec) is subprocess.DEVNULL
    assert env["TMP"] == "/tmp/ambient"
    assert "OPENAI_API_KEY" not in env
    assert "ACME_LOGIN" not in env
    assert "BWS_ACCESS_TOKEN" not in env
    assert "LD_PRELOAD" not in env
    assert "PYTHONPATH" not in env
    assert "NODE_OPTIONS" not in env
    assert "GIT_CONFIG_COUNT" not in env
    assert "SINGULARITYENV_LD_PRELOAD" not in env


def test_active_multiplex_scope_is_authoritative(monkeypatch):
    from agent.secret_scope import (
        reset_secret_scope,
        set_multiplex_active,
        set_secret_scope,
    )

    _plant(monkeypatch)
    set_multiplex_active(True)
    token = set_secret_scope({
        "OPENAI_API_KEY": "profile-b-provider",
        "ACME_LOGIN": "profile-b-arbitrary",
    })
    try:
        env = build_child_process_env(
            trusted_hermes_child_spec(source="test:profile-b")
        )
    finally:
        reset_secret_scope(token)
        set_multiplex_active(False)

    assert env["OPENAI_API_KEY"] == "profile-b-provider"
    assert env["ACME_LOGIN"] == "profile-b-arbitrary"
    assert env.get("HERMES_PROFILE") == "profile-a"
    assert "DISCORD_BOT_TOKEN" not in env


def test_multiplex_without_scope_fails_closed(monkeypatch):
    from agent.secret_scope import set_multiplex_active

    _plant(monkeypatch)
    set_multiplex_active(True)
    with pytest.raises(RuntimeError, match="no active profile secret scope"):
        build_child_process_env(trusted_hermes_child_spec(source="test:unscoped"))


def test_spawn_receipt_contains_keys_not_values(monkeypatch):
    _plant(monkeypatch)
    spec = bws_vault_spec(source="test:receipt")
    env = build_child_process_env(
        spec,
        overrides={"BWS_ACCESS_TOKEN": "receipt-canary-secret"},
    )

    receipt = build_spawn_receipt(
        spec,
        argv=["/usr/bin/bws", "secret", "list"],
        env=env,
    )
    encoded = json.dumps(receipt, sort_keys=True)

    assert "BWS_ACCESS_TOKEN" in encoded
    assert "receipt-canary-secret" not in encoded
    assert "provider-secret" not in encoded
    assert receipt["intent"] == "vault_cli"
    assert len(receipt["policy_sha256"]) == 64


def test_policy_hash_covers_authority_manifest(monkeypatch):
    from tools import child_process_authority as authority

    baseline = authority._policy_hash()
    monkeypatch.setattr(
        authority,
        "_CONTROL_PLANE_EXACT",
        authority._CONTROL_PLANE_EXACT | {"HERMES_NEW_CONTROL_AUTHORITY"},
    )

    assert authority._policy_hash() != baseline


def test_policy_hash_covers_provider_registry_authority(monkeypatch):
    from hermes_cli import auth
    from tools import child_process_authority as authority

    baseline = authority._policy_hash()
    registry = dict(auth.PROVIDER_REGISTRY)
    registry["policy-hash-canary"] = SimpleNamespace(
        api_key_env_vars=("POLICY_HASH_CANARY_API_KEY",),
        base_url_env_var="POLICY_HASH_CANARY_BASE_URL",
    )
    monkeypatch.setattr(auth, "PROVIDER_REGISTRY", registry)

    assert "POLICY_HASH_CANARY_API_KEY" in authority._provider_env_names()
    assert authority._policy_hash() != baseline


def test_spawn_receipt_hash_uses_current_provider_registry(monkeypatch):
    from hermes_cli import auth
    from tools import child_process_authority as authority

    spec = probe_spec(source="test:receipt-policy")
    baseline = authority.build_spawn_receipt(spec, argv=["probe"], env={})[
        "policy_sha256"
    ]
    registry = dict(auth.PROVIDER_REGISTRY)
    registry["receipt-policy-canary"] = SimpleNamespace(
        api_key_env_vars=("RECEIPT_POLICY_CANARY_API_KEY",),
        base_url_env_var="RECEIPT_POLICY_CANARY_BASE_URL",
    )
    monkeypatch.setattr(auth, "PROVIDER_REGISTRY", registry)

    current = authority.build_spawn_receipt(spec, argv=["probe"], env={})[
        "policy_sha256"
    ]
    assert current == authority._policy_hash()
    assert current != baseline


class _DummyProcess:
    def __init__(self):
        self.stdin = None
        self.stdout = None
        self.stderr = None
        self.returncode = 0
        self.pid = 1234


def test_shared_remote_terminal_popen_sanitizes_omitted_env(monkeypatch):
    from tools.environments import base as base_env

    _plant(monkeypatch)
    monkeypatch.setenv("bws_access_token", "lowercase-vault-bootstrap")
    monkeypatch.setenv("OP_SESSION_acme", "op-session-secret")
    calls = []

    def fake_popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _DummyProcess()

    monkeypatch.setattr(base_env.subprocess, "Popen", fake_popen)
    base_env._popen_bash(["bash", "-c", "true"])

    child_env = calls[0][1]["env"]
    assert child_env["PATH"]
    assert "OPENAI_API_KEY" not in child_env
    assert "BWS_ACCESS_TOKEN" not in child_env
    assert "bws_access_token" not in child_env
    assert "OP_SESSION_acme" not in child_env
    assert "DISCORD_BOT_TOKEN" not in child_env
    assert "APPTAINERENV_GATEWAY_RELAY_SECRET" not in child_env


def test_bitwarden_source_uses_minimal_vault_edge(monkeypatch):
    from pathlib import Path

    from agent.secret_sources import bitwarden

    _plant(monkeypatch)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="[]", stderr="")

    monkeypatch.setattr(bitwarden.subprocess, "run", fake_run)
    secrets, warnings = bitwarden._run_bws_list(
        Path("/usr/bin/bws"),
        "edge-token",
        "project-id",
        server_url="https://vault.example.invalid",
    )

    assert secrets == {}
    assert warnings == []
    child_env = calls[0][1]["env"]
    assert child_env["BWS_ACCESS_TOKEN"] == "edge-token"
    assert child_env["BWS_SERVER_URL"] == "https://vault.example.invalid"
    assert "OPENAI_API_KEY" not in child_env
    assert "ACME_LOGIN" not in child_env
    assert calls[0][1]["stdin"] is subprocess.DEVNULL


def test_onepassword_source_uses_minimal_vault_edge(monkeypatch):
    from agent.secret_sources import onepassword

    _plant(monkeypatch)
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "op-token")
    monkeypatch.setenv("OP_SESSION_acme", "session-token")

    env = onepassword._op_child_env("override-token")

    assert env["OP_SERVICE_ACCOUNT_TOKEN"] == "override-token"
    assert env["OP_SESSION_acme"] == "session-token"
    assert "OPENAI_API_KEY" not in env
    assert "ACME_LOGIN" not in env


def test_checkpoint_git_env_isolated_at_real_callsite(monkeypatch, tmp_path):
    from tools import checkpoint_manager

    _plant(monkeypatch)
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_ASKPASS", "/tmp/askpass")
    store = tmp_path / "store"
    work = tmp_path / "work"
    store.mkdir()
    work.mkdir()

    env = checkpoint_manager._git_env(store, str(work))

    assert env["GIT_DIR"] == str(store)
    assert env["GIT_WORK_TREE"] == str(work.resolve())
    assert env["GIT_CONFIG_GLOBAL"] == os.devnull
    assert "GIT_CONFIG_COUNT" not in env
    assert "GIT_ASKPASS" not in env
    assert "OPENAI_API_KEY" not in env


def test_singularity_image_build_uses_explicit_registry_grants(monkeypatch, tmp_path):
    from tools.environments import singularity

    _plant(monkeypatch)
    for index, key in enumerate(CONTAINER_REGISTRY_AUTH_KEYS):
        monkeypatch.setenv(key, f"registry-{index}")
    monkeypatch.setattr(singularity, "_get_apptainer_cache_dir", lambda: tmp_path)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(singularity.subprocess, "run", fake_run)
    result = singularity._get_or_build_sif(
        "docker://example.invalid/private:latest",
        "apptainer",
    )

    assert result.endswith("example.invalid-private-latest.sif")
    child_env = calls[0][1]["env"]
    for index, key in enumerate(CONTAINER_REGISTRY_AUTH_KEYS):
        assert child_env[key] == f"registry-{index}"
    assert "OPENAI_API_KEY" not in child_env
    assert "BWS_ACCESS_TOKEN" not in child_env


def test_codex_version_probe_uses_minimal_closed_edge(monkeypatch):
    from agent.transports import codex_app_server

    _plant(monkeypatch)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="codex-cli 0.125.0\n",
            stderr="",
        )

    monkeypatch.setattr(codex_app_server.subprocess, "run", fake_run)

    ok, version = codex_app_server.check_codex_binary()

    assert ok is True
    assert version == "0.125.0"
    child_env = calls[0][1]["env"]
    assert child_env["PATH"]
    assert "OPENAI_API_KEY" not in child_env
    assert "ACME_LOGIN" not in child_env
    assert "HERMES_KANBAN_TASK" not in child_env
    assert calls[0][1]["stdin"] is subprocess.DEVNULL


def test_bitwarden_libc_probe_uses_minimal_closed_edge(monkeypatch):
    from agent.secret_sources import bitwarden

    _plant(monkeypatch)
    calls = []

    monkeypatch.setattr(bitwarden.platform, "system", lambda: "Linux")
    monkeypatch.setattr(bitwarden.platform, "machine", lambda: "x86_64")

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="ldd (GNU libc) 2.39\n",
            stderr="",
        )

    monkeypatch.setattr(bitwarden.subprocess, "run", fake_run)

    asset = bitwarden._platform_asset_name()

    assert asset.startswith("bws-x86_64-unknown-linux-gnu-")
    child_env = calls[0][1]["env"]
    assert child_env["PATH"]
    assert "OPENAI_API_KEY" not in child_env
    assert "BWS_ACCESS_TOKEN" not in child_env
    assert "ACME_LOGIN" not in child_env
    assert calls[0][1]["stdin"] is subprocess.DEVNULL


def test_host_supervisor_probes_use_minimal_closed_edges(monkeypatch):
    from tui_gateway import host_supervisor

    _plant(monkeypatch)
    calls = []

    def fake_check_output(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd[0] == "git":
            return "abc123\n"
        return "python -m tui_gateway.compute_host\n"

    def fail_read_bytes(_path):
        raise OSError("force ps fallback")

    monkeypatch.setattr(host_supervisor.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(host_supervisor.Path, "read_bytes", fail_read_bytes)

    assert host_supervisor._build_sha() == "abc123"
    assert "tui_gateway.compute_host" in host_supervisor._pid_command(1234)

    assert len(calls) == 2
    for _cmd, kwargs in calls:
        child_env = kwargs["env"]
        assert child_env["PATH"]
        assert "OPENAI_API_KEY" not in child_env
        assert "ACME_LOGIN" not in child_env
        assert "HERMES_KANBAN_TASK" not in child_env
        assert kwargs["stdin"] is subprocess.DEVNULL


def test_checkpoint_store_init_uses_minimal_typed_edge(monkeypatch, tmp_path):
    from tools import checkpoint_manager

    _plant(monkeypatch)
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_ASKPASS", "/tmp/askpass")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(checkpoint_manager.subprocess, "run", fake_run)
    monkeypatch.setattr(
        checkpoint_manager,
        "_run_git",
        lambda *_args, **_kwargs: (True, "", ""),
    )

    store = tmp_path / "checkpoints" / "store"
    error = checkpoint_manager._init_store(store, str(tmp_path))

    assert error is None
    assert len(calls) == 1
    child_env = calls[0][1]["env"]
    assert child_env["GIT_CONFIG_GLOBAL"] == os.devnull
    assert child_env["GIT_CONFIG_SYSTEM"] == os.devnull
    assert child_env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert "GIT_CONFIG_COUNT" not in child_env
    assert "GIT_ASKPASS" not in child_env
    assert "OPENAI_API_KEY" not in child_env
    assert "ACME_LOGIN" not in child_env
    assert calls[0][1]["stdin"] is subprocess.DEVNULL
