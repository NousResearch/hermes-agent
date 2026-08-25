"""Behavioral regressions for the terminal-backend child-process env boundary.

The trusted Hermes process may hold provider, vault, gateway, and service
credentials.  Every model-authored terminal backend (local is covered in
``test_build_subprocess_env.py``; Docker, SSH, and Singularity converge through
``_popen_bash`` here) must launch its host-side child with a sanitized env.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.environments import base as base_env
from tools.environments import docker as docker_env
from tools.environments import local as local_env
from tools.environments import singularity as singularity_env
from tools.environments import ssh as ssh_env


_BLOCKED = {
    "BWS_ACCESS_TOKEN": "fake-bws-bootstrap",
    "DB_PASSWORD": "fake-db-password",
    "PGPASSWORD": "fake-pg-password",
    "MYSQL_PWD": "fake-mysql-password",
    "PASSWORD": "fake-bare-password",
    "OPENAI_API_KEY": "fake-provider-key",
    "GH_TOKEN": "fake-github-token",
    "AUXILIARY_VISION_API_KEY": "fake-aux-key",
    "GATEWAY_RELAY_SECRET": "fake-relay-secret",
}

_SAFE = {
    "PATH": os.environ.get("PATH", ""),
    "PWD": "/safe/cwd",
    "LANG": "C.UTF-8",
    "TERM": "xterm-256color",
    "SSH_AUTH_SOCK": "/safe/ssh-agent.sock",
    "APPTAINER_CACHEDIR": "/safe/apptainer-cache",
    "HERMES_TEST_BENIGN": "keep-me",
}

_CONTAINER_TUNNELS = {
    "APPTAINERENV_BWS_ACCESS_TOKEN": "fake-tunneled-bws",
    "SINGULARITYENV_GH_TOKEN": "fake-tunneled-github",
    "APPTAINERENV_OPENAI_API_KEY": "fake-tunneled-provider",
    "SINGULARITYENV_DB_PASSWORD": "fake-tunneled-password",
}


class _DummyProcess:
    def __init__(self) -> None:
        self.stdin = None
        self.stdout = None
        self.stderr = None
        self.returncode = 0
        self.pid = 12345

    def poll(self):
        return self.returncode


def _capture_popen(monkeypatch):
    calls: list[tuple[list[str], dict]] = []

    def fake_popen(args, **kwargs):
        calls.append((list(args), kwargs))
        return _DummyProcess()

    monkeypatch.setattr(base_env.subprocess, "Popen", fake_popen)
    return calls


def _plant_parent_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    for key, value in {**_SAFE, **_BLOCKED, **_CONTAINER_TUNNELS}.items():
        monkeypatch.setenv(key, value)


def _assert_child_env_is_sanitized(child_env: dict[str, str]) -> None:
    for key in (*_BLOCKED, *_CONTAINER_TUNNELS):
        assert key not in child_env, f"{key} crossed the child-process boundary"
    for key, value in _SAFE.items():
        assert child_env.get(key) == value, f"benign control {key} was not preserved"


def _make_docker_exec_env():
    env = docker_env.DockerEnvironment.__new__(docker_env.DockerEnvironment)
    env._container_id = "container-id"
    env._docker_exe = "docker"
    env._forward_env = []
    env._env = {}
    env._profile_scoped_passthrough = False
    env._init_env_args = []
    return env


def _make_ssh_exec_env(tmp_path: Path):
    env = ssh_env.SSHEnvironment.__new__(ssh_env.SSHEnvironment)
    env.host = "example.invalid"
    env.user = "hermes"
    env.port = 22
    env.key_path = ""
    env.control_socket = tmp_path / "control.sock"
    return env


def _make_singularity_exec_env():
    env = singularity_env.SingularityEnvironment.__new__(
        singularity_env.SingularityEnvironment
    )
    env.executable = "apptainer"
    env.instance_id = "hermes_test"
    env._instance_started = True
    return env


@pytest.mark.parametrize("backend", ["docker", "ssh", "singularity"])
def test_every_remote_terminal_exec_uses_shared_sanitized_popen_boundary(
    backend, monkeypatch, tmp_path
):
    """Exercise each production ``_run_bash`` call shape, not a detached helper."""
    _plant_parent_env(monkeypatch, tmp_path)
    calls = _capture_popen(monkeypatch)

    if backend == "docker":
        _make_docker_exec_env()._run_bash("true")
    elif backend == "ssh":
        _make_ssh_exec_env(tmp_path)._run_bash("true")
    else:
        env = _make_singularity_exec_env()
        env._run_bash("true")
        # Prevent the synthetic object's destructor from invoking a real
        # ``apptainer instance stop`` after the assertion.
        env._instance_started = False

    assert len(calls) == 1
    _assert_child_env_is_sanitized(calls[0][1]["env"])


def test_shared_popen_boundary_sanitizes_caller_supplied_base_env(monkeypatch, tmp_path):
    """An explicit overlay cannot re-open the boundary or skip benign controls."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    calls = _capture_popen(monkeypatch)
    supplied = {**_SAFE, **_BLOCKED, **_CONTAINER_TUNNELS}

    base_env._popen_bash(["bash", "-c", "true"], env=supplied)

    assert len(calls) == 1
    _assert_child_env_is_sanitized(calls[0][1]["env"])


def test_shared_popen_boundary_accepts_empty_base_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    calls = _capture_popen(monkeypatch)

    base_env._popen_bash(["bash", "-c", "true"], env={})

    assert len(calls) == 1
    assert isinstance(calls[0][1]["env"], dict)
    assert not set(_BLOCKED) & set(calls[0][1]["env"])


def test_mixed_case_credential_names_are_denied_for_windows_semantics(
    monkeypatch, tmp_path
):
    """Windows treats env keys case-insensitively; the filter must do the same."""
    monkeypatch.setattr(local_env, "_IS_WINDOWS", True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    base = {
        "Path": "C:/Windows/System32",
        "bWs_AcCeSs_ToKeN": "fake-bws",
        "Db_PaSsWoRd": "fake-password",
        "oPeNaI_aPi_KeY": "fake-provider",
        "gH_tOkEn": "fake-github",
        "Safe_Control": "keep-me",
        "HERMES_HOME": str(tmp_path / "hermes-home"),
    }

    terminal_env = local_env.build_subprocess_env(base=base)
    assert terminal_env.get("Safe_Control") == "keep-me"
    assert terminal_env.get("Path", "").endswith("C:/Windows/System32")
    assert not {k for k in base if k.casefold() in {
        "bws_access_token", "db_password", "openai_api_key", "gh_token"
    }} & set(terminal_env)

    with patch.dict(os.environ, base, clear=True):
        for inherit_credentials in (False, True):
            nonterminal_env = local_env.hermes_subprocess_env(
                inherit_credentials=inherit_credentials
            )
            assert "gH_tOkEn" not in nonterminal_env
            assert "bWs_AcCeSs_ToKeN" not in nonterminal_env
            assert "Db_PaSsWoRd" not in nonterminal_env
            provider_present = any(
                k.casefold() == "openai_api_key" and v == "fake-provider"
                for k, v in nonterminal_env.items()
            )
            if inherit_credentials:
                assert provider_present, (
                    "inherit_credentials=True must preserve the provider "
                    "credential regardless of key casing"
                )
            else:
                assert not provider_present


def test_nested_child_cannot_recover_scrubbed_parent_credentials(monkeypatch, tmp_path):
    """A sanitized child and its inheriting grandchild both lack the secrets."""
    _plant_parent_env(monkeypatch, tmp_path)
    env = local_env.build_subprocess_env()
    grandchild = (
        "import json, os; "
        "print(json.dumps({k: k in os.environ for k in "
        f"{list(_BLOCKED)!r}}}))"
    )
    child = (
        "import subprocess, sys; "
        f"subprocess.run([sys.executable, '-c', {grandchild!r}], check=True)"
    )

    result = subprocess.run(
        [sys.executable, "-c", child],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )

    assert json.loads(result.stdout) == {key: False for key in _BLOCKED}


def _capture_singularity_run(monkeypatch):
    calls: list[tuple[list[str], dict]] = []

    def fake_run(cmd, **kwargs):
        calls.append((list(cmd), kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(singularity_env.subprocess, "run", fake_run)
    return calls


@pytest.mark.parametrize("stage", ["preflight", "start", "cleanup"])
def test_singularity_lifecycle_processes_use_sanitized_env(
    stage, monkeypatch, tmp_path
):
    """The backend's start and control children share the exec scrub contract."""
    _plant_parent_env(monkeypatch, tmp_path)
    calls = _capture_singularity_run(monkeypatch)

    if stage == "preflight":
        monkeypatch.setattr(
            singularity_env, "_find_singularity_executable", lambda: "apptainer"
        )
        assert singularity_env._ensure_singularity_available() == "apptainer"
    else:
        env = _make_singularity_exec_env()
        env.image = "image.sif"
        env._persistent = False
        env._overlay_dir = None
        env._memory = 0
        env._cpu = 0
        if stage == "start":
            env._instance_started = False
            import tools.credential_files as credential_files

            monkeypatch.setattr(credential_files, "get_credential_file_mounts", lambda: [])
            monkeypatch.setattr(credential_files, "get_skills_directory_mount", lambda: [])
            env._start_instance()
        else:
            env.cleanup()

    assert len(calls) == 1
    _assert_child_env_is_sanitized(calls[0][1]["env"])


def test_singularity_image_build_gets_only_explicit_registry_auth(
    monkeypatch, tmp_path
):
    """The deliberate registry-auth exception is narrow, not full inheritance."""
    _plant_parent_env(monkeypatch, tmp_path)
    registry_auth = {
        "APPTAINER_DOCKER_USERNAME": "registry-user",
        "APPTAINER_DOCKER_PASSWORD": "fake-registry-password",
        "SINGULARITY_DOCKER_USERNAME": "legacy-user",
        "SINGULARITY_DOCKER_PASSWORD": "fake-legacy-password",
        "DOCKER_USERNAME": "plain-user",
        "DOCKER_PASSWORD": "fake-plain-password",
    }
    for key, value in registry_auth.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(singularity_env, "_get_apptainer_cache_dir", lambda: tmp_path)
    calls = _capture_singularity_run(monkeypatch)

    result = singularity_env._get_or_build_sif(
        "docker://example.invalid/private:latest", "apptainer"
    )

    assert result.endswith("example.invalid-private-latest.sif")
    assert len(calls) == 1
    child_env = calls[0][1]["env"]
    for key, value in registry_auth.items():
        assert child_env.get(key) == value
    for key in (*_BLOCKED, *_CONTAINER_TUNNELS):
        assert key not in child_env
    assert child_env.get("HERMES_TEST_BENIGN") == "keep-me"


def test_docker_explicit_forward_cannot_export_hermes_internal_secret(
    monkeypatch, tmp_path
):
    """Explicit password passthrough remains valid; BWS bootstrap auth never is."""
    _plant_parent_env(monkeypatch, tmp_path)
    env = _make_docker_exec_env()
    env._forward_env = ["BWS_ACCESS_TOKEN", "DB_PASSWORD"]
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {})

    args = env._build_init_env_args()

    assert "DB_PASSWORD=fake-db-password" in args
    assert not any(arg.startswith("BWS_ACCESS_TOKEN=") for arg in args)
