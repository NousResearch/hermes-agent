"""Contracts for the opt-in macOS Seatbelt wrapper on the local backend."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest


def test_seatbelt_profile_and_argv_are_scoped(tmp_path):
    from tools.environments.local_seatbelt import (
        build_seatbelt_profile,
        seatbelt_spawn_args,
    )

    cwd = tmp_path / 'work "tree"'
    temp_dir = tmp_path / "session-temp"
    runtime = tmp_path / "runtime"
    profile = build_seatbelt_profile(
        cwd=str(cwd),
        temp_dir=str(temp_dir),
        read_paths=[str(runtime)],
        network_policy="deny",
    )

    assert profile.startswith("(version 1)\n(deny default)\n")
    assert '(deny network*)' in profile
    assert f'(subpath {str(cwd)!r})' not in profile  # SBPL paths use escaped strings, not repr.
    assert f'(subpath "{str(temp_dir)}")' in profile
    assert f'(subpath "{str(runtime)}")' in profile
    write_rules = [line for line in profile.splitlines() if line.startswith("(allow file-write")]
    assert write_rules
    assert all("subpath" in line or 'literal "/dev/null"' in line for line in write_rules)

    original = ["/bin/bash", "-c", "printf ok"]
    assert seatbelt_spawn_args(original, None) == original
    assert seatbelt_spawn_args(original, "/tmp/hermes.sb") == [
        "/usr/bin/sandbox-exec",
        "-f",
        "/tmp/hermes.sb",
        *original,
    ]
    allow_profile = build_seatbelt_profile(
        cwd=str(cwd), temp_dir=str(temp_dir), read_paths=[], network_policy="allow"
    )
    assert "(allow network*)" in allow_profile
    with pytest.raises(ValueError, match="OS-user home"):
        build_seatbelt_profile(
            cwd=str(Path.home()), temp_dir=str(temp_dir), read_paths=[], network_policy="deny"
        )


def test_seatbelt_validation_fails_closed_for_platform_binary_and_modes(tmp_path):
    from tools.environments.local_seatbelt import validate_seatbelt_config

    executable = tmp_path / "sandbox-exec"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)

    with pytest.raises(RuntimeError, match="macOS"):
        validate_seatbelt_config("seatbelt", "deny", system_name="Linux", sandbox_exec=str(executable))
    with pytest.raises(RuntimeError, match="not available or executable"):
        validate_seatbelt_config("seatbelt", "deny", system_name="Darwin", sandbox_exec=str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="local_sandbox"):
        validate_seatbelt_config("bubblewrap", "deny", system_name="Darwin", sandbox_exec=str(executable))
    with pytest.raises(ValueError, match="network"):
        validate_seatbelt_config("seatbelt", "sometimes", system_name="Darwin", sandbox_exec=str(executable))


def test_seatbelt_profile_creation_failure_is_actionable(tmp_path, monkeypatch):
    from tools.environments.local_seatbelt import prepare_seatbelt_profile

    def _fail_write(*_args, **_kwargs):
        raise OSError("read-only temp root")

    monkeypatch.setattr(Path, "write_text", _fail_write)
    with pytest.raises(RuntimeError, match="check terminal.temp_dir permissions"):
        prepare_seatbelt_profile(
            cwd=str(tmp_path),
            temp_root=str(tmp_path),
            shell="/bin/bash",
            env={"PATH": "/usr/bin:/bin"},
            network_policy="deny",
        )


def test_seatbelt_config_projects_and_parses_at_runtime(tmp_path):
    from tools.terminal_scope import build_profile_terminal_scope, reset_terminal_scope, set_terminal_scope
    import tools.terminal_tool as terminal_tool

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "terminal:\n"
        "  backend: local\n"
        "  local_sandbox: seatbelt\n"
        "  local_sandbox_network: deny\n",
        encoding="utf-8",
    )
    scope = build_profile_terminal_scope(home)
    assert scope["TERMINAL_LOCAL_SANDBOX"] == "seatbelt"
    assert scope["TERMINAL_LOCAL_SANDBOX_NETWORK"] == "deny"

    token = set_terminal_scope(scope)
    try:
        config = terminal_tool._get_env_config()
    finally:
        reset_terminal_scope(token)
    assert config["local_sandbox"] == "seatbelt"
    assert config["local_sandbox_network"] == "deny"


def test_seatbelt_local_background_uses_environment_transport():
    from tools.terminal_tool_background import _spawn

    class _Registry:
        def __init__(self):
            self.local_calls = []
            self.env_calls = []

        def spawn_local(self, **kwargs):
            self.local_calls.append(kwargs)
            return "local"

        def spawn_via_env(self, **kwargs):
            self.env_calls.append(kwargs)
            return "env"

    class _Env:
        _seatbelt_profile_path = "/private/tmp/profile.sb"

    registry = _Registry()
    result = _spawn(
        registry, env=_Env(), env_type="local", command="printf escaped",
        cwd="/tmp/work", effective_task_id="task", task_id=None,
        session_key="session", effective_pty=False,
    )

    assert result == "env"
    assert not registry.local_calls
    assert registry.env_calls[0]["env"]._seatbelt_profile_path


def test_seatbelt_execute_code_uses_environment_transport(monkeypatch):
    import tools.code_execution_tool as code_execution

    monkeypatch.setattr(
        "tools.terminal_tool._get_env_config",
        lambda: {"env_type": "local", "local_sandbox": "seatbelt"},
    )
    monkeypatch.setattr(
        "tools.terminal_tool._docker_has_host_access", lambda _config: False,
    )
    monkeypatch.setattr(
        "tools.approval.check_execute_code_guard",
        lambda *_args, **_kwargs: {"approved": True},
    )
    calls = []
    monkeypatch.setattr(
        code_execution,
        "_execute_remote",
        lambda code, task_id, enabled_tools, reset=False: calls.append(
            (code, task_id, enabled_tools, reset)
        ) or "sandboxed",
    )

    result = code_execution.execute_code("open('/tmp/escape', 'w').close()", task_id="task")

    assert result == "sandboxed"
    assert calls == [("open('/tmp/escape', 'w').close()", "task", None, False)]


@pytest.mark.macos_only
def test_real_seatbelt_denies_out_of_scope_write_and_allows_cwd(tmp_path):
    if not os.access("/usr/bin/sandbox-exec", os.X_OK):
        pytest.skip("macOS sandbox-exec is unavailable")
    host_probe = subprocess.run(
        [
            "/usr/bin/sandbox-exec",
            "-p",
            "(version 1)(allow default)",
            "/usr/bin/true",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if host_probe.returncode != 0 and "Operation not permitted" in host_probe.stderr:
        pytest.skip("the test runner is already inside a sandbox that forbids nested Seatbelt")

    from tools.environments.local import LocalEnvironment

    cwd = tmp_path / "workspace"
    cwd.mkdir()
    outside = tmp_path / "outside.txt"
    env = LocalEnvironment(
        cwd=str(cwd),
        local_config={"sandbox": "seatbelt", "network": "deny"},
    )
    try:
        legal = env.execute("printf allowed > legal.txt")
        tamper = env.execute(
            f"printf '(version 1)(allow default)' > {shlex.quote(env._seatbelt_profile_path)}"
        )
        denied = env.execute(f"printf denied > {outside}")
    finally:
        env.cleanup()

    assert legal["returncode"] == 0
    assert (cwd / "legal.txt").read_text() == "allowed"
    assert tamper["returncode"] != 0
    assert denied["returncode"] != 0
    assert not outside.exists()
