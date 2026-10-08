"""The tool/container join must never export a native Docker identity."""

from __future__ import annotations

import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from tools.execution_observability import (
    capture_docker_attempt,
    collect_tool_resources,
    docker_exec_started,
    opaque_docker_resource_id,
    record_docker_result,
)
from tools.environments.base import BaseEnvironment
from tools.environments.docker import DockerEnvironment

A = "a" * 64
B = "b" * 64
C = "c" * 64


def test_opaque_docker_id_has_stable_domain_separated_vector():
    assert opaque_docker_resource_id(A) == "bdc14832f12cb1c009fcbfd262eb60e7"
    for invalid in (A[:12], A.upper(), f" {A}", f"{A}\n", None, ""):
        assert opaque_docker_resource_id(invalid) is None


def test_collector_is_specific_only_for_one_known_resource():
    with collect_tool_resources() as collector:
        record_docker_result(A)
        record_docker_result(A)
    payload = collector.annotation()["hermes.execution_environment"]
    assert payload["resource_id"] == opaque_docker_resource_id(A)
    assert payload["resource_id_status"] == "available"
    assert payload["association"] == "docker_exec_target"
    assert A not in str(payload)

    with collect_tool_resources() as collector:
        record_docker_result(A)
        record_docker_result(B)
    payload = collector.annotation()["hermes.execution_environment"]
    assert payload["resource_id_status"] == "ambiguous_multiple"
    assert "resource_id" not in payload

    with collect_tool_resources() as collector:
        record_docker_result("short-id")
    payload = collector.annotation()["hermes.execution_environment"]
    assert payload["resource_id_status"] == "unavailable"
    assert "resource_id" not in payload

    with collect_tool_resources() as collector:
        record_docker_result(A)
        record_docker_result("short-id")
    payload = collector.annotation()["hermes.execution_environment"]
    assert payload["resource_id_status"] == "ambiguous_unknown"
    assert "resource_id" not in payload


def test_collector_is_bounded_and_sealed_after_callback():
    with collect_tool_resources() as collector:
        copied_context = contextvars.copy_context()
        record_docker_result(A)
        record_docker_result(B)
        record_docker_result(C)
        assert len(collector.docker_ids) == 2
    copied_context.run(record_docker_result, "d" * 64)
    assert (
        collector.annotation()["hermes.execution_environment"]["resource_id_status"]
        == "ambiguous_multiple"
    )
    assert len(collector.docker_ids) == 2

    with collect_tool_resources() as collector:
        copied_context = contextvars.copy_context()
        record_docker_result(A)
    copied_context.run(record_docker_result, B)
    assert collector.annotation()["hermes.execution_environment"][
        "resource_id"
    ] == opaque_docker_resource_id(A)


def _bare_docker_environment() -> DockerEnvironment:
    env = object.__new__(DockerEnvironment)
    env._container_id = A
    env._docker_exe = "docker"
    env._profile_scoped_passthrough = False
    env._init_env_args = []
    env._persist_across_processes = True
    env._docker_client_env = lambda _values: None
    return env


def _ready_to_execute_docker_environment(monkeypatch) -> DockerEnvironment:
    env = _bare_docker_environment()
    env.timeout = 1
    env.cwd = "/workspace"
    env._snapshot_ready = True
    env._prefer_nonlogin = False
    env._stdin_mode = "pipe"
    env._wrap_command = lambda command, _cwd: command
    env._update_cwd = lambda _result: None
    env._wait_for_process = lambda *_args, **_kwargs: {
        "returncode": 0,
        "output": "ready",
    }
    monkeypatch.setattr(
        BaseEnvironment,
        "_prepare_command",
        lambda _self, command: (command, None),
    )
    monkeypatch.setattr(
        "tools.environments.docker._popen_bash",
        lambda *_args, **_kwargs: object(),
    )
    return env


def test_real_deadline_context_records_the_main_docker_spawn(monkeypatch):
    env = _ready_to_execute_docker_environment(monkeypatch)
    with collect_tool_resources() as collector:
        assert env.execute("echo ready", timeout=1)["output"] == "ready"
    payload = collector.annotation()["hermes.execution_environment"]
    assert payload["resource_id"] == opaque_docker_resource_id(A)


def test_abandoned_deadline_worker_cannot_claim_a_late_spawn(monkeypatch):
    env = _ready_to_execute_docker_environment(monkeypatch)
    monkeypatch.setattr("tools.environments.base._EXECUTE_WAIT_BOUND_GRACE_S", 0.0)

    def preflight(_self, command):
        docker_exec_started(A)  # a sudo check must not identify the main call
        return command, None

    monkeypatch.setattr(BaseEnvironment, "_prepare_command", preflight)
    entered_spawn = threading.Event()
    release_spawn = threading.Event()
    finished_wait = threading.Event()

    def delayed_spawn(*_args, **_kwargs):
        entered_spawn.set()
        assert release_spawn.wait(5)
        return object()

    def finish_wait(*_args, **_kwargs):
        finished_wait.set()
        return {"returncode": 0, "output": "late"}

    monkeypatch.setattr("tools.environments.docker._popen_bash", delayed_spawn)
    env._wait_for_process = finish_wait
    try:
        with collect_tool_resources() as collector:
            result = env.execute("echo ready", timeout=2)
        assert entered_spawn.wait(5)
        assert result["hermes_timed_out"] is True
        payload = collector.annotation()["hermes.execution_environment"]
        assert payload["resource_id_status"] == "unavailable"
        assert "resource_id" not in payload
    finally:
        release_spawn.set()
        assert finished_wait.wait(5)
    assert collector.annotation()["hermes.execution_environment"] == payload


def test_docker_argv_and_join_use_same_snapshot_when_handle_changes(monkeypatch):
    env = _bare_docker_environment()
    argv = []

    def spawn(command, _stdin, **_kwargs):
        argv.extend(command)
        env._container_id = B
        return object()

    monkeypatch.setattr("tools.environments.docker._popen_bash", spawn)
    with capture_docker_attempt() as attempts:
        env._run_bash("echo ready")
    assert argv[argv.index("exec") + 1] == A
    assert attempts == [A]


def test_recovery_keeps_both_targets_when_first_command_may_have_run(monkeypatch):
    env = _bare_docker_environment()

    def run(_self, *_args, **_kwargs):
        docker_exec_started(env._container_id)
        if env._container_id == A:
            return {"returncode": 1, "output": "No such container"}
        return {"returncode": 0, "output": "ready"}

    def recover():
        env._container_id = B
        return True

    monkeypatch.setattr(BaseEnvironment, "execute", run)
    monkeypatch.setattr(env, "_recreate_container", recover)
    with collect_tool_resources() as collector:
        assert env.execute("echo ready")["output"] == "ready"
    payload = collector.annotation()["hermes.execution_environment"]
    assert payload["resource_id_status"] == "ambiguous_multiple"
    assert "resource_id" not in payload
    assert opaque_docker_resource_id(A) not in str(payload)
    assert opaque_docker_resource_id(B) not in str(payload)


def test_escaping_docker_attempt_remains_correlated(monkeypatch):
    env = _bare_docker_environment()

    def fail(_self, *_args, **_kwargs):
        docker_exec_started(A)
        raise RuntimeError("docker exec failed after spawn")

    monkeypatch.setattr(BaseEnvironment, "execute", fail)
    with collect_tool_resources() as collector:
        with pytest.raises(RuntimeError, match="after spawn"):
            env.execute("echo ready")
    assert collector.annotation()["hermes.execution_environment"][
        "resource_id"
    ] == opaque_docker_resource_id(A)


def test_no_exec_attempt_does_not_invent_an_identity(monkeypatch):
    env = _bare_docker_environment()
    monkeypatch.setattr(
        BaseEnvironment,
        "execute",
        lambda *_args, **_kwargs: {"returncode": 0, "output": ""},
    )
    with collect_tool_resources() as collector:
        env.execute("echo ready")
    assert collector.annotation() is None


def test_timeout_before_spawn_marks_the_resource_unknown(monkeypatch):
    env = _bare_docker_environment()
    monkeypatch.setattr(
        BaseEnvironment,
        "execute",
        lambda *_args, **_kwargs: {
            "returncode": 124,
            "output": "Command timed out",
            "hermes_timed_out": True,
        },
    )
    with collect_tool_resources() as collector:
        env.execute("echo ready")
    payload = collector.annotation()["hermes.execution_environment"]
    assert payload["resource_id_status"] == "unavailable"
    assert "resource_id" not in payload


def test_sudo_preflight_cannot_be_mistaken_for_main_command(monkeypatch):
    env = _bare_docker_environment()

    def prepare(_self, command):
        docker_exec_started(A)  # sudo probe, before the main command
        return command, None

    def timeout_after_probe(_self, command, *_args, **_kwargs):
        _self._prepare_command(command)
        return {
            "returncode": 124,
            "output": "Command timed out",
            "hermes_timed_out": True,
        }

    monkeypatch.setattr(BaseEnvironment, "_prepare_command", prepare)
    monkeypatch.setattr(BaseEnvironment, "execute", timeout_after_probe)
    with collect_tool_resources() as collector:
        env.execute("sudo echo ready")
    payload = collector.annotation()["hermes.execution_environment"]
    assert payload["resource_id_status"] == "unavailable"
    assert "resource_id" not in payload

    def main_after_probe(_self, command, *_args, **_kwargs):
        _self._prepare_command(command)
        docker_exec_started(B)
        return {"returncode": 0, "output": "ready"}

    monkeypatch.setattr(BaseEnvironment, "execute", main_after_probe)
    with collect_tool_resources() as collector:
        env.execute("sudo echo ready")
    payload = collector.annotation()["hermes.execution_environment"]
    assert payload["resource_id"] == opaque_docker_resource_id(B)


@pytest.mark.parametrize("late_spawn", [False, True])
def test_retry_timeout_cannot_claim_only_the_first_container(monkeypatch, late_spawn):
    env = _bare_docker_environment()

    if late_spawn:
        class LateSpawn(list):
            def __getitem__(self, key):
                snapshot = super().__getitem__(key)
                self.append(B)  # deadline worker spawns after this snapshot
                return snapshot

        @contextmanager
        def capture_attempt():
            with capture_docker_attempt() as targets:
                yield LateSpawn() if env._container_id == B else targets

        monkeypatch.setattr(
            "tools.environments.docker.capture_docker_attempt", capture_attempt
        )

    def run(_self, *_args, **_kwargs):
        if env._container_id == A:
            docker_exec_started(A)
            return {"returncode": 1, "output": "No such container"}
        return {
            "returncode": 124,
            "output": "Command timed out",
            "hermes_timed_out": True,
        }

    def recover():
        env._container_id = B
        return True

    monkeypatch.setattr(BaseEnvironment, "execute", run)
    monkeypatch.setattr(env, "_recreate_container", recover)
    with collect_tool_resources() as collector:
        env.execute("echo ready")
    payload = collector.annotation()["hermes.execution_environment"]
    assert payload["resource_id_status"] == "ambiguous_unknown"
    assert "resource_id" not in payload


def test_reused_container_probe_requests_full_id(monkeypatch):
    env = _bare_docker_environment()
    env._labels = {
        "hermes-task-id": "task",
        "hermes-profile": "profile",
        "hermes-environment": "environment",
    }
    observed = []

    def docker_query(argv, **_kwargs):
        observed.extend(argv)
        return SimpleNamespace(stdout=f"{A}\trunning\n")

    monkeypatch.setattr("tools.environments.docker._docker_query", docker_query)
    assert env._find_reusable_container("task", "profile", "off") == (A, "running")
    assert "--no-trunc" in observed


def test_concurrent_calls_on_one_environment_keep_distinct_ids(monkeypatch):
    env = _bare_docker_environment()
    first_started = threading.Event()
    release_first = threading.Event()

    def run(_self, command, *_args, **_kwargs):
        docker_exec_started(env._container_id)
        if command == "first":
            first_started.set()
            assert release_first.wait(5)
        return {"returncode": 0, "output": command}

    def tool(command):
        with collect_tool_resources() as collector:
            env.execute(command)
        return collector.annotation()["hermes.execution_environment"]["resource_id"]

    monkeypatch.setattr(BaseEnvironment, "execute", run)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(tool, "first")
        try:
            assert first_started.wait(5)
            env._container_id = B
            second = pool.submit(tool, "second")
            assert second.result(timeout=5) == opaque_docker_resource_id(B)
        finally:
            release_first.set()
        assert first.result(timeout=5) == opaque_docker_resource_id(A)
