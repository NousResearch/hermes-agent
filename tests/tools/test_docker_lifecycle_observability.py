"""Docker observations describe checked operations without changing their behavior."""

import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools.environments import docker
from tools.environment_observability import (
    capture_observer, environment_acquired, environment_provisioning, observe_environment,
)


CONTAINER = "a" * 64
REPLACEMENT = "b" * 64


@pytest.fixture
def environment():
    observer = Mock()
    observer.provision.return_value = observer
    observer.prepare_cleanup.return_value = observer
    env = object.__new__(docker.DockerEnvironment)
    env._relay_lifecycle_observer = observer
    env._container_id = CONTAINER
    env._docker_exe = "docker"
    env._persist_across_processes = False
    env._persistent = True
    env._image = "test-image"
    env._image_pinned = False
    env._labels = {}
    env._run_env_values = {}
    env.cwd = "/workspace"
    env._run_command = Mock(return_value=["docker", "run", "test-image"])
    env._docker_client_env = Mock(return_value=None)
    env.init_session = Mock()
    env._mark_recreated = Mock()
    try:
        with observe_environment(observer):
            yield env, observer
    finally:
        env.wait_for_cleanup()
        env._container_id = None


def observations(observer):
    return [call.kwargs for call in observer.transition.call_args_list]


def test_constructor_preflight_failure_uses_observer_before_handle_exists(environment, monkeypatch):
    _, observer = environment
    error = docker.EnvironmentConnectionError("Docker unavailable", retry_hint="retry")
    monkeypatch.setattr(docker, "_ensure_docker_available", Mock(side_effect=error))

    with pytest.raises(docker.EnvironmentConnectionError) as caught:
        docker.DockerEnvironment(image="test-image")

    assert caught.value is error
    observer.provision.assert_called_once()
    observer.finish.assert_called_once_with(reason="provision_failed")
    failed = observations(observer)[-1]
    assert (failed["action"], failed["outcome"], failed["evidence"]) == ("provision", "failed", "preflight")
    assert failed["container_id"] is None and failed.get("state_after") is None


def test_constructor_failure_after_preflight_finishes_without_masking_error(environment, monkeypatch):
    _, observer = environment
    error = ValueError("invalid configuration")
    monkeypatch.setattr(docker, "_ensure_docker_available", Mock())
    monkeypatch.setattr(docker.DockerEnvironment, "_resource_args", Mock(side_effect=error))
    observer.finish.side_effect = RuntimeError("observer unavailable")

    with pytest.raises(ValueError) as caught:
        docker.DockerEnvironment(image="test-image")

    assert caught.value is error
    observer.finish.assert_called_once_with(reason="provision_failed")
    observer.transition.assert_not_called()


def test_physical_owner_is_not_replaced_by_another_session():
    owner, first, second = Mock(), Mock(), Mock()
    first.provision.return_value = owner
    env = SimpleNamespace(env_type="docker")
    with observe_environment(first):
        assert environment_provisioning(env) is owner
        assert environment_acquired(env) is env
    with observe_environment(second):
        assert environment_provisioning(env) is owner
        assert environment_acquired(env) is env
        assert capture_observer(env) is owner

    first.provision.assert_called_once_with(env)
    second.provision.assert_not_called()
    second.acquire.assert_called_once_with(env)
    assert env._relay_lifecycle_observer is owner


def test_provisioning_observer_failure_does_not_bind_an_owner():
    observer = Mock()
    observer.provision.side_effect = RuntimeError("observer unavailable")
    env = SimpleNamespace(env_type="docker")
    with observe_environment(observer):
        assert environment_provisioning(env) is None
    assert getattr(env, "_relay_lifecycle_observer", None) is None


def test_provision_records_combined_run_result(environment, monkeypatch):
    env, observer = environment
    run = Mock(return_value=subprocess.CompletedProcess([], 0, REPLACEMENT + "\n", ""))
    monkeypatch.setattr(docker, "run_capture", run)

    assert env._docker_run(env.cwd) == REPLACEMENT

    requested, completed = observations(observer)
    assert (requested["action"], requested["outcome"]) == ("create_and_start", "requested")
    assert completed["container_id"] == REPLACEMENT
    assert (completed["outcome"], completed["state_after"]) == ("succeeded", "running")
    run.assert_called_once_with(env._run_command.return_value, timeout=120, check=True, env=None)


def test_failed_provision_preserves_exception_and_orphan_cleanup(environment, monkeypatch):
    env, observer = environment
    error = subprocess.CalledProcessError(125, "docker run", stderr="private diagnostics")
    monkeypatch.setattr(docker, "run_capture", Mock(side_effect=error))
    cleanup = Mock(return_value=subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(docker.subprocess, "run", cleanup)

    with pytest.raises(subprocess.CalledProcessError) as caught:
        env._docker_run(env.cwd)

    assert caught.value is error
    assert cleanup.call_args.args[0][1:3] == ["rm", "-f"]
    failed = observations(observer)[-1]
    assert (failed["outcome"], failed["error_class"]) == ("failed", "CalledProcessError")
    assert failed.get("state_after") is None
    assert "private diagnostics" not in repr(observations(observer))


def test_constructor_finishes_failed_provision_after_original_orphan_cleanup(environment, monkeypatch):
    _, observer = environment
    error = subprocess.CalledProcessError(125, "docker run")
    monkeypatch.setattr(docker, "_ensure_docker_available", Mock())
    monkeypatch.setattr(docker, "_readonly_skill_mount_args", Mock(return_value=[]))
    monkeypatch.setattr(docker, "_host_user_args", Mock(return_value=[]))
    monkeypatch.setattr(docker, "_image_uses_init_entrypoint", Mock(return_value=False))
    monkeypatch.setattr(docker, "find_docker", Mock(return_value="docker"))
    monkeypatch.setattr(docker.DockerEnvironment, "_resource_args", Mock(return_value=[]))
    monkeypatch.setattr(docker.DockerEnvironment, "_mount_args", Mock(return_value=([], [])))
    def egress(env, _args):
        env._run_env_values = {}
        return "off", [], [], [], []

    monkeypatch.setattr(docker.DockerEnvironment, "_egress_and_env_args", egress)
    monkeypatch.setattr(docker, "run_capture", Mock(side_effect=error))

    def cleanup(argv, **_kwargs):
        observer.finish.assert_not_called()
        assert argv[1:3] == ["rm", "-f"]
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(docker.subprocess, "run", cleanup)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        docker.DockerEnvironment(image="test-image", persist_across_processes=False)

    assert caught.value is error
    observer.finish.assert_called_once_with(reason="provision_failed")
    assert not any(record.get("state_after") == "deleted" for record in observations(observer))


@pytest.mark.parametrize("prior_state", ["running", "exited"])
def test_reuse_records_existing_evidence(environment, monkeypatch, prior_state):
    env, observer = environment
    env._find_reusable_container = Mock(return_value=(CONTAINER, prior_state))
    env._container_image = Mock(return_value="test-image")
    run = Mock(return_value=subprocess.CompletedProcess([], 0, CONTAINER, ""))
    monkeypatch.setattr(docker, "run_capture", run)

    assert env._attach_existing_container("task", "profile", "off", True)

    reused = observations(observer)[-1]
    assert (reused["action"], reused["outcome"], reused["state_after"]) == ("reuse", "succeeded", "running")
    assert reused["evidence"] == ("docker_ps" if prior_state == "running" else "operation_result")
    assert run.call_count == (0 if prior_state == "running" else 1)


def test_failed_start_returns_original_error_without_running_state(environment, monkeypatch):
    env, observer = environment
    error = subprocess.CalledProcessError(1, "docker start")
    monkeypatch.setattr(docker, "run_capture", Mock(side_effect=error))

    assert env._start_container(CONTAINER) is error

    failed = observations(observer)[-1]
    assert (failed["action"], failed["outcome"]) == ("start", "failed")
    assert failed.get("state_after") is None


def test_recovery_records_previous_and_replacement_ids(environment, monkeypatch):
    env, observer = environment
    env._find_reusable_container = Mock(return_value=None)
    monkeypatch.setattr(
        docker, "run_capture",
        Mock(return_value=subprocess.CompletedProcess([], 0, REPLACEMENT + "\n", "")),
    )

    assert env._recreate_container()

    requested, _, _, completed = observations(observer)
    assert requested["container_id"] == CONTAINER
    assert requested.get("state_after") is None
    assert completed["action"] == "recover" and completed["outcome"] == "succeeded"
    assert completed["previous_container_id"] == CONTAINER
    assert completed["container_id"] == REPLACEMENT
    env.init_session.assert_called_once()
    env._mark_recreated.assert_called_once()


def test_failed_recovery_does_not_claim_replacement(environment, monkeypatch):
    env, observer = environment
    env._find_reusable_container = Mock(return_value=None)
    monkeypatch.setattr(docker, "run_capture", Mock(side_effect=OSError("private diagnostics")))

    assert not env._recreate_container()

    failed = observations(observer)[-1]
    assert (failed["action"], failed["outcome"], failed["error_class"]) == ("recover", "failed", "OSError")
    assert failed["previous_container_id"] == CONTAINER and failed["container_id"] is None
    assert failed.get("state_after") is None
    assert "private diagnostics" not in repr(observations(observer))


@pytest.mark.parametrize("returncodes", [(0, 0), (1, 0), (0, 1)])
def test_async_cleanup_reports_checked_results_after_handle_is_cleared(environment, monkeypatch, returncodes):
    env, observer = environment
    run = Mock(side_effect=[subprocess.CompletedProcess([], code, "", "") for code in returncodes])
    monkeypatch.setattr(docker.subprocess, "run", run)

    env.cleanup()
    assert env.wait_for_cleanup()

    assert env._container_id is None
    assert [call.args[0][1] for call in run.call_args_list] == ["stop", "rm"]
    records = observations(observer)
    assert [record["action"] for record in records] == ["stop", "stop", "remove", "remove"]
    assert all(record["container_id"] == CONTAINER for record in records)
    assert all(call.args == (None,) for call in observer.transition.call_args_list)
    for record, code, state in zip(records[1::2], returncodes, ["exited", "deleted"]):
        assert record["outcome"] == ("succeeded" if code == 0 else "failed")
        assert record.get("state_after") == (state if code == 0 else None)
    observer.prepare_cleanup.assert_called_once_with()
    observer.finish.assert_called_once_with(reason="removed" if returncodes[-1] == 0 else "cleanup_failed")


def test_cleanup_reserves_ticket_before_detaching_and_finishes_after_worker(environment, monkeypatch):
    env, owner = environment
    ticket = Mock()
    owner.prepare_cleanup.return_value = ticket
    started, release = threading.Event(), threading.Event()

    def run(argv, **_kwargs):
        owner.prepare_cleanup.assert_called_once_with()
        if argv[1] == "stop":
            started.set()
            assert release.wait(timeout=5)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(docker.subprocess, "run", run)
    try:
        env.cleanup()
        assert started.wait(timeout=5)
        assert env._container_id is None
        ticket.finish.assert_not_called()
    finally:
        release.set()
        assert env.wait_for_cleanup()

    assert [record["action"] for record in observations(ticket)] == ["stop", "stop", "remove", "remove"]
    owner.transition.assert_not_called()
    ticket.finish.assert_called_once_with(reason="removed")


def test_cleanup_observation_failure_does_not_prevent_backend_removal(environment, monkeypatch):
    env, owner = environment
    owner.prepare_cleanup.side_effect = RuntimeError("observer unavailable")
    owner.transition.side_effect = RuntimeError("observer unavailable")
    owner.finish.side_effect = RuntimeError("observer unavailable")
    run = Mock(return_value=subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(docker.subprocess, "run", run)

    env.cleanup()
    assert env.wait_for_cleanup()

    assert [call.args[0][1] for call in run.call_args_list] == ["stop", "rm"]
    owner.finish.assert_called_once_with(reason="removed")


def test_cleanup_thread_start_failure_finishes_and_untracks_ticket(environment, monkeypatch):
    env, owner = environment
    error = RuntimeError("thread unavailable")
    monkeypatch.setattr(threading.Thread, "start", Mock(side_effect=error))

    with pytest.raises(RuntimeError) as caught:
        env.cleanup()

    assert caught.value is error
    owner.prepare_cleanup.assert_called_once_with()
    owner.finish.assert_called_once_with(reason="cleanup_failed")
    owner.transition.assert_not_called()
    assert docker.DockerEnvironment.wait_for_all_teardowns(timeout=0)


def test_cleanup_without_observer_keeps_backend_behavior(environment, monkeypatch):
    env, _ = environment
    env._relay_lifecycle_observer = None
    run = Mock(return_value=subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(docker.subprocess, "run", run)

    with observe_environment(None):
        env.cleanup()
        assert env.wait_for_cleanup()

    assert env._container_id is None
    assert [call.args[0][1] for call in run.call_args_list] == ["stop", "rm"]


def test_cleanup_timeout_does_not_prevent_removal_attempt(environment, monkeypatch):
    env, observer = environment
    run = Mock(side_effect=[
        subprocess.TimeoutExpired("docker stop", 30),
        subprocess.CompletedProcess([], 0, "", ""),
    ])
    monkeypatch.setattr(docker.subprocess, "run", run)

    env.cleanup()
    assert env.wait_for_cleanup()

    failed = observations(observer)[1]
    assert failed["error_class"] == "TimeoutExpired" and failed.get("state_after") is None
    assert observations(observer)[-1]["state_after"] == "deleted"
    assert run.call_count == 2
    observer.finish.assert_called_once_with(reason="removed")


def test_persistent_release_does_not_claim_physical_cleanup(environment, monkeypatch):
    env, observer = environment
    env._persist_across_processes = True
    run = Mock()
    monkeypatch.setattr(docker.subprocess, "run", run)

    env.cleanup()

    assert env._container_id is None
    run.assert_not_called()
    observer.transition.assert_not_called()
    observer.prepare_cleanup.assert_not_called()
    observer.finish.assert_called_once_with(reason="retained")
