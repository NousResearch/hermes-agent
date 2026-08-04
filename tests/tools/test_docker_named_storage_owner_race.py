import subprocess
import threading

from tools.environments import docker as docker_env


def test_storage_owner_barrier_is_held_through_init_session(monkeypatch, tmp_path):
    from tools.environments import base as environment_base

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    monkeypatch.setattr(environment_base, "get_sandbox_dir", lambda: tmp_path)
    docker_env._cgroup_limits_ok = True
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[1] == "version":
            return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
        if cmd[1] == "run":
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f"container-{sum(c[1] == 'run' for c in calls)}\n", stderr="",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", fake_run)
    first_init_started = threading.Event()
    release_first_init = threading.Event()
    second_init_started = threading.Event()
    init_lock = threading.Lock()
    init_count = 0

    def blocking_init(self):
        nonlocal init_count
        with init_lock:
            init_count += 1
            ordinal = init_count
        if ordinal == 1:
            first_init_started.set()
            assert release_first_init.wait(timeout=5)
        else:
            second_init_started.set()

    monkeypatch.setattr(docker_env.DockerEnvironment, "init_session", blocking_init)
    created = []
    failures = []

    def construct():
        try:
            created.append(docker_env.DockerEnvironment(
                image="python:3.12-slim",
                cwd="/workspace",
                persistent_filesystem=True,
                persist_across_processes=True,
                storage_task_id="profile-target-storage",
                task_id="runtime",
            ))
        except Exception as exc:
            failures.append(exc)

    first = threading.Thread(target=construct)
    second = threading.Thread(target=construct)
    first.start()
    assert first_init_started.wait(timeout=5)
    second.start()

    assert not second_init_started.wait(timeout=0.2)
    assert second.is_alive()

    release_first_init.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert failures == []
    assert second_init_started.is_set()
    assert len(created) == 2
    for env in created:
        env.cleanup()
