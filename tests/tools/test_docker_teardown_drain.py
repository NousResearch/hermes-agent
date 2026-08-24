"""#86317: a container-teardown worker must be joined at exit even after the idle reaper popped its
env out of ``_active_environments`` (the atexit drain only knew the registry, so ``docker rm`` for a
detached env died with the interpreter and left a stopped, labeled container behind)."""

import subprocess
import threading

import tools.environments.docker as docker_env
import tools.terminal_tool as terminal_tool


def _env_with_slow_teardown(monkeypatch, release: threading.Event, seen: list):
    docker_env._cgroup_limits_ok = True
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")

    def _run(cmd, **kwargs):
        argv = list(cmd)
        if argv[1] == "rm":
            release.wait(5)  # the slow part: interpreter used to exit before it finished
        if argv[1] in ("stop", "rm"):
            seen.append(argv[1])
        out = "fake-cid\n" if argv[1] == "run" else ""
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    monkeypatch.setattr(docker_env, "run_capture", lambda cmd, **kw: _run(cmd))
    monkeypatch.setattr(docker_env.DockerEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(docker_env.DockerEnvironment, "_remove_bind_dirs", lambda self: None)
    return docker_env.DockerEnvironment(
        image="python:3.11", cwd="/root", timeout=5, task_id="t-detach", persistent_filesystem=False,
        persist_across_processes=False)


def test_atexit_drain_joins_worker_of_env_already_popped_from_registry(monkeypatch):
    release, seen = threading.Event(), []
    env = _env_with_slow_teardown(monkeypatch, release, seen)
    # The reaper's shape: pop first, then cleanup() — the registry no longer references env.
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    env.cleanup()
    assert env._cleanup_thread.is_alive()

    joined = {}

    def _drain():
        joined["result"] = None
        terminal_tool._atexit_cleanup()
        joined["result"] = True

    drainer = threading.Thread(target=_drain)
    drainer.start()
    drainer.join(0.5)
    assert drainer.is_alive(), "drain returned while a detached teardown worker was still running"
    release.set()
    drainer.join(5)
    assert joined["result"] is True and seen == ["stop", "rm"]
    assert not env._cleanup_thread.is_alive()


def test_finished_workers_do_not_accumulate():
    t = threading.Thread(target=lambda: None)
    with docker_env._TEARDOWN_LOCK:
        docker_env._TEARDOWN_THREADS.add(t)
    t.start(); t.join()
    assert docker_env.DockerEnvironment.wait_for_all_teardowns(timeout=1) is True
    with docker_env._TEARDOWN_LOCK:
        assert t not in docker_env._TEARDOWN_THREADS
