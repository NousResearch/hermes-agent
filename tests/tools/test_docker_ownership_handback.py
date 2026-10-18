"""Root-run Docker sandboxes hand their bind-mounted files back to the host user after the container
stops and before it is removed; the next root container takes back exactly those files."""

import importlib
import json
import os
import subprocess

import pytest

from tools.environments import docker as docker_env


_ROOT_OWNED = b"/root/.cache/x\0/workspace/out.txt\0"  # what a root walk of the sandbox finds
_HOST_OWNED = b"/workspace/repo/HEAD\0"  # a no-symlink walk for host-owned entries finds only this


def _fake_docker(monkeypatch, security_options: str, fail: str = ""):
    docker_env._cgroup_limits_ok = True
    try:  # the daemon-mode probe is cached per process
        importlib.import_module("tools.environments.docker_ownership")._remapping_cache.clear()
    except ImportError:
        pass
    calls: list[tuple[list[str], bytes]] = []

    def _inspect() -> str:  # like `docker inspect`: the real user and every -v, extra args included
        run = next(argv for argv, _ in calls if argv[1] == "run")
        mounts = [{"Destination": run[i + 1].split(":")[1]} for i, a in enumerate(run[:-1]) if a == "-v"]
        user = next((run[i + 1] for i, a in enumerate(run[:-1]) if a == "--user"), "")
        return json.dumps([{"Config": {"User": user}, "Image": "sha256:img", "Mounts": mounts}])

    def _run(cmd, **kwargs):
        argv = [str(a) for a in cmd]
        calls.append((argv, kwargs.get("input") or b""))
        sub = argv[1] if len(argv) > 1 else ""
        rc = 0
        if "/bin/chown" in argv and fail == "chown-timeout":
            raise subprocess.TimeoutExpired(argv, 10)
        if "/bin/chown" in argv and fail == "chown":
            rc = 1  # e.g. a container without CAP_CHOWN
        out = {"run": "fake-cid\n", "info": security_options}.get(sub, "")
        if sub == "inspect" and argv[-1] == "fake-cid":
            out = _inspect() if len(argv) == 3 else json.dumps(json.loads(_inspect())[0]["Mounts"])
        if "/usr/bin/find" in argv and "-user" in argv:
            out = (_ROOT_OWNED if argv[argv.index("-user") + 1] == "0" else _HOST_OWNED).decode()
            rc = 1 if fail == "find" else 0  # e.g. find missing from the image
        if kwargs.get("text") or kwargs.get("encoding"):
            return subprocess.CompletedProcess(argv, rc, stdout=out, stderr="")
        return subprocess.CompletedProcess(argv, rc, stdout=out.encode(), stderr=b"")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env.DockerEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(os, "geteuid", lambda: 1000)  # Hermes itself is not root
    monkeypatch.setattr(os, "getegid", lambda: 1000)
    return calls


_PLAIN = "[name=apparmor name=seccomp,profile=builtin name=cgroupns]"
_EXTRA_USER = ["--user", "1000:1000"]  # docker_extra_args, not docker_run_as_host_user


def _to_root(calls):
    return [stdin for argv, stdin in calls if "/bin/chown" in argv and "0:0" in argv[argv.index("/bin/chown"):]]


@pytest.mark.platforms("linux")
@pytest.mark.parametrize(("run_as_host_user", "user_args", "security_options", "scenario"), [
    (False, [], _PLAIN, "handback"),
    (True, [], _PLAIN, "none"),
    (False, _EXTRA_USER, _PLAIN, "none"),  # the container's real user, however it was selected
    (False, [], "[name=seccomp,profile=builtin name=rootless name=cgroupns]", "none"),
    (False, [], "[name=apparmor name=userns]", "none"),
    (False, [], _PLAIN, "chown-timeout"),  # paths are recorded before they change owner
    (False, [], _PLAIN, "reclaim"),  # only recorded paths a no-symlink walk still finds go back to root
    (True, [], _PLAIN, "reclaim"),  # a non-root container keeps the handed-back owner
    (False, _EXTRA_USER, _PLAIN, "reclaim"),
    (False, [], _PLAIN, "reclaim-fails-find"),  # a failed reclaim keeps the record; a retry succeeds
    (False, [], _PLAIN, "reclaim-fails-chown"),
])
def test_root_container_teardown_hands_every_sandbox_mount_back_before_removal(
        monkeypatch, tmp_path, run_as_host_user, user_args, security_options, scenario):
    monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(tmp_path / "sandboxes"))
    fail = scenario.rsplit("-", 1)[1] if "fails" in scenario else scenario if scenario == "chown-timeout" else ""
    calls = _fake_docker(monkeypatch, security_options, fail=fail)
    nested_ro, shared = tmp_path / "skills", tmp_path / "shared"
    nested_ro.mkdir()
    shared.mkdir()
    record = tmp_path / "sandboxes" / "docker" / "chown-back" / ".hermes-owner-handback"
    if scenario.startswith("reclaim"):  # an earlier root teardown handed these back; "link" is now a symlink
        record.parent.mkdir(parents=True)
        record.write_bytes(_HOST_OWNED + b"/workspace/link/escaped\0")

    def _env():
        return docker_env.DockerEnvironment(
            image="python:3.12-slim", cwd="/root", timeout=5, task_id="chown-back", persistent_filesystem=True,
            persist_across_processes=False, run_as_host_user=run_as_host_user,
            volumes=[f"{nested_ro}:/root/.hermes/skills:ro"],
            extra_args=["-v", f"{shared}:/workspace/shared", *user_args])
    env = _env()
    if scenario.startswith("reclaim-fails"):
        assert record.exists(), calls
        retry_calls = _fake_docker(monkeypatch, security_options)
        retry = _env()  # held: a collected env tears down and hands back again
        assert _to_root(retry_calls) == [_HOST_OWNED] and not record.exists(), retry_calls
        assert retry._container_id
        return
    if scenario == "reclaim":
        assert _to_root(calls) == ([] if run_as_host_user or user_args else [_HOST_OWNED]), calls
        assert all("PATH=/usr/sbin:/usr/bin:/sbin:/bin" in argv for argv, _ in calls if "/usr/bin/find" in argv), calls
        assert not record.exists()
        return
    env.cleanup()
    assert env.wait_for_cleanup(5)

    subcommands = [argv[1] for argv, _ in calls]
    chowns = [(argv, stdin) for argv, stdin in calls if "/bin/chown" in argv]
    if scenario == "none":
        assert chowns == [], chowns
        assert subcommands[-1] == "rm" and "stop" in subcommands
        return
    assert record.read_bytes() == _ROOT_OWNED
    if scenario == "chown-timeout":  # the CLI is killed; the named throwaway must not run on
        name = chowns[0][0][chowns[0][0].index("--name") + 1]
        assert name.startswith("hermes-handback-fake-cid-") and [docker_env.find_docker(), "rm", "-f", name] in [
            argv for argv, _ in calls], calls
        return
    (walk,) = [argv for argv, _ in calls if "/usr/bin/find" in argv]
    ((chown, stdin),) = chowns
    # After stop (no writer left, shutdown writes included), before rm, from the stopped container's volumes.
    stop, rm = subcommands.index("stop"), subcommands.index("rm")
    assert stop < calls.index((walk, b"")) < rm and stop < calls.index((chown, stdin)) < rm
    for argv in (walk, chown):
        assert argv[argv.index("--volumes-from") + 1] == "fake-cid" and "--user" in argv and "sha256:img" in argv
    for argv in (walk, chown):  # a binary planted on the image PATH under /root or /workspace never runs
        joined = " ".join(argv[:argv.index("sha256:img")])
        assert "--cap-drop ALL" in joined and "no-new-privileges" in joined and "--pids-limit" in argv, argv
        assert "-e PATH=/usr/sbin:/usr/bin:/sbin:/bin -e LD_PRELOAD=" in joined and "--cap-add MKNOD" not in joined
    assert chown[chown.index("sha256:img") + 1:][:1] == ["/usr/bin/xargs"]
    assert walk[walk.index("/usr/bin/find") + 1:walk.index("-xdev")] == ["-P", "/root", "/workspace"]
    assert " ".join(walk).endswith('( -type f -o -type d -o -type l ) -user 0 -print0'), walk  # no device nodes
    pruned = [walk[i + 1] for i, a in enumerate(walk) if a == "-path"]
    assert {"/root/.hermes/skills", "/workspace/shared"} <= set(pruned)  # docker_volumes and extra-args binds
    assert "1000:1000" in chown and stdin == _ROOT_OWNED
