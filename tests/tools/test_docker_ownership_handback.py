"""Root-run Docker sandboxes hand their bind-mounted files back to the host user after the container
stops and before it is removed; the next root container takes back exactly those files."""

import importlib
import json
import logging
import os
import shutil
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

    def _inspect() -> str:  # like `docker inspect` of the creating run: user, labels, caps, every -v
        run = next(argv for argv, _ in calls if argv[1:3] == ["run", "-d"])
        mounts = [{"Type": "bind", "Source": run[i + 1].split(":")[0], "Destination": run[i + 1].split(":")[1]}
                  for i, a in enumerate(run[:-1]) if a == "-v"]
        flag = lambda name: [run[i + 1] for i, a in enumerate(run[:-1]) if a == name]
        labels = dict(v.split("=", 1) for v in flag("--label"))
        host = {"CapAdd": flag("--cap-add"), "Privileged": "--privileged" in run}
        return json.dumps([{"Config": {"User": (flag("--user") or [""])[0], "Labels": labels}, "HostConfig": host,
                            "Image": "sha256:img", "Mounts": mounts}])

    def _run(cmd, **kwargs):
        argv = [str(a) for a in cmd]
        calls.append((argv, kwargs.get("input") or b""))
        sub = argv[1] if len(argv) > 1 else ""
        rc = 0
        if "/usr/bin/xargs" in argv and fail == "chown-timeout":
            raise subprocess.TimeoutExpired(argv, 10)
        if "/usr/bin/xargs" in argv and fail == "chown":
            rc = 1  # e.g. a container without CAP_CHOWN
        out = {"run": "fake-cid\n", "info": security_options}.get(sub, "")
        if sub == "ps" and "status=exited" in argv:  # the orphan sweep finds the detached container
            out = "fake-cid\n"
        if sub == "inspect" and argv[-1] == "fake-cid":
            out = _inspect() if len(argv) == 3 else "2020-01-01T00:00:00Z" if "FinishedAt" in argv[3] else json.dumps(
                json.loads(_inspect())[0]["Mounts"])
        if "--mount" in argv and fail == "strip":
            rc = 1  # e.g. an image without find
        elif "/usr/bin/find" in argv and "-user" in argv and "-print0" in argv:
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


_CAP_ARGS = {name: args for name, args in [  # a container that could write file capabilities
    ("setfcap", ["--cap-add", "SETFCAP"]), ("cap_setfcap", ["--cap-add", "cap_setfcap"]),
    ("all", ["--cap-add", "ALL"]), ("privileged", ["--privileged"])]}


def _assert_strip(argv, sandbox, caps):
    """One hardened throwaway that mounts only the creator's existing sandbox dirs and needs no stat."""
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--mount"] == [
        f"type=bind,src={sandbox}/home,dst=/root", f"type=bind,src={sandbox}/workspace,dst=/workspace"], argv
    assert "--volumes-from" not in argv and "-v" not in argv and "--cap-drop" in argv, argv
    tail = argv[argv.index("sha256:img") + 1:]
    if caps:  # every regular file re-chowned to its own uid clears caps too; no owner changes
        assert tail[:2] == ["/bin/sh", "-c"] and "-perm" not in tail and 'chown -h "$u"' in tail[2], tail
    else:
        assert tail[0] == "find" and tail[-9:] == ["-type", "f", "-perm", "/6000", "-exec", "chmod", "ug-s", "{}", "+"], tail


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
    (False, [], _PLAIN, "persist"),  # never removed: stripped on detach and reattach, owners unchanged
    (True, [], _PLAIN, "persist-none"),
    (False, [], _PLAIN, "persist-shared"),  # another profile attaching strips the creator's dirs
    (False, [], _PLAIN, "persist-reap"),  # the orphan sweep, with TERMINAL_SANDBOX_DIR behind a symlink
    (False, [], _PLAIN, "persist-gone"),  # deleted sandbox dirs are never recreated root-owned
    (False, [], _PLAIN, "persist-fails-strip"),  # warned once, not at debug
    (False, [], _PLAIN, "persist-label"),  # container metadata never widens what the root strip touches
    *[(False, args, _PLAIN, scenario) for scenario in ("handback", "persist") for args in _CAP_ARGS.values()],
])
def test_root_container_teardown_hands_every_sandbox_mount_back_before_removal(
        monkeypatch, tmp_path, caplog, run_as_host_user, user_args, security_options, scenario):
    real_sandbox = tmp_path / "sandboxes"
    real_sandbox.mkdir()
    if scenario == "persist-reap":
        (tmp_path / "link").symlink_to(real_sandbox)
    monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(tmp_path / ("link" if scenario == "persist-reap" else "sandboxes")))
    if scenario == "persist-shared":  # profiles of one Hermes root, each with its default sandbox dir
        monkeypatch.delenv("TERMINAL_SANDBOX_DIR")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "root" / "profiles" / "creator"))
    caps = any(a in user_args for args in _CAP_ARGS.values() for a in args[-1:])
    fail = scenario.rsplit("-", 1)[1] if "fails" in scenario else scenario if scenario == "chown-timeout" else ""
    calls = _fake_docker(monkeypatch, security_options, fail=fail)
    nested_ro, shared, decoy = tmp_path / "skills", tmp_path / "shared", tmp_path / "outside" / "a" / "decoy"
    nested_ro.mkdir()
    shared.mkdir()
    decoy.mkdir(parents=True)
    if scenario == "persist-label":  # extra args (or an image LABEL) claim a host dir as a sandbox root
        user_args = ["--label", f"hermes-sandbox={tmp_path / 'outside'}", "-v", f"{decoy}:/workspace"]
    record = tmp_path / "sandboxes" / "docker" / "chown-back" / ".hermes-owner-handback"
    if scenario.startswith("reclaim"):  # an earlier root teardown handed these back; "link" is now a symlink
        record.parent.mkdir(parents=True)
        record.write_bytes(_HOST_OWNED + b"/workspace/link/escaped\0")

    def _env():
        return docker_env.DockerEnvironment(
            image="python:3.12-slim", cwd="/root", timeout=5, task_id="chown-back", persistent_filesystem=True,
            persist_across_processes=scenario.startswith("persist"), run_as_host_user=run_as_host_user,
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
    sandbox = os.path.realpath(env._home_dir.rsplit("/", 1)[0])  # mounted resolved: a later symlink swap is inert
    if scenario == "persist-gone":
        shutil.rmtree(sandbox)
    env.cleanup()
    assert env.wait_for_cleanup(5)
    if scenario.startswith("persist"):  # a running container: strip bits/caps in place, give nothing away
        if scenario == "persist-shared":
            monkeypatch.setenv("HERMES_HOME", str(tmp_path / "root" / "profiles" / "other"))
        monkeypatch.setattr(docker_env.DockerEnvironment, "_find_reusable_container", lambda *a: ("fake-cid", "running"))
        if scenario == "persist-gone":
            docker_env.reap_orphan_containers(max_age_seconds=0)
            assert not os.path.exists(sandbox)
        else:
            reattached = _env()
            assert reattached._container_id == "fake-cid"
        if scenario == "persist-reap":
            docker_env.reap_orphan_containers(max_age_seconds=0)
        # Filtered to this sandbox: an env another test holds may tear down (and strip) mid-test.
        strips = [argv for argv, _ in calls if "--rm" in argv and f"{sandbox}/home" in " ".join(argv)]
        assert "stop" not in [argv[1] for argv, _ in calls] and not record.exists(), calls
        assert len(strips) == {"persist-none": 0, "persist-gone": 0, "persist-reap": 3}.get(scenario, 2), calls
        assert not [argv for argv, _ in calls if "--mount" in argv and str(decoy) in " ".join(argv)], calls
        if scenario == "persist-label":
            return
        for argv in strips:  # only the sandbox dirs are mounted: a swap mid-walk cannot reach host data
            _assert_strip(argv, sandbox, caps)
        warned = [r.levelno for r in caplog.records if "Could not strip" in r.getMessage() and sandbox in r.getMessage()]
        assert warned == ([logging.WARNING] if scenario == "persist-fails-strip" else []), warned  # once, then debug
        return

    subcommands = [argv[1] for argv, _ in calls]
    chowns = [(argv, stdin) for argv, stdin in calls if "/usr/bin/xargs" in argv]
    if scenario == "none":
        assert chowns == [], chowns
        assert subcommands[-1] == "rm" and "stop" in subcommands
        return
    assert record.read_bytes() == _ROOT_OWNED
    # Handback first (its chown clears the bits it touches), then the strip on its own budget, before rm.
    (strip,) = [argv for argv, _ in calls if "--mount" in argv]
    _assert_strip(strip, sandbox, caps)
    if scenario == "chown-timeout":  # the CLI is killed; the named throwaway must not run on
        name = chowns[0][0][chowns[0][0].index("--name") + 1]
        assert name.startswith("hermes-handback-fake-cid-") and [docker_env.find_docker(), "rm", "-f", name] in [
            argv for argv, _ in calls], calls
        assert calls.index((chowns[0][0], chowns[0][1])) < calls.index((strip, b""))  # a timeout never skips it
        return
    (walk,) = [argv for argv, _ in calls if "/usr/bin/find" in argv and "-print0" in argv]
    ((chown, stdin),) = chowns
    # After stop (no writer left, shutdown writes included), before rm, from the stopped container's volumes.
    stop, rm = subcommands.index("stop"), subcommands.index("rm")
    assert stop < calls.index((walk, b"")) < calls.index((chown, stdin)) < calls.index((strip, b"")) < rm
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
