"""Reuse candidate selection must not let an irremovable running box mask a
stopped compatible sibling, recovery must keep the tracked cwd, and snap label
staging must not leak dirs across SIGKILL (#84969 review)."""

import gc
import os
import subprocess
import time
from pathlib import Path

import pytest

from tools.environments import docker as docker_env
from tests.tools.test_docker_environment import _labels_in_run_args, _make_dummy_env


@pytest.fixture(autouse=True)
def _teardowns_finish_under_the_fakes():
    """Join daemon teardown workers before monkeypatch unwinds (same contract as
    test_docker_ownership_handback)."""
    yield
    gc.collect()
    assert docker_env.DockerEnvironment.wait_for_all_teardowns(timeout=10)


def _stateful_docker(monkeypatch):
    """Fake docker boundary with per-container state, labels, image and network."""
    containers: dict[str, dict] = {}
    calls: list[list[str]] = []

    def _ps(cmd):
        filters = [cmd[i + 1][6:] for i, arg in enumerate(cmd[:-1]) if arg == "--filter"]
        return "".join(
            f"{cid}\t{rec['state']}\n" for cid, rec in containers.items()
            if all(rec["labels"].get(k) == v for k, v in
                   (pair.split("=", 1) for pair in filters))
        ), 0

    def _create(cmd):
        cid = f"cid-{len(containers)}"
        containers[cid] = {
            "image": cmd[-3],
            "network": "none" if "--network=none" in cmd else "bridge",
            "state": "running",
            "labels": dict(pair.split("=", 1) for pair in _labels_in_run_args(cmd)),
        }
        label_file = Path(cmd[cmd.index("--label-file") + 1])
        containers[cid]["labels"].update(
            pair.split("=", 1) for pair in label_file.read_text(encoding="utf-8").splitlines())
        return cid, 0

    def _inspect(cmd):
        rec = containers[cmd[-1]]
        fmt = cmd[cmd.index("--format") + 1]
        if fmt == "{{.Config.Image}}":
            return rec["image"], 0
        if fmt == "{{.HostConfig.NetworkMode}}":
            return rec["network"], 0
        assert fmt == '{{index .Config.Labels "hermes-runtime-fingerprint"}}'
        return rec["labels"].get("hermes-runtime-fingerprint", ""), 0

    def _image_inspect(cmd):
        assert cmd[2] == "inspect"
        return ("null" if "Entrypoint" in cmd[-2] else "sha256:fake\n"), 0

    def _set_state(state):
        def _handler(cmd):
            cid = cmd[-1]
            if cid not in containers:
                return "", 1
            containers[cid]["state"] = state
            return "", 0
        return _handler

    def _rm(cmd):
        cid = cmd[-1]
        if cid not in containers:
            return "", 0
        if "-f" in cmd or containers[cid]["state"] != "running":
            del containers[cid]
            return "", 0
        return "", 1  # plain rm refuses a running container, like the daemon

    handlers = {
        "version": lambda cmd: ("Docker version", 0),
        "info": lambda cmd: ("vfs", 0),  # not overlay2: --storage-opt stays off, deterministically
        "ps": _ps,
        "run": _create,
        "inspect": _inspect,
        "image": _image_inspect,
        "pull": lambda cmd: ("", 0),
        "start": _set_state("running"),
        "stop": _set_state("exited"),
        "rm": _rm,
    }

    def _run(cmd, **kwargs):
        calls.append(list(cmd))
        sub = cmd[1] if len(cmd) > 1 else ""
        handler = handlers.get(sub)
        stdout, code = handler(cmd) if handler else ("", 0)
        return subprocess.CompletedProcess(cmd, code, stdout=stdout, stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_cgroup_limits_ok", True)
    monkeypatch.setattr(docker_env.DockerEnvironment, "init_session", lambda self: None)
    return containers, calls


def test_stopped_compatible_candidate_is_not_masked_by_running_drifted_sibling(monkeypatch):
    """Shared key + pinned image drift: the running old-image box cannot be rm'ed,
    so the stopped box on the requested image must win instead of a fresh create."""
    containers, _calls = _stateful_docker(monkeypatch)
    shared = {"shared_container_key": "trusted-team", "persistent_filesystem": True,
              "task_id": "shared-img"}
    first = _make_dummy_env(image="python:3.11", image_pinned=True, **shared)
    second = _make_dummy_env(image="python:3.12", image_pinned=True, **shared)
    assert second._container_id != first._container_id
    assert first._container_id in containers  # drifted box preserved, never killed

    containers[second._container_id]["state"] = "exited"  # idle-stopped later
    third = _make_dummy_env(image="python:3.12", image_pinned=True, **shared)
    assert third._container_id == second._container_id
    assert containers[third._container_id]["state"] == "running"


def test_stopped_airgapped_candidate_is_not_masked_by_running_bridge_sibling(monkeypatch):
    """Shared key + docker_network=false: a running bridge box is not removable, so
    the stopped ``--network=none`` sibling must be selected."""
    containers, _calls = _stateful_docker(monkeypatch)
    shared = {"shared_container_key": "trusted-team", "persistent_filesystem": True,
              "task_id": "shared-net"}
    first = _make_dummy_env(network=True, **shared)
    second = _make_dummy_env(network=False, **shared)
    assert second._container_id != first._container_id
    assert containers[second._container_id]["network"] == "none"

    containers[second._container_id]["state"] = "exited"
    third = _make_dummy_env(network=False, **shared)
    assert third._container_id == second._container_id


def test_recreate_container_starts_in_the_tracked_cwd(monkeypatch):
    """Recovery after out-of-band removal must recreate at the session's current
    cwd, not the original creation cwd — otherwise ``pwd`` silently regresses."""
    containers, calls = _stateful_docker(monkeypatch)
    env = _make_dummy_env(task_id="cwd-recovery", cwd="/root", persistent_filesystem=True)
    env.cwd = "/opt/transient"  # the session cd'd after creation
    containers.clear()  # container removed out-of-band

    assert env._recreate_container()
    run_cmds = [c for c in calls if len(c) > 1 and c[1] == "run"]
    assert run_cmds, "recovery must create a fresh container"
    assert run_cmds[-1][run_cmds[-1].index("-w") + 1] == "/opt/transient"


def test_snap_label_sweep_removes_only_stale_owned_dirs(monkeypatch, tmp_path):
    """A SIGKILLed snap-mode creator leaves its 0700 label dir under $HOME; the
    once-per-process sweep removes only stale dirs with our prefix and mode."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(docker_env, "_snap_label_dirs_swept", False)
    old = time.time() - docker_env._SNAP_LABEL_STALE_SECONDS - 60

    def _dir(name, mode):
        path = home / name
        path.mkdir(mode=mode)
        os.chmod(path, mode)
        os.utime(path, (old, old))
        return path

    stale = home / "hermes-docker-label-stale"
    stale.mkdir(mode=0o700)
    os.chmod(stale, 0o700)
    (stale / "runtime.label").write_text("k=v\n", encoding="utf-8")
    os.utime(stale, (old, old))  # last: writing inside bumps the dir mtime
    fresh = home / "hermes-docker-label-fresh"
    fresh.mkdir(mode=0o700)
    foreign = _dir("other-prefix-stale", 0o700)
    wide = _dir("hermes-docker-label-wide", 0o755)

    docker_env._sweep_stale_snap_label_dirs()

    assert not stale.exists()
    assert fresh.exists() and foreign.exists() and wide.exists()
