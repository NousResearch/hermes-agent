"""Persistent Docker reuse is bound to the final runtime configuration (#84969)."""

import subprocess
from pathlib import Path

import pytest

from tools.environments import docker as docker_env
from tests.tools.test_docker_environment import _labels_in_run_args, _make_dummy_env


@pytest.mark.parametrize("stored_fingerprint", ["stale-fingerprint", "<no value>"])
@pytest.mark.parametrize("state", ["running", "exited"])
@pytest.mark.parametrize("recovery", [False, True])
def test_reuse_rejects_container_when_runtime_fingerprint_drifts(
    monkeypatch, stored_fingerprint, state, recovery,
):
    """Drifted and legacy unlabeled containers must not be reattached."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    docker_env._cgroup_limits_ok = True
    calls = []

    def _run(cmd, **kwargs):
        calls.append((list(cmd) if isinstance(cmd, list) else cmd, kwargs))
        if isinstance(cmd, list) and len(cmd) >= 2:
            sub = cmd[1]
            if sub == "version":
                return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
            if sub == "ps":
                return subprocess.CompletedProcess(
                    cmd, 0, stdout=f"reused-cid\t{state}\n", stderr="",
                )
            if sub == "inspect":
                format_arg = cmd[cmd.index("--format") + 1]
                if format_arg == "{{.Config.Image}}":
                    return subprocess.CompletedProcess(cmd, 0, stdout="python:3.11\n", stderr="")
                assert format_arg == (
                    '{{index .Config.Labels "hermes-runtime-fingerprint"}}'
                )
                return subprocess.CompletedProcess(
                    cmd, 0, stdout=f"{stored_fingerprint}\n", stderr="",
                )
            if sub == "rm":
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if sub == "run":
                return subprocess.CompletedProcess(cmd, 0, stdout="fresh-cid\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    env = _make_dummy_env(task_id="reuse-drift", memory=2048, extra_args=["--pids-limit=64"])
    if recovery:
        assert env._recreate_container()
    assert env._container_id == "fresh-cid"
    assert not any(c[0][1] in ("rm", "start", "stop") for c in calls)
    assert any(isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run" for c in calls)


@pytest.mark.parametrize("change", [
    "unchanged", "memory", "cpu", "mount", "cwd", "env", "extra",
    "snap", "user", "network", "env-file", "inherited", "joined-inherited",
    "file-inherited", "default-image", "default-s6-image", "missing-image", "pinned-image",
    "default-image-env", "missing-image-env",
    "shared-memory", "shared-env", "shared-image", "shared-mount", "shm",
])
def test_runtime_configuration_controls_persistent_reuse(monkeypatch, tmp_path, change):
    """#84969: run -> config change -> attach -> recovery keeps only compatible boxes.

    Only the external Docker boundary is simulated. The real constructor,
    mounts, env resolution, fingerprint, selection and recovery run together.
    """
    containers = {}
    calls = []
    env_file = tmp_path / "docker.env"
    env_file.write_text("STATIC=value-a\nINHERITED_VALUE\n", encoding="utf-8")
    monkeypatch.setenv("INHERITED_VALUE", "value-a")
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_cgroup_limits_ok", True)

    def _run(cmd, **kwargs):
        calls.append((list(cmd), kwargs))
        stdout = ""
        code = 0
        if cmd[1] == "version":
            stdout = "Docker version"
        elif cmd[1] == "ps":
            filters = [cmd[i + 1][6:] for i, arg in enumerate(cmd[:-1]) if arg == "--filter"]
            stdout = "".join(
                f"{cid}\trunning\n" for cid, record in containers.items()
                if all(record["labels"].get(pair.split("=", 1)[0]) == pair.split("=", 1)[1]
                       for pair in filters)
            )
        elif cmd[1] == "run":
            if change.startswith("missing-image") and cmd[-3] == "python:3.12":
                raise subprocess.CalledProcessError(125, cmd, stderr="image unavailable")
            cid = f"cid-{len(containers)}"
            containers[cid] = {
                "image": cmd[-3],
                "network": "none" if "--network=none" in cmd else "bridge",
                "labels": dict(pair.split("=", 1) for pair in _labels_in_run_args(cmd)),
            }
            label_file = Path(cmd[cmd.index("--label-file") + 1])
            containers[cid]["labels"].update(
                pair.split("=", 1) for pair in label_file.read_text(encoding="utf-8").splitlines()
            )
            stdout = cid
        elif cmd[1] == "inspect":
            record = containers[cmd[-1]]
            fmt = cmd[cmd.index("--format") + 1]
            if fmt == "{{.Config.Image}}":
                stdout = record["image"]
            elif fmt == "{{.HostConfig.NetworkMode}}":
                stdout = record["network"]
            else:
                assert fmt == '{{index .Config.Labels "hermes-runtime-fingerprint"}}'
                stdout = record["labels"]["hermes-runtime-fingerprint"]
        elif cmd[1:3] == ["image", "inspect"]:
            if "{{.Id}}" in cmd:
                code = 1 if change.startswith("missing-image") else 0
            elif cmd[3] == "s6/image:1":
                stdout = '["/init"]'
        elif cmd[1] == "pull":
            code = 1 if change.startswith("missing-image") else 0
        elif cmd[1] == "rm":
            if cmd[-1] in containers:
                assert "-f" not in cmd, "a config change must never kill a sibling's running sandbox"
                code = 1  # Docker refuses plain rm on a running container.
        return subprocess.CompletedProcess(cmd, code, stdout=stdout, stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    initial: dict[str, str | int | bool | list[str] | dict[str, str]] = {
        "image": "python:3.11", "persistent_filesystem": True,
    }
    if change.startswith("shared-"):
        initial["shared_container_key"] = "trusted-team"
    if change == "env":
        initial["env"] = {"MY_TOKEN": "old-value"}
    if change == "env-file" or change == "file-inherited":
        initial["extra_args"] = ["--env-file", str(env_file)]
    elif change == "inherited":
        initial["extra_args"] = ["-e", "INHERITED_VALUE"]
    elif change == "joined-inherited":
        initial["extra_args"] = ["-iteINHERITED_VALUE"]
    first = _make_dummy_env(**initial)
    changed = dict(initial)
    changes = {
        "memory": {"memory": 2048}, "cpu": {"cpu": 2},
        "mount": {"volumes": [f"{tmp_path}:/data"]}, "cwd": {"cwd": "/workspace"},
        "env": {"env": {"MY_TOKEN": "new-value"}},
        "extra": {"extra_args": ["--pids-limit=64"]}, "snap": {"snap_compat": True},
        "user": {"extra_args": ["--user", "1234:5678"]}, "network": {"network": False},
        "default-image": {"image": "python:3.12"},
        "default-s6-image": {"image": "s6/image:1"},
        "missing-image": {"image": "python:3.12", "image_pinned": True},
        "pinned-image": {"image": "python:3.12", "image_pinned": True},
        "default-image-env": {"image": "python:3.12", "env": {"MY_TOKEN": "new-value"}},
        "missing-image-env": {"image": "python:3.12", "image_pinned": True,
                              "env": {"MY_TOKEN": "new-value"}},
        "shared-memory": {"memory": 2048},
        "shared-env": {"env": {"MY_TOKEN": "new-value"}},
        "shared-image": {"image": "python:3.12"},
        "shared-mount": {"volumes": [f"{tmp_path}:/data"]},
        "shm": {"shm_size": "256m"},
    }
    changed.update(changes.get(change, {}))
    if change == "env-file":
        env_file.write_text("STATIC=value-b\nINHERITED_VALUE\n", encoding="utf-8")
    elif change in ("inherited", "joined-inherited", "file-inherited"):
        monkeypatch.setenv("INHERITED_VALUE", "value-b")
    if change.startswith("missing-image"):
        # Upstream's environment filter already separates images before attach.
        # A failed fresh image pull must leave the existing sandbox untouched.
        with pytest.raises(subprocess.CalledProcessError):
            _make_dummy_env(**changed)
        assert first._container_id in containers
        assert not any(cmd[1] in ("rm", "stop") and cmd[-1] == first._container_id
                       for cmd, _kw in calls)
        return
    second = _make_dummy_env(**changed)
    should_reuse = change == "unchanged" or change.startswith("shared-")
    assert (second._container_id == first._container_id) is should_reuse
    if should_reuse:
        assert second._image == "python:3.11"
    third = _make_dummy_env(**changed)
    assert third._container_id == second._container_id, "the matching box wins over preserved stale ones"
    assert third._recreate_container()
    assert third._container_id == second._container_id
    assert all("-f" not in cmd for cmd, _kw in calls if cmd[1] == "rm")
    # Values in the client environment must not leak into label/argv delivery.
    assert all("new-value" not in arg for cmd, _kw in calls for arg in cmd)
    assert all(record["labels"]["hermes-runtime-fingerprint"] not in arg
               for record in containers.values() if record["labels"]["hermes-runtime-fingerprint"]
               for cmd, _kw in calls for arg in cmd)
    assert all(not Path(cmd[cmd.index("--label-file") + 1]).exists()
               for cmd, _kw in calls if cmd[1] == "run")
