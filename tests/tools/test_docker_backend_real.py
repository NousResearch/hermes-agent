"""The Docker terminal backend against a real daemon and a real container.

Every other Docker backend test mocks ``subprocess`` or the docker CLI. These run
``terminal_tool`` and the file tools with ``TERMINAL_ENV=docker`` end to end, so
a broken ``docker run``/``docker exec`` argv, snapshot bootstrap, cwd tracking or
env forwarding fails here instead of on a user's machine.

Armed by ``HERMES_TEST_DOCKER_IMAGE`` (a small public image with bash, e.g.
``debian:13-slim``); the remote-backends CI lane sets it. Skipped otherwise. Once
armed a missing daemon FAILS rather than skips: a lane that reports green over
zero coverage is the state this file exists to end.
"""

import json
import os
import shlex
import subprocess
import uuid

import pytest

_IMAGE = os.getenv("HERMES_TEST_DOCKER_IMAGE", "")

pytestmark = pytest.mark.skipif(not _IMAGE, reason="HERMES_TEST_DOCKER_IMAGE not set")


@pytest.fixture
def docker_task(monkeypatch):
    """A fresh docker-backed task: own container, removed (not left running) afterwards."""
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_DOCKER_IMAGE", _IMAGE)
    monkeypatch.setenv("TERMINAL_DOCKER_IMAGE_PINNED", "1")
    # Remove the container at cleanup instead of leaving it for cross-process reuse,
    # and never sweep the host's other hermes containers.
    monkeypatch.setenv("TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES", "false")
    monkeypatch.setenv("TERMINAL_DOCKER_ORPHAN_REAPER", "false")
    monkeypatch.setenv("TERMINAL_CONTAINER_DISK", "0")
    monkeypatch.setenv("TERMINAL_DOCKER_ENV", json.dumps({"HERMES_RB_STATIC": "from-config"}))
    monkeypatch.setenv("TERMINAL_DOCKER_FORWARD_ENV", json.dumps(["HERMES_RB_FORWARDED"]))
    monkeypatch.setenv("HERMES_RB_FORWARDED", "from-host")
    monkeypatch.setenv("HERMES_RB_NOT_FORWARDED", "must-stay-on-host")
    task_id = f"rb-docker-{uuid.uuid4().hex[:8]}"
    yield task_id
    _teardown(task_id)


def _env_key(task_id) -> str:
    # Without a session key, persistent Docker collapses every task onto the profile's
    # container key ("default" here), exactly as a CLI run does.
    from tools.terminal_tool import _resolve_container_task_id

    return _resolve_container_task_id(task_id)


def _teardown(task_id):
    from tools.environments.docker import DockerEnvironment
    from tools.terminal_tool_lifecycle import cleanup_vm

    cleanup_vm(_env_key(task_id), force_remove=True)
    assert DockerEnvironment.wait_for_all_teardowns(timeout=60)


def _run(command, task_id, **kwargs):
    from tools.terminal_tool import terminal_tool

    return json.loads(terminal_tool(command, task_id=task_id, **kwargs))


def _container_id(task_id) -> str | None:
    """The live env's container, confirmed running by the daemon itself."""
    from tools.terminal_tool import _active_environments

    env = _active_environments.get(_env_key(task_id))
    cid = getattr(env, "_container_id", None)
    if not cid:
        return None
    state = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", cid],
                           capture_output=True, text=True, timeout=30, check=False)
    return cid if state.stdout.strip() == "true" else None


def _container_exists(cid: str) -> bool:
    return subprocess.run(["docker", "inspect", cid], capture_output=True, timeout=30,
                          check=False).returncode == 0


def test_exec_runs_in_the_container_and_reports_exit_codes(docker_task):
    r = _run("echo hello-docker; cat /etc/os-release | head -1", docker_task)
    assert r["exit_code"] == 0, r
    assert "hello-docker" in r["output"]
    assert "Debian" in r["output"]  # ran in the image, not on the host
    assert _container_id(docker_task)

    failed = _run("echo before; exit 7", docker_task)
    assert failed["exit_code"] == 7
    assert "before" in failed["output"]


def test_cwd_persists_and_workdir_does_not(docker_task):
    assert _run("pwd", docker_task)["output"].strip() == "/root"
    _run("mkdir -p /workspace/proj && cd /workspace/proj", docker_task)
    assert _run("pwd", docker_task)["output"].strip() == "/workspace/proj"

    once = _run("pwd", docker_task, workdir="/usr")
    assert once["output"].strip() == "/usr"
    assert _run("pwd", docker_task)["output"].strip() == "/workspace/proj"


def test_exported_env_persists_between_commands(docker_task):
    _run("export HERMES_RB_EXPORTED='kept value'", docker_task)
    assert _run('echo "$HERMES_RB_EXPORTED"', docker_task)["output"].strip() == "kept value"
    _run("unset HERMES_RB_EXPORTED", docker_task)
    assert _run('echo "${HERMES_RB_EXPORTED-gone}"', docker_task)["output"].strip() == "gone"


def test_configured_and_forwarded_env_reach_the_container_and_nothing_else(docker_task):
    r = _run('echo "s=$HERMES_RB_STATIC f=$HERMES_RB_FORWARDED n=${HERMES_RB_NOT_FORWARDED:-unset}"',
             docker_task)
    assert r["output"].strip() == "s=from-config f=from-host n=unset"


def test_quoting_survives_the_docker_exec_hop(docker_task):
    payload = "it's \"quoted\" $HOME `tick` \\back"
    r = _run(f"printf '%s\\n' {shlex.quote(payload)}", docker_task)
    assert r["output"].rstrip("\n") == payload


def test_file_tools_write_and_read_inside_the_container(docker_task):
    from tools.file_tools import read_file_tool, write_file_tool

    content = "first line\nsecond: ünïcödé ✓\n"
    wrote = json.loads(write_file_tool("/workspace/notes.txt", content, task_id=docker_task))
    assert wrote.get("success", True) and "error" not in wrote, wrote

    # Byte-exact on disk (the tool output itself is trailing-newline-trimmed).
    assert _run("cat /workspace/notes.txt; echo; wc -c < /workspace/notes.txt", docker_task)[
        "output"].splitlines()[-1].strip() == str(len(content.encode()))
    assert _run("cat /workspace/notes.txt", docker_task)["output"] == content.rstrip("\n")
    read = json.loads(read_file_tool("/workspace/notes.txt", task_id=docker_task))
    assert "second: ünïcödé ✓" in read["content"], read


def test_home_survives_a_container_replacement(docker_task):
    """container_persistent (default on): /root is a host bind mount, so a file written
    there is still present after the container is removed and a new one starts."""
    _run("echo durable > /root/state.txt", docker_task)
    first = _container_id(docker_task)
    assert first
    _teardown(docker_task)
    assert not _container_exists(first)

    r = _run("cat /root/state.txt", docker_task)
    assert r["output"].strip() == "durable"
    second = _container_id(docker_task)
    assert second and second != first
