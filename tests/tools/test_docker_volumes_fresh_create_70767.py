"""Regression tests for #69575 / supersedes #70767.

The previously proposed patch only fixed the *reuse* path: a cached
container with stale bind-mounts got `rm -f`'d. The repro the issue
actually describes goes through the *fresh-create* path (``docker rm -f
<existing>; start new container``), where ``TERMINAL_DOCKER_VOLUMES``
(the env var the gateway snapshots at boot) carries the old list and the
freshly-created container silently omits any mount added to
``config.yaml`` since the gateway started.

Two behavioral contracts:

* ``_resolved_docker_volumes`` re-reads current config.yaml on each
  call and unions with the passed-in list, so a freshly-created container
  picks up mounts added since the gateway started.
* The reuse path now inspects the cached container's actual bind-mount
  destinations and ``rm -f``'s the container when a required destination
  is missing — then the fresh-create path's re-resolution takes effect.
"""

import logging
import subprocess

import pytest

from tools.environments import docker as docker_env


# ---- helpers ----

class _FakeConfig:
    """Minimal stand-in for hermes_cli.config.load_config() result."""

    def __init__(self, terminal):
        self._terminal = terminal

    def get(self, key, default=None):
        if key == "terminal":
            return self._terminal
        return default


def _stub_load_config(monkeypatch, terminal_section):
    """Patch ``hermes_cli.config.load_config`` to return a fake config."""
    cfg = _FakeConfig(terminal_section)
    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda: cfg, raising=False
    )
    # Also patch the import path that _resolved_docker_volumes uses
    # (lazy import inside the function body).
    monkeypatch.setattr(
        "tools.environments.docker.load_config",
        lambda: cfg,
        raising=False,
    )
    # Direct attribute path used by the function:
    monkeypatch.setattr(docker_env, "load_config", lambda: cfg, raising=False)


def _args_str(call):
    cmd, _ = call
    return " ".join(cmd) if isinstance(cmd, list) else ""


def _find_run_call(calls):
    runs = [c for c in calls if isinstance(c[0], list) and c[0][1] == "run"]
    assert runs, "docker run should have been called"
    return runs[0]


def _has_arg_pair(cmd, flag, value):
    """True if ``[flag, value]`` appears consecutively in the argv list."""
    for i, tok in enumerate(cmd):
        if tok == flag and i + 1 < len(cmd) and cmd[i + 1] == value:
            return True
    return False


# ---- unit: _resolved_docker_volumes ----

def test_resolved_volumes_unions_passed_and_config(monkeypatch):
    """``_resolved_docker_volumes`` unions caller-passed with config.yaml's
    current list, dedupes, preserves order."""
    _stub_load_config(
        monkeypatch,
        {"docker_volumes": ["/data/cfg1:/data/c", "/data/shared:/data/s"]},
    )
    out = docker_env._resolved_docker_volumes(
        ["/data/old:/data/old", "/data/shared:/data/s"]
    )
    # caller-passed first (in order), then config extras (in order), deduped
    assert out == [
        "/data/old:/data/old",
        "/data/shared:/data/s",
        "/data/cfg1:/data/c",
    ]


def test_resolved_volumes_dedupes_within_passed(monkeypatch):
    _stub_load_config(monkeypatch, {"docker_volumes": []})
    out = docker_env._resolved_docker_volumes(["/a:/x", "/a:/x", "/b:/y"])
    assert out == ["/a:/x", "/b:/y"]


def test_resolved_volumes_falls_back_when_config_unreadable(monkeypatch):
    """If config read raises, return passed_volumes only."""

    def _raise():
        raise RuntimeError("boom")

    monkeypatch.setattr(docker_env, "load_config", _raise, raising=False)
    out = docker_env._resolved_docker_volumes(["/x:/y", "/a:/b"])
    assert out == ["/x:/y", "/a:/b"]


def test_resolved_volumes_handles_none_passed(monkeypatch):
    _stub_load_config(monkeypatch, {"docker_volumes": ["/a:/b"]})
    out = docker_env._resolved_docker_volumes(None)
    assert out == ["/a:/b"]


def test_volume_destination_parses_three_segment():
    """host:container[:mode] -> container only."""
    assert docker_env._volume_destination("/host/dir:/container/dir") == "/container/dir"
    assert docker_env._volume_destination("/host/dir:/container/dir:ro") == "/container/dir"
    assert docker_env._volume_destination("/host/dir:/container/dir:rw") == "/container/dir"
    assert docker_env._volume_destination("/host/dir:/container/dir:z") == "/container/dir"
    # Single segment: no colon -> no container side
    assert docker_env._volume_destination("/just-a-host-path") is None
    # Empty
    assert docker_env._volume_destination("") is None


# ---- contract: fresh-create path picks up mounts added since boot ----

def test_create_path_includes_volumes_added_to_config_after_boot(monkeypatch, tmp_path):
    """The issue-#69575 repro: operator edits config.yaml to add a new mount,
    runs ``docker rm -f`` on the cached container, restarts the gateway.
    The new mount must land on the freshly-created container despite the
    bootstrap env var having the old list.

    Black-box: drive the real ``DockerEnvironment(...)`` constructor. It
    builds ``volume_args`` from ``_resolved_docker_volumes(volumes)`` and,
    finding no reusable container (``docker ps`` empty), issues a real
    ``docker run``. We assert the *captured* ``docker run`` argv carries
    BOTH the env-var-listed mount AND the config-only mount — exercising the
    fix at the actual call site rather than only via the helper.
    """
    docker_env._cgroup_limits_ok = True
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")

    calls = []

    def _run(cmd, **kwargs):
        calls.append((list(cmd), kwargs))
        # ps / inspect: no existing container
        if isinstance(cmd, list) and len(cmd) >= 2 and cmd[1] == "ps":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if isinstance(cmd, list) and len(cmd) >= 2 and cmd[1] == "inspect":
            return subprocess.CompletedProcess(cmd, 0, stdout="[]", stderr="")
        # run: produce fake container id
        if isinstance(cmd, list) and len(cmd) >= 2 and cmd[1] == "run":
            return subprocess.CompletedProcess(
                cmd, 0, stdout="fake-container-id\n", stderr=""
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    _stub_load_config(
        monkeypatch,
        {"docker_volumes": ["/host/NEW:/c/NEW"]},
    )

    env = docker_env.DockerEnvironment(
        image="python:3.11",
        cwd="/root",
        timeout=60,
        cpu=0,
        memory=0,
        disk=0,
        persistent_filesystem=False,
        task_id="test-task",
        # Bootstrap env var still carries the OLD list (no NEW mount)
        volumes=["/host/OLD:/c/OLD"],
        forward_env=None,
        network=True,
        host_cwd=None,
        auto_mount_cwd=False,
        env=None,
        run_as_host_user=False,
        extra_args=[],
        persist_across_processes=True,
    )

    # The constructor found no reusable container and fell through to a fresh
    # ``docker run``. Assert that real argv mounted BOTH the stale bootstrap
    # mount AND the config-only mount added since boot — a wiring mistake at
    # the call site (e.g. bypassing _resolved_docker_volumes) would drop one.
    run_cmd = _find_run_call(calls)[0]
    assert _has_arg_pair(run_cmd, "-v", "/host/OLD:/c/OLD"), \
        "fresh docker run must carry the bootstrap (OLD) mount"
    assert _has_arg_pair(run_cmd, "-v", "/host/NEW:/c/NEW"), \
        "fresh docker run must carry the config-added (NEW) mount"


# ---- contract: reuse path removes stale-bind-mount containers ----

def test_reuse_path_removes_container_missing_a_bind_mount(monkeypatch, tmp_path):
    """If the cached container is missing a configured bind-mount destination
    (operator edited config.yaml post-creation), the reuse check should
    ``rm -f`` the container and fall through to a fresh create that lands the
    new mount.

    Black-box: drive the real ``DockerEnvironment(...)`` constructor. The
    mocked ``docker ps`` reports a stale running container, ``docker inspect``
    reports only the OLD bind-mount destination, so the constructor's reuse
    branch must issue ``docker rm -f stale-cid`` and then a fresh ``docker
    run`` carrying the missing NEW mount. We assert both subprocess
    invocations — a wiring mistake at either call site would leave this red.
    """
    docker_env._cgroup_limits_ok = True
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")

    calls = []

    def _run(cmd, **kwargs):
        calls.append((list(cmd), kwargs))
        if isinstance(cmd, list) and len(cmd) >= 2:
            # Reuse probe: report one stale, running container. Egress is off
            # so _find_reusable_container uses the 3-field format
            # ID\tState\tEgressLabel; an empty egress label is accepted.
            if cmd[1] == "ps":
                return subprocess.CompletedProcess(
                    cmd, 0, stdout="stale-cid\trunning\t\n", stderr=""
                )
            # Bind-mount inspection: only the OLD destination is present
            # (the operator added /c/NEW to config.yaml after creation).
            if cmd[1] == "inspect":
                return subprocess.CompletedProcess(
                    cmd, 0, stdout="bind:/c/OLD\n", stderr=""
                )
            if cmd[1] == "rm":
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if cmd[1] == "run":
                return subprocess.CompletedProcess(
                    cmd, 0, stdout="new-cid\n", stderr=""
                )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    _stub_load_config(
        monkeypatch,
        {"docker_volumes": ["/host/NEW:/c/NEW", "/host/OLD:/c/OLD"]},
    )

    env = docker_env.DockerEnvironment(
        image="python:3.11",
        cwd="/root",
        timeout=60,
        cpu=0,
        memory=0,
        disk=0,
        persistent_filesystem=False,
        task_id="test-task",
        # Bootstrap env var is stale: carries only OLD, not the NEW mount
        # the operator just added to config.yaml.
        volumes=["/host/OLD:/c/OLD"],
        forward_env=None,
        network=True,
        host_cwd=None,
        auto_mount_cwd=False,
        env=None,
        run_as_host_user=False,
        extra_args=[],
        persist_across_processes=True,
    )

    # (a) The reuse branch removed the stale container. Assert the real
    # ``docker rm -f stale-cid`` argv was issued (not just that a helper
    # computed a ``missing`` set).
    rm_calls = [
        c[0] for c in calls
        if isinstance(c[0], list) and len(c[0]) >= 4
        and c[0][1] == "rm" and c[0][2] == "-f" and c[0][3] == "stale-cid"
    ]
    assert rm_calls, "constructor must `docker rm -f stale-cid` the stale container"

    # (b) After the rm, a fresh ``docker run`` carried the previously-missing
    # NEW mount (and the OLD one).
    run_cmd = _find_run_call(calls)[0]
    assert _has_arg_pair(run_cmd, "-v", "/host/NEW:/c/NEW"), \
        "fresh docker run after rm must carry the previously-missing NEW mount"
    assert _has_arg_pair(run_cmd, "-v", "/host/OLD:/c/OLD"), \
        "fresh docker run after rm must still carry the OLD mount"


# ---- regression: Windows drive-letter volume specs (#73814 review) ----

def test_volume_destination_parses_windows_drive_letter():
    """Windows bind-mount specs like ``C:\\Users:/data`` must yield
    ``/data`` as the container destination, not ``\\Users``.

    The drive-letter colon must not be mistaken for the host/container
    field separator.  Covers forward-slash, backslash, and mode-suffixed
    variants.
    """
    cases = [
        ("C:\\Users:/data", "/data"),
        ("C:\\Users:/data:ro", "/data"),
        ("C:/Users:/data", "/data"),
        ("C:/Users:/data:rw", "/data"),
        ("D:\\code\\repo:/workspace", "/workspace"),
        ("E:/media:/mnt:z", "/mnt"),
    ]
    for spec, expect in cases:
        got = docker_env._volume_destination(spec)
        assert got == expect, (
            f"_volume_destination({spec!r}) returned {got!r}, expected {expect!r}"
        )


def test_volume_destination_preserves_linux_specs():
    """Linux/POSIX specs are unaffected by the Windows drive-letter fix."""
    assert docker_env._volume_destination("/host/dir:/container/dir") == "/container/dir"
    assert docker_env._volume_destination("/host/dir:/container/dir:ro") == "/container/dir"
    assert docker_env._volume_destination("named-volume:/data") == "/data"
    assert docker_env._volume_destination("/just-a-host-path") is None
    assert docker_env._volume_destination("") is None


# ---- regression: inspect failure skips stale-mount check (#73814 review) ----

def test_reuse_path_skips_rm_when_inspect_fails(monkeypatch, tmp_path):
    """When ``docker inspect`` fails (non-zero exit, timeout, or OSError),
    ``_container_bind_mounts`` returns ``None`` and the constructor must
    **skip** the stale-mount check rather than treating every configured
    destination as missing and ``rm -f``'ing a potentially-healthy
    container.

    Black-box: drive the real ``DockerEnvironment(...)`` constructor.
    Mock ``docker ps`` to report a running container, ``docker inspect``
    to fail (non-zero exit), and assert that **no** ``docker rm -f`` is
    issued — the container is reused as-is (fail-open).
    """
    docker_env._cgroup_limits_ok = True
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")

    calls = []

    def _run(cmd, **kwargs):
        calls.append((list(cmd), kwargs))
        if isinstance(cmd, list) and len(cmd) >= 2:
            if cmd[1] == "ps":
                return subprocess.CompletedProcess(
                    cmd, 0, stdout="healthy-cid\trunning\t\n", stderr=""
                )
            # inspect fails — daemon blip / permission error / etc.
            if cmd[1] == "inspect":
                return subprocess.CompletedProcess(
                    cmd, 1, stdout="", stderr="Error: No such container"
                )
            if cmd[1] == "rm":
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if cmd[1] == "run":
                return subprocess.CompletedProcess(
                    cmd, 0, stdout="new-cid\n", stderr=""
                )
            # start / other — succeed
            if cmd[1] == "start":
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    _stub_load_config(
        monkeypatch,
        {"docker_volumes": ["/host/OLD:/c/OLD", "/host/NEW:/c/NEW"]},
    )

    env = docker_env.DockerEnvironment(
        image="python:3.11",
        cwd="/root",
        timeout=60,
        cpu=0,
        memory=0,
        disk=0,
        persistent_filesystem=False,
        task_id="test-task",
        volumes=["/host/OLD:/c/OLD"],
        forward_env=None,
        network=True,
        host_cwd=None,
        auto_mount_cwd=False,
        env=None,
        run_as_host_user=False,
        extra_args=[],
        persist_across_processes=True,
    )

    # No docker rm -f should have been issued — inspect failed, so the
    # stale-mount check is skipped (fail-open) and the healthy container
    # is reused.
    rm_calls = [
        c[0] for c in calls
        if isinstance(c[0], list) and len(c[0]) >= 4
        and c[0][1] == "rm" and c[0][2] == "-f"
    ]
    assert not rm_calls, (
        "constructor must NOT issue `docker rm -f` when inspect fails — "
        "the stale-mount check should be skipped (fail-open)"
    )


def test_reuse_path_skips_rm_when_inspect_times_out(monkeypatch, tmp_path):
    """Same fail-open contract, but ``docker inspect`` raises
    ``TimeoutExpired`` instead of returning non-zero."""
    docker_env._cgroup_limits_ok = True
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")

    calls = []

    def _run(cmd, **kwargs):
        calls.append((list(cmd), kwargs))
        if isinstance(cmd, list) and len(cmd) >= 2:
            if cmd[1] == "ps":
                return subprocess.CompletedProcess(
                    cmd, 0, stdout="healthy-cid\trunning\t\n", stderr=""
                )
            if cmd[1] == "inspect":
                raise subprocess.TimeoutExpired(cmd, 10)
            if cmd[1] == "start":
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    _stub_load_config(
        monkeypatch,
        {"docker_volumes": ["/host/OLD:/c/OLD"]},
    )

    env = docker_env.DockerEnvironment(
        image="python:3.11",
        cwd="/root",
        timeout=60,
        cpu=0,
        memory=0,
        disk=0,
        persistent_filesystem=False,
        task_id="test-task",
        volumes=["/host/OLD:/c/OLD"],
        forward_env=None,
        network=True,
        host_cwd=None,
        auto_mount_cwd=False,
        env=None,
        run_as_host_user=False,
        extra_args=[],
        persist_across_processes=True,
    )

    rm_calls = [
        c[0] for c in calls
        if isinstance(c[0], list) and len(c[0]) >= 4
        and c[0][1] == "rm" and c[0][2] == "-f"
    ]
    assert not rm_calls, (
        "constructor must NOT issue `docker rm -f` when inspect times out — "
        "fail-open"
    )
