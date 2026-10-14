"""The Bot Desktop sandbox marker must track whichever container actually backs the live screen
right now, not just whatever ``start()`` recorded once at cold-start.

Confirmed live on wian (follow-up to #t_2ca05370's reattach-hint fix): the marker named a
container that had been gone for hours (``docker inspect``: no such object), while TWO other,
differently-labeled containers sat alive in the meantime. The reattach hint added by
959b400df6 makes RECONNECTING after a restart tolerant of a drifted task-id key, but nothing
ever refreshed the marker itself once a live call resolved to a different container than the
one it already named -- so the marker calcified on its first-ever value and the hint it feeds
degraded into a guaranteed miss.
"""
from __future__ import annotations

import json

import pytest

from tools.bot_desktop import sandbox_host


class _FakeDockerEnv:
    _docker_exe = "docker"

    def __init__(self, container_id):
        self._container_id = container_id

    def get_temp_dir(self):
        return "/tmp"


class _FakeSshEnv:
    """A non-Docker backend: no ``_container_id`` to track at all."""


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _write_marker(container: str):
    sandbox_host._marker().parent.mkdir(parents=True, exist_ok=True)
    sandbox_host._marker().write_text(
        json.dumps({"display": ":20", "dir": "/tmp/hermes-bot-desktop/wian", "profile": "wian",
                    "backend": "DockerEnvironment", "container": container, "docker": "/usr/bin/docker"}),
        encoding="utf-8")


def test_refresh_owner_rewrites_a_drifted_marker(isolated_home):
    """THE BUG, directly: the marker names a container that is no longer the one in use (it may
    not even exist anymore) -- the next call to resolve the sandbox must correct it."""
    _write_marker("stale-and-gone")
    env = _FakeDockerEnv("fresh-actual-container")

    sandbox_host.refresh_owner(env, "wian", {"DISPLAY": ":20"})

    assert sandbox_host._read_marker()["container"] == "fresh-actual-container"


def test_refresh_owner_is_a_no_op_when_nothing_drifted(isolated_home, monkeypatch):
    """The common case (every call, once the marker is already correct) must not thrash a disk
    write on every single computer_use call."""
    _write_marker("same-container")
    env = _FakeDockerEnv("same-container")
    before_mtime = sandbox_host._marker().stat().st_mtime_ns

    def _boom(*a, **k):
        raise AssertionError("_record must not be called when the marker already matches")
    monkeypatch.setattr(sandbox_host, "_record", _boom)

    sandbox_host.refresh_owner(env, "wian", {"DISPLAY": ":20"})

    assert sandbox_host._marker().stat().st_mtime_ns == before_mtime


def test_refresh_owner_writes_a_fresh_marker_when_none_existed(isolated_home):
    """No marker yet (first-ever resolution via this path, bypassing the cold-start start()
    call) -- still must not crash, and must produce a correct one."""
    env = _FakeDockerEnv("brand-new")
    assert sandbox_host._read_marker() == {}

    sandbox_host.refresh_owner(env, "wian", {"DISPLAY": ":20"})

    assert sandbox_host._read_marker()["container"] == "brand-new"


def test_refresh_owner_ignores_backends_without_a_container_id(isolated_home, monkeypatch):
    """SSH/apptainer backends have no ``_container_id`` -- nothing to track, must not crash or
    write a bogus marker."""
    _write_marker("some-docker-container")

    def _boom(*a, **k):
        raise AssertionError("_record must not be called for a backend with no container id")
    monkeypatch.setattr(sandbox_host, "_record", _boom)

    sandbox_host.refresh_owner(_FakeSshEnv(), "wian", {"DISPLAY": ":20"})

    assert sandbox_host._read_marker()["container"] == "some-docker-container"  # untouched


def test_sandbox_cua_invocation_refreshes_the_marker_after_resolving_a_drifted_container(monkeypatch, isolated_home):
    """Wiring: the actual computer_use entry point (sandbox_mcp_invocation / sandbox_cli_invocation,
    both routed through _sandbox_cua_invocation) must call refresh_owner with whatever env it just
    resolved -- not just leave the marker as a stale one-time snapshot from cold-start."""
    from tools.bot_desktop import placement, runtime as bd_runtime
    from tools.computer_use import cua_backend as cb

    _write_marker("stale-and-gone")
    fresh_env = _FakeDockerEnv("freshly-resolved-container")

    monkeypatch.setattr(bd_runtime, "tool_placement", lambda: placement.TERMINAL)
    monkeypatch.setattr(bd_runtime, "published_env", lambda: {"DISPLAY": ":20"})
    monkeypatch.setattr(bd_runtime, "_sandbox_env", lambda create=True: fresh_env)
    monkeypatch.setattr(bd_runtime, "_profile_name", lambda: "wian")
    monkeypatch.setattr(bd_runtime, "touch_activity", lambda: None)
    monkeypatch.setattr(sandbox_host, "exec_invocation", lambda *a, **k: ("docker", ["exec", "-i", "x"]))

    cb.sandbox_mcp_invocation()

    assert sandbox_host._read_marker()["container"] == "freshly-resolved-container"
