"""Regression for #141630: the bot-desktop re-attach binds to the container the marker records and
never asks the terminal planner to acquire one — for session-scoped sandboxes the planner's
label-based reuse is disabled (persist_across_processes off), so every acquire from a process
without the registry entry built a DUPLICATE container while the screen's real container kept
running (and runtime.stop() then stopped the throwaway one)."""

from __future__ import annotations

from tools.bot_desktop import runtime, sandbox_host
from tools.environments import streams
from tools.environments.docker import DockerEnvironment


def _marker() -> dict:
    return {"backend": "DockerEnvironment", "container": "c" * 64, "docker": "/usr/bin/docker",
            "dir": "/tmp/hermes-bot-desktop/demo", "display": ":20", "profile": "demo"}


def test_sandbox_env_reattaches_to_marker_container_without_a_planner_acquire(monkeypatch):
    """An alive recorded container IS the screen's sandbox: bind to it, and never build another one —
    not even with create=True (the old behaviour grew the container count 1 -> 2)."""
    marker = _marker()
    monkeypatch.setattr(sandbox_host, "_read_marker", lambda: marker)
    monkeypatch.setattr(sandbox_host, "marker_sandbox_alive", lambda m: True)

    def _no_acquire(*, create: bool):
        raise AssertionError(f"planner acquire create={create} would risk a duplicate container")

    monkeypatch.setattr(runtime.placement, "terminal_environment", _no_acquire)
    env = runtime._sandbox_env(create=True)
    assert isinstance(env, DockerEnvironment)
    assert env._container_id == marker["container"] and env._docker_exe == marker["docker"]
    owned = runtime._owned_sandbox_env()
    assert isinstance(owned, DockerEnvironment) and owned._container_id == marker["container"]


def test_sandbox_env_create_still_builds_only_when_no_marker_names_a_live_container(monkeypatch):
    """The fail side of the contract: a fresh screen start (no marker) still acquires through the
    planner, and create=False never builds anything."""
    monkeypatch.setattr(sandbox_host, "_read_marker", dict)
    acquires: list = []

    def _planner(*, create: bool):
        acquires.append(create)
        return "fresh-env" if create else None

    monkeypatch.setattr(runtime.placement, "terminal_environment", _planner)
    assert runtime._sandbox_env(create=False) is None
    assert acquires == [False]
    assert runtime._sandbox_env(create=True) == "fresh-env"
    assert acquires == [False, False, True]


def test_marker_env_routes_exec_through_the_recorded_container():
    """Every sandbox_host probe/stop/stream goes through streams.exec_prefix: the bound handle must
    exec into the recorded container id, so stop() targets the screen's real container."""
    marker = _marker()
    env = sandbox_host.marker_env(marker)
    assert env is not None
    argv = streams.exec_prefix(env, user="pn", interactive=True)
    assert argv == [marker["docker"], "exec", "-i", "-u", "pn", marker["container"]]
    assert env.get_temp_dir() == "/tmp"


def test_marker_env_is_none_without_a_recorded_container():
    assert sandbox_host.marker_env({"backend": "SSHEnvironment"}) is None
