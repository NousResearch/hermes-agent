"""Bot Desktop placement: the screen (and with it computer_use and the browser) follows the terminal backend.

Two invariants, both about the boundary a sandboxed user chose:
  1. A sandbox backend that cannot host a screen never falls back to the gateway host silently; it refuses
     unless ``bot_desktop.placement: gateway`` opts in. Docker/ssh/singularity resolve to the sandbox.
  2. Every RFB byte between the pane and a sandbox-hosted Xvnc, and cua-driver's MCP stdio, go through the
     terminal backend's exec prefix, never a host socket or a host binary.
"""
from __future__ import annotations

import subprocess

import pytest

from tools.bot_desktop import placement, runtime
from tools.environments import streams


@pytest.mark.parametrize(
    ("setting", "backend", "expected"),
    [
        ("auto", "local", placement.GATEWAY),
        ("auto", "docker", placement.TERMINAL),
        ("auto", "ssh", placement.TERMINAL),
        ("auto", "singularity", placement.TERMINAL),
        ("auto", "modal", placement.REFUSED),
        ("auto", "daytona", placement.REFUSED),
        ("gateway", "modal", placement.GATEWAY),
        ("gateway", "docker", placement.GATEWAY),
        ("terminal", "docker", placement.TERMINAL),
        ("terminal", "local", placement.GATEWAY),
    ],
)
def test_placement_follows_the_terminal_backend_and_refuses_unhostable_sandboxes(monkeypatch, setting, backend, expected):
    monkeypatch.setattr(placement, "_setting", lambda: setting)
    monkeypatch.setattr(placement, "_terminal_backend", lambda: backend)
    where = placement.resolve()
    assert where.where == expected
    if expected == placement.REFUSED:
        assert backend in where.reason and "placement: gateway" in where.reason


def test_refused_placement_blocks_start_and_names_the_opt_in(monkeypatch):
    monkeypatch.setattr(placement, "_setting", lambda: "auto")
    monkeypatch.setattr(placement, "_terminal_backend", lambda: "modal")
    with pytest.raises(RuntimeError, match="placement: gateway"):
        runtime.start()


class _FakeDocker:
    """Stand-in for DockerEnvironment: only what ``streams.exec_prefix`` reads."""
    _docker_exe = "docker"
    _container_id = "c0ffee"

    def get_temp_dir(self):
        return "/tmp"  # no-tmp: ok — sandbox-side path


def test_sandbox_rfb_and_cua_ride_the_exec_prefix(monkeypatch):
    from tools.environments.docker import DockerEnvironment
    env = _FakeDocker()
    monkeypatch.setattr(streams, "exec_prefix", lambda e, *, user=None, interactive=True:
                        ["docker", "exec", "-i", "-u", user, e._container_id] if user else ["docker", "exec", "-i", e._container_id])
    spawned: list[list[str]] = []

    class _P:
        stdin = stdout = None
        def kill(self):
            pass

    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kw: spawned.append(argv) or _P())
    from tools.bot_desktop import sandbox_host
    monkeypatch.setattr(sandbox_host, "_user_for", lambda e: "pn")
    sandbox_host.open_rfb_stream(env, "p1")
    assert spawned[0][:6] == ["docker", "exec", "-i", "-u", "pn", "c0ffee"]
    assert "rfb.sock" in spawned[0][-1] and "python3" in spawned[0][-1]

    command, args = sandbox_host.cua_mcp_invocation(env, "p1", {"DISPLAY": ":20"})
    assert command == "docker" and args[:5] == ["exec", "-i", "-u", "pn", "c0ffee"]
    assert "cua-driver mcp" in args[-1] and "export DISPLAY=:20" in args[-1]
    assert not isinstance(env, DockerEnvironment)  # the fake never touched a real daemon
