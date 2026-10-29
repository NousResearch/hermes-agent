"""Regression for #141629: the RFB stream for a sandbox-hosted screen attaches to the container the
marker records when this process has no terminal env registry entry (gateway restart / another
process owns the screen). Before, open_rfb_stream() raised RuntimeError there while
sandbox_screen_running() said the screen was up, and the display bridge turned that into a
silent black pane."""

from __future__ import annotations

import subprocess

import pytest

from tools.bot_desktop import runtime, sandbox_host


def _marker() -> dict:
    return {"backend": "DockerEnvironment", "container": "c" * 64, "docker": "/usr/bin/docker",
            "dir": "/tmp/hermes-bot-desktop/demo", "display": ":20", "profile": "demo"}


def test_open_rfb_stream_attaches_to_marker_container_without_registry_entry(monkeypatch):
    """The screen's container outlives the creating process's registry entry; the stream must still
    reach it — the viewer must not black-screen on a process that did not start the sandbox."""
    marker = _marker()
    monkeypatch.setattr(runtime, "_sandbox_env", lambda *, create: None)
    monkeypatch.setattr(sandbox_host, "_read_marker", lambda: marker)
    monkeypatch.setattr(sandbox_host, "marker_sandbox_alive", lambda m: True)
    seen: dict = {}
    monkeypatch.setattr(sandbox_host, "open_rfb_stream_marker",
                        lambda m, profile: seen.update(marker=m, profile=profile) or "stream")
    assert runtime.open_rfb_stream() == "stream"
    assert seen["marker"] is marker


def test_open_rfb_stream_still_raises_when_the_markers_sandbox_is_gone(monkeypatch):
    """The reachability contract keeps its fail-closed side: no registry env AND no live recorded
    container is "not running", and the bridge maps it to the 4001 close."""
    monkeypatch.setattr(runtime, "_sandbox_env", lambda *, create: None)
    monkeypatch.setattr(sandbox_host, "_read_marker", dict)
    with pytest.raises(RuntimeError):
        runtime.open_rfb_stream()
    monkeypatch.setattr(sandbox_host, "_read_marker", _marker)
    monkeypatch.setattr(sandbox_host, "marker_sandbox_alive", lambda m: False)
    with pytest.raises(RuntimeError):
        runtime.open_rfb_stream()


def test_open_rfb_stream_marker_execs_into_the_recorded_container(monkeypatch):
    """The marker path binds to the RECORDED container id (not a planner acquire, which builds a
    duplicate container for session-scoped sandboxes) and relays its rfb.sock."""
    marker = _marker()
    monkeypatch.setattr(sandbox_host.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout=b"1000\n"))
    captured: dict = {}

    class _Popen:
        def __init__(self, argv, **kwargs):
            captured["argv"] = argv

    monkeypatch.setattr(sandbox_host.subprocess, "Popen", _Popen)
    sandbox_host.open_rfb_stream_marker(marker, "demo")
    argv = captured["argv"]
    assert argv[:4] == [marker["docker"], "exec", "-i", "-u"]
    assert argv[4] == sandbox_host.DESKTOP_USER and argv[5] == marker["container"]
    assert "python3" in argv and argv[-1] == f"{marker['dir']}/rfb.sock"


def test_open_rfb_stream_marker_survives_a_user_probe_failure(monkeypatch):
    """An image without DESKTOP_USER falls back to the exec default rather than giving up the
    stream entirely."""
    marker = _marker()

    def _boom(*a, **k):
        raise OSError("docker unavailable")

    monkeypatch.setattr(sandbox_host.subprocess, "run", _boom)
    captured: dict = {}

    class _Popen:
        def __init__(self, argv, **kwargs):
            captured["argv"] = argv

    monkeypatch.setattr(sandbox_host.subprocess, "Popen", _Popen)
    sandbox_host.open_rfb_stream_marker(marker, "demo")
    argv = captured["argv"]
    assert argv[:3] == [marker["docker"], "exec", "-i"] and argv[3] == marker["container"]
