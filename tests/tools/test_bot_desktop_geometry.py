"""Hand-back restores ``bot_desktop.geometry``: a lease holder may resize the screen (noVNC
``resizeSession``, Xvnc ``-AcceptSetDesktopSize``) and Xvnc keeps that size after they leave."""

from __future__ import annotations

import subprocess

import pytest

from tools.bot_desktop import lease, runtime


@pytest.fixture(autouse=True)
def _clean_lease():
    lease._reset_for_tests()
    yield
    lease._reset_for_tests()


def test_release_after_a_takeover_restores_the_size_once(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime, "restore_geometry", lambda profile_home=None: calls.append(profile_home) or True)
    lease.acquire("phone")
    lease.release("someone-else")          # a stale viewer: nothing handed back
    assert calls == []
    assert lease.release("phone").holder == lease.AGENT
    assert calls == [None]
    lease.release()                        # already the agent's: no transition, no restore
    assert calls == [None]


def _fake_xrandr(query_stdout, set_ok=False, fail=()):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv[1:])
        if argv[1] == "-q":
            return subprocess.CompletedProcess(argv, 0, query_stdout, "")
        if argv[1] == "-s":
            return subprocess.CompletedProcess(argv, 0 if set_ok else 1, "", "Size not found in available modes")
        return subprocess.CompletedProcess(argv, 1 if argv[1] in fail else 0, "", "")
    return run, calls


_RESIZED = "Screen 0: minimum 32 x 32, current 393 x 607, maximum 32768 x 32768\nVNC-0 connected 393x607+0+0 0mm x 0mm\n"
_AT_SIZE = "Screen 0: minimum 32 x 32, current 1440 x 900, maximum 32768 x 32768\nVNC-0 connected 1440x900+0+0\n"


@pytest.fixture
def screen_up(monkeypatch):
    monkeypatch.setattr(runtime, "published_env", lambda: {"DISPLAY": ":20", "XAUTHORITY": "/x"})
    monkeypatch.setattr(runtime, "geometry", lambda: "1440x900")


def test_restore_adds_the_configured_mode_back_when_xvnc_dropped_it(monkeypatch, screen_up):
    run, calls = _fake_xrandr(_RESIZED)
    monkeypatch.setattr(runtime.subprocess, "run", run)
    assert runtime.restore_geometry() is True
    assert calls[0] == ["-q"] and calls[1] == ["-s", "1440x900"]
    assert calls[2][:2] == ["--newmode", "hermes-1440x900"] and calls[2][3] == "1440" and calls[2][7] == "900"
    assert calls[3] == ["--addmode", "VNC-0", "hermes-1440x900"]
    assert calls[4] == ["--output", "VNC-0", "--mode", "hermes-1440x900"]


def test_restore_uses_a_listed_mode_first_and_does_nothing_at_size(monkeypatch, screen_up):
    run, calls = _fake_xrandr(_RESIZED, set_ok=True)
    monkeypatch.setattr(runtime.subprocess, "run", run)
    assert runtime.restore_geometry() is True
    assert calls == [["-q"], ["-s", "1440x900"]]

    run, calls = _fake_xrandr(_AT_SIZE)
    monkeypatch.setattr(runtime.subprocess, "run", run)
    assert runtime.restore_geometry() is True
    assert calls == [["-q"]]


def test_restore_never_raises_and_skips_a_screen_that_is_down(monkeypatch):
    monkeypatch.setattr(runtime, "published_env", lambda: {})
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: pytest.fail("no xrandr without a screen"))
    assert runtime.restore_geometry() is False

    monkeypatch.setattr(runtime, "published_env", lambda: {"DISPLAY": ":20"})
    monkeypatch.setattr(runtime, "geometry", lambda: "1440x900")

    def boom(*a, **k):
        raise FileNotFoundError("xrandr")
    monkeypatch.setattr(runtime.subprocess, "run", boom)
    assert runtime.restore_geometry() is False


def test_restore_targets_the_profile_it_is_given(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(runtime, "published_env", lambda: seen.append(str(runtime.get_hermes_home())) or {})
    runtime.restore_geometry(str(tmp_path))
    assert seen == [str(tmp_path)]
