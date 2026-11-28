"""A computer_use backend nobody uses is released after its profile's Bot Desktop idle window.

A cached backend keeps its cua-driver alive for the whole session; on a screen inside a pausable terminal
sandbox that driver is an exec stream which keeps the sandbox awake (and billed) until the session ends."""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.bot_desktop import browser, runtime as desktop
from tools.computer_use import tool as cu
from tools.computer_use_tool import registry
from tests.bot_desktop_lease import reset_lease
from tests.computer_use_fakes import NoopBackend, reset_backend


@contextmanager
def _profile(home):
    token = set_hermes_home_override(str(home))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _call(sid="s"):
    return json.loads(registry.dispatch("computer_use", {"action": "list_apps"}, session_id=sid))


def _home(path, idle_minutes):
    (path / "bot-desktop").mkdir(parents=True)
    (path / "bot-desktop" / "env").write_text("DISPLAY=:41\n", encoding="utf-8")
    (path / "config.yaml").write_text(
        f"computer_use:\n  permission_mode: standard\nbot_desktop:\n  idle_stop_minutes: {idle_minutes}\n",
        encoding="utf-8")
    return path


@pytest.fixture
def backends(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(_home(tmp_path / "main", 1)))
    monkeypatch.setattr(desktop, "_launcher_pid", lambda: 1)  # the device boundary only
    monkeypatch.setattr(browser, "executable", lambda: None)
    created = []

    class Backend(NoopBackend):
        def __init__(self, mode):
            super().__init__()
            self.stopped = False
            self.stopped_in = None
            self.during_call = lambda: None
            created.append(self)

        def stop(self):
            from hermes_constants import get_hermes_home_override
            self.stopped, self.stopped_in = True, get_hermes_home_override()

        def list_apps(self):
            self.during_call()
            assert not self.stopped, "the backend was stopped during its own call"
            return []

    reset_backend()
    monkeypatch.setattr(cu, "_new_backend", Backend)
    yield created
    reset_backend()
    reset_lease()


def test_an_unused_backend_is_released_and_the_next_call_starts_a_new_one(backends):
    assert "error" not in _call()
    first = backends[0]
    assert cu.release_idle_backends(now=time.monotonic() + 59) == []
    assert not first.stopped
    assert cu.release_idle_backends(now=time.monotonic() + 61) == ["s"]
    assert first.stopped
    assert "error" not in _call()
    assert len(backends) == 2 and not backends[1].stopped


def test_each_call_restarts_the_idle_window(backends, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(cu, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=time.sleep))
    _call()
    clock[0] += 50
    _call()  # used again: measured from here
    assert cu.release_idle_backends(now=1000.0 + 61) == []
    assert not backends[0].stopped
    assert cu.release_idle_backends(now=1050.0 + 61) == ["s"]


def test_each_profile_keeps_its_own_idle_window(backends, tmp_path):
    """One gateway serves several profiles: the sweep runs outside any profile, so it must apply each
    backend's own profile's window, not the launch profile's."""
    off, short = _home(tmp_path / "off", 0), _home(tmp_path / "short", 1)
    with _profile(off):
        _call()
    with _profile(short):
        _call()
    released = cu.release_idle_backends(now=time.monotonic() + 3600)
    assert len(released) == 1 and released[0].startswith("s@")
    kept, gone = backends
    assert not kept.stopped and gone.stopped


def test_a_call_running_past_the_window_keeps_its_backend(backends):
    """A long action is use, not idleness: retiring its backend mid-call would let the next call start a
    second driver for the same session while the first action still runs."""
    _call()
    first = backends[0]
    inside, finish = threading.Event(), threading.Event()
    first.during_call = lambda: (inside.set(), finish.wait(10))
    worker = threading.Thread(target=_call, daemon=True)
    worker.start()
    try:
        assert inside.wait(10)
        assert cu.release_idle_backends(now=time.monotonic() + 3600) == []
        assert not first.stopped
    finally:
        finish.set()
        worker.join(10)
    assert len(backends) == 1


def test_teardown_runs_in_the_owning_profile(backends, tmp_path):
    """The sweep runs outside any profile; the driver's stop reads profile state, so it must see its own."""
    other = _home(tmp_path / "other", 1)
    with _profile(other):
        _call()
    assert len(cu.release_idle_backends(now=time.monotonic() + 3600)) == 1
    assert backends[0].stopped_in == str(other)


def test_a_cached_backend_does_not_keep_the_installing_call_alive(backends):
    """Backends outlive agent eviction; whatever the installing call had bound (progress sinks that hold
    a parent agent) must not live as long as they do."""
    import contextvars
    import gc
    import weakref

    class Sink:
        pass

    var = contextvars.ContextVar("test_call_binding")
    sink = Sink()
    ref = weakref.ref(sink)
    token = var.set(sink)
    try:
        _call()
    finally:
        var.reset(token)
    del sink
    gc.collect()
    assert ref() is None


def test_a_profile_that_cannot_be_bound_still_has_its_backend_stopped(backends, tmp_path, monkeypatch):
    """The backend is out of the cache before teardown: if binding its profile fails, nothing else would
    ever stop it, and its driver (a remote exec stream) would keep the sandbox awake."""
    from contextlib import contextmanager
    from agent import runtime_scope

    @contextmanager
    def unavailable(home, **kw):
        raise RuntimeError("secret source unavailable")
        yield

    with _profile(_home(tmp_path / "other", 1)):
        _call()
    monkeypatch.setattr(runtime_scope, "bind_profile_scope", unavailable)
    cu.release_idle_backends(now=time.monotonic() + 3600)
    assert backends[0].stopped
