"""Backend admission during Bot Screen rebind/release; extracted from #114565 for #108914."""

from __future__ import annotations

import contextvars
import json
import threading
from concurrent.futures import Future
from contextlib import contextmanager

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.bot_desktop import browser, lease, runtime as desktop
from tools.computer_use import tool as cu
from tools.computer_use_tool import registry


@contextmanager
def _profile(home):
    token = set_hermes_home_override(str(home))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _spawn(fn, name):
    future = Future()
    context = contextvars.copy_context()

    def run():
        try:
            future.set_result(context.run(fn))
        except BaseException as exc:
            future.set_exception(exc)

    thread = threading.Thread(target=run, name=name, daemon=True)
    thread.start()
    return thread, future


def _call():
    return json.loads(registry.dispatch("computer_use", {"action": "list_apps"}, session_id="shared"))


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    homes = [tmp_path / "a", tmp_path / "b"]
    for home, display in zip(homes, (":37", ":38")):
        (home / "bot-desktop").mkdir(parents=True)
        (home / "bot-desktop" / "env").write_text(f"DISPLAY={display}\n", encoding="utf-8")
        (home / "config.yaml").write_text("computer_use:\n  permission_mode: standard\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(homes[0]))
    # Fake the device boundary only. Profile env publication, spawn identity,
    # config, backend caches, dispatch and persisted human leases stay real.
    monkeypatch.setattr(desktop, "_launcher_pid", lambda: 1)
    monkeypatch.setattr(browser, "executable", lambda: None)
    created = []

    class Backend(cu._NoopBackend):
        def __init__(self, mode):
            super().__init__()
            self.stopped = False
            self.count = 0
            self.before_action = lambda: None
            created.append(self)

        def stop(self):
            self.stopped = True

        def list_apps(self):
            assert not self.stopped, "dispatch used a retired backend"
            self.count += 1
            self.before_action()
            assert not self.stopped, "teardown ran during dispatch"
            return [{"name": str(created.index(self))}]

    cu.reset_backend_for_tests()
    monkeypatch.setattr(cu, "_new_backend", Backend)
    yield homes, created
    cu.reset_backend_for_tests()
    lease._reset_for_tests()


def test_slow_profile_start_leaves_other_profile_lifecycle_available(runtime, monkeypatch):
    homes, _ = runtime
    a_starting, finish_a = threading.Event(), threading.Event()
    created = []
    threads = []

    with _profile(homes[0]):
        owner_a = cu._scoped_sid("shared")

    class Backend(cu._NoopBackend):
        def __init__(self, owner):
            super().__init__()
            self.owner = owner
            self.stopped = False
            created.append(self)

        def start(self):
            if self.owner == owner_a:
                a_starting.set()
                assert finish_a.wait(10), "test did not release profile A startup"

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(cu, "_new_backend", lambda mode: Backend(cu._scoped_sid("shared")))

    with _profile(homes[1]):
        old_b = cu._get_backend("shared")
    with _profile(homes[0]):
        thread, started_a = _spawn(lambda: cu._get_backend("shared"), "slow-start-a")
        threads.append(thread)

    try:
        assert a_starting.wait(10)
        with _profile(homes[1]):
            thread, released_b = _spawn(
                lambda: cu.release_computer_use_session("shared"), "release-b"
            )
            threads.append(thread)
        assert released_b.result(timeout=2) is True
        assert old_b.stopped

        with _profile(homes[1]):
            thread, restarted_b = _spawn(lambda: cu._get_backend("shared"), "restart-b")
            threads.append(thread)
        new_b = restarted_b.result(timeout=2)
        assert new_b is not old_b
        assert getattr(new_b, "owner") != owner_a
        assert not getattr(new_b, "stopped")
    finally:
        finish_a.set()
        for thread in threads:
            thread.join(timeout=10)
            assert not thread.is_alive()

    assert getattr(started_a.result(timeout=1), "owner") == owner_a


def test_owner_start_retry_and_reset_keep_single_flight_bounded(runtime, monkeypatch):
    homes, _ = runtime
    first_starting, close_starting = threading.Event(), threading.Event()
    finish_first, finish_close = threading.Event(), threading.Event()
    follower_waiting = threading.Event()
    created = []
    threads = []
    active_starts = peak_starts = 0
    state_lock = threading.Lock()

    class UserCounts(dict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            if value == 2:
                follower_waiting.set()

    user_counts = UserCounts()
    monkeypatch.setattr(cu, "_backend_owner_lock_users", user_counts)

    class Backend(cu._NoopBackend):
        def __init__(self):
            super().__init__()
            self.index = len(created)
            self.stopped = False
            created.append(self)

        def start(self):
            nonlocal active_starts, peak_starts
            with state_lock:
                active_starts += 1
                peak_starts = max(peak_starts, active_starts)
            try:
                if self.index == 0:
                    first_starting.set()
                    assert finish_first.wait(10), "test did not release failed startup"
                    raise RuntimeError("planned startup failure")
                if self.index == 1:
                    assert created[0].stopped, "retry began before failed-start cleanup"
                if self.index == 2:
                    close_starting.set()
                    assert finish_close.wait(10), "test did not release cancelled startup"
            finally:
                with state_lock:
                    active_starts -= 1

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(cu, "_new_backend", lambda mode: Backend())

    with _profile(homes[0]):
        sid = cu._scoped_sid("shared")
        thread, first = _spawn(lambda: cu._get_backend("shared"), "first-start")
        threads.append(thread)
    try:
        assert first_starting.wait(10)
        with _profile(homes[0]):
            thread, retry = _spawn(lambda: cu._get_backend("shared"), "same-owner-retry")
            threads.append(thread)
        assert follower_waiting.wait(10)
        assert len(created) == 1
        finish_first.set()
        with pytest.raises(RuntimeError, match="planned startup failure"):
            first.result(timeout=10)
        replacement = retry.result(timeout=10)
        assert replacement is created[1]
        assert peak_starts == 1

        with _profile(homes[0]):
            assert cu.release_computer_use_session("shared") is True
            thread, closing_start = _spawn(lambda: cu._get_backend("shared"), "closing-start")
            threads.append(thread)
        assert close_starting.wait(10)
        thread, reset = _spawn(cu.reset_backend_for_tests, "backend-reset")
        threads.append(thread)
        assert reset.result(timeout=2) is None
        finish_close.set()
        with pytest.raises(RuntimeError, match="cancelled by shutdown"):
            closing_start.result(timeout=10)
        assert created[2].stopped
        assert sid not in cu._backends
    finally:
        finish_first.set()
        finish_close.set()
        for thread in threads:
            thread.join(timeout=10)
            assert not thread.is_alive()

    assert not cu._backend_owner_locks
    assert not user_counts


@pytest.mark.parametrize("pause_at", ["after_lookup", "before_acquire"])
@pytest.mark.parametrize("change", ["display", "mode", "release", "queued_display"])
def test_dispatch_rechecks_admission(runtime, monkeypatch, pause_at, change):
    homes, created = runtime
    paused, resume = threading.Event(), threading.Event()
    real_get = cu._get_backend
    old = real_get("shared")
    old_lock = cu._backend_call_locks["shared"]

    def pause():
        if threading.current_thread().name == "waiting-call" and not paused.is_set():
            paused.set()
            assert resume.wait(10), "test did not release the waiting call"

    def get(session_id=""):
        backend = real_get(session_id)
        if pause_at == "after_lookup":
            pause()
        return backend

    class PausingLock:
        def __enter__(self):
            pause()
            old_lock.acquire()
            return self

        def __exit__(self, *_):
            old_lock.release()

    monkeypatch.setattr(cu, "_get_backend", get)
    if pause_at == "before_acquire":
        cu._backend_call_locks["shared"] = PausingLock()
    worker, result = _spawn(_call, "waiting-call")
    try:
        assert paused.wait(10)
        if change == "release":
            assert cu.release_computer_use_session("shared")
        else:
            if change == "mode":
                (homes[0] / "config.yaml").write_text("computer_use:\n  permission_mode: bounded\n", encoding="utf-8")
            else:
                (homes[0] / "bot-desktop" / "env").write_text("DISPLAY=:39\n", encoding="utf-8")
            if change != "queued_display":
                assert real_get("shared") is not old
        if change != "queued_display":
            assert old.stopped
        resume.set()
        value = result.result(timeout=10)
        assert "error" not in value, value
        assert old.count == 0
        assert sum(backend.count for backend in created) == 1
        assert value["apps"] == [{"name": str(created.index(real_get("shared")))}]
    finally:
        resume.set()
        worker.join(timeout=10)
        assert not worker.is_alive()


@pytest.mark.parametrize("fail_action", [False, True])
def test_release_waits_without_blocking_another_profile_or_replaying(runtime, monkeypatch, fail_action):
    homes, created = runtime
    action_entered, finish_action, stop_attempted = (threading.Event() for _ in range(3))
    threads = []
    with _profile(homes[0]):
        old = cu._get_backend("shared")

        def action():
            action_entered.set()
            assert finish_action.wait(10)
            if fail_action:
                raise RuntimeError("action outcome is unknown; do not replay")

        old.before_action = action
        real_stop = cu._stop_backend

        def stop(backend, lock, on_error):
            if backend is old:
                stop_attempted.set()
            return real_stop(backend, lock, on_error)

        monkeypatch.setattr(cu, "_stop_backend", stop)
        try:
            thread, result = _spawn(_call, "admitted-call")
            threads.append(thread)
            assert action_entered.wait(10)
            thread, released = _spawn(lambda: cu.release_computer_use_session("shared"), "release")
            threads.append(thread)
            assert stop_attempted.wait(10)
            assert not old.stopped and not released.done()
            with _profile(homes[1]):
                thread, other = _spawn(_call, "other-profile")
                threads.append(thread)
                assert "error" not in other.result(timeout=10)
                assert not old.stopped
            finish_action.set()
            value = result.result(timeout=10)
            assert ("error" in value) is fail_action
            assert released.result(timeout=10) is True
            assert old.stopped and old.count == 1
            # A -> B -> A: the released profile gets its own fresh backend.
            assert "error" not in _call()
            assert cu._get_backend("shared") is not old
            with _profile(homes[1]):
                assert not cu._get_backend("shared").stopped
        finally:
            finish_action.set()
            for thread in threads:
                thread.join(timeout=10)
                assert not thread.is_alive()


@pytest.mark.parametrize("hand_back", [False, True])
def test_backend_retry_preserves_the_original_human_lease_epoch(runtime, monkeypatch, hand_back):
    _, created = runtime
    paused, resume = threading.Event(), threading.Event()
    real_get = cu._get_backend
    old = real_get("shared")

    def get(session_id=""):
        backend = real_get(session_id)
        if threading.current_thread().name == "waiting-call" and not paused.is_set():
            paused.set()
            assert resume.wait(10)
        return backend

    monkeypatch.setattr(cu, "_get_backend", get)
    worker, result = _spawn(_call, "waiting-call")
    try:
        assert paused.wait(10)
        assert cu.release_computer_use_session("shared")
        assert old.stopped
        lease.acquire("human")
        if hand_back:
            lease.release("human")
        resume.set()
        assert result.result(timeout=10)["code"] == "human_has_control"
        assert all(backend.count == 0 for backend in created)
    finally:
        resume.set()
        worker.join(timeout=10)
        assert not worker.is_alive()
