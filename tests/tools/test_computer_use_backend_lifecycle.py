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
    monkeypatch.setattr(cu, "_new_backend", lambda sid, mode, provider: Backend(mode))
    yield homes, created
    cu.reset_backend_for_tests()
    lease._reset_for_tests()


@pytest.mark.parametrize("pause_at", ["after_lookup", "before_acquire"])
@pytest.mark.parametrize("change", ["display", "mode", "release", "queued_display"])
def test_dispatch_rechecks_admission(runtime, monkeypatch, pause_at, change):
    homes, created = runtime
    paused, resume = threading.Event(), threading.Event()
    real_get = cu._get_backend
    old = real_get("shared")
    owner = cu._backend_owner_key("shared")
    old_lock = cu._backend_call_locks[owner]

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
        cu._backend_call_locks[owner] = PausingLock()
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
        assert old.count == 0
        if change == "release":
            # Explicit release revokes this acquisition generation. A new call,
            # not the queued one, may acquire authority again.
            assert "released" in value.get("error", ""), value
            assert sum(backend.count for backend in created) == 0
            assert "error" not in _call()
            return
        assert "error" not in value, value
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
    homes, created = runtime
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
        # Rebind retries admission within the same generation; explicit release
        # is instead covered by the revocation cases above.
        (homes[0] / "bot-desktop" / "env").write_text("DISPLAY=:39\n", encoding="utf-8")
        assert real_get("shared") is not old
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
