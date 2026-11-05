"""The session's ``_run_thread`` handle is read from other threads by ``session.interrupt``, session close and the
exit stop, which call ``is_alive()`` and ``join()`` on it. It must refer only to a started thread, including while
``prompt.submit``'s dispatch thread hands the turn to its ``prompt-turn-*`` worker."""

import threading
from unittest.mock import MagicMock

import pytest

from tui_gateway import server


class _HookedStartThread(threading.Thread):
    """A turn thread whose ``start()`` calls ``before_start`` first and ``after_start`` after the thread has
    started."""

    def __init__(self, *args, before_start=None, after_start=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._before_start = before_start
        self._after_start = after_start

    def start(self):
        if self._before_start is not None:
            self._before_start()
        super().start()
        if self._after_start is not None:
            self._after_start()


def _hook_turn_threads(monkeypatch, **hooks_by_name):
    def spawn(target, *, name, **_kwargs):
        return _HookedStartThread(target=target, name=name, daemon=True, **hooks_by_name.get(name, {}))

    monkeypatch.setattr("agent.memory_provider.spawn_context_thread", spawn)


@pytest.fixture
def registered_session():
    sid = "turn-thread-handle"
    session = {
        "history_lock": threading.Lock(),
        "running": True,
        "queued_prompt": None,
        "session_key": "turn-thread-handle-key",
        "agent": MagicMock(),
        "_run_thread": None,
    }
    with server._sessions_lock:
        server._sessions[sid] = session
    try:
        yield session
    finally:
        with server._sessions_lock:
            server._sessions.pop(sid, None)


def test_exit_stop_while_the_turn_worker_starts_leaves_the_turn_to_end_itself(monkeypatch, registered_session):
    monkeypatch.setattr("tools.environments.base.kill_live_foreground_processes", lambda **_kwargs: 0)
    # prompt.submit published its dispatch thread, which now starts the worker. This test thread plays the
    # dispatch thread.
    registered_session["_run_thread"] = threading.current_thread()
    outcomes = []

    def exit_stop():
        try:
            server._stop_turns_before_exit(budget_s=0)
        except RuntimeError as exc:
            outcomes.append(str(exc))
        else:
            outcomes.append(None)

    def stop_from_another_thread():
        stopper = threading.Thread(target=exit_stop, name="exit-stop")
        stopper.start()
        stopper.join(10)

    _hook_turn_threads(monkeypatch, **{"prompt-turn-sid": {"before_start": stop_from_another_thread}})
    worker = server._start_session_work(lambda: None, name="prompt-turn-sid", session=registered_session)
    worker.join(10)

    # A live turn clears `running` itself when it ends; only a stuck flag is cleared by the stop.
    assert (outcomes, registered_session["running"], registered_session["_run_thread"]) == ([None], True, worker)


@pytest.mark.parametrize("worker_publishes_first", [True, False])
def test_the_worker_stays_published_whichever_thread_publishes_first(
    monkeypatch, registered_session, worker_publishes_first
):
    worker_published, worker_starting, dispatch_published = (threading.Event() for _ in range(3))
    workers = []

    def dispatch():
        workers.append(server._start_session_work(lambda: None, name="worker", session=registered_session))
        worker_published.set()

    if worker_publishes_first:
        hooks = {"dispatch": {"after_start": lambda: worker_published.wait(10)}}
    else:
        # The dispatch thread reads the session's handle before prompt.submit, played by this test thread,
        # publishes the dispatch thread.
        hooks = {
            "dispatch": {"after_start": lambda: worker_starting.wait(10)},
            "worker": {"before_start": lambda: (worker_starting.set(), dispatch_published.wait(10))},
        }
    _hook_turn_threads(monkeypatch, **hooks)
    dispatcher = server._start_session_work(dispatch, name="dispatch", session=registered_session)
    dispatch_published.set()
    dispatcher.join(10)
    workers[0].join(10)

    assert registered_session["_run_thread"] is workers[0]
