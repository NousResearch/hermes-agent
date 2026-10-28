"""Unobserved presentation work sleeps; reconnect and stdio still receive changes.

The clock/condition are driven explicitly so no assertion depends on a quiet machine.
The real watcher, transport registry, signatures and broadcast path run on a worker thread.
"""

import queue
import threading
import time
from types import SimpleNamespace

import pytest

from agent import memory_provider
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tui_gateway import server


class _StopWatcher(BaseException):
    pass


@pytest.fixture
def watcher(tmp_path, monkeypatch):
    waits = queue.Queue()
    stopped = threading.Event()
    threads = []
    clock = [10.0]

    class ControlledCondition(threading.Condition):
        def wait(self, timeout=None):
            waits.put(timeout)
            result = super().wait()  # the test advances the connected-client deadline
            if stopped.is_set():
                raise _StopWatcher
            return result

    condition = ControlledCondition()

    def make_thread(target, **kwargs):
        def run():
            try:
                target()
            except _StopWatcher:
                pass
        thread = threading.Thread(target=run, **kwargs)
        threads.append(thread)
        return thread

    def sleep(seconds):  # exercise the old loop too, for the red-on-base proof
        with condition:
            condition.wait(seconds)

    monkeypatch.setattr(memory_provider, "spawn_context_thread",
                        lambda target, **kw: make_thread(memory_provider.ctx_bound(target), **kw))
    monkeypatch.setattr(server, "threading", SimpleNamespace(Thread=make_thread))
    monkeypatch.setattr(server, "time", SimpleNamespace(monotonic=lambda: clock[0], time=time.time, sleep=sleep))
    monkeypatch.setattr(server, "_hermes_home", str(tmp_path))
    monkeypatch.setattr(server, "_cfg_cache", None)
    monkeypatch.setattr(server, "_live_transports_lock", condition)
    monkeypatch.setattr(server, "_live_transports", set())
    monkeypatch.setattr(server, "_stdio_is_rpc_channel", False)
    monkeypatch.setattr(server, "_skin_watcher_started", False)
    monkeypatch.setattr(server, "_last_skin_sig", None)
    monkeypatch.setattr(server, "_skin_sig_by_home", {})
    monkeypatch.setattr(server, "_change_sigs", {})
    monkeypatch.setattr(server, "_change_checked_at", {})
    monkeypatch.setattr(server, "_change_broadcast_at", {})
    monkeypatch.setattr(server, "_CHANGE_WATCHES", {
        "cron.changed": (1.0, lambda: server._home_mtime_ns("cron", "jobs.json"), dict),
    })
    (tmp_path / "config.yaml").write_text("display: {}\n")
    (tmp_path / "cron").mkdir()

    def at_wait(expected):
        assert waits.get(timeout=5) == expected
        with condition:  # the worker has released the registry lock and is asleep
            pass

    def tick():
        with condition:
            clock[0] += 10
            condition.notify_all()

    try:
        yield SimpleNamespace(home=tmp_path, at_wait=at_wait, tick=tick, threads=threads)
    finally:
        with condition:
            stopped.set()
            condition.notify_all()
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()


def test_last_client_parks_scans_and_reconnect_catches_external_changes(watcher, monkeypatch):
    # A detached running/queued turn is not permission to keep display scans alive,
    # nor is disconnect permission to interrupt that work.
    work = {"detached": {"running": True, "queued_prompts": ["next"], "_auto_continue_scheduled": True}}
    monkeypatch.setattr(server, "_sessions", work)
    server._ensure_skin_watcher()
    server._ensure_skin_watcher()
    watcher.at_wait(None)
    assert len(watcher.threads) == 1
    assert not server._change_checked_at

    frames = []
    class Peer:
        def write(self, frame):
            frames.append(frame)

    first, second = Peer(), Peer()
    server.register_live_transport(first)
    watcher.at_wait(0.5)
    server.register_live_transport(first)  # duplicate registration is not another client
    watcher.at_wait(0.5)
    server.register_live_transport(second)
    watcher.at_wait(0.5)
    server.unregister_live_transport(first)
    watcher.at_wait(0.5)  # the other window still needs updates
    server.unregister_live_transport(first)  # duplicate disconnect must not park it
    watcher.at_wait(0.5)
    server.unregister_live_transport(second)
    watcher.at_wait(None)

    checked = dict(server._change_checked_at)
    (watcher.home / "cron" / "jobs.json").write_text("[]")
    watcher.tick()  # even a spurious wake cannot scan with no consumer
    watcher.at_wait(None)
    assert server._change_checked_at == checked
    assert not frames

    server.register_live_transport(second)
    watcher.at_wait(0.5)
    assert [frame["params"]["type"] for frame in frames] == ["cron.changed"]
    assert server._sessions is work
    assert work["detached"]["running"] and work["detached"]["queued_prompts"] == ["next"]


def test_stdio_receives_changes_without_websocket_clients(watcher, monkeypatch):
    frames = []
    monkeypatch.setattr(server, "_stdio_is_rpc_channel", True)
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload: frames.append((event, sid, payload)))
    server._broadcast_watched_changes(now=0.0)
    # The first ensure can originate in a request's profile scope; the one process-wide
    # watcher must still observe the launch home, with sibling skins routed separately.
    token = set_hermes_home_override(str(watcher.home / "profiles" / "other"))
    try:
        server._ensure_skin_watcher()
    finally:
        reset_hermes_home_override(token)
    watcher.at_wait(0.5)
    assert not server._live_transports

    (watcher.home / "cron" / "jobs.json").write_text("[]")
    watcher.tick()
    watcher.at_wait(0.5)
    assert frames == [("cron.changed", "", {})]
