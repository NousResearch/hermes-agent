"""A blocked pruned pipe close must not starve other registry users.

Regression for #130889, reported by pwwang. A Windows pipe's close()
can wait for a live reader. The concurrent RPC must still obtain the
registry lock while the spawn caller is stuck releasing that pipe.
"""

import threading
import time

from tools.process_registry import FINISHED_TTL_SECONDS, ProcessRegistry, ProcessSession


def test_blocked_prune_close_does_not_starve_registry(monkeypatch):
    registry = ProcessRegistry()
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)

    closing = threading.Event()
    allow_close = threading.Event()
    rpc_done = threading.Event()
    errors = []

    class BlockingStream:
        def close(self):
            closing.set()
            if not allow_close.wait(timeout=10):
                raise AssertionError("test never released blocked close")

    class FakeProcess:
        stdout = BlockingStream()
        stderr = None
        stdin = None

    stale = ProcessSession(
        id="proc_expired",
        command="echo expired",
        task_id="test",
        started_at=time.time() - FINISHED_TTL_SECONDS - 60,
        exited=True,
        exit_code=0,
    )
    stale.process = FakeProcess()
    registry._finished[stale.id] = stale

    newcomer = ProcessSession(
        id="proc_current",
        command="echo current",
        task_id="test",
        started_at=time.time(),
    )

    def spawn():
        try:
            registry._track_started(newcomer, lambda session: None, "test-reader")
        except BaseException as exc:
            errors.append(exc)

    def query():
        try:
            registry.list_sessions()
        except BaseException as exc:
            errors.append(exc)
        finally:
            rpc_done.set()

    spawner = threading.Thread(target=spawn, daemon=True)
    spawner.start()
    querier = None
    try:
        assert closing.wait(timeout=2), "prune did not reach stream.close()"
        querier = threading.Thread(target=query, daemon=True)
        querier.start()
        assert rpc_done.wait(timeout=2), (
            "list_sessions blocked behind the stream.close() in the spawn thread"
        )
    finally:
        allow_close.set()
        spawner.join(timeout=2)
        if querier is not None:
            querier.join(timeout=2)

    assert not spawner.is_alive()
    assert querier is not None and not querier.is_alive()
    assert errors == []
