"""A pool-routed RPC handler that raises answers -32000 and leaves its traceback in the log, as the
inline WS path and the stdio path already do."""

import logging
import queue
from concurrent.futures import ThreadPoolExecutor


class _Transport:
    def __init__(self):
        self.frames = queue.Queue()

    def write(self, frame):
        self.frames.put(frame)
        return True


def test_pool_handler_failure_is_logged_with_traceback(monkeypatch, caplog):
    from hermes_agent.tui_gateway import server

    def handler(rid, params):
        raise RuntimeError("store unavailable")

    monkeypatch.setitem(server._methods, "test.pool_failure", handler)
    monkeypatch.setattr(server, "_LONG_HANDLERS", {"test.pool_failure"})
    transport = _Transport()
    with ThreadPoolExecutor(max_workers=1) as pool, caplog.at_level(logging.ERROR, logger="hermes_agent.tui_gateway"):
        monkeypatch.setattr(server, "_pool", pool)
        assert server.dispatch({"id": "r1", "method": "test.pool_failure"}, transport) is None
        reply = transport.frames.get(timeout=10)

    assert reply["error"] == {"code": -32000, "message": "handler error: store unavailable"}
    record = next(r for r in caplog.records if "test.pool_failure" in r.getMessage())
    assert record.exc_info is not None and "store unavailable" in str(record.exc_info[1])
