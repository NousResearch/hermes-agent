import threading
import types

from tui_gateway import server
from tui_gateway.transport import Transport


def test_context_breakdown_dispatch_releases_reader_and_preserves_response(monkeypatch):
    """A slow breakdown must not hold the RPC reader thread."""
    sid = "context-breakdown-pool"
    expected = {
        "categories": [{"name": "Messages", "tokens": 123}],
        "context_max": 4096,
        "context_percent": 3.0,
        "context_used": 123,
        "estimated_total": 123,
        "context_estimated": True,
        "context_source": "local_estimate",
        "model": "fixture",
    }
    server._sessions[sid] = {
        "agent": types.SimpleNamespace(),
        "session_key": "context-breakdown-session",
        "history": [{"role": "user", "content": "hello"}],
        "history_lock": threading.Lock(),
    }

    compute_started = threading.Event()
    release_compute = threading.Event()
    reader_returned = threading.Event()
    response_written = threading.Event()
    writes = []
    dispatch_result = {}

    def slow_breakdown(agent, history):
        assert agent is server._sessions[sid]["agent"]
        assert history == [{"role": "user", "content": "hello"}]
        compute_started.set()
        assert release_compute.wait(timeout=5.0)
        return expected

    class RecordingTransport(Transport):
        def write(self, obj: dict) -> bool:
            writes.append(obj)
            response_written.set()
            return True

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "agent.context_breakdown.compute_session_context_breakdown", slow_breakdown
    )
    transport = RecordingTransport()
    request = {
        "id": "breakdown",
        "method": "session.context_breakdown",
        "params": {"session_id": sid},
    }

    def read_request():
        dispatch_result["value"] = server.dispatch(request, transport)
        reader_returned.set()

    reader = threading.Thread(target=read_request)
    reader.start()
    try:
        assert compute_started.wait(timeout=2.0)
        assert reader_returned.wait(timeout=1.0), "breakdown blocked the RPC reader"
        assert dispatch_result["value"] is None
        assert response_written.is_set() is False

        ping = server.dispatch({"id": "ping", "method": "ping", "params": {}}, transport)
        assert ping == {"jsonrpc": "2.0", "id": "ping", "result": {"pong": True}}
    finally:
        release_compute.set()
        reader.join(timeout=5.0)
        server._sessions.pop(sid, None)

    assert response_written.wait(timeout=1.0)
    assert writes == [{"jsonrpc": "2.0", "id": "breakdown", "result": expected}]
