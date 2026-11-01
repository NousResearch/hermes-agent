import threading

# FanoutTransport._drain imports tui_gateway.ws lazily, which loads tui_gateway.server. Load it here so the
# drain thread's first write doesn't pay a cold server import inside the 3s waits below.
import tui_gateway.ws  # noqa: F401
from tui_gateway.transport import FanoutTransport


def test_global_coalescing_preserves_distinct_reclaimed_sessions():
    entered = threading.Event()
    release = threading.Event()
    done = threading.Event()
    received = []

    class Peer:
        def close(self):
            pass

        def write(self, obj: dict) -> bool:
            frame = obj
            if not received:
                entered.set()
                assert release.wait(3)
            received.append(frame)
            if len(received) == 3:
                done.set()
            return True

    fanout = FanoutTransport(Peer(), coalesce_events=True)
    try:
        fanout.write({"method": "event", "params": {"type": "skin.changed"}})
        assert entered.wait(3)
        for sid in ("a", "b"):
            fanout.write({"method": "event", "params": {"type": "session.reclaimed", "payload": {"session_id": sid}}})
        release.set()
        assert done.wait(3)
        assert [frame["params"]["payload"]["session_id"] for frame in received[1:]] == ["a", "b"]
    finally:
        release.set()
        fanout.close()


def test_global_full_state_invalidations_coalesce_before_backlog_limit():
    entered = threading.Event()
    release = threading.Event()
    done = threading.Event()
    received = []

    class Peer:
        def close(self):
            pass

        def write(self, obj: dict) -> bool:
            if not received:
                entered.set()
                assert release.wait(3)
            received.append(obj)
            if len(received) == 2:
                done.set()
            return True

    fanout = FanoutTransport(Peer(), coalesce_events=True)
    try:
        fanout.write({"method": "event", "params": {"type": "skin.changed"}})
        assert entered.wait(3)
        for tick in range(300):
            assert fanout.write({"method": "event", "params": {"type": "sessions.changed", "payload": {"tick": tick}}})
        release.set()
        assert done.wait(3)
        assert received[1]["params"]["payload"] == {"tick": 299}
    finally:
        release.set()
        fanout.close()
