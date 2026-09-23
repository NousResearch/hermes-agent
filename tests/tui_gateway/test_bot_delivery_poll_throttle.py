"""Idle notification poller must not probe bot-delivery on every 0.5s queue tick (#108005)."""

from __future__ import annotations

import threading
import time
import queue

import tui_gateway.server as server
from tools.process_registry import process_registry


def test_idle_poller_throttles_bot_live_delivery_probe(monkeypatch):
    """Drive the bound poller: ~1.2s of 0.5s idle ticks must not call the probe every iteration.

    Pre-fix every loop iteration hits ``_poll_bot_live_delivery_once`` (~2/s → 3 calls).
    Post-fix the first iteration fires promptly, then the 5.0s cadence holds (exactly 1).
    """
    calls: list[float] = []

    def _probe(_sid, _session):
        calls.append(time.monotonic())
        return False

    monkeypatch.setattr(server, "_poll_bot_live_delivery_once", _probe)
    monkeypatch.setattr(server, "_notif_poll_kanban", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_maybe_fire_tui_loop_tick", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_maybe_fire_tui_heartbeat_tick", lambda *_a, **_k: None)
    monkeypatch.setattr(process_registry, "completion_queue", queue.Queue())

    session = {
        "session_key": "sid-bot-delivery-throttle",
        "history_lock": threading.Lock(),
        "_finalized": False,
    }
    stop = threading.Event()
    thread = threading.Thread(
        target=server._notification_poller_loop,
        args=(stop, "sid-bot-delivery-throttle", session),
        daemon=True,
        name="tui-notif-poller-bot-delivery-throttle",
    )
    thread.start()
    try:
        time.sleep(1.2)
    finally:
        stop.set()
        thread.join(timeout=5)

    assert thread.is_alive() is False
    assert len(calls) == 1, f"idle bot-delivery probe ran {len(calls)} times in ~1.2s; expected 1, not every 0.5s tick"
