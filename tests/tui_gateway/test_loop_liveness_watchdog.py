"""Tests for tui_gateway.loop_liveness_watchdog — the serve-loop wedge guard.

Unit tests drive ``_check_once(now=...)`` with an explicit clock (deterministic,
no sleeps); the integration tests run a real loop in a thread and spin it with a
monotonic busy-wait to prove the OS-thread watchdog fires while the loop is wedged.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any

import pytest

from tests.waiting import wait_until
from tui_gateway.loop_liveness_watchdog import (
    LoopLivenessHandle,
    arm_loop_liveness_watchdog,
    disarm_loop_liveness_watchdog,
    _reset_for_tests,
)


@pytest.fixture(autouse=True)
def _isolate_singleton():
    _reset_for_tests()
    yield
    _reset_for_tests()


def _make_handle(tmp_path, *, clients=False, **kw) -> Any:
    defaults: dict = dict(
        stall_threshold_s=30.0, check_interval_s=5.0, exit_strikes=3,
        client_grace_s=60.0, min_uptime_s=120.0, tick_interval_s=2.0,
        redump_interval_s=60.0, exit_code=0,
    )
    defaults.update(kw)
    h = LoopLivenessHandle(None, lambda: clients, tmp_path / "watchdog.log", **defaults)
    h.armed_at = 0.0
    h.last_tick = 0.0
    h.exits = []
    h._exit = lambda code: h.exits.append(code)
    h._dump_stacks = lambda fh: fh.write("FAKE-STACKS\n")
    return h


def _records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.startswith("{")]


def test_healthy_loop_never_dumps_or_exits(tmp_path):
    h = _make_handle(tmp_path)
    for now in (10.0, 20.0, 29.0):
        h.last_tick = now - 1.0
        h._check_once(now=now)
    assert h.exits == []
    assert _records(h._dump_path) == []
    assert h._stall_strikes == 0


def test_stall_with_live_client_dumps_but_never_exits(tmp_path):
    h = _make_handle(tmp_path, clients=True)
    for now in (200.0, 260.0, 320.0, 400.0, 500.0):
        h._check_once(now=now)
    assert h.exits == [], "a backend serving a live window must NEVER self-exit"
    records = _records(h._dump_path)
    assert records and all(r["action"] == "dump_only" for r in records)
    assert all(r["has_clients"] is True for r in records)


def test_young_backend_never_exits(tmp_path):
    h = _make_handle(tmp_path, min_uptime_s=1000.0)
    for now in (200.0, 260.0, 320.0):
        h._check_once(now=now)
    assert h.exits == []


def test_client_grace_not_met_never_exits(tmp_path):
    h = _make_handle(tmp_path, client_grace_s=1000.0)
    for now in (200.0, 260.0, 320.0):
        h._check_once(now=now)
    assert h.exits == []


def test_strike_gate_holds_until_required_strikes(tmp_path):
    h = _make_handle(tmp_path, exit_strikes=4)
    for now in (200.0, 260.0, 320.0):
        h._check_once(now=now)
    assert h.exits == [], "3 strikes must not fire when 4 are required"
    h._check_once(now=380.0)
    assert h.exits == [0]


def test_orphaned_wedged_backend_fires_and_dumps(tmp_path):
    h = _make_handle(tmp_path)
    for now in (200.0, 260.0, 320.0):
        h._check_once(now=now)
    assert h.exits == [0]
    records = _records(h._dump_path)
    fired = [r for r in records if r["tag"] == "loop_watchdog.fired"]
    assert len(fired) == 1
    assert fired[0]["action"] == "self_exit"
    assert fired[0]["exit_code"] == 0
    assert fired[0]["gap_s"] > 30.0
    text = h._dump_path.read_text()
    assert "FAKE-STACKS" in text


def test_client_probe_failure_is_fail_safe(tmp_path):
    h = _make_handle(tmp_path)
    def _boom():
        raise RuntimeError("registry unreadable")
    h._has_clients_fn = _boom
    for now in (200.0, 260.0, 320.0, 400.0):
        h._check_once(now=now)
    assert h.exits == [], "an unreadable client registry must never license an exit"
    assert all(r["action"] == "dump_only" for r in _records(h._dump_path))


def test_dump_rate_limit_and_episode_reset(tmp_path):
    h = _make_handle(tmp_path, redump_interval_s=1000.0)
    h._check_once(now=200.0)          # first dump of the episode
    h._check_once(now=210.0)          # within redump interval: suppressed
    assert len(_records(h._dump_path)) == 1
    h.last_tick = 220.0
    h._check_once(now=220.0)          # loop breathing: episode cleared
    assert h._stall_strikes == 0 and not h._stall_episode_dumped
    h._check_once(now=260.0)          # new episode dumps despite the rate limit
    assert len(_records(h._dump_path)) == 2


def test_disarmed_handle_stops_evaluating(tmp_path):
    h = _make_handle(tmp_path)
    h.disarm()
    for now in (200.0, 260.0, 320.0):
        h._check_once(now=now)
    assert h.exits == []
    assert _records(h._dump_path) == []


def test_real_faulthandler_dump_contains_thread_stacks(tmp_path):
    h = _make_handle(tmp_path)
    del h._dump_stacks  # restore the real faulthandler seam
    h._write_dump({"tag": "loop_stall.detected", "pid": 1})
    text = h._dump_path.read_text()
    assert "loop_stall.detected" in text
    assert "thread" in text.lower(), "faulthandler must write real thread stacks"


def test_exit_escort_bounds_wedged_forensics(tmp_path):
    h = _make_handle(tmp_path)
    dump_started = threading.Event()
    release = threading.Event()
    def _wedged_dump(fh):
        dump_started.set()
        release.wait(timeout=10)  # forensics hang (hung disk / held lock)
    h._dump_stacks = _wedged_dump
    slept = []
    h._sleep = lambda seconds: slept.append(seconds)  # escort's bound elapses immediately
    fire = threading.Thread(target=h._fire, args=(320.0, 300.0, 320.0), daemon=True)
    fire.start()
    assert dump_started.wait(timeout=5)
    # The escort must hard-exit even though _fire is still stuck in the dump.
    wait_until(lambda: h.exits, what="escort hard-exit", timeout=5)
    assert h.exits == [0]
    assert slept, "escort must wait its bound before exiting"
    release.set()
    fire.join(timeout=5)


class _StubTimerHandle:
    def cancel(self):
        pass


class _StubLoop:
    def call_soon_threadsafe(self, fn, *args):
        fn(*args)
    def call_later(self, delay, fn, *args):
        return _StubTimerHandle()


def test_arm_idempotent_and_config_disable(monkeypatch, tmp_path):
    import tui_gateway.loop_liveness_watchdog as mod
    monkeypatch.setattr(mod, "_resolve_config", dict)
    monkeypatch.setattr(mod, "get_dump_path", lambda home=None: tmp_path / "w.log")
    loop = _StubLoop()
    h1 = arm_loop_liveness_watchdog(loop, lambda: True)
    assert h1 is not None and h1.is_alive()
    h2 = arm_loop_liveness_watchdog(loop, lambda: True)
    assert h2 is h1, "arm must be idempotent"
    disarm_loop_liveness_watchdog()
    wait_until(lambda: not h1.is_alive(), what="watchdog thread stopped", timeout=5)
    monkeypatch.setattr(mod, "_resolve_config", lambda: {"enabled": False})
    assert arm_loop_liveness_watchdog(loop, lambda: True) is None


# ── real-loop integration: the watchdog must work against a live asyncio loop ──


def _loop_thread(loop):
    ready = threading.Event()
    def _run():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()
    thread = threading.Thread(target=_run, daemon=True, name="test-serve-loop")
    thread.start()
    assert ready.wait(timeout=5)
    return thread


def _stop_loop(loop, thread):
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


def test_tick_advances_while_loop_runs(tmp_path):
    loop = asyncio.new_event_loop()
    thread = _loop_thread(loop)
    try:
        h = LoopLivenessHandle(loop, lambda: True, tmp_path / "w.log",
                               tick_interval_s=0.05, check_interval_s=0.05,
                               stall_threshold_s=30.0)
        assert h._start()
        baseline = h.last_tick
        wait_until(lambda: h.last_tick > baseline, what="loop tick stamped", timeout=5)
        h.disarm()
        wait_until(lambda: not h.is_alive(), what="watchdog stopped", timeout=5)
        assert _records(h._dump_path) == []
    finally:
        _stop_loop(loop, thread)


def test_wedged_loop_spin_fires_watchdog_end_to_end(tmp_path):
    """The exact incident class: the loop spins inside a callback (no timers run),
    no client is connected — the OS-thread watchdog must dump and self-exit."""
    loop = asyncio.new_event_loop()
    thread = _loop_thread(loop)
    try:
        h = LoopLivenessHandle(loop, lambda: False, tmp_path / "w.log",
                               stall_threshold_s=0.2, check_interval_s=0.05,
                               exit_strikes=2, client_grace_s=0.1, min_uptime_s=0.0,
                               tick_interval_s=0.05, redump_interval_s=60.0)
        exited = threading.Event()
        codes = []
        def _record_exit(code):
            codes.append(code)
            exited.set()
        h._exit = _record_exit
        assert h._start()

        def _spin():
            # Busy-wait on the monotonic clock: a GIL-holding callback, no sleeps.
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                pass

        loop.call_soon_threadsafe(_spin)
        assert exited.wait(timeout=10), "watchdog must fire while the loop is spinning"
        assert codes == [0]
        records = _records(h._dump_path)
        assert any(r["tag"] == "loop_watchdog.fired" for r in records)
        assert any(r["action"] == "self_exit" for r in records)
    finally:
        _stop_loop(loop, thread)


def test_wedged_loop_with_client_dumps_but_stays_alive(tmp_path):
    loop = asyncio.new_event_loop()
    thread = _loop_thread(loop)
    try:
        h = LoopLivenessHandle(loop, lambda: True, tmp_path / "w.log",
                               stall_threshold_s=0.2, check_interval_s=0.05,
                               exit_strikes=2, client_grace_s=0.1, min_uptime_s=0.0,
                               tick_interval_s=0.05, redump_interval_s=60.0)
        exited = threading.Event()
        h._exit = lambda code: exited.set()
        assert h._start()

        def _spin():
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                pass

        loop.call_soon_threadsafe(_spin)
        wait_until(lambda: _records(h._dump_path), what="stall dump written", timeout=10)
        assert not exited.is_set(), "a connected window must keep the backend alive"
        assert all(r["action"] == "dump_only" for r in _records(h._dump_path))
        h.disarm()
    finally:
        _stop_loop(loop, thread)
