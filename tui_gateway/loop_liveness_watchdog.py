"""Runtime loop-liveness watchdog for the Desktop/dashboard ``serve`` backend.

The serve event loop can wedge after the Desktop client disconnects: its main
thread keeps burning CPU inside a callback (or parked on a lock) while accepts and
WS sends stop completing. The peer is gone, the app has already spawned a fresh
backend, but this orphan keeps holding the chat's ``session_turn_leases`` row, so
every later turn in that chat blocks and then interrupts ("session stops midway").

The existing CF-1 heartbeat (``hermes_cli.web_server``) is a ``call_later`` tick ON
the loop, so it cannot fire during a true spin -- it only logs drift once the loop
breathes again. This watchdog runs an INDEPENDENT OS daemon thread that watches a
loop-stamped timestamp:

* On a sustained stall it dumps all-thread stacks (``faulthandler``) plus a JSON
  record to ``<HERMES_HOME>/logs/desktop-loop-watchdog.log`` so the exact spinning
  code is captured -- even on a backend that still has a live window (dump-only).
* It self-exits ONLY an orphaned backend: no live WS client for a sustained grace,
  the stall confirmed across several checks, and past a minimum uptime. A backend
  serving a window the user is watching is NEVER exited, only dumped.

Modelled on ``hermes_startup_watchdog.py``: the fire path does no imports (the dump
path and the client probe are captured at arm time, when the loop is alive and the
import lock is free), and a lock-free escort thread bounds the whole forensic path
so a wedged logging handler or hung disk cannot strand the exit.

Behaviour config lives in ``config.yaml`` under ``desktop.loop_liveness_watchdog``
(read once at arm time); there are no new environment variables. Everything is
best-effort: a watchdog failure must never affect the backend it observes.
"""

from __future__ import annotations

import faulthandler
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Conservative defaults. stall_threshold sits well above the 10s WS worker wait and
# typical GIL-heavy turn drift, so a legitimate busy turn is not mistaken for a wedge.
DEFAULT_STALL_THRESHOLD_S = 30.0
DEFAULT_CHECK_INTERVAL_S = 5.0
DEFAULT_EXIT_STRIKES = 3
DEFAULT_CLIENT_GRACE_S = 60.0
DEFAULT_MIN_UPTIME_S = 120.0
DEFAULT_TICK_INTERVAL_S = 2.0
DEFAULT_REDUMP_INTERVAL_S = 60.0
DEFAULT_EXIT_CODE = 0

# Upper bound on the ENTIRE forensic fire path; the escort hard-exits past it.
_FIRE_EXIT_BOUND_S = 10.0

_DUMP_RELATIVE = ("logs", "desktop-loop-watchdog.log")

_ARMED = "armed"
_DISARMED = "disarmed"
_FIRING = "firing"

# Module singleton: one serve loop per process; arm is idempotent.
_handle_lock = threading.Lock()
_handle: Optional[LoopLivenessHandle] = None


def _process_hermes_home() -> Path:
    from hermes_constants import get_process_hermes_home
    return get_process_hermes_home()


def get_dump_path(home: Optional[Path] = None) -> Path:
    base = home if home is not None else _process_hermes_home()
    return base.joinpath(*_DUMP_RELATIVE)


def _log_quietly(level: str, msg: str, *args) -> None:
    """Log without ever raising: the wedged main thread may hold the logging lock."""
    try:
        getattr(logger, level)(msg, *args)
    except Exception:
        pass


class LoopLivenessHandle:
    """Disarm/inspect handle for the armed runtime loop-liveness watchdog."""

    def __init__(
        self,
        loop,
        has_clients: Callable[[], bool],
        dump_path: Path,
        *,
        stall_threshold_s: float = DEFAULT_STALL_THRESHOLD_S,
        check_interval_s: float = DEFAULT_CHECK_INTERVAL_S,
        exit_strikes: int = DEFAULT_EXIT_STRIKES,
        client_grace_s: float = DEFAULT_CLIENT_GRACE_S,
        min_uptime_s: float = DEFAULT_MIN_UPTIME_S,
        tick_interval_s: float = DEFAULT_TICK_INTERVAL_S,
        redump_interval_s: float = DEFAULT_REDUMP_INTERVAL_S,
        exit_code: int = DEFAULT_EXIT_CODE,
    ) -> None:
        self._loop = loop
        self._has_clients_fn = has_clients
        self._dump_path = dump_path
        self.stall_threshold_s = stall_threshold_s
        self.check_interval_s = check_interval_s
        self.exit_strikes = max(1, int(exit_strikes))
        self.client_grace_s = client_grace_s
        self.min_uptime_s = min_uptime_s
        self.tick_interval_s = tick_interval_s
        self.redump_interval_s = redump_interval_s
        self.exit_code = exit_code

        self.armed_at = self._monotonic()
        self.last_tick = self.armed_at
        self._no_clients_since: Optional[float] = None
        self._stall_strikes = 0
        self._stall_episode_dumped = False
        self._last_dump = 0.0
        self._state = _ARMED
        self._state_lock = threading.Lock()
        self._disarmed_event = threading.Event()
        self._fire_done = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tick_handle = None

    # ── seams (tests override these; production is the bare stdlib call) ──────────
    @staticmethod
    def _monotonic() -> float:
        return time.monotonic()

    @staticmethod
    def _exit(code: int) -> None:
        os._exit(code)

    @staticmethod
    def _sleep(seconds: float) -> None:
        time.sleep(seconds)

    def _dump_stacks(self, fh) -> None:
        faulthandler.dump_traceback(file=fh, all_threads=True)

    # ── lifecycle ────────────────────────────────────────────────────────────────
    def disarm(self) -> None:
        with self._state_lock:
            if self._state == _ARMED:
                self._state = _DISARMED
        self._disarmed_event.set()
        handle = self._tick_handle
        if handle is not None:
            try:
                handle.cancel()
            except Exception:
                pass

    @property
    def disarmed(self) -> bool:
        return self._state == _DISARMED

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _schedule_tick(self) -> None:
        # Runs on the loop thread (call_soon_threadsafe from arm).
        if self._state != _ARMED:
            return
        self.last_tick = self._monotonic()
        self._tick_handle = self._loop.call_later(self.tick_interval_s, self._tick)

    def _tick(self) -> None:
        if self._state != _ARMED:
            return
        self.last_tick = self._monotonic()
        self._tick_handle = self._loop.call_later(self.tick_interval_s, self._tick)

    def _start(self) -> bool:
        try:
            self._loop.call_soon_threadsafe(self._schedule_tick)
        except Exception:
            _log_quietly("debug", "loop-liveness tick scheduling failed")
            return False
        thread = threading.Thread(target=self._run, daemon=True, name="desktop-loop-liveness-watchdog")
        try:
            thread.start()
        except Exception:
            _log_quietly("debug", "loop-liveness watchdog thread start failed")
            return False
        self._thread = thread
        return True

    def _run(self) -> None:
        while True:
            with self._state_lock:
                if self._state != _ARMED:
                    return
            if self._disarmed_event.wait(self.check_interval_s):
                return
            try:
                self._check_once()
            except Exception:
                _log_quietly("debug", "loop-liveness check failed")

    # ── core evaluation (one pass; tests drive this directly) ─────────────────────
    def _has_clients(self) -> bool:
        try:
            return bool(self._has_clients_fn())
        except Exception:
            # Fail-safe: an unreadable registry must never license an exit.
            return True

    def _check_once(self, now: Optional[float] = None) -> None:
        now = self._monotonic() if now is None else now
        with self._state_lock:
            if self._state != _ARMED:
                return
        gap = now - self.last_tick
        uptime = now - self.armed_at
        clients = self._has_clients()
        if clients:
            self._no_clients_since = None
        elif self._no_clients_since is None:
            self._no_clients_since = now

        if gap <= self.stall_threshold_s:
            # Loop is breathing: clear the stall episode.
            self._stall_strikes = 0
            self._stall_episode_dumped = False
            return

        self._stall_strikes += 1
        exit_pending = (
            not clients
            and uptime >= self.min_uptime_s
            and self._no_clients_since is not None
            and (now - self._no_clients_since) >= self.client_grace_s
            and self._stall_strikes >= self.exit_strikes
        )
        self._maybe_dump(now, gap, uptime, clients, exit_pending)
        if exit_pending:
            self._fire(now, gap, uptime)

    def _maybe_dump(self, now: float, gap: float, uptime: float, clients: bool, exit_pending: bool) -> None:
        # One dump per stall episode, then rate-limited re-dumps while still stalled.
        if self._stall_episode_dumped and (now - self._last_dump) < self.redump_interval_s:
            return
        self._stall_episode_dumped = True
        self._last_dump = now
        self._write_dump({
            "ts": datetime.now(UTC).isoformat(),
            "tag": "loop_stall.detected",
            "pid": os.getpid(),
            "gap_s": round(gap, 3),
            "uptime_s": round(uptime, 3),
            "has_clients": clients,
            "stall_strikes": self._stall_strikes,
            "action": "self_exit" if exit_pending else "dump_only",
        })

    def _write_dump(self, record: dict) -> None:
        """Append a JSON record + all-thread stacks. No imports, no logging lock: the
        wedged main thread may hold both, so this touches only a fresh file handle and
        faulthandler (C-level). Best-effort; the escort bounds the fire path regardless."""
        try:
            self._dump_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._dump_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
                self._dump_stacks(fh)
                fh.write("\n")
        except Exception:
            _log_quietly("debug", "loop-liveness dump write failed")

    def _fire(self, now: float, gap: float, uptime: float) -> None:
        with self._state_lock:
            if self._state != _ARMED:
                return
            self._state = _FIRING
        # Escort FIRST: a lock-free hard-exit if forensics wedge.
        try:
            threading.Thread(target=self._exit_escort, daemon=True,
                             name="desktop-loop-liveness-exit-escort").start()
        except Exception:
            pass
        _log_quietly(
            "critical",
            "Desktop serve loop wedged %.1fs with no live client for %.1fs (uptime %.1fs); "
            "dumping all thread stacks and self-exiting orphaned backend (code %d).",
            gap, now - (self._no_clients_since or now), uptime, self.exit_code,
        )
        self._write_dump({
            "ts": datetime.now(UTC).isoformat(),
            "tag": "loop_watchdog.fired",
            "pid": os.getpid(),
            "gap_s": round(gap, 3),
            "uptime_s": round(uptime, 3),
            "no_clients_s": round(now - (self._no_clients_since or now), 3),
            "exit_code": self.exit_code,
            "action": "self_exit",
        })
        self._fire_done.set()
        self._exit(self.exit_code)

    def _exit_escort(self) -> None:
        self._sleep(_FIRE_EXIT_BOUND_S)
        if not self._fire_done.is_set():
            self._exit(self.exit_code)


def _resolve_config() -> dict:
    """Read ``desktop.loop_liveness_watchdog`` from config.yaml at arm time (never in
    the fire path). Defaults on any error; ``enabled`` defaults True."""
    cfg: dict = {}
    try:
        from hermes_cli.config import load_config
        raw = (load_config() or {}).get("desktop", {})
        if isinstance(raw, dict):
            section = raw.get("loop_liveness_watchdog", {})
            if isinstance(section, dict):
                cfg = dict(section)
    except Exception:
        _log_quietly("debug", "loop-liveness config read failed; using defaults")
    return cfg


def arm_loop_liveness_watchdog(loop, has_clients: Callable[[], bool]) -> Optional[LoopLivenessHandle]:
    """Arm the runtime watchdog on the serve loop. Idempotent; never raises.

    ``has_clients`` must be a cheap, import-free callable returning True while any
    live WS client is connected (captured by the caller, which already holds the
    server module). Returns the handle, or None when disabled or already armed.
    """
    global _handle
    try:
        cfg = _resolve_config()
        if cfg.get("enabled", True) is False:
            return None
        with _handle_lock:
            if _handle is not None and _handle.is_alive():
                return _handle
            try:
                dump_path = get_dump_path()
            except Exception:
                dump_path = Path(os.path.expanduser("~/.hermes")).joinpath(*_DUMP_RELATIVE)

            def _num(key, default):
                try:
                    value = float(cfg.get(key, default))
                    return value if value > 0 else default
                except (TypeError, ValueError):
                    return default

            handle = LoopLivenessHandle(
                loop, has_clients, dump_path,
                stall_threshold_s=_num("stall_threshold_s", DEFAULT_STALL_THRESHOLD_S),
                check_interval_s=_num("check_interval_s", DEFAULT_CHECK_INTERVAL_S),
                exit_strikes=int(_num("exit_strikes", DEFAULT_EXIT_STRIKES)),
                client_grace_s=_num("client_grace_s", DEFAULT_CLIENT_GRACE_S),
                min_uptime_s=_num("min_uptime_s", DEFAULT_MIN_UPTIME_S),
                tick_interval_s=_num("tick_interval_s", DEFAULT_TICK_INTERVAL_S),
                redump_interval_s=_num("redump_interval_s", DEFAULT_REDUMP_INTERVAL_S),
                exit_code=int(cfg.get("exit_code", DEFAULT_EXIT_CODE)),
            )
            if not handle._start():
                return None
            _handle = handle
            _log_quietly("info", "Desktop loop-liveness watchdog armed (stall>%.0fs, exit only when orphaned)",
                         handle.stall_threshold_s)
            return handle
    except Exception:
        _log_quietly("debug", "Failed to arm loop-liveness watchdog")
        return None


def disarm_loop_liveness_watchdog() -> None:
    global _handle
    try:
        with _handle_lock:
            handle = _handle
            _handle = None
        if handle is not None:
            handle.disarm()
    except Exception:
        _log_quietly("debug", "Failed to disarm loop-liveness watchdog")


def _reset_for_tests() -> None:
    global _handle
    with _handle_lock:
        handle = _handle
        _handle = None
    if handle is not None:
        handle.disarm()
