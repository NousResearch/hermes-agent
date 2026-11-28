"""Boot-time auto-continue sweep for desktop sessions (unattended operation).

Upstream already has the pieces: ``turn_marker`` writes a durable marker when a turn starts and
retires it on any conclusion, and ``session_auto_continue._maybe_schedule_auto_continue`` resumes a
turn from that marker. The resume, however, only ever runs from ``session.resume`` (client attach):
after a backend restart nobody opens the app, so nothing resumes the interrupted turn (#135286,
#127919).

This module closes that gap from the serve boot path: wait N seconds, scan each home's
``desktop/interrupted_turns.json``, and for sessions whose writer is gone, whose marker is still
fresh and whose attempts are not exhausted, call the registered ``session.resume`` handler
in-process (same pattern as ``methods_onboarding.py:206``) so the existing resume path does the work.

Every predicate is reused rather than reinvented:

* writer liveness -- ``turn_marker.marker_writer_state`` (pid + process start time, #94778)
* freshness / enable switch / attempt budget -- ``desktop.auto_continue`` config
* a client already attached -- ``session.resume`` skips it on its own

Disable with ``desktop.auto_continue.enabled: false`` or ``HERMES_AUTO_CONTINUE_SWEEP=0``.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DELAY_SECS = 20.0


def hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


def _notify(text: str) -> None:
    """Surface the sweep outcome. The backend owns no user-facing channel of its own, so this is a
    warning log line; hosts that want a push can tail the log or attach a logging handler."""
    logger.warning("auto-continue: %s", " | ".join(text.splitlines()))


def _targets() -> list[tuple[Path, str]]:
    """(home, profile); default first, then profiles/*."""
    root = hermes_home()
    out: list[tuple[Path, str]] = [(root, "default")]
    profiles = root / "profiles"
    try:
        if profiles.is_dir():
            out += [(p, p.name) for p in sorted(profiles.iterdir()) if p.is_dir()]
    except OSError:
        pass
    return out


def sweep_once() -> int:
    """Scan every home's interrupted-turn markers once and resume whatever is resumable.

    Single-flight across processes: a host can run more than one desktop backend (isolated / SSH
    children) and two sweeps would resume the same session twice, so take a per-home file lock and
    stand down for this round when another backend holds it. Returns the number of resumes started.
    """
    if os.environ.get("HERMES_AUTO_CONTINUE_SWEEP", "1") == "0":
        return 0
    import fcntl
    lock_path = hermes_home() / "state" / "auto-continue-sweep.lock"
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(lock_path, "a+")
    except OSError as exc:
        logger.warning("auto-continue sweep: cannot open lock %s (%s), skipped", lock_path, exc)
        return 0
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            logger.info("auto-continue sweep: another backend is sweeping, skipped")
            return 0
        return _sweep_once_locked()
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        fh.close()


def _sweep_once_locked() -> int:
    """The actual scan, holding the per-home lock."""
    from . import turn_marker as tm

    from . import server  # deferred: only then is _methods populated by the @method decorators

    # method_ctx.bind_module rebinds each function and publishes it on server, so the submodule's
    # own copy stays unbound (calling it raises NameError: _load_cfg). Use the server-bound copy.
    cfg_fn = getattr(server, "_auto_continue_config", None)
    if cfg_fn is None:
        from .session_auto_continue import _auto_continue_config as cfg_fn
    try:
        enabled, freshness_secs, max_attempts, max_resumes = cfg_fn()
    except NameError:                       # binding not done yet (very early boot) — skip, never guess defaults
        logger.warning("auto-continue sweep: config helper not bound yet, skipped")
        return 0
    if not enabled or max_attempts <= 0:
        logger.info("auto-continue sweep: disabled (enabled=%s max_attempts=%s)", enabled, max_attempts)
        return 0

    resume = server._methods.get("session.resume")
    if resume is None:                      # handler not registered yet (early boot) — skip this round
        logger.warning("auto-continue sweep: session.resume not registered yet")
        return 0

    resumed = 0
    now = time.time()
    resumed_list: list[tuple[str, float]] = []      # (session_key, seconds interrupted)
    abandoned: list[tuple[str, str]] = []           # (session_key, reason)
    for home, profile in _targets():
        path = tm._marker_path(home)
        entries = tm._load(path)
        for session_key, entry in list((entries or {}).items()):
            try:
                if not isinstance(entry, dict):
                    logger.info("auto-continue sweep: %s skip (not a dict)", session_key)
                    continue
                state = tm.marker_writer_state(entry)
                age = now - float(entry.get("started_at") or 0)
                if state != "dead":   # writer alive or unknowable — leave it (a sibling backend may be running)
                    logger.info("auto-continue sweep: %s skip (writer_state=%s pid=%s)",
                                session_key, state, entry.get("writer_pid"))
                    continue
                started = float(entry.get("started_at") or 0)
                if started <= 0 or (now - started) > freshness_secs:
                    logger.info("auto-continue sweep: %s skip (stale: age=%.0fs > %.0fs)",
                                session_key, age, freshness_secs)
                    abandoned.append((session_key, f"interrupted {age/60:.0f} min ago, past the {freshness_secs/60:.0f} min window"))
                    continue
                attempts = int(entry.get("attempts") or 0)
                resumes = int(entry.get("resumes") or 0)
                if attempts >= max_attempts:
                    logger.warning("auto-continue sweep: %s reached max_attempts=%s "
                                   "(consecutive runs with no progress), skipped", session_key, max_attempts)
                    abandoned.append((session_key, f"{max_attempts} consecutive resumes produced nothing, giving up"))
                    continue
                if resumes >= max_resumes:
                    logger.warning("auto-continue sweep: %s reached max_resumes=%s, skipped",
                                   session_key, max_resumes)
                    abandoned.append((session_key, f"already auto-resumed {max_resumes} times without finishing"
                                                  f" (the backend keeps restarting — stopping here)"))
                    continue
                resume(f"boot-ac:{session_key[:12]}", {
                    "session_id": session_key,
                    "source": "desktop",
                    "profile": profile,
                })
                resumed += 1
                resumed_list.append((session_key, now - started))
                logger.info("auto-continue sweep: resumed session=%s profile=%s age=%.0fs",
                            session_key, profile, now - started)
            except Exception:
                logger.exception("auto-continue sweep: resume failed for %s", session_key)
                abandoned.append((session_key, "the resume call raised; see gui.log"))
    logger.info("auto-continue sweep: %d resumed", resumed)

    # Report: with no client attached the log is the only witness, so record the outcome either way.
    if resumed_list:
        lines = [f"auto-continue: {len(resumed_list)} session(s) interrupted by the restart were resumed:"]
        lines += [f"- {k} (interrupted {a:.0f}s ago)" for k, a in resumed_list]
        _notify("\n".join(lines))
    if abandoned:
        lines = [f"auto-continue: {len(abandoned)} session(s) were NOT resumed and need a human:"]
        lines += [f"· {k}：{why}" for k, why in abandoned]
        _notify("\n".join(lines))
    return resumed


def schedule(delay: float | None = None) -> threading.Thread | None:
    """Background thread: sweep once after delay seconds (never blocks serve startup)."""
    if os.environ.get("HERMES_AUTO_CONTINUE_SWEEP", "1") == "0":
        return None
    try:
        secs = float(os.environ.get("HERMES_AUTO_CONTINUE_SWEEP_DELAY", _DEFAULT_DELAY_SECS)) \
            if delay is None else float(delay)
    except (TypeError, ValueError):
        secs = _DEFAULT_DELAY_SECS

    def _run() -> None:
        time.sleep(max(0.0, secs))
        try:
            sweep_once()
        except Exception:
            logger.exception("auto-continue sweep failed")

    t = threading.Thread(target=_run, daemon=True, name="auto-continue-sweep")
    t.start()
    return t
