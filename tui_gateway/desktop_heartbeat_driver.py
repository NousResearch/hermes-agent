"""Desktop-owned Heartbeat scheduling through the normal TUI turn path."""

from __future__ import annotations

import threading
import time
import uuid

from .method_ctx import bind_module


_HEARTBEAT_POLL_SECONDS = 5.0
_desktop_heartbeat_driver_started = False
_desktop_heartbeat_driver_lock = threading.Lock()


def _poll_desktop_heartbeats_once() -> int:
    """Start one due Heartbeat turn per idle live Desktop session.

    The session lock covers the persisted fire claim and the in-memory ``running``
    claim. That makes a user prompt win whenever it arrived first, and prevents two
    poll passes from incrementing the same Heartbeat more than once.
    """
    fired = 0
    for sid, session in list(_sessions.items()):
        if not isinstance(session, dict):
            continue
        session_key = str(session.get("session_key") or "")
        if not session_key:
            continue
        lock = session.get("history_lock")
        if lock is None:
            continue
        try:
            with lock:
                if (
                    session.get("_closing")
                    or session.get("running")
                    or session.get("queued_prompt") is not None
                    or session.get("queued_prompts")
                ):
                    continue
                with _session_profile_runtime_scope(session):
                    from hermes_cli.heartbeat import HeartbeatManager

                    prompt = HeartbeatManager(session_key).due_prompt()
                    if not prompt:
                        continue
                    control = _snapshot_control(session_key)
                session["running"] = True
        except Exception as exc:
            logger.debug("desktop heartbeat check failed for %s: %s", sid, exc)
            continue

        try:
            _emit("session.control.update", sid, {"control": control})
            _run_prompt_submit(f"heartbeat-{uuid.uuid4().hex}", sid, session, prompt)
            fired += 1
        except Exception as exc:
            with lock:
                session["running"] = False
            logger.debug("desktop heartbeat dispatch failed for %s: %s", sid, exc)
    return fired


def _start_desktop_heartbeat_driver() -> None:
    """Start the one daemon poller that drives active Desktop session Heartbeats."""
    global _desktop_heartbeat_driver_started
    with _desktop_heartbeat_driver_lock:
        if _desktop_heartbeat_driver_started:
            return
        _desktop_heartbeat_driver_started = True

    def _loop() -> None:
        while True:
            try:
                _poll_desktop_heartbeats_once()
            except Exception:
                logger.debug("desktop heartbeat driver tick failed", exc_info=True)
            time.sleep(_HEARTBEAT_POLL_SECONDS)

    threading.Thread(target=_loop, name="desktop-heartbeat-driver", daemon=True).start()


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
