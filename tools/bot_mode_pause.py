"""Bot Mode pause: a bot whose user paused it stops working on its own.

The flag is ``ui_meta['hermes-bots'].paused`` in the bot's own ``profile.yaml``: the Bot Mode
metadata the Desktop already writes (``profiles.configure``) and syncs to every desktop, read
here from the profile home, so each profile's pause stays its own (profiles are islands).

While paused:

- its scheduled jobs do not fire. Each recurring slot that falls due is consumed (its
  ``next_run_at`` advances) instead of run, so a resume never bursts the missed runs; a due
  one-shot is held and meets the scheduler's ordinary one-shot grace after the resume. The
  pause's start and end are recorded when ``profiles.configure`` flips the flag
  (:func:`record_pause_change`), so a slot that fell due inside a pause no tick observed (shorter
  than a tick, or with the ticker down) is still skipped at the first tick after the resume.
- a teammate's ``message_agent`` delivery is refused with :data:`RECIPIENT_PAUSED`
  (``gateway/session_bot.py``), a definite no-send the sender is told about.

The user can still open and chat with a paused bot: pause stops the work the bot does on its
own (schedules, teammate traffic), not the human's own conversation. ``hermes pause`` (the
global emergency stop) is the switch that holds every turn. Hidden stays display-only.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from tools.bot_failure_reasons import RECIPIENT_PAUSED

__all__ = ["RECIPIENT_PAUSED", "hold_paused_profile_jobs", "is_bot_paused", "record_pause_change"]

# Pause intervals ``[{"start": ts, "end": ts | None}]``, newest last, in the profile home.
_WINDOWS_FILE = "bot_pause_windows.json"
_MAX_WINDOWS = 16

# Profile homes whose pause was logged already: once per engagement, re-armed on resume.
_logged_lock = threading.Lock()
_logged_homes: set[str] = set()


def is_bot_paused(home: str | os.PathLike) -> bool:
    """True when the profile at ``home`` is a paused Bot Mode bot. A missing or unreadable
    ``profile.yaml`` is not paused: pause is an opt-in the user set, never a failure mode."""
    from tools.bot_mode_probe import _bots_meta, _read_yaml_dict

    meta = _bots_meta(_read_yaml_dict(Path(home) / "profile.yaml", "paused"))
    return bool(meta) and meta.get("paused") is True


def _read_windows(home: Path) -> list[dict]:
    try:
        windows = json.loads((home / _WINDOWS_FILE).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return []
    return [w for w in windows if isinstance(w, dict)] if isinstance(windows, list) else []


def record_pause_change(home: str | os.PathLike, paused: bool) -> None:
    """Stamp a pause's start (``paused``) or end in the profile home. ``profiles.configure`` calls
    this when it flips the flag, so the tick can tell which slots fell due during the pause even
    when it never ran while the bot was paused."""
    from utils import atomic_json_write

    home = Path(home)
    windows = _read_windows(home)
    still_open = bool(windows) and windows[-1].get("end") is None
    if paused and not still_open:
        windows.append({"start": time.time(), "end": None})
    elif not paused and still_open:
        windows[-1]["end"] = time.time()
    else:
        return
    atomic_json_write(home / _WINDOWS_FILE, windows[-_MAX_WINDOWS:])


def _pause_end_covering(due: float, windows: list[dict]) -> float | None:
    # An open window is skipped: with the window written before the flag, "unpaused, open window"
    # only means a pause that has not taken effect (or whose flag write failed).
    return next((w["end"] for w in windows if isinstance(w.get("end"), (int, float))
                 and w.get("start", due + 1) <= due <= w["end"]), None)


def _held_by_pause(job: dict, windows: list[dict], now: float) -> bool:
    """True when every slot of the recurring ``job`` due by ``now`` fell inside a recorded pause.
    A slot due after the resume leaves the job on the ordinary due path, which fires it once as
    it fires any missed slot; only the pause's own slots are consumed."""
    from cron.jobs import _parse_aware, compute_next_run, parse_schedule

    raw = job.get("schedule")
    schedule = parse_schedule(raw) if isinstance(raw, str) else raw if isinstance(raw, dict) else {}
    slot, held = _parse_aware(job.get("next_run_at")), False
    while slot is not None and slot.timestamp() <= now:
        end = _pause_end_covering(slot.timestamp(), windows)
        if end is None:
            return False
        held = True  # each step lands strictly past a window's end, so this ends within len(windows)
        slot = _parse_aware(compute_next_run(schedule, datetime.fromtimestamp(end, UTC).isoformat()))
    return held


def _note_paused(home: Path, paused: bool, logger: logging.Logger) -> None:
    key = str(home)
    with _logged_lock:
        first = paused and key not in _logged_homes
        if paused:
            _logged_homes.add(key)
        else:
            _logged_homes.discard(key)
    if first:
        logger.info("Cron dispatch held for %s: its bot is paused (due slots are skipped, not run)", home)


def hold_paused_profile_jobs(due_jobs: list, logger: logging.Logger) -> list:
    """The due jobs the tick may run for the profile in scope: all of them, or none while that
    profile's bot is paused. Called under the tick lock with the profile's home and cron store
    bound (``_profile_cron_scope``); a paused profile's recurring jobs are advanced past their
    due slot so the resume finds nothing overdue."""
    from cron.constants import is_recurring
    from cron.jobs import advance_next_runs
    from cron.scheduler_gateway_gate import _note_fire_refused
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    paused = is_bot_paused(home)
    _note_paused(home, paused, logger)
    if not due_jobs:
        return due_jobs
    if paused:
        held, why = due_jobs, "Skipped: this bot is paused"
    else:
        # Resumed: a recurring job whose due slots all fell inside recorded pauses is consumed; one
        # with a slot due since the resume runs, as any missed slot does.
        windows = _read_windows(Path(home))
        now = time.time()
        held = [job for job in due_jobs if is_recurring(job) and _held_by_pause(job, windows, now)]
        why = "Skipped: it fell due while this bot was paused"
        if not held:
            return due_jobs
    # Never handed to the dispatcher: drop the due scan's pending-slot mark (else the next scan
    # "restores" the slot as a lost dispatch) and stamp why, as the headless gate does.
    _note_fire_refused([str(job["id"]) for job in held], why)
    advance_next_runs([job["id"] for job in held if is_recurring(job)])
    return [job for job in due_jobs if job not in held]
