"""Everything a workflow reacts to arrives as a named event, through one door.

``publish`` writes the event to a durable inbox under ``HERMES_HOME/workflows/inbox``; the reactor
(``workflow/reactor.py``) in whichever process holds the workflow lease drains it. The inbox is
what makes the door the same from everywhere: the CLI, a cron script, the messaging gateway's
webhook adapter and the desktop backend are separate processes, and an event published while no
host is up waits on disk instead of vanishing.

Names are dotted (``github.pull_request.merged``, ``hermes.session.ended``,
``workflow.run.finished``). A pattern is a case-insensitive shell glob (``github.*``).

``workflow.cmd.*`` names are the commands that move runs — start, pause, resume, cancel, answer —
so a Play from the canvas, ``hermes workflow approve`` and ``/workflow approve`` in a chat all
travel the same road as a webhook.
"""

from __future__ import annotations

import fnmatch
import itertools
import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from utils import atomic_write_text

from workflow.store import load_documents, workflows_dir
from workflow.topology import config_of, kind_of, scenario_of, steps_of

logger = logging.getLogger(__name__)

COMMAND_PREFIX = "workflow.cmd."
START = COMMAND_PREFIX + "start"
PAUSE = COMMAND_PREFIX + "pause"
RESUME = COMMAND_PREFIX + "resume"
CANCEL = COMMAND_PREFIX + "cancel"
ANSWER = COMMAND_PREFIX + "answer"

APPROVAL_REQUESTED = "workflow.approval.requested"
APPROVAL_ANSWERED = "workflow.approval.answered"
RUN_FINISHED = "workflow.run.finished"
STEP_FINISHED = "workflow.step.finished"

_WAKE = threading.Event()
_counter = itertools.count()


def inbox_dir(home: Path | None = None) -> Path:
    path = workflows_dir(home) / "inbox"
    path.mkdir(parents=True, exist_ok=True)
    return path


def wake() -> None:
    _WAKE.set()


def wait_for_wake(timeout: float) -> None:
    _WAKE.wait(timeout)
    _WAKE.clear()


def publish(name: str, payload: Any = None, *, source: str = "event", home: Path | None = None) -> dict[str, Any]:
    """Queue one event for the reactor of ``home`` (this profile's by default). Returns it, id
    included."""
    clean = str(name or "").strip()
    if not clean:
        raise ValueError("an event needs a name")
    now = int(time.time() * 1000)
    event = {"id": uuid.uuid4().hex, "name": clean, "payload": payload, "source": source, "ts": now}
    # The counter keeps one process's events in publish order within a millisecond (a cancel
    # before the Play that follows it).
    stem = f"{now:013d}-{next(_counter):08d}-{event['id']}"
    atomic_write_text(inbox_dir(home) / f"{stem}.json", json.dumps(event, ensure_ascii=False))
    wake()
    return event


def publish_if_wanted(
    name: str, payload: Any = None, *, source: str, home: Path | None = None,
) -> dict[str, Any] | None:
    """For Hermes's own happenings (a session ended, a run finished): queue the event only when
    some workflow listens for it, so a busy agent does not fill the inbox for nobody."""
    return publish(name, payload, source=source, home=home) if wanted(name, home=home) else None


def publish_for_run(name: str, run_id: str, **fields: Any) -> dict[str, Any] | None:
    """Publish a command about one run; returns the run's state as it stands (None: no such run)."""
    from workflow.store import load_run

    state = load_run(run_id)
    if state is not None:
        publish(name, {"runId": run_id, **fields}, source="desktop")
    return state


def drain() -> list[dict[str, Any]]:
    """Take every queued event, oldest first. Only the lease holder calls this."""
    events = []
    for path in sorted(inbox_dir().glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            raw = None
        try:
            path.unlink()
        except OSError:
            logger.debug("could not remove inbox event %s", path, exc_info=True)
            continue
        if isinstance(raw, dict) and raw.get("name"):
            events.append(raw)
    return events


def matches(pattern: str, name: str) -> bool:
    text = str(pattern or "").strip().lower()
    return bool(text) and fnmatch.fnmatchcase(str(name or "").strip().lower(), text)


def event_patterns(scenario: dict) -> list[str]:
    """What a scenario listens for: its event triggers and its event waits."""
    from workflow.topology import parse_poll

    out = []
    for step in steps_of(scenario):
        cfg = config_of(step)
        kind = kind_of(step)
        if kind == "trigger":
            on = cfg.get("on") or {}
            if on.get("type") == "event" and str(on.get("spec") or "").strip():
                out.append(str(on["spec"]).strip())
        elif kind == "wait":
            until = cfg.get("until") or {}
            spec = str(until.get("spec") or "").strip()
            if until.get("type") == "event" and spec:
                out.append(spec)
            elif until.get("type") == "poll" and spec and parse_poll(spec) is None:
                out.append(spec)
    return out


_wanted_cache: dict[str, tuple[int, tuple[str, ...]]] = {}
_wanted_lock = threading.Lock()


def _wanted_patterns(home: Path | None) -> tuple[str, ...]:
    path = workflows_dir(home) / "documents.json"
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return ()
    with _wanted_lock:
        cached = _wanted_cache.get(str(path))
    if cached is not None and cached[0] == stamp:
        return cached[1]
    docs = load_documents(home)["docs"]
    patterns = tuple(p for doc in docs for p in event_patterns(scenario_of(doc)))
    with _wanted_lock:
        _wanted_cache[str(path)] = (stamp, patterns)
    return patterns


def wanted(name: str, *, home: Path | None = None) -> bool:
    return any(matches(pattern, name) for pattern in _wanted_patterns(home))
