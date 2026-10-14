"""Durable pre-admission parking for local Bot Chat DMs (#128996).

A local ``message_agent`` DM whose recipient's Bot Chat is held by a surface
that has not advertised live delivery (``bot_live_delivery_consumer``) must
not spawn a competing CLI copy. The DM is parked here, under the RECIPIENT's
home, and drained by exactly one runner:

- the sender's own delivery runner waits a bounded budget: when a live owner
  appears the record converts to a mailbox ticket (the standard live-owner
  contract); when NO surface holds the chat the record is claimed and the
  sanctioned one-shot CLI turn runs;
- a live-owner session that begins advertising after the runner lapsed picks
  the record up on its mailbox poll (``_poll_bot_live_delivery_once``).

Same contract as ``tools/bot_live_delivery``: a single private record per
delivery id advances queued -> claimed -> terminal (settled | failed) or
transferred, claims never expire, receipts are permanent, and an id reused
with a different payload is an error — never an overwrite. Mirrors
``cron/bot_chat_delivery`` (the cron-side precedent) but parks pre-admission:
the record holds the raw argv + message + dm file so whichever path drains it
can execute the original delivery exactly once.
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from utils import atomic_json_write, fsync_directory

from hermes_cli.active_sessions import _FileLock

logger = logging.getLogger(__name__)

PENDING_DIR_NAME = "bot_dm_pending"
_TERMINAL = frozenset({"settled", "failed", "transferred"})
_warned_unreadable: set[Path] = set()


def _root(profile_home: Path | str) -> Path:
    return Path(profile_home).resolve() / "runtime" / PENDING_DIR_NAME


def has_pending(profile_home: Path | str) -> bool:
    """Cheap pre-check for the live-owner poller: no dir means nothing parked."""
    return _root(profile_home).is_dir()


@contextmanager
def _locked(profile_home: Path | str):
    root = _root(profile_home)
    created = not root.is_dir()
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(mode=0o700, exist_ok=True)
    root.chmod(0o700)
    if created:
        fsync_directory(root.parent)
    lock = root / ".lock"
    fd = os.open(lock, os.O_CREAT | os.O_WRONLY, 0o600)
    os.close(fd)
    with _FileLock(lock):
        yield root


def _read(root: Path, delivery_id: str) -> dict[str, Any] | None:
    """Exact-id read: absent → None; unreadable or not a JSON object → raises (fail closed)."""
    try:
        record = json.loads((root / f"{delivery_id}.json").read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return None
    if not isinstance(record, dict):
        raise ValueError(f"pending DM {delivery_id} is not a JSON object ({type(record).__name__})")
    return record


def _scan_read(path: Path) -> dict[str, Any] | None:
    """Bulk-scan variant: one damaged receipt must not wedge the whole dir."""
    try:
        record = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(record, dict):
            raise ValueError(f"expected a JSON object, got {type(record).__name__}")
    except (OSError, ValueError) as exc:  # ValueError: corrupt JSON and invalid UTF-8 alike
        level = logging.DEBUG if path in _warned_unreadable else logging.WARNING
        _warned_unreadable.add(path)
        logger.log(level, "Unreadable pending DM receipt %s: %s", path, exc)
        return None
    _warned_unreadable.discard(path)
    return record


def _records(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    return [(path, record) for path in sorted(root.glob("*.json"))
            if (record := _scan_read(path)) is not None]


def _write(path: Path, record: dict[str, Any]) -> None:
    atomic_json_write(path, record, indent=None, sort_keys=True, fsync_dir=True, mode=0o600)


def park(delivery_id: str, home: Path | str, *, argv: list[str], dm_file: str,
         message: str, author: dict[str, Any] | None = None) -> dict[str, Any]:
    """Park (or re-inspect) a DM for a Bot Chat held by an unadvertised surface.

    Retry with the same id, home, argv, dm_file and message to inspect the
    existing record; a differing payload under the same id is an error.
    """
    key = str(delivery_id)
    payload = dict(id=key, status="queued", home=str(Path(home).resolve()),
                   argv=list(argv), dm_file=str(dm_file), message=message)
    if author:
        payload["author"] = dict(author)
    with _locked(home) as root:
        path = root / f"{key}.json"
        record = _read(root, key)
        if record is not None:
            if (record.get("home") != payload["home"] or record.get("argv") != payload["argv"]
                    or record.get("dm_file") != payload["dm_file"]
                    or record.get("message") != payload["message"]
                    or record.get("author") != payload.get("author")):
                raise ValueError("delivery id already belongs to a different payload")
            return record
        _write(path, payload)
        return payload


def read_pending(home: Path | str, delivery_id: str) -> dict[str, Any] | None:
    """Exact-id read under the record lock (None when absent)."""
    with _locked(home) as root:
        return _read(root, str(delivery_id))


def claim(home: Path | str, delivery_id: str, *, reason: str) -> dict[str, Any] | None:
    """Claim a queued record for the CLI drain; None when already claimed/terminal
    (the claim is the at-most-once fence — a lost runner leaves it claimed, never
    re-executed)."""
    key = str(delivery_id)
    with _locked(home) as root:
        record = _read(root, key)
        if record is None or record["status"] != "queued":
            return None
        record.update(status="claimed", claimed_by=reason, claimed_at=time.time_ns())
        _write(root / f"{key}.json", record)
        return record


def settle(home: Path | str, delivery_id: str, *, status: str, reply: str = "",
           error: str = "", reason: str = "") -> dict[str, Any]:
    """Persist the terminal receipt for a CLI-drained record; immutable after."""
    key = str(delivery_id)
    if status not in ("settled", "failed"):
        raise ValueError("invalid pending DM status")
    with _locked(home) as root:
        record = _read(root, key)
        if record is None:
            raise FileNotFoundError(f"pending DM not found: {key}")
        if record["status"] in _TERMINAL:
            return record
        record.update(status=status, reply=reply, error=error, reason=reason)
        _write(root / f"{key}.json", record)
        return record


def convert_to_live_owner(home: Path | str, delivery_id: str, owner: dict[str, Any],
                          *, author: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Transfer a queued record into the live-owner mailbox once a consumer advertises.

    Writes a mailbox ticket pinned to ``owner`` (idempotent: the mailbox's own
    admission dedupe applies), then marks the pending record ``transferred`` so
    no CLI drain may ever execute it. None when the record is absent or no
    longer queued. ``author`` (the sending bot) rides to the mailbox ticket so
    the recipient's turn carries the same attribution a direct admission had.
    """
    from tools.bot_live_delivery import deliver_to_live_owner

    key = str(delivery_id)
    home_path = Path(home).resolve()
    with _locked(home) as root:
        record = _read(root, key)
        if record is None:
            return None
        if record["status"] == "transferred":
            return record
        if record["status"] != "queued":
            return None
        ticket_author = record.get("author") or author
        deliver_to_live_owner(home_path, owner, record["message"],
                              delivery_id=key, author=ticket_author or None)
        record.update(status="transferred", transferred_to=owner["lease_id"])
        _write(root / f"{key}.json", record)
        return record


def pending_records_for_home(home: Path | str) -> list[dict[str, Any]]:
    """Every still-queued record parked for this home (live-owner poller's pickup list)."""
    if not has_pending(home):
        return []
    with _locked(home) as root:
        return [record for _, record in _records(root) if record.get("status") == "queued"]
