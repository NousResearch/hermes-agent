"""Session-scoped, one-conversation handoffs created through the memory tool.

A pending note is keyed by the session that created it.  Only an explicit
``/new`` transition moves that note to the newly minted session id; prompt
rebuilds, compression children, resumes, and branches therefore cannot claim
another conversation's note.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from hermes_constants import get_hermes_home, mkdir_under_hermes_home

try:  # pragma: no cover - platform-specific import
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        msvcrt = None
else:  # pragma: no cover - the Windows name is only used there
    msvcrt = None


_PROCESS_LOCK = threading.RLock()
_NEXT_SESSION_MARKER = re.compile(
    r"\b(?:next (?:session|conversation)|bir sonraki (?:oturum|sohbet))\b", re.IGNORECASE)
_DIRECTIVE = re.compile(
    r"\b(?:task|to-?do|start|begin|continue|resume|do|handle|follow up|work on|remind me|"
    r"görev|başla|devam et|yap|ele al|hatırlat)\b",
    re.IGNORECASE,
)


def looks_like_next_session_handoff(content: str) -> bool:
    """Recognize an explicit next-session directive, not a factual mention."""
    normalized = " ".join(str(content or "").strip().split())
    if not normalized:
        return False
    marker = _NEXT_SESSION_MARKER.search(normalized)
    if marker is None:
        return False
    lo, hi = max(0, marker.start() - 100), min(len(normalized), marker.end() + 100)
    return _DIRECTIVE.search(normalized[lo:hi]) is not None


def _handoff_dir() -> Path:
    return get_hermes_home() / "handoffs" / "next-session"


def _session_path(kind: str, session_id: str) -> Path:
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    return _handoff_dir() / kind / f"{digest}.json"


@contextmanager
def _locked() -> Iterator[None]:
    """Serialize handoff read-modify-write operations across threads/processes."""
    directory = _handoff_dir()
    mkdir_under_hermes_home(directory)
    lock_path = directory / ".lock"
    with _PROCESS_LOCK, open(lock_path, "a+b") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        elif msvcrt is not None:  # pragma: no cover - Windows
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover - Windows
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


def _read(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    mkdir_under_hermes_home(path.parent)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".handoff_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def queue_next_session_handoffs(
    messages: list[str], *, source_session_id: str, source: str = "memory"
) -> dict[str, Any]:
    """Append deduplicated notes to one source session's pending handoff."""
    session_id = str(source_session_id or "").strip()
    clean = [str(message).strip() for message in messages if str(message).strip()]
    if not session_id:
        return {"success": False, "error": "A current session id is required for a next-session handoff."}
    if not clean:
        return {"success": False, "error": "A next-session handoff cannot be empty."}
    path = _session_path("pending", session_id)
    with _locked():
        previous = _read(path) or {}
        queued = [str(item).strip() for item in previous.get("messages", []) if str(item).strip()]
        for message in clean:
            if message not in queued:
                queued.append(message)
        payload = {
            "version": 1,
            "source_session_id": session_id,
            "messages": queued,
            "source": source,
            "created_at": previous.get("created_at") or datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "use_once": True,
        }
        _atomic_write(path, payload)
    return {
        "success": True,
        "done": True,
        "handoff": True,
        "queued_count": len(payload["messages"]),
        "message": "Saved for the next session only; it was not added to durable memory.",
        "note": "Write saved. This update is complete — do not repeat it.",
    }


def transfer_next_session_handoff(source_session_id: str, target_session_id: str) -> bool:
    """Atomically assign one source's pending note to an explicit ``/new`` target."""
    source_id, target_id = str(source_session_id or "").strip(), str(target_session_id or "").strip()
    if not source_id or not target_id or source_id == target_id:
        return False
    pending = _session_path("pending", source_id)
    assigned = _session_path("assigned", target_id)
    with _locked():
        payload = _read(pending)
        if payload is None:
            return False
        existing = _read(assigned) or {}
        messages = [str(item).strip() for item in existing.get("messages", []) if str(item).strip()]
        for item in payload.get("messages", []):
            message = str(item).strip()
            if message and message not in messages:
                messages.append(message)
        payload.update({
            "messages": messages,
            "target_session_id": target_id,
            "assigned_at": datetime.now(UTC).isoformat(),
        })
        _atomic_write(assigned, payload)
        pending.unlink(missing_ok=True)
        # The source conversation is now over. Its own assigned copy has
        # already been frozen into that session's persisted system prompt.
        _session_path("assigned", source_id).unlink(missing_ok=True)
    return True


def next_session_handoff_prompt(session_id: str) -> str | None:
    """Return the note assigned to this exact session (idempotent within it)."""
    target_id = str(session_id or "").strip()
    if not target_id:
        return None
    payload = _read(_session_path("assigned", target_id))
    if payload is None or payload.get("target_session_id") != target_id:
        return None
    messages = [str(item).strip() for item in payload.get("messages", []) if str(item).strip()]
    if not messages:
        return None
    bullets = "\n".join(f"- {message}" for message in messages)
    return (
        "## Next Session Handoff\n"
        "One-shot carryover from the immediately previous session. Use it only if it remains "
        "relevant to the user's current message.\n"
        f"{bullets}"
    )
