"""
Persistent Todo Store — file-backed todo list storage.

Wraps the in-memory TodoStore with JSON file persistence so that
todo state survives across agent sessions and can be queried by
external clients (e.g., the API server's /api/todos endpoints).

Storage layout:
    ~/.hermes/todos/{session_id}.json

Each file contains:
    {
        "session_id": "...",
        "created_at": "ISO8601",
        "updated_at": "ISO8601",
        "todos": [ {id, content, status}, ... ]
    }
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.todo_tool import TodoStore, VALID_STATUSES

logger = logging.getLogger(__name__)

HERMES_DIR = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
TODOS_DIR = HERMES_DIR / "todos"


def _secure_dir(path: Path):
    """Set directory to owner-only access (0700). No-op on Windows."""
    try:
        os.chmod(path, 0o700)
    except (OSError, NotImplementedError):
        pass


def _secure_file(path: Path):
    """Set file to owner-only read/write (0600). No-op on Windows."""
    try:
        if path.exists():
            os.chmod(path, 0o600)
    except (OSError, NotImplementedError):
        pass


def ensure_dirs():
    """Ensure todos directory exists with secure permissions."""
    TODOS_DIR.mkdir(parents=True, exist_ok=True)
    _secure_dir(TODOS_DIR)


def _todos_file(session_id: str) -> Path:
    """Return the path to a session's todo file."""
    # Sanitize session_id for safe filesystem use
    safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_.")
    return TODOS_DIR / f"{safe_id}.json"


# =========================================================================
# File-level CRUD (used by both PersistentTodoStore and API endpoints)
# =========================================================================

def load_session_todos(session_id: str) -> Optional[Dict[str, Any]]:
    """Load a session's todo data from disk. Returns None if not found."""
    path = _todos_file(session_id)
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load todos for session %s: %s", session_id, e)
        return None


def save_session_todos(session_id: str, todos: List[Dict[str, str]], created_at: Optional[str] = None) -> Dict[str, Any]:
    """
    Atomically save a session's todos to disk.

    Returns the full stored document.
    """
    ensure_dirs()
    now = datetime.now(timezone.utc).isoformat()

    doc = {
        "session_id": session_id,
        "created_at": created_at or now,
        "updated_at": now,
        "todos": todos,
    }

    path = _todos_file(session_id)
    # Atomic write: write to temp file then rename
    fd, tmp_path = tempfile.mkstemp(dir=TODOS_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
        _secure_file(path)
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return doc


def list_all_sessions() -> List[Dict[str, Any]]:
    """
    List all sessions that have persisted todos.

    Returns a list of summary dicts (without full todo items) sorted by
    updated_at descending.
    """
    ensure_dirs()
    sessions = []
    for path in TODOS_DIR.glob("*.json"):
        try:
            with open(path, "r") as f:
                doc = json.load(f)
            todos = doc.get("todos", [])
            pending = sum(1 for t in todos if t.get("status") == "pending")
            in_progress = sum(1 for t in todos if t.get("status") == "in_progress")
            completed = sum(1 for t in todos if t.get("status") == "completed")
            cancelled = sum(1 for t in todos if t.get("status") == "cancelled")
            sessions.append({
                "session_id": doc.get("session_id", path.stem),
                "created_at": doc.get("created_at"),
                "updated_at": doc.get("updated_at"),
                "summary": {
                    "total": len(todos),
                    "pending": pending,
                    "in_progress": in_progress,
                    "completed": completed,
                    "cancelled": cancelled,
                },
            })
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping malformed todo file %s: %s", path, e)
            continue

    sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
    return sessions


def delete_session_todos(session_id: str) -> bool:
    """Delete a session's todo file. Returns True if found and deleted."""
    path = _todos_file(session_id)
    if path.exists():
        try:
            path.unlink()
            return True
        except OSError as e:
            logger.warning("Failed to delete todos for session %s: %s", session_id, e)
    return False


# =========================================================================
# PersistentTodoStore — drop-in replacement for TodoStore
# =========================================================================

class PersistentTodoStore(TodoStore):
    """
    TodoStore subclass that persists state to disk after every write.

    On init, loads any existing todos from disk for the given session_id.
    On every write(), saves the updated list back to disk.
    Read operations are served from memory (fast) and always consistent
    because writes are the only mutation path.
    """

    def __init__(self, session_id: str):
        super().__init__()
        self._session_id = session_id
        self._created_at: Optional[str] = None

        # Load existing state from disk
        existing = load_session_todos(session_id)
        if existing:
            self._created_at = existing.get("created_at")
            for item in existing.get("todos", []):
                validated = self._validate(item)
                self._items.append(validated)

    def write(self, todos: List[Dict[str, Any]], merge: bool = False) -> List[Dict[str, str]]:
        """Write todos and persist to disk."""
        result = super().write(todos, merge)
        self._persist()
        return result

    def _persist(self):
        """Save current state to disk."""
        try:
            save_session_todos(
                self._session_id,
                self.read(),
                created_at=self._created_at,
            )
        except Exception as e:
            logger.warning("Failed to persist todos for session %s: %s", self._session_id, e)
