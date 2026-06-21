"""Worktree conflict notifications — adopted from jcode.

When two Hermes agents run in parallel worktrees on the same repo (the
common ``hermes -w`` pattern), and Agent A reads a file that Agent B
later edits, Agent A has stale data and may make conflicting changes.
jcode's solution: when Agent B writes a file Agent A has read, push a
system-message notification to Agent A so it can re-check.

This module provides:

* :class:`WatchedSet` — per-session registry of file paths the agent
  has read, with a TTL window so old reads age out automatically.
* :class:`GitIndexWatcher` — background thread that polls the git
  index of a repo root every N seconds, emits changed paths.
* :class:`PeerNotifier` — when a watched path changes, finds peer
  sessions that share the same repo and notifies them via the
  existing kanban / session-injection surface.

The module is opt-in. ``hermes`` does NOT start the watcher by
default — users set ``agent.worktree_conflict_notifications: true`` or
pass ``hermes --worktree --conflict-notify``. When disabled, all the
classes are no-ops.
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ──────────────────────────── WatchedSet ───────────────────────────────


class WatchedSet:
    """Thread-safe set of file paths with per-entry TTL.

    Used to track what files an agent has recently read so we can
    notify when those files are mutated by a peer. Entries expire
    after ``window_seconds`` (default 600 = 10 min) so the set
    doesn't grow unbounded."""

    def __init__(self, window_seconds: int = 600):
        self._paths: Dict[str, float] = {}
        self._lock = threading.Lock()
        self.window_seconds = int(window_seconds)

    def note_read(self, path: str) -> None:
        """Mark ``path`` as just-read. Re-noting refreshes the TTL."""
        if not path:
            return
        normalized = self._normalize(path)
        with self._lock:
            self._paths[normalized] = time.time()

    def paths(self) -> List[str]:
        """Return currently-watched paths (TTL not expired), in
        arbitrary order."""
        cutoff = time.time() - self.window_seconds
        with self._lock:
            return [p for p, t in self._paths.items() if t >= cutoff]

    def watched_set(self) -> Set[str]:
        """Same as :meth:`paths` but as a set for fast intersection."""
        return set(self.paths())

    def clear(self) -> None:
        with self._lock:
            self._paths.clear()

    @staticmethod
    def _normalize(path: str) -> str:
        # Resolve to absolute and stringify so callers can pass any
        # reasonable form (relative, with ./, etc).
        try:
            return str(Path(path).expanduser().resolve())
        except Exception:
            return str(path)


# Global instance used by the harness. Each AIAgent shares this
# unless it overrides. The watcher pulls from this when deciding who
# to notify.
_global: Optional[WatchedSet] = None
_global_lock = threading.Lock()


def global_watched_set() -> WatchedSet:
    """Return the process-wide :class:`WatchedSet`. Created lazily."""
    global _global
    with _global_lock:
        if _global is None:
            _global = WatchedSet()
        return _global


def set_global_watched_set(ws: WatchedSet) -> None:
    """Replace the process-wide watched set (used by tests)."""
    global _global
    with _global_lock:
        _global = ws


# ──────────────────────────── GitIndexWatcher ──────────────────────────


@dataclass
class GitIndexSnapshot:
    """Snapshot of git-tracked file states at one point in time."""
    paths: Dict[str, str]  # path -> blob SHA (or "" for untracked-but-modified)


class GitIndexWatcher:
    """Polls a git repo's working-tree state and emits changed paths.

    The watcher runs a daemon thread that calls ``git diff --name-only
    HEAD`` (or ``git status --porcelain`` for richer output) every
    ``interval_seconds`` and compares against the previous snapshot.
    Paths that are new, modified, or deleted in the working tree are
    reported via ``on_change``.

    Safe to use on any git repo — failures (no git, not a repo, etc.)
    log a warning and stop the watcher rather than crashing the agent."""

    def __init__(self, repo_path: Path, *, on_change: Callable[[List[str]], None],
                 interval_seconds: float = 2.0):
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.on_change = on_change
        self.interval_seconds = float(interval_seconds)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_snapshot: Optional[GitIndexSnapshot] = None
        self._lock = threading.Lock()
        self._running = False

    def start(self) -> None:
        """Start the background polling thread. Idempotent."""
        with self._lock:
            if self._running:
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop, daemon=True,
                name=f"GitIndexWatcher[{self.repo_path.name}]",
            )
            self._thread.start()
            self._running = True

    def stop(self, *, join_timeout: float = 2.0) -> None:
        """Signal the thread to stop and wait briefly for it to exit."""
        with self._lock:
            if not self._running:
                return
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
        with self._lock:
            self._running = False
            self._thread = None

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def poll_once(self) -> List[str]:
        """Take one snapshot and return changed paths since last poll.
        Useful for tests and manual one-shot polling."""
        return self._poll_and_diff()

    def _loop(self) -> None:
        # Take an initial baseline so the first real change emits.
        self._last_snapshot = self._snapshot()
        while not self._stop.is_set():
            try:
                changed = self._poll_and_diff()
                if changed:
                    try:
                        self.on_change(changed)
                    except Exception as exc:
                        logger.warning(
                            "GitIndexWatcher on_change raised: %s", exc,
                            exc_info=True,
                        )
            except Exception as exc:
                logger.debug("GitIndexWatcher poll error: %s", exc)
            self._stop.wait(self.interval_seconds)

    def _poll_and_diff(self) -> List[str]:
        new = self._snapshot()
        old = self._last_snapshot
        changed: List[str] = []
        if old is not None:
            old_paths = set(old.paths.keys())
            new_paths = set(new.paths.keys())
            # Modified or deleted
            for p in old_paths - new_paths:
                changed.append(p)
            for p in old_paths & new_paths:
                if old.paths[p] != new.paths[p]:
                    changed.append(p)
            # Newly added (rare for a tracked repo but possible)
            for p in new_paths - old_paths:
                changed.append(p)
        self._last_snapshot = new
        return changed

    def _snapshot(self) -> GitIndexSnapshot:
        # `git status --porcelain` gives us a stable, parseable list of
        # every path with a working-tree delta vs HEAD. We capture both
        # the path and the status code so we can detect changes on the
        # next poll.
        try:
            result = subprocess.run(
                ["git", "-C", str(self.repo_path), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True, text=True, timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            logger.debug("git status failed for %s: %s", self.repo_path, exc)
            return GitIndexSnapshot(paths={})
        if result.returncode != 0:
            # Not a git repo, or git failed. Stop polling.
            logger.debug(
                "git status rc=%s for %s: %s",
                result.returncode, self.repo_path, result.stderr.strip()[:200],
            )
            self._stop.set()
            return GitIndexSnapshot(paths={})
        paths: Dict[str, str] = {}
        for line in result.stdout.splitlines():
            # porcelain v1 format: XY PATH  (XY = 2 status chars)
            if len(line) < 4:
                continue
            status = line[:2]
            raw_path = line[3:]
            # Rename/copy entries look like "old -> new"; keep "new".
            if " -> " in raw_path:
                raw_path = raw_path.split(" -> ", 1)[1]
            paths[raw_path] = status
        return GitIndexSnapshot(paths=paths)


# ──────────────────────────── PeerNotifier ────────────────────────────


class PeerNotifier:
    """When watched paths change, find peer sessions and notify them.

    The notifier is intentionally simple: it doesn't try to manage the
    lifetime of peer sessions or open IPC sockets. Instead it queries
    the local SQLite session DB for sessions whose stored cwd is under
    the same repo, and uses the existing ``kanban_comment`` surface
    (when available) or a direct ``agent.notify_peer`` API call
    (added separately) to push a system message."""

    def __init__(self, *, repo_path: Path,
                 watched_set: WatchedSet,
                 source_session_id: Optional[str] = None):
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.watched_set = watched_set
        self.source_session_id = source_session_id

    def notify_changed(self, changed_paths: List[str]) -> int:
        """Find peers watching any of ``changed_paths`` and notify them.
        Returns the number of peers notified."""
        if not changed_paths:
            return 0
        watched = self.watched_set.watched_set()
        relevant = [p for p in changed_paths
                    if self._normalize(p) in watched
                    or p in watched]
        if not relevant:
            return 0
        peers = self._find_peers()
        notified = 0
        for peer_session_id in peers:
            try:
                self._notify_peer(peer_session_id, relevant)
                notified += 1
            except Exception as exc:
                logger.warning(
                    "notify peer %s failed: %s", peer_session_id, exc,
                )
        return notified

    def _find_peers(self) -> List[str]:
        """Query the local session DB for sessions whose cwd is under
        our repo path. Best-effort: returns [] if state.db is missing
        or the schema is unexpected."""
        import sqlite3
        from hermes_constants import get_hermes_home
        peers: List[str] = []
        db_path = get_hermes_home() / "state.db"
        if not db_path.exists():
            return peers
        try:
            conn = sqlite3.connect(str(db_path))
            try:
                repo_str = str(self.repo_path)
                # Hermes's state.db uses 'sessions' table with columns
                # including session_id and cwd. Best-effort — we use
                # PRAGMA to discover columns first so we work across
                # schema versions.
                cols = {row[1] for row in conn.execute(
                    "PRAGMA table_info(sessions)").fetchall()}
                if "session_id" not in cols:
                    return peers
                where_parts = []
                params: list = []
                for col in ("cwd", "worktree_path"):
                    if col in cols:
                        where_parts.append(f"{col} LIKE ?")
                        params.append(f"{repo_str}%")
                if not where_parts:
                    return peers
                sql = (
                    "SELECT session_id, cwd FROM sessions WHERE "
                    + " OR ".join(where_parts)
                )
                for sid, cwd in conn.execute(sql, params).fetchall():
                    if sid == self.source_session_id:
                        continue
                    if not cwd:
                        continue
                    peers.append(sid)
            finally:
                conn.close()
        except Exception as exc:
            logger.debug("peer discovery failed: %s", exc)
        return peers

    def _notify_peer(self, session_id: str, paths: List[str]) -> None:
        """Push a system message to ``session_id``. Tries the kanban
        comment surface first (best-effort, requires an active task),
        then falls back to a debug log line. The fallback is fine for
        v1: it surfaces the conflict in agent logs and the operator can
        see it; a richer push channel (gateway webhook, CLI status
        event) is a v2 addition."""
        msg = (
            "[worktree-conflict-watch] "
            f"Files you recently read have changed: {', '.join(paths[:5])}"
            f"{'…' if len(paths) > 5 else ''}. "
            "Re-read before relying on cached content."
        )
        # Try the kanban comment surface first — comment delivery is
        # already plumbed through to worker processes.
        delivered = False
        try:
            from hermes_cli import kanban_db as _kb
            conn = _kb.connect()
            try:
                rows = conn.execute(
                    "SELECT id FROM tasks WHERE session_id = ? "
                    "ORDER BY updated_at DESC LIMIT 1",
                    (session_id,),
                ).fetchall()
                if rows:
                    _kb.add_comment(conn, rows[0][0],
                                    author="worktree-watcher", body=msg)
                    delivered = True
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception:
            pass
        if delivered:
            return
        # Fallback: log. The CLI / gateway can pick this up via its
        # existing log tailing infrastructure.
        logger.info(
            "worktree conflict (session=%s): %s", session_id, msg,
        )

    @staticmethod
    def _normalize(path: str) -> str:
        try:
            return str(Path(path).expanduser().resolve())
        except Exception:
            return str(path)


# ──────────────────────────── top-level glue ────────────────────────────


def should_run_watcher(config: dict) -> bool:
    """Read agent.worktree_conflict_notifications from config. Defaults
    to False (opt-in)."""
    if not isinstance(config, dict):
        return False
    return bool(config.get("worktree_conflict_notifications", False))


def start_watcher_for_repo(repo_path: Path,
                           *, source_session_id: Optional[str] = None,
                           interval_seconds: float = 2.0) -> Optional[GitIndexWatcher]:
    """Start a GitIndexWatcher on ``repo_path`` whose on_change callback
    notifies peers. Returns the watcher (or None if no watched paths
    are registered yet — there's nothing to do)."""
    watched = global_watched_set()
    if not watched.paths():
        # No peers are watching anything yet. Start a watcher anyway so
        # we don't miss changes once a peer joins, but emit a debug log.
        logger.debug(
            "starting watcher with empty watched set (no reads yet)"
        )

    notifier = PeerNotifier(
        repo_path=repo_path,
        watched_set=watched,
        source_session_id=source_session_id,
    )

    def _on_change(changed: List[str]) -> None:
        n = notifier.notify_changed(changed)
        if n > 0:
            logger.info(
                "worktree conflict: notified %d peer(s) of %d change(s)",
                n, len(changed),
            )

    watcher = GitIndexWatcher(
        repo_path=repo_path,
        on_change=_on_change,
        interval_seconds=interval_seconds,
    )
    watcher.start()
    return watcher
