"""Host-owned lifecycle state for terminal environments."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterable


class TerminalEnvironmentManager:
    """Own environment caches and serialize creation per task.

    Terminal, file, and code tools use the same instance. A backend plugin
    never receives these caches or locks.
    """

    def __init__(self) -> None:
        self.active_environments: dict[str, Any] = {}
        self.last_activity: dict[str, float] = {}
        self.lock = threading.Lock()
        self.creation_locks: dict[str, threading.Lock] = {}
        self.creation_locks_lock = threading.Lock()
        self._creation_lock_users: dict[str, int] = {}

    @contextmanager
    def lifecycle_lock(self, primary_key: str):
        """Serialize create, lookup, and cleanup for one environment key."""
        with self.creation_locks_lock:
            task_lock = self.creation_locks.setdefault(primary_key, threading.Lock())
            self._creation_lock_users[primary_key] = (
                self._creation_lock_users.get(primary_key, 0) + 1
            )
        try:
            with task_lock:
                yield
        finally:
            with self.creation_locks_lock:
                users = self._creation_lock_users.get(primary_key, 1) - 1
                if users <= 0:
                    self._creation_lock_users.pop(primary_key, None)
                    if self.creation_locks.get(primary_key) is task_lock:
                        self.creation_locks.pop(primary_key, None)
                else:
                    self._creation_lock_users[primary_key] = users

    def get(self, primary_key: str, aliases: Iterable[str] = ()) -> Any | None:
        with self.lock:
            for key in (primary_key, *aliases):
                env = self.active_environments.get(key)
                if env is not None:
                    self.last_activity[key] = time.time()
                    return env
        return None

    def get_or_create(
        self,
        primary_key: str,
        create: Callable[[], Any],
        *,
        aliases: Iterable[str] = (),
    ) -> tuple[Any, bool]:
        """Return one live environment and whether this call created it."""
        alias_keys = tuple(key for key in aliases if key and key != primary_key)
        with self.lifecycle_lock(primary_key):
            existing = self.get(primary_key, alias_keys)
            if existing is not None:
                return existing, False

            env = create()
            with self.lock:
                self.active_environments[primary_key] = env
                self.last_activity[primary_key] = time.time()
            return env, True

    def cleanup(
        self,
        primary_key: str,
        cleanup: Callable[[Any], None],
        *,
        aliases: Iterable[str] = (),
        inactive_before: float | None = None,
    ) -> bool:
        """Remove and clean one environment without racing its creation."""
        alias_keys = tuple(key for key in aliases if key and key != primary_key)
        keys = (primary_key, *alias_keys)
        with self.lifecycle_lock(primary_key):
            with self.lock:
                found_key = next(
                    (key for key in keys if key in self.active_environments),
                    None,
                )
                if found_key is None:
                    return False
                if inactive_before is not None:
                    last_used = self.last_activity.get(found_key, time.time())
                    if last_used >= inactive_before:
                        return False
                env = self.active_environments[found_key]
                previous_activity = self.last_activity.get(found_key, time.time())
                for key in keys:
                    if self.active_environments.get(key) is env:
                        self.active_environments.pop(key, None)
                        self.last_activity.pop(key, None)

            try:
                cleanup(env)
            except Exception:
                with self.lock:
                    if primary_key not in self.active_environments:
                        self.active_environments[primary_key] = env
                        self.last_activity[primary_key] = previous_activity
                raise
            return True


environment_manager = TerminalEnvironmentManager()
