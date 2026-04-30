"""Shared coordination for mutable Hermes skills state.

Hermes supports multiple agent processes sharing one ``HERMES_HOME``.  Skill
files are ordinary filesystem state, so multi-process deployments need a small
coordination layer around writes and prompt-cache invalidation.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import random
import time
from typing import Iterator

from hermes_constants import get_hermes_home

try:  # pragma: no cover - Windows is not a supported production target.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


DEFAULT_LOCK_TIMEOUT_SECONDS = 120.0


def _lock_timeout_seconds() -> float:
    raw = os.getenv("HERMES_SKILLS_LOCK_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_LOCK_TIMEOUT_SECONDS
    try:
        parsed = float(raw)
    except ValueError:
        return DEFAULT_LOCK_TIMEOUT_SECONDS
    return max(1.0, parsed)


def skills_lock_path() -> Path:
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    return home / ".skills_write.lock"


def skills_epoch_path() -> Path:
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    return home / ".skills_epoch"


@contextmanager
def shared_skill_write_lock(*, timeout_seconds: float | None = None) -> Iterator[None]:
    """Serialize skill tree mutations across Hermes processes."""
    lock_path = skills_lock_path()
    timeout = _lock_timeout_seconds() if timeout_seconds is None else max(1.0, timeout_seconds)
    deadline = time.monotonic() + timeout

    with lock_path.open("a+", encoding="utf-8") as handle:
        if fcntl is None:  # pragma: no cover
            yield
            return

        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for Hermes skills lock at {lock_path}") from exc
                time.sleep(random.uniform(0.02, 0.15))

        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def mark_skills_changed() -> None:
    """Touch the shared skill epoch so other processes drop skill caches."""
    path = skills_epoch_path()
    now = time.time_ns()
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{now}.tmp")
    tmp.write_text(str(now), encoding="utf-8")
    os.replace(tmp, path)


def skills_epoch_value() -> tuple[int, int]:
    """Return a stable cache-key value for the current shared skill epoch."""
    path = skills_epoch_path()
    try:
        st = path.stat()
    except OSError:
        return (0, 0)
    return (st.st_mtime_ns, st.st_size)
