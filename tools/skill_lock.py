"""Advisory write lock for skill operations.

Uses POSIX ``fcntl.flock`` to prevent concurrent agent turns from colliding
on the same skill file.  This is an advisory lock — processes that don't
call ``skill_write_lock`` won't be blocked.

Design assumptions:
  - Concurrency scenario: two agent turns (e.g. background curator +
    foreground agent) both call ``skill_manage(action='edit', name='foo')``
    at the same time.  Without a lock, one silently wins and the other's
    write is lost.
  - The lock file lives in the registry's ``.hub/`` directory (excluded
    from skill discovery) so it never appears as a skill.
  - Timeout is a hard limit — callers see a RuntimeError immediately rather
    than blocking forever.  5 seconds is generous for local disk ops.
  - flock is POSIX-only (Linux + macOS).  On Windows, fcntl is absent;
    the lock degrades to a no-op warning (Windows concurrency is not a
    supported use case today).
  - Lock release is guaranteed by the context manager ``finally`` block.

Usage::

    from tools.skill_lock import skill_write_lock
    from pathlib import Path

    skill_dir = Path("~/.hermes/skills/my-skill").expanduser()
    registry_path = Path("~/.hermes/skills").expanduser()

    with skill_write_lock("my-skill", registry_path):
        # safe to write SKILL.md here
        (skill_dir / "SKILL.md").write_text(content)
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import fcntl
    _FLOCK_AVAILABLE = True
except ImportError:
    # Windows — advisory locking not supported; degrade to no-op.
    fcntl = None  # type: ignore[assignment]
    _FLOCK_AVAILABLE = False


@contextmanager
def skill_write_lock(skill_name: str, registry_path: Path, timeout: int = 5):
    """Acquire an advisory POSIX flock for *skill_name* in *registry_path*.

    Args:
        skill_name: Canonical skill name (used as lock file name).
        registry_path: Root path of the registry containing the skill.
            The lock file is created under ``registry_path/.hub/``.
        timeout: Maximum seconds to wait for the lock.  Raises
            ``RuntimeError`` immediately if the lock is held (LOCK_NB),
            so this parameter is currently unused but reserved for
            a future blocking-with-timeout implementation.

    Raises:
        RuntimeError: When the lock cannot be acquired within ``timeout``
            seconds (currently: immediately, because LOCK_NB is used).

    Yields:
        None.  The lock is released when the context exits.

    Note:
        On Windows (no fcntl), this context manager is a no-op that logs
        a WARNING.  Concurrent writes on Windows are not prevented.
    """
    if not _FLOCK_AVAILABLE:
        logger.warning(
            "fcntl not available (Windows?). Advisory write lock for '%s' is a no-op. "
            "Concurrent writes to the same skill are not prevented.",
            skill_name,
        )
        yield
        return

    lock_dir = registry_path / ".hub"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = lock_dir / f"{skill_name}.lock"

    try:
        fd = open(lock_file, "w", encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not open lock file %s: %s. Proceeding without lock.", lock_file, exc)
        yield
        return

    try:
        # LOCK_EX | LOCK_NB: exclusive, non-blocking.
        # If another process holds the lock, IOError is raised immediately.
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        logger.debug("Acquired write lock for skill '%s' at %s", skill_name, lock_file)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            logger.debug("Released write lock for skill '%s'", skill_name)
    except (IOError, OSError) as exc:
        raise RuntimeError(
            f"Skill '{skill_name}' is being modified by another process. "
            f"Retry in a moment. (Lock: {lock_file})"
        ) from exc
    finally:
        fd.close()
