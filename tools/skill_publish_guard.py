"""Minimal cross-process serialization for live skill publication.

Scope (Hermes 0.20.5 A1G+A1B minimal re-derivation):
  - One canonical-name-derived file lock under ~/.hermes/locks/.
  - Bounded non-blocking acquisition (POSIX fcntl / Windows msvcrt).
  - Context manager that releases on exit; fd-close also releases on POSIX,
    explicit LK_UNLCK on Windows.
  - Lock-acquisition failures surface as ``SkillPublishLockError`` with a
    stable ``kind`` discriminator (``CONTENTION`` vs
    ``HARD_ACQUISITION_FAILURE``) and the original cause so the caller can
    distinguish retryable contention from hard infrastructure failure.

Deliberately NOT ported from the historical 1830-line implementation:
  - separate per-target + global-name lock namespaces (one canonical-name
    lock is sufficient for the create path)
  - multi-stage failure taxonomy (one CONTENTION-vs-HARD distinction
    covers the create-path contract)
  - repair-guard, replacement_policy literal, duplicate-scan API
    (those served the broader Phase C architecture freeze; the create
    path doesn't exercise them)

The lock file lives OUTSIDE every skills root by construction
(~/.hermes/locks/), so external/read-only roots are never mutated as a
side effect of publication protection.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Iterator, Optional

# fcntl is Unix-only; on Windows use msvcrt for file locking. Neither import
# may be unconditional: ``fcntl`` does not exist on Windows (Tier-1 platform)
# and ``msvcrt`` does not exist on POSIX, so an unconditional import of
# either makes this module unimportable on half the supported hosts. Same
# shape as ``cron/scheduler.py``, which is the in-tree precedent for this
# lock. Both names stay module-level (and None on the other platform) so the
# platform branch is a cheap truthiness check and so tests can patch the
# primitive they care about.
try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None
try:
    import msvcrt
except ImportError:  # pragma: no cover - exercised on POSIX
    msvcrt = None

from hermes_constants import get_hermes_home


_logger = logging.getLogger("tools.skill_publish_guard")

_LOCK_NAMESPACE = "hermes-skill-publish-v1"
_LOCK_DIR_NAME = "locks"

# Failure classification for SkillPublishLockError. Stable string literals
# so caller payloads can surface them directly. CONTENTION is short-term
# retryable; HARD_ACQUISITION_FAILURE is not — the user-facing error
# wording must NOT recommend an immediate retry for HARD.
LOCK_KIND_CONTENTION = "CONTENTION"
LOCK_KIND_HARD = "HARD_ACQUISITION_FAILURE"


class SkillPublishLockError(PermissionError):
    """Raised when the canonical-name publication lock cannot be acquired.

    Distinct from a normal duplicate-name refusal so callers can surface
    a meaningful error instead of silently converting contention into
    "skill already exists". The ``kind`` attribute distinguishes
    ``CONTENTION`` (retryable; another process holds the lock right now)
    from ``HARD_ACQUISITION_FAILURE`` (NOT retryable in the short term —
    e.g. EIO from the kernel, EACCES opening the lock file). The legacy
    ``lock_acquisition_failure`` attribute is preserved for backward
    compatibility with existing callers and tests.
    """

    def __init__(
        self,
        canonical_name: str,
        lock_path: Path,
        cause: BaseException,
        kind: str,
    ):
        self.canonical_name = canonical_name
        self.lock_path = lock_path
        self.lock_acquisition_failure = True
        self.kind = kind
        self.cause_exception = cause
        if kind == LOCK_KIND_CONTENTION:
            base = (
                f"could not acquire skill publication lock for "
                f"{canonical_name!r} at {lock_path}: another publisher "
                f"holds the lock (retry shortly)"
            )
        else:
            base = (
                f"could not acquire skill publication lock for "
                f"{canonical_name!r} at {lock_path}: hard infrastructure "
                f"failure ({type(cause).__name__}: {cause})"
            )
        super().__init__(base)


def _lock_root() -> Path:
    """Return the directory holding publication lock files.

    Lives outside every skills root and outside external_dirs by
    construction (HERMES_HOME/locks), so it cannot be reached by a
    poisoned skills-tree symlink and cannot collide with content a
    publisher might legitimately write.
    """
    root = get_hermes_home() / _LOCK_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def _canonical_lock_path(canonical_name: str) -> Path:
    """Derive a stable lock path from the canonical skill name.

    The canonical name must already pass the L1 strict regex
    ``^[a-z0-9][a-z0-9._-]*$`` enforced by
    ``tools.skill_manager_tool._validate_name``. We do not re-normalize
    here — the caller has already done that.
    """
    digest = hashlib.sha256(
        f"{_LOCK_NAMESPACE}\0{canonical_name}".encode("utf-8")
    ).hexdigest()[:32]
    return _lock_root() / f"skill-publish-{digest}.lock"


def _classify_flock_failure(exc: OSError) -> str:
    """Map an OSError from the lock-acquire syscall to a kind.

    POSIX ``fcntl.flock(LOCK_EX | LOCK_NB)`` reports EWOULDBLOCK / EAGAIN
    when another publisher holds the lock. Windows
    ``msvcrt.locking(LK_NBLCK)`` reports EACCES / EDEADLK for the same
    condition. Those are CONTENTION (another publisher holds the lock and we
    cannot wait — the caller decides whether to retry).

    Everything else is a hard infrastructure failure (EIO, ENOMEM, ENOSPC,
    ENOSYS, EMFILE/ENFILE, ...) and must NOT be reported as retryable: a
    tight retry loop cannot fix a hard fault, it just spins. The per-platform
    errno sets follow ``cron/scheduler.py::_is_lock_contention_errno``, the
    in-tree taxonomy precedent for exactly this lock primitive.
    """
    if _use_msvcrt():
        return (
            LOCK_KIND_CONTENTION
            if exc.errno in (errno.EACCES, errno.EDEADLK)
            else LOCK_KIND_HARD
        )
    if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
        return LOCK_KIND_CONTENTION
    return LOCK_KIND_HARD


def _use_msvcrt() -> bool:
    """True when the Windows locking primitive is the one to use.

    ``fcntl`` is preferred whenever it is importable so POSIX behaviour is
    byte-for-byte unchanged; ``msvcrt`` is used only when ``fcntl`` is
    genuinely absent (Windows) and ``msvcrt`` is present. Reads the
    module-level bindings rather than caching a boolean so a test that
    patches them is honoured.
    """
    return fcntl is None and msvcrt is not None


def _acquire_lock(fd: int) -> None:
    """Take the exclusive publication lock on *fd*, non-blocking.

    Raises OSError on failure; the caller classifies it via
    ``_classify_flock_failure``. Never blocks, never retries.

    On Windows ``msvcrt.locking`` is a BYTE-RANGE lock relative to the
    CURRENT file position, unlike ``flock`` which is whole-file. Two
    consequences the POSIX path does not have:
      - a zero-length lock file has no byte to lock, so we materialise byte 0
        first (same as ``hermes_cli/managed_uv.py::_acquire_repair_lock``);
      - the descriptor must be seeked to 0 before locking, or two publishers
        could lock disjoint ranges and both believe they won.
    """
    if _use_msvcrt():
        if os.fstat(fd).st_size == 0:
            os.write(fd, b"\0")
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        return
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_lock(fd: int) -> None:
    """Release the publication lock on *fd*.

    Raises OSError on failure; ``_safe_unlock`` turns that into an observable
    secondary diagnostic (MF4) rather than letting it mask the body.

    Windows requires an EXPLICIT unlock — relying on fd close is not the
    documented contract — and the unlock must cover the same one-byte range
    at the same offset as the acquire, so we reposition first.
    """
    if _use_msvcrt():
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return
    fcntl.flock(fd, fcntl.LOCK_UN)


def _safe_unlock(fd: int, lock_path: Path) -> Optional[BaseException]:
    """Release the publication lock, never masking the caller's outcome.

    Returns the unlock OSError if one was raised, otherwise None. The
    caller is responsible for attaching the returned exception to the
    primary outcome (chained cause, logger.warning, etc.) so the release
    failure is observable without masking the body's success/failure.
    """
    try:
        _release_lock(fd)
        return None
    except OSError as exc:
        # POSIX fcntl releases on fd close regardless, so the worst case
        # is transient contention on the next acquirer. On Windows the
        # explicit unlock IS the release, so a failure here is more
        # serious. Either way we MUST surface it as a secondary
        # diagnostic rather than swallowing it.
        _logger.warning(
            "skill publication lock release (unlock) failed for %s: %s",
            lock_path, exc, exc_info=exc,
        )
        return exc


def _safe_close(fd: int, lock_path: Path) -> Optional[BaseException]:
    """Close the guard-owned fd, never masking the caller's outcome.

    Returns the close OSError if one was raised, otherwise None. Mirrors
    ``_safe_unlock`` semantics: secondary diagnostic on failure, never
    propagate upward from cleanup. The caller is responsible for
    attaching the returned exception to the primary outcome.
    """
    try:
        os.close(fd)
        return None
    except OSError as exc:
        _logger.warning(
            "skill publication lock fd close failed for %s: %s",
            lock_path, exc, exc_info=exc,
        )
        return exc


def _attach_cleanup_error(
    primary: BaseException,
    cleanup_exc: BaseException,
) -> None:
    """Chain ``cleanup_exc`` onto ``primary`` via ``__context__``.

    We use ``__context__`` (not ``__cause__``) because the cleanup error
    is a sibling that occurred while handling ``primary`` — Python's
    implicit context chaining already populates ``__context__`` when an
    exception is raised inside an except block. We set it explicitly so
    the caller of the context manager sees the cleanup failure in the
    cause chain regardless of how the body's exception was raised.

    ``primary`` is mutated in place; ``cleanup_exc`` is preserved as-is
    so its traceback survives.
    """
    # Walk to the tail of the existing chain so we don't clobber an
    # earlier __context__ that the body's own code may have set.
    tail = primary
    while tail.__context__ is not None:
        tail = tail.__context__
    if tail is cleanup_exc:
        # Already chained (would create a cycle if we set it again).
        return
    tail.__context__ = cleanup_exc


@contextlib.contextmanager
def live_skill_publish_guard(
    canonical_name: str,
    *,
    target: Path,
) -> Iterator[None]:
    """Serialize the validate→mutate window for one canonical skill name.

    Two processes that enter this guard with the same canonical_name at
    the same time cannot both proceed; the loser observes
    ``SkillPublishLockError`` (not a duplicate-name refusal). Two
    processes with different canonical_names do not contend.

    The lock is acquired non-blocking (LOCK_NB). On acquisition failure
    we raise immediately so the caller can decide — we never sleep, wait,
    or retry. The lock is released on context-manager exit (success or
    exception); POSIX fcntl additionally releases when the file
    descriptor is closed by the GC if the holder process crashes. On
    Windows the explicit one-byte LK_UNLCK is the release.

    Failure semantics:
      - ``os.open`` for the lock file raises → ``SkillPublishLockError``
        with ``kind=HARD_ACQUISITION_FAILURE`` (infrastructure, not
        contention).
      - the acquire primitive reports a platform contention errno
        (POSIX ``EWOULDBLOCK``/``EAGAIN``; Windows ``EACCES``/``EDEADLK``)
        → ``kind=CONTENTION`` (another publisher holds it).
      - the acquire primitive raises any other OSError →
        ``kind=HARD_ACQUISITION_FAILURE``.
      - the release primitive raises during cleanup → NEVER masks the
        body's outcome; recorded as a secondary diagnostic via a module
        logger warning AND chained onto the primary exception via
        ``__context__`` (if the body raised).
      - ``os.close(fd)`` raises during cleanup → same contract as
        unlock failure.

    The ``target`` argument is the directory the publication would write
    to; it is accepted for caller-friendliness and validated downstream
    by ``tools.skill_manager_tool._validate_publish_target`` which lives
    INSIDE the guarded region (so the identity check is itself
    serialized). The guard itself does not depend on ``target`` for
    correctness.
    """
    lock_path = _canonical_lock_path(canonical_name)
    # Open in append mode so writes (none here) would not truncate.
    # We never write to the lock file; the kernel state is the lock.
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    except OSError as exc:
        # Cannot even open the lock file (parent dir missing/writable,
        # parent dir's perms denied, etc.). Surface as a structured
        # acquisition failure so callers never see this as a duplicate.
        raise SkillPublishLockError(
            canonical_name, lock_path, exc, LOCK_KIND_HARD,
        ) from exc
    # We must close ``fd`` on every exit path. Track the primary outcome
    # ourselves so we never let a cleanup failure mask it.
    body_error: Optional[BaseException] = None
    try:
        try:
            _acquire_lock(fd)
        except OSError as exc:
            kind = _classify_flock_failure(exc)
            raise SkillPublishLockError(
                canonical_name, lock_path, exc, kind,
            ) from exc
        try:
            yield
        except BaseException as body_exc:
            body_error = body_exc
            raise
    finally:
        # Release and close in deterministic order. The PRIMARY rule is:
        # never let a cleanup failure mask the body's outcome. We collect
        # both into ``release_exc`` / ``close_exc`` and chain them onto
        # the primary exception (if any) via __context__ so the release
        # failure is observable without ever replacing the primary.
        release_exc = _safe_unlock(fd, lock_path)
        close_exc = _safe_close(fd, lock_path)
        if body_error is not None:
            if release_exc is not None:
                _attach_cleanup_error(body_error, release_exc)
            if close_exc is not None:
                _attach_cleanup_error(body_error, close_exc)