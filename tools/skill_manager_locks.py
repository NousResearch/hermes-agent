"""Per-skill mutation locks for ``skill_manage``.

Every mutation takes the profile's own per-skill lock, exactly as before. It also takes one lock for
the skill folder the write will change, kept in one directory every profile of this Hermes install
shares (``<hermes root>/.skill_locks``) and named by a digest of the folder's real path. Every
profile whose write resolves to one folder therefore takes one common lock, whatever name it used,
whichever roots it configured, and whatever SKILL.md files sit above the folder (a skill nested in
another skill's folder, a folder becoming or ceasing to be a skill mid-batch); and no lock is ever
placed inside the skills tree. Origin-owned state (``_skills_dir``, ``_find_skill``,
``_resolve_skill_dir``) is reached lazily via ``tools.skill_manager_tool`` so test patches keep
working."""

import errno
import hashlib
import logging
import os
import time
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path
from typing import Optional

logger = logging.getLogger("tools.skill_manager_tool")


def _digest(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8", "surrogatepass")).hexdigest()


def _skill_lock_path(name: str) -> Path:
    """The profile's own per-skill lock file under ``<skills>/.locks/`` (same idiom as the usage
    ledger's ``.usage.json.lock``), never inside the skill dir so delete/recreate cannot unlink it
    under a waiting writer. Keyed by a digest of the basename so ``foo`` and ``category/foo`` share
    one lock and no name can hit a filesystem limit (callers validate the basename first)."""
    from tools import skill_manager_tool as _smt
    return _smt._skills_dir() / ".locks" / f"{_digest(Path(name).name)}.lock"


def _lock_target(name: str, create_target: Optional[Path] = None) -> Path:
    """The skill folder a write to *name* changes: the folder the mutation's own lookup finds, or, for
    a create, the folder it will create: *create_target* when the caller knows it (a create with a
    category: ``_resolve_skill_dir(name, category)``), else the uncategorized one. ``skills.create_dir``
    may not exist yet."""
    from tools import skill_manager_tool as _smt
    existing = _smt._find_skill(name)  # the same lookup the mutation itself uses
    if existing:
        return Path(existing["path"])
    return Path(create_target) if create_target is not None else _smt._resolve_skill_dir(Path(name).name)


def _folder_lock(target: Path) -> Path:
    """The lock of skill folder *target*: in the lock directory of this Hermes install (shared by
    all its profiles, outside every skills tree), named by a digest of the folder's real path. It
    depends on nothing but that path: not on the roots a profile lists, the name form used, or which
    folders above it hold a SKILL.md, so no write a batch makes can move it."""
    from hermes_constants import get_default_hermes_root
    key = os.path.normcase(os.path.realpath(target))
    return get_default_hermes_root() / ".skill_locks" / f"{_digest(key)}.lock"


def _target_folder_lock(name: str, create_target: Optional[Path] = None) -> Optional[Path]:
    """The folder lock of the skill folder a write to *name* changes; None when the lookup fails (the
    profile's own lock alone then applies, as before). A lock this process cannot acquire (read-only
    Hermes root, another user's lock file) is skipped when it is taken: see
    :func:`_skill_mutation_locks`."""
    try:
        return _folder_lock(_lock_target(name, create_target))
    except Exception:
        logger.debug("skill lock target lookup failed for %s; using the local lock only", name, exc_info=True)
        return None


def _lock_identity(path: Path) -> str:
    """One lock file can be reached by several spellings (a symlinked skills dir or Hermes root).
    ``skill_file_lock`` is re-entrant per path string, so taking one file twice under two spellings
    would deadlock this thread on itself; locks are therefore deduplicated and ordered by real path."""
    return os.path.realpath(path)


def _skill_lock_paths(names, create_targets: Optional[dict] = None) -> list[Path]:
    """Every lock mutations of *names* take: each one's own lock and folder lock (a create's folder
    from *create_targets*, name -> folder, when given). One path per file (the first spelling met,
    so the profile's own lock keeps main's path), in real-path order."""
    return [path for path, _ in _skill_lock_plan(names, create_targets)]


def _skill_lock_plan(names, create_targets: Optional[dict] = None) -> list[tuple[Path, bool]]:
    """:func:`_skill_lock_paths` with, per lock, whether it is only a folder lock (additive: a write
    whose folder lock cannot be acquired keeps the profile's own lock, as on main)."""
    if isinstance(names, str):
        names = [names]
    targets = create_targets or {}
    paths: dict[str, Path] = {}
    own: set[str] = set()
    for name in names:
        own_lock = _skill_lock_path(name)
        own.add(_lock_identity(own_lock))
        for path in (own_lock, _target_folder_lock(name, targets.get(name))):
            if path is not None:
                paths.setdefault(_lock_identity(path), path)
    return [(paths[k], k not in own) for k in sorted(paths)]


# Every process takes its locks in real-path order, so a wait can only close a cycle when this thread
# already holds a lock that sorts AFTER the one it needs: a nested call, e.g. a lifecycle hook writing a
# skill from inside another skill's mutation, or an operation inside a batch whose name now resolves to
# a folder the batch did not lock. Such a lock is waited for this long, then the call is refused
# (a refusal result at skill_manage's entry) instead of deadlocking.
_OUT_OF_ORDER_WAIT = 10.0

# What a non-blocking lock attempt reports when ANOTHER holder has the lock: fcntl.flock gives
# EWOULDBLOCK/EAGAIN (EACCES on some systems); msvcrt.locking gives EACCES, or EDEADLOCK after its own
# retries. Anything else (ENOLCK, no locking support, an open that fails) means the lock is unusable
# here, not busy.
_CONTENTION = frozenset({errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES, errno.EDEADLK})


# How one acquisition ended. A VALUE, decided where the lock attempt's error code is read: never an
# exception type that later code (a stream's cleanup, a context manager's exit) could replace.
_ACQUIRED, _BUSY, _UNUSABLE = "acquired", "busy", "unusable"

_BUSY_MESSAGE = ("skill lock {} is busy; it is needed out of lock order (a write nested in another skill "
                 "write, or a batch operation on a folder the batch did not lock), so the write is refused "
                 "rather than risk a deadlock; retry it")


class _LockBusy(TimeoutError):
    """Raised only by the :func:`_skill_mutation_locks` wrapper (tests, in-process helpers). skill_manage
    never sees it: :func:`_enter_skill_locks` returns its refusal as a value."""


def _lock_attempt(su, fd: int) -> None:
    """One non-blocking attempt on the raw descriptor *fd*; raises the platform's OSError."""
    if su.fcntl:
        su.fcntl.flock(fd, su.fcntl.LOCK_EX | su.fcntl.LOCK_NB)
        return
    if os.fstat(fd).st_size == 0:  # msvcrt locks a byte, so give the file one. O_APPEND puts it at the
        os.write(fd, b" ")  # end (never truncating), and os.write is unbuffered: nothing is written later
    os.lseek(fd, 0, os.SEEK_SET)
    su.msvcrt.locking(fd, su.msvcrt.LK_NBLCK, 1)


def _release(su, held: set, lock_path: Path, fd: int) -> None:
    """Forget, unlock and close an acquired lock; errors are swallowed (nothing is buffered)."""
    held.discard(lock_path)
    with suppress(OSError):
        if su.fcntl:
            su.fcntl.flock(fd, su.fcntl.LOCK_UN)
        else:
            os.lseek(fd, 0, os.SEEK_SET)
            su.msvcrt.locking(fd, su.msvcrt.LK_UNLCK, 1)
    with suppress(OSError):
        os.close(fd)


def _acquire_skill_lock(su, lock_path: Path, timeout: Optional[float]):
    """``(outcome, fd, error)`` for the lock file *lock_path*, through a raw unbuffered descriptor (the
    same file and lock as ``skill_file_lock``, so the two exclude each other). Non-blocking attempts are
    decided by their OS error code: one in ``_CONTENTION`` means another holder has the lock, so it is
    retried, for *timeout* seconds (None: no limit), and still held after that is ``_BUSY``; any other
    OSError (from the attempt, the msvcrt initializing byte, mkdir or open) is ``_UNUSABLE`` with that
    error. ``_ACQUIRED`` returns the locked descriptor, for :func:`_release`. (Kept here because
    tools/skill_usage.py is already over the repo's file-size cap and may not grow.)"""
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0), 0o666)
    except OSError as exc:
        return _UNUSABLE, None, exc
    deadline = None if timeout is None else time.monotonic() + timeout
    outcome = None
    try:
        while outcome is None:
            try:
                _lock_attempt(su, fd)
                return _ACQUIRED, fd, None
            except OSError as exc:
                if exc.errno not in _CONTENTION:
                    outcome = (_UNUSABLE, None, exc)
                elif deadline is not None and time.monotonic() >= deadline:
                    outcome = (_BUSY, None, None)
                else:
                    time.sleep(0.05)
    except BaseException:
        with suppress(OSError):
            os.close(fd)
        raise
    with suppress(OSError):  # never locked: nothing to unlock, and closing cannot change the outcome
        os.close(fd)
    return outcome


def _enter_skill_locks(stack: ExitStack, names, create_targets: Optional[dict] = None) -> Optional[str]:
    """Take every lock of an atomic batch (or one mutation) into *stack*, in one stable path order:
    every process orders them by real path, so no two writers can wait on each other in a cycle.
    Returns the refusal message when one stays busy past the bounded out-of-order wait (the locks
    already taken are released with *stack*), None otherwise. The ONE entry every skill_manage call
    uses, single or batch, so a busy lock refuses the call before it writes and never raises past it.

    A lock this thread already holds under any spelling (an operation inside its own batch) is not
    taken again: re-entry is per path string, and locking one file through a second descriptor would
    block on itself. The profile's own lock, in order, is taken exactly as on main
    (``skill_file_lock``). Every folder lock, and an own lock needed out of order, goes through
    :func:`_acquire_skill_lock`: busy -> refusal; unusable -> a folder lock is skipped, so the write
    keeps the profile's own lock as on main, and an own lock raises its error as on main."""
    from tools import skill_usage as su
    held = getattr(su._held_locks, "paths", None)
    if held is None:
        held = su._held_locks.paths = set()
    held_ids = {_lock_identity(p) for p in held}
    top = max(held_ids, default="")
    for lock_path, folder_only in _skill_lock_plan(list(names), create_targets):
        if _lock_identity(lock_path) in held_ids:
            continue
        wait = _OUT_OF_ORDER_WAIT if _lock_identity(lock_path) < top else None  # out of order: bounded
        if not folder_only and wait is None:
            stack.enter_context(su.skill_file_lock(lock_path))  # the profile's own lock: main's path
            continue
        if su.fcntl is None and su.msvcrt is None:
            continue  # no locking on this platform, as skill_file_lock
        outcome, fd, error = _acquire_skill_lock(su, lock_path, wait)
        if outcome == _BUSY:
            return _BUSY_MESSAGE.format(lock_path)
        if outcome == _UNUSABLE:
            if not folder_only:
                raise error
            logger.debug("cannot acquire %s; using the local lock only", lock_path, exc_info=error)
            continue
        held.add(lock_path)
        stack.callback(_release, su, held, lock_path, fd)
    return None


def _skill_mutation_lock(name: str, create_target: Optional[Path] = None):
    """Exclusive locks held across one skill's whole read-modify-write; thread-re-entrant.
    *create_target*: the folder a create will make (see :func:`_lock_target`)."""
    return _skill_mutation_locks([name], {name: create_target} if create_target is not None else None)


@contextmanager
def _skill_mutation_locks(names, create_targets: Optional[dict] = None):
    """:func:`_enter_skill_locks` as a context manager, raising _LockBusy for its refusal (for tests
    and in-process helpers; skill_manage uses _enter_skill_locks and gets the refusal as a value)."""
    with ExitStack() as stack:
        busy = _enter_skill_locks(stack, names, create_targets)
        if busy is not None:
            raise _LockBusy(busy)
        yield
