"""Per-skill mutation locks for ``skill_manage``.

Every mutation takes the profile's own per-skill lock, exactly as before. It also takes the lock of
the skill folder the write will change, in the skills root that holds that folder (the profile's own,
an ``external_dirs`` root, a shared ``create_dir``), keyed by the folder's name. Every profile whose
write resolves to one folder therefore takes one common lock, whatever name it used and whichever
profile owns the root. Origin-owned state (``_skills_dir``, ``_find_skill``, ``_resolve_skill_dir``)
is reached lazily via ``tools.skill_manager_tool`` so test patches keep working."""

import hashlib
import logging
import os
import time
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path
from typing import Optional

logger = logging.getLogger("tools.skill_manager_tool")


def _lock_file(root: Path, key: str) -> Path:
    return root / ".locks" / f"{hashlib.sha256(key.encode('utf-8', 'surrogatepass')).hexdigest()}.lock"


def _skill_lock_path(name: str) -> Path:
    """The profile's own per-skill lock file under ``<skills>/.locks/`` (same idiom as the usage
    ledger's ``.usage.json.lock``), never inside the skill dir so delete/recreate cannot unlink it
    under a waiting writer. Keyed by a digest of the basename so ``foo`` and ``category/foo`` share
    one lock and no name can hit a filesystem limit (callers validate the basename first)."""
    from tools import skill_manager_tool as _smt
    return _lock_file(_smt._skills_dir(), Path(name).name)


def _target_root_lock(name: str) -> Optional[Path]:
    """The lock of the skill folder a write to *name* changes, in the root that holds it, keyed by the
    folder's name: every profile that resolves its write to that folder (by any name form, including
    a display name, and whether the root is its own or an external one) takes the same file. A create
    uses the folder it will create. None when the lookup fails or this process cannot open the lock
    the way ``skill_file_lock`` does (read-only root, another OS user's file): the profile's own lock
    alone then applies, as before."""
    from tools import skill_manager_tool as _smt
    from tools.skill_manager_guards import _containing_skills_root
    try:
        existing = _smt._find_skill(name)  # the same lookup the mutation itself uses
        if existing:
            target = Path(existing["path"])
            root = _containing_skills_root(target)
        else:  # a create; skills.create_dir may not exist yet
            target = _smt._resolve_skill_dir(Path(name).name)
            root = target.parent
        lock = _lock_file(Path(root), target.name)
    except Exception:
        logger.debug("skill lock root lookup failed for %s; using the local lock only", name, exc_info=True)
        return None
    if _can_open_lock(lock):
        return lock
    logger.debug("cannot open %s for locking; using the local lock only for %s", lock, name)
    return None


def _can_open_lock(lock: Path) -> bool:
    """Try the access ``skill_file_lock`` needs (create the dir, open read+write; its Windows "r+"
    needs the same rights) instead of predicting it from permission bits."""
    try:
        lock.parent.mkdir(parents=True, exist_ok=True)
        with open(lock, "a+", encoding="utf-8"):
            return True
    except OSError:
        return False


def _lock_identity(path: Path) -> str:
    """One lock file can be reached by several spellings (a symlinked skills dir). ``skill_file_lock``
    is re-entrant per path string, so taking one file twice under two spellings would deadlock this
    thread on itself; locks are therefore deduplicated and ordered by their real path."""
    return os.path.realpath(path)


def _skill_lock_paths(names) -> list[Path]:
    """Every lock mutations of *names* take: each one's own lock and target-folder lock. One path per
    file (the first spelling met, so the profile's own lock keeps main's path), in real-path order."""
    if isinstance(names, str):
        names = [names]
    paths: dict[str, Path] = {}
    for name in names:
        for path in (_skill_lock_path(name), _target_root_lock(name)):
            if path is not None:
                paths.setdefault(_lock_identity(path), path)
    return [paths[k] for k in sorted(paths)]


# Every process takes its locks in real-path order, so a wait can only close a cycle when this thread
# already holds a lock that sorts AFTER the one it needs: a nested call, e.g. a lifecycle hook writing a
# skill from inside another skill's mutation. Such a lock is waited for this long, then the nested call
# fails (TimeoutError) instead of deadlocking; hooks run under suppress(), so the outer write completes.
_OUT_OF_ORDER_WAIT = 10.0


@contextmanager
def _skill_file_lock_within(lock_path: Path, timeout: float):
    """``skill_file_lock`` that gives up after *timeout* seconds (TimeoutError). Same file, open mode
    and per-thread held set, so it excludes and re-enters exactly like it. (Kept here because
    tools/skill_usage.py is already over the repo's file-size cap and may not grow.)"""
    from tools import skill_usage as su
    held = getattr(su._held_locks, "paths", None)
    if held is None:
        held = su._held_locks.paths = set()
    if (su.fcntl is None and su.msvcrt is None) or lock_path in held:
        yield
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if su.msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
        lock_path.write_text(" ", encoding="utf-8")  # msvcrt needs a non-empty byte range to lock
    deadline = time.monotonic() + timeout
    with open(lock_path, "r+" if su.msvcrt else "a+", encoding="utf-8") as fd:
        while True:
            try:
                if su.fcntl:
                    su.fcntl.flock(fd, su.fcntl.LOCK_EX | su.fcntl.LOCK_NB)
                else:
                    fd.seek(0)
                    su.msvcrt.locking(fd.fileno(), su.msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"skill lock {lock_path} is busy; it is needed out of lock order "
                                       "(a write nested in another skill write), so the nested write fails "
                                       "rather than risk a deadlock; retry it") from None
                time.sleep(0.05)
        held.add(lock_path)
        try:
            yield
        finally:
            held.discard(lock_path)
            with suppress(OSError):
                su._flock(fd, False)


def _skill_mutation_lock(name: str):
    """Exclusive locks held across one skill's whole read-modify-write; thread-re-entrant."""
    return _skill_mutation_locks([name])


@contextmanager
def _skill_mutation_locks(names):
    """Every lock of an atomic batch (or one mutation), acquired on entry in one stable path order:
    every process orders them by real path, so no two writers can wait on each other in a cycle. A
    failure part-way releases the locks already taken. A lock this thread already holds under any
    spelling (an operation inside its own batch) is not taken again: ``skill_file_lock`` is re-entrant
    only per path string, and flocking one file through a second descriptor would block on itself."""
    from tools.skill_usage import _held_locks, skill_file_lock
    held = {_lock_identity(p) for p in getattr(_held_locks, "paths", ())}
    top = max(held, default="")
    with ExitStack() as stack:
        for lock_path in _skill_lock_paths(list(names)):
            identity = _lock_identity(lock_path)
            if identity in held:
                continue
            if identity < top:  # out of order: bounded wait (see _OUT_OF_ORDER_WAIT)
                stack.enter_context(_skill_file_lock_within(lock_path, _OUT_OF_ORDER_WAIT))
            else:
                stack.enter_context(skill_file_lock(lock_path))
        yield
