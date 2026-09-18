"""Coordinate native skill transactions within one active Hermes profile.

The public context manager lets native callers check a previously observed file
under the same lock as skill_manage and its batch rollback. Reentrancy belongs
to an OS thread, not a copied ContextVar. External editors and separate profiles
sharing an external skill directory do not participate in this profile lock.
"""

from contextlib import contextmanager
import errno
from functools import wraps
import math
import os
from pathlib import Path
import stat
import threading
import time

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None
try:
    import msvcrt
except ImportError:  # POSIX
    msvcrt = None


class SkillMutationError(RuntimeError):
    """A skill transaction could not obtain its lock or its expected preimage."""


_local = threading.local()
_open_fds = set()


def _after_fork():
    # Never inherit reentrancy into a new process. Closing an inherited fd
    # (without LOCK_UN) leaves the parent's open file description locked.
    global _local
    for fd in tuple(_open_fds):
        try:
            os.close(fd)
        except OSError:
            pass
    _open_fds.clear()
    _local = threading.local()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_after_fork)


def _check_expected(expected):
    for target, before in (expected or {}).items():
        if before is not None and not isinstance(before, bytes):
            raise SkillMutationError('skill_preimage_requires_bytes_or_none')
        path = Path(target)
        try:
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise SkillMutationError('skill_preimage_target_changed')
            actual = path.read_bytes() if path.exists() else None
        except OSError as exc:
            raise SkillMutationError('skill_preimage_unavailable') from exc
        if actual != before:
            raise SkillMutationError('skill_preimage_changed')


def _acquire(fd, timeout):
    if fcntl is None and msvcrt is None:
        raise SkillMutationError('skill_mutation_lock_unavailable')
    deadline = time.monotonic() + timeout
    if fcntl is None and os.fstat(fd).st_size == 0:
        os.write(fd, b'\0')
    while True:
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise SkillMutationError('skill_mutation_lock_unavailable') from exc
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SkillMutationError('skill_mutation_busy') from exc
            time.sleep(min(0.02, remaining))


@contextmanager
def skill_mutation_lock(*, expected=None, timeout=30.0):
    """Hold the active profile's skill write lock and verify optional preimages.

    ``expected`` maps target paths to exact bytes, or None for an absent file.
    Mismatch and a bounded lock wait raise SkillMutationError without mutation.
    Keep prepare, mutation, postcondition checks and rollback inside this scope.
    The reserved lock file must remain on disk so contenders lock the same inode.
    """
    from hermes_constants import get_hermes_home
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < 0:
        raise SkillMutationError('invalid_skill_mutation_timeout')
    home = get_hermes_home().resolve()
    key = str(home / '.skill-mutation.lock')
    held = getattr(_local, 'held', None)
    if held is None:
        _local.held = held = {}
    if key in held:
        _check_expected(expected)
        yield
        return
    try:
        home.mkdir(parents=True, exist_ok=True)
        fd = os.open(key, os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    except OSError as exc:
        raise SkillMutationError('skill_mutation_lock_unavailable') from exc
    owner_pid = os.getpid()
    _open_fds.add(fd)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise SkillMutationError('invalid_skill_mutation_lock_file')
        if hasattr(os, 'fchmod'):
            os.fchmod(fd, 0o600)
        _acquire(fd, timeout)
        held[key] = fd
        _check_expected(expected)
        yield
    finally:
        if os.getpid() == owner_pid:
            held.pop(key, None)
            _open_fds.discard(fd)
            os.close(fd)


def serialized_skill_mutation(handler):
    """Keep a JSON tool handler and its signature behind the native shared lock."""
    @wraps(handler)
    def guarded(*args, **kwargs):
        try:
            with skill_mutation_lock():
                return handler(*args, **kwargs)
        except SkillMutationError as exc:
            from tools.registry import tool_error
            return tool_error(str(exc), success=False)
    # Callers can detect an upstream checkout that restored the entrypoints
    # while leaving this local helper behind. This is not a tool-schema field.
    guarded._skill_mutation_lock_version = 1
    return guarded
