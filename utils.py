"""Shared utility functions for hermes-agent."""

import contextvars
import errno
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import sys
import tempfile
import threading
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, NamedTuple, Union
from urllib.parse import urlparse

import hermes_yaml as yaml

logger = logging.getLogger(__name__)


TRUTHY_STRINGS = frozenset({"1", "true", "yes", "on"})


def is_truthy_value(value: Any, default: bool = False) -> bool:
    """Coerce bool-ish values using the project's shared truthy string set."""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in TRUTHY_STRINGS
    return bool(value)


def env_var_enabled(name: str, default: str = "") -> bool:
    """Return True when an environment variable is set to a truthy value."""
    return is_truthy_value(os.getenv(name, default), default=False)


def file_signature(st: os.stat_result) -> tuple[int, int, int, int]:
    """Change-detection key for a stat result: ``(st_mtime_ns, st_size, st_ino, st_ctime_ns)``.

    mtime + size alone miss a replacement that preserves both (``cp -p``, ``rsync -t``, a tar
    restore, a script pinning the timestamp with ``os.utime``). The inode changes on an atomic
    replace and ctime cannot be backdated from user space, so the pair catches those writers.
    On Windows ``st_ino`` may be 0 and ``st_ctime_ns`` is the creation time — both stable across
    an in-place rewrite, so the key degrades to mtime + size there rather than misfiring.
    """
    return (st.st_mtime_ns, st.st_size, st.st_ino, st.st_ctime_ns)


# Filesystem timestamps tick coarsely (~4 ms; whole seconds on some filesystems), so a write in the
# same tick as a stat leaves the signature unchanged. A signature this fresh is "racily clean" (as in
# git's index) and must not be trusted as "nothing changed since".
RACY_MTIME_WINDOW_NS = 2_000_000_000


def mtime_is_racy(mtime_ns: int) -> bool:
    """True while ``mtime_ns`` is too recent to rule out a same-tick write after the stat."""
    return time.time_ns() - mtime_ns < RACY_MTIME_WINDOW_NS


def content_digest(data: bytes) -> str:
    """Short digest of ``data`` for cache keys (not a security primitive)."""
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def _frozen_config_worker(path: str | Path) -> bool:
    """True for a ``config.yaml`` read inside a worker whose config is a frozen snapshot: such a
    worker never opens a profile config file, so a cache key there stays stat-only."""
    if Path(path).name != "config.yaml":
        return False
    policy = sys.modules.get("agent.safe_worker_policy")  # binding a snapshot imports it first
    return policy is not None and policy.worker_config_snapshot() is not None


def _file_digest(path: str | Path) -> str:
    """:func:`content_digest` of the file's bytes. A file that stats but cannot be read here
    (permission, removed since the stat, a directory) gets ``<ErrorName>``, which no hex digest
    matches: the key differs from any cached one, so the caller's own read runs and meets the error
    in its normal recovery, as it did when the key was stat-only and never opened the file."""
    try:
        with open(path, "rb") as f:
            return content_digest(f.read())
    except OSError as exc:
        return f"<{type(exc).__name__}>"


def racy_content_digest(path: str | Path, sig: tuple[int, int, int, int]) -> str:
    """``""`` once the :func:`file_signature` ``sig`` has settled; while it is racy, a digest of the
    file's bytes (:func:`_file_digest`, so it never raises). A cache keyed on a stat signature appends
    it, because two same-size in-place writes in one timestamp tick share mtime, ctime, size and inode
    (kernels without multigrain timestamps). Racy is judged on the later of mtime and ctime: userland
    can restore an older mtime after a write (``cp -p``, ``rsync --inplace -t``), never ctime. Read
    before the parse it guards: a write in between then costs one re-parse instead of pinning the
    older content."""
    if not mtime_is_racy(max(sig[0], sig[3])) or _frozen_config_worker(path):
        return ""
    return _file_digest(path)


def file_cache_version(path: str | Path, st: os.stat_result | None = None, *,
                       always_digest: bool = False) -> tuple[Any, ...]:
    """Key for anything derived from a file's content: :func:`file_signature` plus a content digest.

    The digest is :func:`racy_content_digest` (empty once the file is 2 s old), which suits a cache
    re-validated on every read: a racily-stored entry costs one re-parse when the file settles.
    ``always_digest`` hashes on every call, for a signature stored and compared much later (a
    change detector whose false "changed" restarts something, or a "still the broken file" record),
    where the racy digest turning empty would read as a change. A worker with a frozen config
    snapshot never hashes a ``config.yaml`` (the key is stat-only there, as before). Raises
    ``OSError`` only from the stat (none given ``st``): a file that cannot be read gets the error's
    name as its digest (:func:`_file_digest`)."""
    sig = file_signature(Path(path).stat() if st is None else st)
    if always_digest and not _frozen_config_worker(path):
        return (*sig, _file_digest(path))
    return (*sig, racy_content_digest(path, sig))


def _preserve_file_mode(path: Path) -> int | None:
    """Permission bits of *path* if it exists, else ``None``."""
    try:
        return stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    except OSError:
        return None


def _preserve_file_owner(path: Path) -> tuple[int, int] | None:
    """Owning ``(uid, gid)`` of *path* on POSIX, else ``None``."""
    try:
        st = path.stat() if os.name == "posix" else None
    except OSError:
        return None
    return (st.st_uid, st.st_gid) if st else None


def _restore_file_metadata(path: Path, owner: tuple[int, int] | None, mode: int | None) -> None:
    """Best-effort re-apply of uid/gid and permission bits after an atomic replace.

    Docker/NAS installs often run some commands as root on a volume owned by the runtime user;
    ``os.replace`` swaps in the temp file's owner, so privileged callers chown it back. ``mkstemp``
    creates 0o600 files; without re-applying *mode* the target would inherit that and break
    volume mounts relying on broader permissions.
    """
    if owner is not None and hasattr(os, "chown"):
        with suppress(OSError):
            os.chown(path, owner[0], owner[1])
    if mode is not None:
        with suppress(OSError):
            os.chmod(path, mode)


def default_new_file_mode() -> int | None:
    """The mode ``open(path, "w")`` gives a file it has to create (``0o666 & ~umask``); ``None``
    when the umask cannot be read or on non-POSIX hosts (Windows mode bits are synthesized).

    ``mkstemp`` always creates at 0600, so publishing a *new* non-secret file through a temp
    file would tighten it to owner-only — the Docker/NAS volume-mount hazard
    :func:`_restore_file_metadata` documents. The transient mask is 0o077: a thread that opens
    a file in the read window gets a tighter file, never a looser one.
    """
    if os.name != "posix":
        return None
    try:
        current = os.umask(0o077)
        os.umask(current)
    except OSError:
        return None
    return 0o666 & ~current


def _restore_file_owner(path: Path, owner: tuple[int, int] | None) -> None:
    _restore_file_metadata(path, owner, None)


def _restore_file_mode(path: Path, mode: int | None) -> None:
    _restore_file_metadata(path, None, mode)


_IS_WINDOWS = os.name == "nt"
# Windows rename failures possibly caused by another handle on the target. CPython opens files
# without FILE_SHARE_DELETE, so ``os.replace`` onto an open file is denied with 5 ERROR_ACCESS_DENIED
# (what a held *target* handle actually reports — measured: a plain reader yields 5, NOT 32),
# 32 ERROR_SHARING_VIOLATION (the *source* temp file is held) or 33 ERROR_LOCK_VIOLATION (byte-range
# lock on the target). Ambiguous (a real ACL denial is also 5), so recovery is bounded and a
# still-failing write is re-raised unchanged rather than classified up front.
_WINDOWS_CONTENDED_REPLACE_ERRORS = frozenset({5, 32, 33})
# Retry budget for the atomic rename. A rename that wins here keeps the write fully atomic, so the
# budget covers a realistic hold (desktop auth-init holds auth.json >100 ms): ~200 ms recovered
# atomically, ~310 ms worst case. The cap matters as much as the count — gateway_state.json is
# rewritten every turn, so a permanently-held target pays the full budget per write. Jittered so
# concurrent writers don't retry in lockstep.
_REPLACE_RETRY_ATTEMPTS = 4
_REPLACE_RETRY_BASE_DELAY_S = 0.02
_REPLACE_RETRY_MAX_DELAY_S = 0.1
_CROSS_DEVICE_ERRNOS = (errno.EXDEV, errno.EBUSY)


def _is_contended_windows_replace_error(exc: OSError) -> bool:
    """Candidate-only: winerror 5 also covers a genuine ACL denial."""
    return _IS_WINDOWS and getattr(exc, "winerror", None) in _WINDOWS_CONTENDED_REPLACE_ERRORS


def _rewrite_in_place(tmp_str: str, real_path: str) -> None:
    """Overwrite *real_path* through the existing file — last resort for a still-held target.

    Not atomic (a smaller window than a copy, not none), so it runs only after the rename has
    genuinely failed. Writing through the target also preserves its ACL, which ``os.replace``
    does not (the temp file's inherited ACL wins there).
    """
    with open(tmp_str, "rb") as src:
        data = src.read()
    fd = os.open(real_path, os.O_WRONLY | getattr(os, "O_BINARY", 0))
    try:
        written = 0
        while written < len(data):
            written += os.write(fd, data[written:])
        os.ftruncate(fd, len(data))
        with suppress(OSError):
            os.fsync(fd)
    finally:
        os.close(fd)
    os.unlink(tmp_str)


def _copy_fallback(tmp_str: str, real_path: str) -> None:
    """Copy/fsync/unlink fallback for cross-device and bind-mount renames."""
    shutil.copyfile(tmp_str, real_path)
    with suppress(OSError):
        shutil.copystat(tmp_str, real_path)
    with suppress(OSError), open(real_path, "rb") as f:
        os.fsync(f.fileno())
    os.unlink(tmp_str)


def atomic_replace(tmp_path: str | Path, target: str | Path) -> str:
    """Atomically move *tmp_path* onto *target*, preserving symlinks.

    Resolves a symlink first so ``os.replace`` writes the real file in place and the symlink
    survives. Otherwise identical to ``os.replace`` unless the rename fails with EXDEV/EBUSY
    (cross-device, bind-mount, busy file: copy/fsync/unlink immediately — these never clear on
    retry) or a Windows rename contended by another open handle (winerror 5/32/33: bounded retry,
    then in-place rewrite).
    """
    target_str = str(target)
    real_path = _publish_path(target_str)
    tmp_str = str(tmp_path)
    try:
        os.replace(tmp_str, real_path)
        return real_path
    except OSError as exc:
        contended = _is_contended_windows_replace_error(exc)
        if exc.errno not in _CROSS_DEVICE_ERRNOS and not contended:
            raise
        if contended:
            # Lazy: keeps ``utils`` free of a package-level dependency on ``agent``.
            from agent.retry_utils import jittered_backoff
            for attempt in range(1, _REPLACE_RETRY_ATTEMPTS + 1):
                time.sleep(jittered_backoff(attempt, base_delay=_REPLACE_RETRY_BASE_DELAY_S, max_delay=_REPLACE_RETRY_MAX_DELAY_S))
                try:
                    os.replace(tmp_str, real_path)
                    return real_path
                except OSError as retry_exc:
                    exc = retry_exc
                    if retry_exc.errno in _CROSS_DEVICE_ERRNOS:
                        contended = False  # not contention after all — stop burning the budget
                        break
                    if not _is_contended_windows_replace_error(retry_exc):
                        raise
        logger.debug("atomic_replace: %s -> %s failed with %s; falling back to %s", tmp_str, real_path,
                     getattr(exc, "winerror", None) or errno.errorcode.get(exc.errno or 0, exc.errno),
                     "in-place rewrite" if contended else "copy")
        # The rewrite re-raises its own error, so an ACL denial is reported as such, not as contention.
        (_rewrite_in_place if contended else _copy_fallback)(tmp_str, real_path)
    return real_path


def _publish_path(target_str: str) -> str:
    """The path :func:`atomic_replace` renames onto: a symlink's real file, else the target itself."""
    return os.path.realpath(target_str) if os.path.islink(target_str) else target_str


def mkstemp_beside(target: str | Path, **kw: Any) -> tuple[int, str]:
    """``tempfile.mkstemp`` in the directory :func:`atomic_replace` will rename into.

    A temp staged next to a symlink whose target lives on another filesystem turns the publish
    rename into EXDEV, and atomic_replace's copy fallback then rewrites the file in place (torn on
    a crash). Staging beside the resolved target keeps the rename atomic. If that directory is not
    writable to us (the file itself may still be), stage beside the link instead: the save keeps
    working through the non-atomic copy fallback, exactly as before.
    """
    target_str = str(target)
    link_dir = str(Path(target_str).parent)
    stage_dir = os.path.dirname(_publish_path(target_str)) or link_dir
    try:
        return tempfile.mkstemp(dir=stage_dir, **kw)
    except PermissionError:
        if stage_dir == link_dir:
            raise
        return tempfile.mkstemp(dir=link_dir, **kw)


def fsync_directory(path: str | Path) -> None:
    """Best-effort fsync of a directory entry so a just-renamed file survives power loss.

    No-op on Windows (directories can't be opened with ``os.open``; the file fsync still applies)
    and on any OSError — durability of the directory entry is never worth failing a write that
    has already been replaced into place.
    """
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        with suppress(OSError):
            os.fsync(fd)
    finally:
        os.close(fd)


def unlink_files_older_than(directory: str | Path, pattern: str, max_age_seconds: float, *,
                            now: float | None = None) -> int:
    """Delete regular files in *directory* matching *pattern* whose mtime is more than
    *max_age_seconds* before *now* (default: the current time); return how many were removed.

    The one age-based cache sweep. Never raises: a missing directory, or an entry that cannot be
    stat'ed or unlinked, is skipped (the next sweep retries it).
    """
    cutoff = (time.time() if now is None else now) - max_age_seconds
    removed = 0
    with suppress(OSError):
        for path in Path(directory).glob(pattern):
            with suppress(OSError):
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
    return removed


def rmtree_readonly(path: str | Path, *, ignore_errors: bool = False) -> None:
    """``shutil.rmtree`` that can also delete read-only trees.

    ``shutil.rmtree`` stops at the first entry it cannot unlink.  Git marks
    loose object files read-only on Windows (``WinError 5``), and package
    installs (Nix store, deb/rpm) are copied ``r--r--r--`` into ``0555``
    directories on POSIX, where unlinking needs a writable *parent*.  Clear the
    write bit on the failing path and on its parent, then retry the exact
    operation that failed.  Only ``PermissionError`` is retried: every other
    failure keeps ``shutil.rmtree``'s semantics (and ``ignore_errors``).
    """

    def _on_error(func, fpath, exc_info):
        # ``onerror`` (3.11) passes ``exc_info``, ``onexc`` (3.12+) the exception.
        exc = exc_info[1] if isinstance(exc_info, tuple) else exc_info
        if not isinstance(exc, PermissionError):
            raise exc
        for candidate in (os.path.dirname(fpath), fpath):
            if candidate:
                with suppress(OSError):
                    os.chmod(candidate, os.stat(candidate).st_mode | stat.S_IWUSR | stat.S_IXUSR)
        func(fpath)

    try:
        try:
            shutil.rmtree(path, onexc=_on_error)
        except TypeError:  # ``onexc`` is 3.12+; 3.11 only knows ``onerror``
            shutil.rmtree(path, onerror=_on_error)
    except OSError:
        # ``ignore_errors`` still gets the read-only recovery; it only swallows
        # whatever is left after the retry.
        if not ignore_errors:
            raise


def _atomic_write(path: Path, write, *, prefix: str, encoding: str = "utf-8", mode: int | None = None,
                  preserve_owner: bool = True, binary: bool = False, fsync_dir: bool = False) -> None:
    """Temp file + fsync + :func:`atomic_replace`, then re-apply owner/mode.

    *write(f)* emits the payload into the open handle (text, or bytes when *binary*). The temp file
    is created by ``mkstemp`` — ``O_CREAT|O_EXCL`` at 0600 regardless of umask — so a secret is
    never readable at process umask, not even between create and chmod. *mode* is fchmod'd onto
    the temp fd BEFORE the replace so the target never transits through mkstemp's 0600 (fchmod is
    Unix-only; the post-replace chmod is the sole path on Windows). With no *mode* a NEW target
    gets what ``open(path, "w")`` would have given it (process umask) — the callers this replaced
    wrote at umask, and silently tightening every fresh cache/state file to 0600 breaks shared
    volume mounts; an existing target with no *mode* keeps mkstemp's bits, as before. *fsync_dir*
    also fsyncs the resolved target's parent so the rename itself is durable. The temp file is
    removed on any failure — ``BaseException`` on purpose, so KeyboardInterrupt / SystemExit still
    clean up.
    """
    # A profile delete leaves a tombstone beside its removed home.  Background
    # writers may retain that home in a context variable, so a plain mkdir here
    # would resurrect the profile before the write can fail.
    from hermes_constants import mkdir_under_hermes_home

    mkdir_under_hermes_home(path.parent)
    if mode is None and not path.exists():
        mode = default_new_file_mode()
    original_owner = _preserve_file_owner(path) if preserve_owner else None
    fd, tmp_path = mkstemp_beside(path, prefix=prefix, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb" if binary else "w", encoding=None if binary else encoding) as f:
            if mode is not None and hasattr(os, "fchmod"):
                os.fchmod(f.fileno(), mode)
            write(f)
            f.flush()
            os.fsync(f.fileno())
            # Taken from the descriptor that is about to be renamed into place: the identity of what THIS call
            # published (only paid for when a caller asked to record publications).
            identity = _identity_of_fd(f.fileno()) if _publications.get() is not None else None
        replaced = Path(atomic_replace(tmp_path, path))  # symlink-preserving actual destination
        if identity is not None:
            _note_publication(path, identity)
        _restore_file_metadata(replaced, original_owner, mode)
        if fsync_dir:
            fsync_directory(replaced.parent)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_path)
        raise


def _mode_for_write(path: Path, create_mode: int | None, preserve: bool = True) -> int | None:
    """Existing permission bits of *path* (when *preserve*), else *create_mode* for a new file."""
    mode = _preserve_file_mode(path) if preserve else None
    return mode if mode is not None or path.exists() else create_mode


def atomic_write_text(path: str | Path, content: str, *, encoding: str = "utf-8", tmp_prefix: str = ".tmp_",
                      preserve_mode: bool = False, create_mode: int | None = None, mode: int | None = None,
                      fsync_dir: bool = False) -> None:
    """Write *content* to *path* via temp file + fsync + atomic rename.

    The target is never left partially written on crash/interrupt. Shared by every destructive
    file rewrite (memory store, skill manager, agent importer, ...). *mode* forces the final
    permission bits (secret files: ``0o600``) regardless of what exists; *create_mode* applies only
    when the target is new and *preserve_mode* carries an existing file's bits and owner across.
    """
    path = Path(path)
    _atomic_write(path, lambda f: f.write(content), prefix=tmp_prefix, encoding=encoding,
                  mode=mode if mode is not None else _mode_for_write(path, create_mode, preserve=preserve_mode),
                  preserve_owner=preserve_mode, fsync_dir=fsync_dir)


# errno values meaning "this filesystem cannot hard-link" (FAT/exFAT, some SMB/FUSE mounts) — not "the name is taken".
_NO_HARDLINK_ERRNOS = frozenset({errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS, errno.EMLINK, errno.EINVAL})
_NO_HARDLINK_WINERRORS = frozenset({1, 50})  # ERROR_INVALID_FUNCTION, ERROR_NOT_SUPPORTED

# Directory descriptors (``dir_fd``) + ``O_NOFOLLOW`` let a create/cleanup pin ITS parent directory once, so a
# symlink swapped into the parent's place afterwards is never followed. Where they are missing (Windows) the
# same code runs on paths, guarded by a symlink check only: a narrower, check-then-use protection.
_PINNED_DIRS = (hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY") and os.link in os.supports_follow_symlinks
                and {os.open, os.link, os.rename, os.unlink, os.stat} <= os.supports_dir_fd)
_OPEN_FLAGS = getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)


class FileIdentity(NamedTuple):
    """The file a publish left behind: ``(st_dev, st_ino, st_size, st_mtime_ns)`` plus the SHA-256 of its bytes.
    Size and mtime can be forged by an in-place edit; the digest cannot. ``parent_dev``/``parent_ino`` name the
    directory the file was published INTO when it was pinned by descriptor (else ``None``): a path that now
    leads to a different directory is not where this file is."""
    dev: int
    ino: int
    size: int
    mtime_ns: int
    sha256: str
    parent_dev: int | None = None
    parent_ino: int | None = None


class RemovalResult(NamedTuple):
    """Outcome of :func:`unlink_if_unchanged`; truthy only when the file was removed.

    ``status``: ``removed``; ``absent`` (nothing to remove); ``changed`` (what is there is not the file we
    published, so it was KEPT — ``preserved_at`` names where, when it could not go back under its own name);
    ``quarantined`` (the file is this call's inode but its bytes cannot be vouched for, so it was moved to
    ``preserved_at`` instead of deleted); ``failed`` (could not remove it; ``error`` says why, ``preserved_at``
    names a leftover aside copy if any).
    """
    status: str
    preserved_at: str | None = None
    error: str | None = None

    def __bool__(self) -> bool:
        return self.status == "removed"


# --- who published what -------------------------------------------------------------------------------
# A cleanup may only delete a file it can prove it published. That proof has to be taken AT the publish — from
# the descriptor that was written and then linked/renamed into place — not from a later read of the path, which
# would happily "adopt" a file an independent writer swapped in the moment after the publish returned. A caller
# that needs it (the skill batch) opens ``record_publications()``; every successful atomic create/write in that
# context is noted here with the identity of the exact inode it published.
_publications: contextvars.ContextVar[dict | None] = contextvars.ContextVar("_publications", default=None)


@contextmanager
def record_publications():
    """Collect ``{Path: FileIdentity}`` for every file published by :func:`atomic_create_text` /
    :func:`atomic_write_text` (and friends) inside this context, from the publish itself."""
    log: dict = {}
    token = _publications.set(log)
    try:
        yield log
    finally:
        _publications.reset(token)


def _note_publication(path: str | Path, identity: FileIdentity) -> None:
    if (log := _publications.get()) is not None:
        log[Path(path)] = identity


def _forget_publication(path: str | Path) -> None:
    if (log := _publications.get()) is not None:
        log.pop(Path(path), None)


class _ParentDir:
    """The directory a create/cleanup works in, pinned by descriptor where the platform allows it."""

    def __init__(self, path: Path):
        self.path, self.fd, self.ident = Path(path), None, None
        if _PINNED_DIRS:
            try:
                self.fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY | _OPEN_FLAGS)
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.ENOTDIR) and self.path.is_symlink():
                    raise OSError(errno.ELOOP, f"refusing to write through {self.path}: it is a symbolic link, "
                                               "not a real directory") from exc
                raise
            st = os.fstat(self.fd)
            self.ident = (st.st_dev, st.st_ino)
        elif self.path.is_symlink():
            raise OSError(errno.ELOOP, f"refusing to write through {self.path}: it is a symbolic link, "
                                       "not a real directory")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def close(self) -> None:
        if self.fd is not None:
            with suppress(OSError):
                os.close(self.fd)
            self.fd = None

    def moved(self) -> bool:
        """True when the pinned directory is no longer what ``path`` leads to (renamed away, replaced).

        The descriptor follows the directory wherever it went, ``path`` follows the name: after a rename the
        two disagree and anything done through the descriptor lands somewhere ``path`` does not point to.
        Only answerable where a descriptor pins the directory; without one this is ``False``. It is a check,
        not a guarantee: a rename right after it is not seen.
        """
        if self.ident is None:
            return False
        try:
            st = os.lstat(self.path)
        except OSError:
            return True
        return (st.st_dev, st.st_ino) != self.ident

    def open(self, name: str, flags: int, mode: int = 0o600) -> int:
        flags |= _OPEN_FLAGS
        if self.fd is None:
            return os.open(self.path / name, flags, mode)
        return os.open(name, flags, mode, dir_fd=self.fd)

    def lstat(self, name: str) -> os.stat_result:
        if self.fd is None:
            return os.lstat(self.path / name)
        return os.stat(name, dir_fd=self.fd, follow_symlinks=False)

    def link(self, src: str, dst: str) -> None:
        if self.fd is None:
            os.link(self.path / src, self.path / dst)
        else:  # never follow a symlink that replaced ``src``: link the name itself
            os.link(src, dst, src_dir_fd=self.fd, dst_dir_fd=self.fd, follow_symlinks=False)

    def rename(self, src: str, dst: str) -> None:
        if self.fd is None:
            os.rename(self.path / src, self.path / dst)
        else:
            os.rename(src, dst, src_dir_fd=self.fd, dst_dir_fd=self.fd)

    def unlink(self, name: str) -> None:
        if self.fd is None:
            os.unlink(self.path / name)
        else:
            os.unlink(name, dir_fd=self.fd)

    def private_temp(self, prefix: str) -> tuple[str, int]:
        """A new 0600 file (``O_EXCL``, never an existing name) beside the target; ``(name, fd)``."""
        for _ in range(100):
            name = f"{prefix}{os.urandom(6).hex()}.tmp"
            try:
                return name, self.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                continue
        raise FileExistsError(errno.EEXIST, f"no free temp name beside {self.path}")


def _read_all(fd: int) -> bytes:
    chunks = []
    while chunk := os.read(fd, 1 << 16):
        chunks.append(chunk)
    return b"".join(chunks)


def _identity_of_fd(fd: int, parent: "_ParentDir | None" = None) -> FileIdentity:
    st = os.fstat(fd)
    os.lseek(fd, 0, os.SEEK_SET)
    parent_dev, parent_ino = (parent.ident or (None, None)) if parent is not None else (None, None)
    return FileIdentity(st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns,
                        hashlib.sha256(_read_all(fd)).hexdigest(), parent_dev, parent_ino)


def _identity_matches(identity: FileIdentity, st: os.stat_result, data: bytes) -> bool:
    return ((st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns) == tuple(identity[:4])
            and hashlib.sha256(data).hexdigest() == identity[4])


def _remove_if(parent: _ParentDir, name: str, matches, ambiguous=None) -> RemovalResult:
    """Remove ``name`` from ``parent`` only if ``matches(stat, bytes)`` accepts the file that is really there.

    Check-then-unlink cannot be made atomic (POSIX has no compare-and-delete), and a writer that swaps in
    its own file between the two steps would lose it. So the file is first renamed — atomically — to a
    private random name, and THAT inode is verified (inode, size, mtime and content hash): only a verified
    file is deleted; anything else is hard-linked back under its name (``link`` never overwrites, so a
    third writer that claimed the name meanwhile is not clobbered) and otherwise stays under the private
    name, reported through ``preserved_at``. A file that is not verified but for which
    ``ambiguous(stat, bytes)`` holds (same inode this call created, bytes nobody can vouch for) is NOT put
    back and NOT deleted: it stays under the private name (``quarantined``) so a loader never sees it as the
    real file and nothing is lost.

    Residual risk, unavoidable without kernel support (no POSIX compare-and-delete), bounded here and not
    closed: (1) between the verification and the final ``unlink`` of the private name, a process that
    already knows that random hidden name — or already holds the verified file open — can still replace or
    write into it, and that data goes with the unlink; (2) the name is briefly absent while the file sits
    under the private name.
    """
    try:
        st = parent.lstat(name)
    except FileNotFoundError:
        return RemovalResult("absent")
    except OSError as exc:
        return RemovalResult("failed", error=str(exc))
    if not stat.S_ISREG(st.st_mode):
        return RemovalResult("changed")  # a symlink or directory there was never ours
    aside = f".{name}.removing-{os.urandom(8).hex()}"
    try:
        parent.rename(name, aside)
    except FileNotFoundError:
        return RemovalResult("absent")
    except OSError as exc:
        return RemovalResult("failed", error=str(exc))
    aside_path = str(parent.path / aside)
    aside_stat, data, verify_error = None, b"", None
    try:
        fd = parent.open(aside, os.O_RDONLY)
        try:
            aside_stat, data = os.fstat(fd), _read_all(fd)
        finally:
            os.close(fd)
        verified = stat.S_ISREG(aside_stat.st_mode) and matches(aside_stat, data)
    except OSError as exc:  # cannot prove it is ours: keep it
        verified, verify_error = False, str(exc)
    if verified:
        try:
            parent.unlink(aside)
        except OSError as exc:
            return RemovalResult("failed", preserved_at=aside_path, error=str(exc))
        return RemovalResult("removed")
    if (ambiguous is not None and aside_stat is not None and stat.S_ISREG(aside_stat.st_mode)
            and ambiguous(aside_stat, data)):
        return RemovalResult("quarantined", preserved_at=aside_path)
    try:
        parent.link(aside, name)  # exclusive: fails if anything took the name in the meantime
    except OSError:
        return RemovalResult("changed", preserved_at=aside_path, error=verify_error)
    with suppress(OSError):
        parent.unlink(aside)
    return RemovalResult("changed", error=verify_error)


def _create_text_in_place(parent: _ParentDir, name: str, content: str, *, encoding: str,
                          mode: int | None) -> FileIdentity:
    """``O_EXCL`` create-and-write for filesystems without hard links: exclusive, but not crash-atomic."""
    fd = parent.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)  # FileExistsError for anything there
    created = os.fstat(fd)
    try:
        with os.fdopen(fd, "w+", encoding=encoding) as f:
            if mode is not None and hasattr(os, "fchmod"):
                os.fchmod(f.fileno(), mode)  # an error here is an error: never report the wrong mode as success
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
            return _identity_of_fd(f.fileno(), parent)
    except BaseException as exc:
        expected = content.encode(encoding)

        def normalized(data: bytes) -> bytes:
            return data.replace(b"\r\n", b"\n") if os.linesep != "\n" else data

        def same_inode(st: os.stat_result) -> bool:
            return (st.st_dev, st.st_ino) == (created.st_dev, created.st_ino)

        def nothing_to_lose(st: os.stat_result, data: bytes) -> bool:
            # Our inode holding either nothing or exactly the bytes we meant to write: deleting it loses nobody's
            # data, even if somebody saved the same text over it.
            return same_inode(st) and (data == b"" or normalized(data) == expected)

        # Anything else on our inode — a prefix of our text included — may be a person's in-place save, and a
        # prefix cannot tell the two apart: it is kept (quarantined), never deleted, and the error names it.
        removal = _remove_if(parent, name, nothing_to_lose, lambda st, data: same_inode(st))
        if removal.status in ("changed", "failed", "quarantined"):
            where = removal.preserved_at or str(parent.path / name)
            logger.warning("atomic_create_text: kept %s after a failed in-place create (%s)", where,
                           removal.error or removal.status)
            exc.add_note(f"A file that cannot be proven to be this call's unfinished text was kept, not deleted: "
                         f"{where}")
            exc.preserved_at = where  # type: ignore[attr-defined]  # callers put it in their own error
        raise


def _published(parent: _ParentDir, path: Path, identity: FileIdentity) -> FileIdentity:
    """A create just landed ``identity`` in the pinned ``parent``: if ``path`` no longer leads there, undo it.

    The descriptor follows a directory that was renamed after it was opened; the file would then sit in a
    directory ``path`` does not name — invisible to the caller, yet loadable by anything that scans for it.
    """
    if parent.moved():
        removal = _remove_if(parent, path.name,
                             lambda st, data: _identity_matches(identity, st, data))
        error = OSError(errno.ESTALE, f"{path.parent} was moved or replaced while {path.name} was being created, "
                                      f"so nothing was published at {path}")
        if removal.status not in ("removed", "absent"):
            where = removal.preserved_at or "the moved directory"
            error.add_note(f"the file that landed in the moved directory could not be removed and was kept: {where}")
            error.preserved_at = where  # type: ignore[attr-defined]
        raise error
    _note_publication(path, identity)
    return identity


def atomic_create_text(path: str | Path, content: str, *, encoding: str = "utf-8", tmp_prefix: str = ".tmp_",
                       mode: int | None = None) -> FileIdentity:
    """Create *path* holding *content* — and never replace anything that is already there.

    Unlike :func:`atomic_write_text` (``os.replace`` overwrites whatever is at *path*, so a file another
    writer created a moment earlier is silently lost), the content is staged in a private temp file
    beside *path* and published with ``os.link``, which fails with ``FileExistsError`` when *path*
    exists in ANY form: regular file, directory, or symlink (live or dangling — it is never followed).
    The parent directory must already exist; nothing is created or removed except this call's temp file.
    The parent is opened once without following a symlink and every step runs relative to that
    descriptor, so a symlink swapped into its place afterwards is not written through (without descriptor
    support — Windows — only a symlink check beforehand protects). A parent that IS a symlink is refused.
    A parent renamed away or replaced before the publish is detected right after it (the file is taken
    back out of the moved directory and ``OSError(ESTALE)`` is raised); a rename after that check is not.
    *mode* is applied before publication, so *path* never shows a looser or tighter mode than asked;
    ``None`` gives what ``open(path, "w")`` would (process umask).

    On a filesystem that cannot hard-link, the fallback is an ``O_EXCL`` create-and-write at *path*:
    still exclusive, but a reader can briefly see a partial file and a crash can leave one. Its failure
    cleanup deletes the unfinished file only when it is empty or byte-identical to what was to be written;
    any other content on that inode is kept under a hidden name and named in the error (``preserved_at``,
    plus an exception note), because a person's in-place save over it cannot be told from our partial write.

    Returns the :class:`FileIdentity` of the file published, taken from the publish itself, for
    :func:`unlink_if_unchanged`.
    """
    path = Path(path)
    if mode is None:
        mode = default_new_file_mode()
    with _ParentDir(path.parent) as parent:
        tmp_name, fd = parent.private_temp(tmp_prefix)
        try:
            with os.fdopen(fd, "w+", encoding=encoding) as f:
                if mode is not None and hasattr(os, "fchmod"):
                    os.fchmod(f.fileno(), mode)
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
                identity = _identity_of_fd(f.fileno(), parent)
            if mode is not None and not hasattr(os, "fchmod"):
                os.chmod(parent.path / tmp_name, mode)
            try:
                parent.link(tmp_name, path.name)
            except OSError as exc:
                if isinstance(exc, FileExistsError) or not (
                        exc.errno in _NO_HARDLINK_ERRNOS or getattr(exc, "winerror", None) in _NO_HARDLINK_WINERRORS):
                    raise
                logger.debug("atomic_create_text: no hard links for %s (%s); exclusive in-place create", path, exc)
                return _published(parent, path, _create_text_in_place(parent, path.name, content,
                                                                      encoding=encoding, mode=mode))
            return _published(parent, path, identity)
        finally:
            with suppress(OSError):
                parent.unlink(tmp_name)


def file_identity(path: str | Path) -> FileIdentity | None:
    """:class:`FileIdentity` of the regular file at *path* now (symlinks never followed), or ``None``."""
    path = Path(path)
    try:
        with _ParentDir(path.parent) as parent:
            fd = parent.open(path.name, os.O_RDONLY)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    return None
                return _identity_of_fd(fd, parent)
            finally:
                os.close(fd)
    except OSError:
        return None


def file_matches_identity(path: str | Path, identity: FileIdentity | tuple) -> bool:
    """True only if the regular file at *path* (symlinks never followed) is exactly the one *identity* names
    (inode, size, mtime and content hash). A read-only check: nothing is moved or removed."""
    path = Path(path)
    try:
        with _ParentDir(path.parent) as parent:
            fd = parent.open(path.name, os.O_RDONLY)
            try:
                st = os.fstat(fd)
                return stat.S_ISREG(st.st_mode) and _identity_matches(identity, st, _read_all(fd))
            finally:
                os.close(fd)
    except OSError:
        return False


def unlink_if_unchanged(path: str | Path, identity: FileIdentity | tuple) -> RemovalResult:
    """Remove *path* only if it still is the file :func:`atomic_create_text` returned *identity* for.

    A different file at *path* (another writer replaced ours, or edited it in place — even keeping size and
    mtime) is KEPT and reported as ``changed``; a directory that is no longer the one the file was published
    into (it was moved or replaced) is ``failed``, since the file is not where *path* points; a removal that
    cannot be done is reported as ``failed``. Nothing here raises for an OS error: the caller decides what an
    incomplete cleanup means. See :func:`_remove_if` for how the check-then-delete race is closed and what
    remains.
    """
    path = Path(path)
    try:
        parent = _ParentDir(path.parent)
    except FileNotFoundError:
        return RemovalResult("absent")
    except OSError as exc:
        return RemovalResult("failed", error=str(exc))
    with parent:
        published_in = tuple(identity[5:7]) if len(identity) >= 7 else (None, None)
        if parent.ident is not None and None not in published_in and parent.ident != published_in:
            return RemovalResult("failed", error=f"{path.parent} is no longer the directory the file was published "
                                                 "into (it was moved or replaced); the file is not at this path")
        removal = _remove_if(parent, path.name, lambda st, data: _identity_matches(identity, st, data))
    if removal.status in ("removed", "absent"):
        _forget_publication(path)
    return removal


def atomic_write_bytes(path: str | Path, content: bytes, *, tmp_prefix: str = ".tmp_",
                       mode: int | None = None, fsync_dir: bool = False) -> None:
    """Bytes variant of :func:`atomic_write_text` (encrypted blobs, key material)."""
    path = Path(path)
    _atomic_write(path, lambda f: f.write(content), prefix=tmp_prefix, binary=True, preserve_owner=False,
                  mode=mode if mode is not None else _preserve_file_mode(path), fsync_dir=fsync_dir)


def _dump_json(data: Any, f, *, indent: int | None, ensure_ascii: bool, dump_kwargs: dict) -> None:
    """``json.dump`` that survives surrogate-escaped strings.

    ``os.fsdecode`` of a non-UTF-8 filename/argv yields lone surrogates (``'\\udcff'``); a utf-8
    text handle rejects them with ``UnicodeEncodeError`` — a ValueError, which callers guarding
    ``except OSError`` never see. ``ensure_ascii=True`` escapes them as ``\\udcff`` and
    ``json.loads`` restores the identical str, so the retry round-trips; ``surrogateescape``
    would emit a raw 0xFF byte that the reader's utf-8 decode rejects. Serializing to a str first
    keeps the failure before any byte reaches the file, so no partial payload is left behind.
    """
    text = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii, **dump_kwargs)
    try:
        f.write(text)
    except UnicodeEncodeError:
        f.write(json.dumps(data, indent=indent, ensure_ascii=True, **dump_kwargs))


def atomic_json_write(
    path: str | Path, data: Any, *, indent: int = 2, mode: int | None = None,
    ensure_ascii: bool = False, fsync_dir: bool = False, **dump_kwargs: Any,
) -> None:
    """Write JSON to *path* atomically (temp file + fsync + replace).

    Surrogate-escaped strings (non-UTF-8 argv/paths) are always persisted: the write falls back
    to ``ensure_ascii=True`` escapes for that payload only, so normal content keeps its raw UTF-8
    bytes. ``mode=0o600`` is the private-credential form: the temp file is 0600 from creation
    (mkstemp), so the payload is never umask-readable.
    """
    path = Path(path)
    _atomic_write(path, lambda f: _dump_json(data, f, indent=indent, ensure_ascii=ensure_ascii, dump_kwargs=dump_kwargs),
                  prefix=f".{path.stem}_", mode=mode if mode is not None else _preserve_file_mode(path),
                  fsync_dir=fsync_dir)


def read_json_or_empty(path: str | Path) -> dict:
    """The JSON object at *path*, or ``{}`` when the file is missing, unreadable, malformed or
    not an object. The read half of every ``read → merge → atomic_json_write`` config store
    (memory-provider ``save_config``), so a corrupt sidecar degrades to defaults instead of
    taking the provider down."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))  # utf-8-sig: a Windows-editor BOM must not wipe the config
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def warn_if_credential_file_broadly_readable(path: str | Path, *, label: str = "", log: logging.Logger | None = None) -> bool:
    """Warn when a credential file is group/world-readable; True when a warning was emitted.

    Hand-made secret files (or ones older Hermes wrote without an explicit mode) commonly end up
    0o644 under the default umask; call this before loading any token/credential file. No-op on
    non-POSIX (Windows ACLs don't map onto group/other bits; st_mode there is synthesized), when
    the file is missing, or when permissions are already tight.
    """
    p = Path(path)
    try:
        file_mode = p.stat().st_mode
    except OSError:
        return False
    if os.name != "posix" or not (file_mode & (stat.S_IRGRP | stat.S_IROTH)):
        return False
    (log or logger).warning("%s%s is group/world-readable (mode 0%o) and contains secrets. Run: chmod 600 %s",
                            f"{label} " if label else "", p.name, stat.S_IMODE(file_mode), p)
    return True


def atomic_yaml_write(path: str | Path, data: Any, *, default_flow_style: bool = False, sort_keys: bool = False,
                      extra_content: str | None = None, create_mode: int | None = None) -> None:
    """Write YAML to *path* atomically (temp file + fsync + replace)."""
    path = Path(path)

    def _write(f) -> None:
        yaml.safe_dump(data, f, default_flow_style=default_flow_style, sort_keys=sort_keys)
        if extra_content:
            f.write(extra_content)

    _atomic_write(path, _write, prefix=f".{path.stem}_", mode=_mode_for_write(path, create_mode))


def _roundtrip_load(path: Path):
    """``(yaml_rt, CommentedMap)``: a ruamel round-trip loader keeping quotes/Unicode with 2-space
    indents, plus *path* loaded through it (empty map when missing/blank)."""
    from ruamel.yaml.comments import CommentedMap

    yaml_rt = yaml.roundtrip_yaml()
    data = yaml_rt.load(path.read_text(encoding="utf-8")) if path.exists() else None
    return yaml_rt, data if isinstance(data, CommentedMap) else CommentedMap(data or {})


def _roundtrip_dump(path: Path, yaml_rt, config, *, extra_content: str | None = None) -> None:
    def _write(f) -> None:
        yaml_rt.dump(config, f)
        if extra_content:
            f.write(extra_content)

    _atomic_write(path, _write, prefix=f".{path.stem}_", mode=_preserve_file_mode(path))


def atomic_roundtrip_yaml_update(path: str | Path, key_path: str, value: Any) -> None:
    """Update one dotted YAML key while preserving comments, ordering, quoting and Unicode.

    Narrower than :func:`atomic_yaml_write` on purpose: for user-edited config files where a
    single setting mutation must not disturb the rest. Still writes via temp file + atomic replace.
    ``value=None`` removes the key (a ``key: null`` leftover reads as absent everywhere but
    litters the file and diverges from whole-document writers that drop the key).
    """
    from ruamel.yaml.comments import CommentedMap
    # Honor escaped dots and prefer existing literal dotted keys (model IDs like ``glm-5.3``) over
    # blind splitting — same navigation as ``hermes config set``'s ``_set_nested``; otherwise
    # /model + TUI persistence wrote ``glm-5: {'3': ...}`` phantom siblings.
    # See #91607.
    from hermes_cli.config import _greedy_literal_match, _split_key_path

    path = Path(path)
    from hermes_constants import mkdir_under_hermes_home

    mkdir_under_hermes_home(path.parent)
    yaml_rt, config = _roundtrip_load(path)
    current = config
    keys = _split_key_path(key_path)
    i = 0
    while True:
        remaining = keys[i:]
        seg, consumed = _greedy_literal_match(dict(current), remaining) or (remaining[0], 1)
        if i + consumed == len(keys):
            if value is None:
                current.pop(seg, None)
            else:
                current[seg] = value
            break
        next_value = current.get(seg)
        if not isinstance(next_value, CommentedMap):
            if value is None:
                return  # nothing to remove under a missing/scalar parent
            next_value = CommentedMap()
            current[seg] = next_value
        current = next_value
        i += consumed
    _roundtrip_dump(path, yaml_rt, config)


# ruamel's round-trip dumper resolves plain scalars under YAML 1.2, where only true/false/null are
# reserved — so a str like "off" or "yes" is emitted unquoted. Every other config reader here
# (PyYAML, yaml.safe_load sites) parses under YAML 1.1, where on/off/yes/no are booleans: an
# unquoted ``approvals.mode: off`` would silently round-trip back as ``False``.
_YAML11_AMBIGUOUS_WORDS = frozenset({"y", "n", "yes", "no", "true", "false", "on", "off", "null", "~"})


def _rt_value(value: Any) -> Any:
    """Plain Python value → ruamel node: YAML 1.1-ambiguous strings force-quoted at every depth
    (a bare ``off`` inside a list reads back as ``False`` just like one at the top level)."""
    from ruamel.yaml.comments import CommentedMap, CommentedSeq
    from ruamel.yaml.scalarstring import DoubleQuotedScalarString

    if isinstance(value, dict):
        node = CommentedMap()
        for k, v in value.items():
            node[k] = _rt_value(v)
        return node
    if isinstance(value, (list, tuple)):
        return CommentedSeq(_rt_value(v) for v in value)
    if isinstance(value, str) and value.lower() in _YAML11_AMBIGUOUS_WORDS:
        return DoubleQuotedScalarString(value)
    return value


def atomic_roundtrip_yaml_save(path: str | Path, new_state: dict, *,
                               extra_content_on_create: str | None = None) -> None:
    """Persist a full config-state dict while preserving comments and ordering.

    THE on-disk primitive for ``config.yaml`` (production callers reach it through
    ``hermes_cli.config.atomic_config_write`` or ``atomic_config_replace``): the document is
    loaded through ruamel round-trip mode and *new_state* is merged onto it, so comments, key order, quotes, blank
    lines and readable Unicode survive. Only nodes whose value actually changed are reassigned;
    an untouched scalar or list keeps its inline comments and formatting. Keys absent from
    *new_state* are deleted ("explicit absence": ``cfg.pop(k)`` + save removes ``k`` from disk).
    ``extra_content_on_create`` (commented example blocks) is appended only when the file is
    being created — re-appending it on every rewrite is how the stock boilerplate replaced
    users' own comments (#92554).
    """
    from ruamel.yaml.comments import CommentedMap, CommentedSeq
    from hermes_cli.config import require_readable_config_before_write

    path = Path(path)
    from hermes_constants import mkdir_under_hermes_home

    mkdir_under_hermes_home(path.parent)
    require_readable_config_before_write(path)
    creating = not path.exists() or not path.read_text(encoding="utf-8").strip()
    yaml_rt, existing = _roundtrip_load(path)

    def _unchanged(current: Any, value: Any) -> bool:
        # ``True == 1`` in Python; a bool↔int flip is a real change for YAML readers.
        return current == value and isinstance(current, bool) is isinstance(value, bool)

    def _merge_seq(dst: CommentedSeq, src: list) -> None:
        # Element-wise so appending/editing one entry keeps the comments on its siblings.
        for i, value in enumerate(src):
            if i < len(dst):
                _merge_item(dst, i, value)
            else:
                dst.append(_rt_value(value))
        del dst[len(src):]

    def _merge_item(dst, key, value) -> None:
        current = dst[key]
        if isinstance(value, dict) and isinstance(current, CommentedMap):
            _merge(current, value)
        elif isinstance(value, list) and isinstance(current, CommentedSeq):
            _merge_seq(current, value)
        elif not _unchanged(current, value):
            dst[key] = _rt_value(value)
        # else: unchanged — keep the existing node and the comments/quoting attached to it

    def _merge(dst: CommentedMap, src: dict) -> None:
        for key, value in src.items():
            if key in dst:
                _merge_item(dst, key, value)
            else:
                dst[key] = _rt_value(value)
        for key in [k for k in dst if k not in src]:
            del dst[key]

    _merge(existing, new_state)
    _roundtrip_dump(path, yaml_rt, existing, extra_content=extra_content_on_create if creating else None)


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Parse JSON, returning *default* on any parse error."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return default


def fast_safe_load(stream: Any) -> Any:
    """Use the shared safe reader (which selects ruamel's C parser when available)."""
    return yaml.safe_load(stream)


_YAML_FILE_CACHE: dict = {}
_YAML_FILE_CACHE_LOCK = threading.Lock()


def load_yaml_file_readonly(path: str | Path) -> Any:
    """``fast_safe_load`` of a file, re-parsed only when its :func:`file_cache_version` changes.

    Returns the cached object itself — callers must never mutate it. Parse errors propagate and
    are not cached; a missing file raises ``FileNotFoundError`` like ``open`` does."""
    path = Path(path)
    sig = file_cache_version(path)
    key = str(path)
    with _YAML_FILE_CACHE_LOCK:
        cached = _YAML_FILE_CACHE.get(key)
        if cached is not None and cached[0] == sig:
            return cached[1]
    with open(path, encoding="utf-8") as f:
        data = fast_safe_load(f)
    with _YAML_FILE_CACHE_LOCK:
        _YAML_FILE_CACHE[key] = (sig, data)
    return data


def _env_number(key: str, default, cast):
    raw = os.getenv(key, "").strip()
    try:
        return cast(raw) if raw else default
    except (ValueError, TypeError):
        return default


def env_int(key: str, default: int = 0) -> int:
    """Read an environment variable as an integer, with fallback."""
    return _env_number(key, default, int)


def env_float(key: str, default: float = 0.0) -> float:
    """Read an environment variable as a float, with fallback."""
    return _env_number(key, default, float)


def env_bool(key: str, default: bool = False) -> bool:
    """Read an environment variable as a boolean."""
    return is_truthy_value(os.getenv(key, ""), default=default)


_PROXY_ENV_KEYS = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy")
_NO_PROXY_ENV_KEYS = ("NO_PROXY", "no_proxy")


def normalize_proxy_url(proxy_url: str | None) -> str | None:
    """Normalize proxy URLs for httpx/aiohttp: WSL/Clash export ``socks://``, httpx needs ``socks5://``."""
    candidate = str(proxy_url or "").strip()
    if candidate.lower().startswith("socks://"):
        return f"socks5://{candidate[len('socks://'):]}"
    return candidate or None


def _bare_ipv6_literal(entry: str) -> str | None:
    """The bare IPv6 literal a NO_PROXY entry should become, or None when the entry is not an
    IPv6 form httpx cannot compile (``[::1]``, ``[::1]:8080``, ``[2001:db8::1]/64``, ``::1/128``).

    httpx 0.28.1's ``get_environment_proxies`` routes bracketed entries through its wildcard
    branch (``all://*[::1]`` → ``InvalidURL: Invalid port`` at ``Client.__init__``) and cannot
    compile IPv6 CIDR bypasses at all, so both degrade to the bare literal — which httpx compiles
    into a working ``all://[<v6>]`` bypass and urllib, requests and ``agent.proxy_bypass`` all
    understand. A ``[<v6>]:port`` entry loses its port: httpx has no port-scoped IPv6 bypass form,
    and the portless literal still bypasses every port (the operator's intent superset). Bracketed
    or CIDR IPv4 is left alone (httpx compiles those forms fine).
    """
    import ipaddress

    candidate = str(entry or "").strip()
    m = (re.fullmatch(r"\[([0-9A-Fa-f:.]+)\](?:[/:].*)?", candidate)
         or re.fullmatch(r"([0-9A-Fa-f:.]+)/\d+", candidate))
    if not m:
        return None
    try:
        ip = ipaddress.ip_address(m.group(1))
    except ValueError:
        return None  # bracketed junk: httpx's wildcard branch compiles it; not ours to touch
    return m.group(1) if ip.version == 6 else None


def sanitize_no_proxy_entries(no_proxy_value: str | None) -> str:
    """Rewrite NO_PROXY entries httpx 0.28.1 cannot compile into the equivalent bare-IPv6 forms.

    Every other entry passes through verbatim, and the original string is returned byte-stable
    when nothing needed rewriting (an env without bracketed/CIDR IPv6 entries is untouched).
    """
    raw = str(no_proxy_value or "")
    entries = [part for part in re.split(r"[\s,]+", raw.strip()) if part]
    if not entries or "*" in entries:  # a wildcard bypasses everything; httpx returns no mounts
        return raw
    rewritten: list[str] = []
    for entry in entries:
        bare = _bare_ipv6_literal(entry)
        if bare is None:
            rewritten.append(entry)  # not ours to touch: IPv4, IPv4 CIDR, domains, junk
        elif bare not in rewritten:
            rewritten.append(bare)
    if rewritten == entries:
        return raw
    return ",".join(rewritten)


def normalize_proxy_env_vars() -> None:
    """Rewrite supported proxy env vars to canonical URL forms in-place."""
    for key in _PROXY_ENV_KEYS:
        value = os.getenv(key, "")
        normalized = normalize_proxy_url(value)
        if normalized and normalized != value:
            os.environ[key] = normalized
    for key in _NO_PROXY_ENV_KEYS:
        value = os.getenv(key)
        if value:
            sanitized = sanitize_no_proxy_entries(value)
            if sanitized != value:
                os.environ[key] = sanitized


def _parse_base_url(base_url: str):
    """``urlparse`` that tolerates a bare ``host[:port][/path]`` (no scheme)."""
    raw = (base_url or "").strip()
    return urlparse(raw if "://" in raw else f"//{raw}") if raw else None


def _hostname_of(parsed) -> str:
    return (parsed.hostname or "").lower().rstrip(".") if parsed else ""


def base_url_hostname(base_url: str) -> str:
    """Lowercased hostname for a base URL, or ``""`` if absent.

    Compare exact hostnames against provider hosts instead of substring-matching the raw URL:
    ``https://api.openai.com.example/v1`` or ``https://proxy.test/api.openai.com/v1`` would
    otherwise pass as native endpoints and mis-route api_mode and auth.
    """
    return _hostname_of(_parse_base_url(base_url))


def base_url_path(base_url: str) -> str:
    """Lowercased URL path without the trailing slash (``""`` for a bare host); same scheme
    tolerance as :func:`base_url_hostname` so host and path checks agree on one URL."""
    parsed = _parse_base_url(base_url)
    return (parsed.path if parsed else "").lower().rstrip("/")


def model_forces_max_completion_tokens(model: str) -> bool:
    """True for OpenAI families that reject ``max_tokens`` (HTTP 400 ``unsupported_parameter``)."""
    m = (model or "").strip().lower().rsplit("/", 1)[-1]
    return m.startswith(("gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "o4"))


def base_url_origin(base_url: str) -> tuple[str, str, int]:
    """``(scheme, hostname, effective_port)`` for a base URL; ``("", "", 0)`` on no host/bad port.

    Origin, not just host: ``https://h`` vs ``http://h`` and two ports on one host are different
    trust boundaries, so handing a bearer secret to a new URL must compare all three — hostname
    alone would authorise an HTTPS→HTTP downgrade. Port defaults to 443/80 so ``https://h``
    equals ``https://h:443``.
    """
    parsed = _parse_base_url(base_url)
    hostname = _hostname_of(parsed)
    if not hostname:
        return ("", "", 0)
    scheme = (parsed.scheme or "").lower()
    try:
        port = parsed.port
    except ValueError:  # out-of-range or non-numeric port — not a usable origin
        return ("", "", 0)
    return (scheme, hostname, {"https": 443, "http": 80}.get(scheme, 0) if port is None else port)


def base_url_host_matches(base_url: str, domain: str) -> bool:
    """True when the base URL's hostname is ``domain`` or a subdomain.

    Safer than ``domain in base_url`` (``evil.com/moonshot.ai`` / ``moonshot.ai.evil`` must not
    match). Accepts bare hosts, full URLs, and URLs with paths.
    """
    hostname = base_url_hostname(base_url)
    domain = (domain or "").strip().lower().rstrip(".")
    return bool(hostname and domain) and (hostname == domain or hostname.endswith("." + domain))
