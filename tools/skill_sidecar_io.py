"""Symlink-safe, durable I/O primitives for skill lifecycle sidecars."""

from __future__ import annotations

import errno
import os
import stat
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

_LOCAL_LOCK_GUARD = threading.Lock()
_LOCAL_LOCKS: dict[str, threading.Lock] = {}
_DIR_FD_SUPPORTED = (
    os.open in os.supports_dir_fd
    and os.rename in os.supports_dir_fd
    and os.unlink in os.supports_dir_fd
)


def _local_lock(skill_dir: Path, name: str) -> threading.Lock:
    key = str(skill_dir / name)
    with _LOCAL_LOCK_GUARD:
        return _LOCAL_LOCKS.setdefault(key, threading.Lock())


def _directory_flags() -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return flags


def _file_flags(flags: int) -> int:
    return flags | getattr(os, "O_NOFOLLOW", 0)


@contextmanager
def _open_skill_dir(skill_dir: Path) -> Iterator[int]:
    """Pin and identity-check the final directory before sidecar operations."""
    expected = skill_dir.stat(follow_symlinks=False)
    if not stat.S_ISDIR(expected.st_mode):
        raise OSError(f"skill path is not a directory: {skill_dir}")
    fd = os.open(skill_dir, _directory_flags())
    try:
        actual = os.fstat(fd)
        if not stat.S_ISDIR(actual.st_mode):
            raise OSError(f"skill path is not a directory: {skill_dir}")
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise OSError(f"skill directory changed during sidecar open: {skill_dir}")
        yield fd
    finally:
        os.close(fd)


def _fsync_dir(fd: int) -> None:
    try:
        os.fsync(fd)
    except OSError as exc:
        unsupported = {
            errno.EINVAL,
            getattr(errno, "ENOTSUP", errno.EINVAL),
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }
        if exc.errno not in unsupported:
            raise


def _check_single_link(fd: int, label: str) -> None:
    if os.fstat(fd).st_nlink != 1:
        raise OSError(f"refusing sidecar hard link: {label}")


def _open_child(dir_fd: int, name: str, flags: int, mode: int = 0o600) -> int:
    try:
        return os.open(name, _file_flags(flags), mode, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise OSError(f"refusing sidecar symlink: {name}") from exc
        raise


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short sidecar write")
        view = view[written:]


def _contains_marker(fd: int, marker: bytes) -> bool:
    os.lseek(fd, 0, os.SEEK_SET)
    overlap = b""
    while True:
        chunk = os.read(fd, 64 * 1024)
        if not chunk:
            return False
        combined = overlap + chunk
        if marker in combined:
            return True
        overlap = combined[-(len(marker) - 1) :] if len(marker) > 1 else b""


@contextmanager
def _process_lock(fd: int) -> Iterator[None]:
    """Exclusive advisory lock for cooperating processes on POSIX and Windows."""
    if os.name == "nt":
        import msvcrt

        if os.fstat(fd).st_size == 0:
            _write_all(fd, b"0")
            os.fsync(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)


def secure_sidecar_io_available() -> bool:
    """Return whether race-safe directory-relative sidecar I/O is supported."""
    return _DIR_FD_SUPPORTED


@contextmanager
def sidecar_lock(
    skill_dir: Path, lock_name: str, *, mode: int = 0o600
) -> Iterator[None]:
    """Hold a named thread/process lock inside a pinned skill directory."""
    if "/" in lock_name or "\\" in lock_name:
        raise ValueError("sidecar lock names must be direct children")
    if not _DIR_FD_SUPPORTED:
        raise OSError("secure dir_fd sidecar operations are unavailable")

    with _local_lock(skill_dir, lock_name), _open_skill_dir(skill_dir) as dir_fd:
        lock_fd = _open_child(dir_fd, lock_name, os.O_RDWR | os.O_CREAT, mode)
        try:
            _check_single_link(lock_fd, lock_name)
            if hasattr(os, "fchmod"):
                os.fchmod(lock_fd, mode)
            _fsync_dir(dir_fd)
            with _process_lock(lock_fd):
                yield
        finally:
            os.close(lock_fd)


def append_sidecar(
    skill_dir: Path,
    name: str,
    data: bytes,
    *,
    lock_name: str,
    mode: int = 0o600,
    dedupe_marker: bytes | None = None,
) -> Path:
    """Append one complete block under thread and cross-process locks."""
    if "/" in name or "\\" in name or "/" in lock_name or "\\" in lock_name:
        raise ValueError("sidecar names must be direct children")
    if not _DIR_FD_SUPPORTED:
        raise OSError("secure dir_fd sidecar operations are unavailable")

    with _local_lock(skill_dir, name), _open_skill_dir(skill_dir) as dir_fd:
        lock_fd = _open_child(
            dir_fd,
            lock_name,
            os.O_RDWR | os.O_CREAT,
            mode,
        )
        try:
            _check_single_link(lock_fd, lock_name)
            if hasattr(os, "fchmod"):
                os.fchmod(lock_fd, mode)
            with _process_lock(lock_fd):
                target_fd = _open_child(
                    dir_fd,
                    name,
                    os.O_RDWR | os.O_APPEND | os.O_CREAT,
                    mode,
                )
                try:
                    _check_single_link(target_fd, name)
                    if hasattr(os, "fchmod"):
                        os.fchmod(target_fd, mode)
                    original_size = os.fstat(target_fd).st_size
                    if not (
                        dedupe_marker and _contains_marker(target_fd, dedupe_marker)
                    ):
                        try:
                            _write_all(target_fd, data)
                            os.fsync(target_fd)
                        except BaseException:
                            os.ftruncate(target_fd, original_size)
                            os.fsync(target_fd)
                            raise
                finally:
                    os.close(target_fd)
                _fsync_dir(dir_fd)
        finally:
            os.close(lock_fd)
    return skill_dir / name


def read_sidecar(
    skill_dir: Path,
    name: str,
    *,
    max_bytes: int,
    tail: bool = False,
) -> tuple[Optional[bytes], bool]:
    """Read a bounded sidecar and report whether older bytes were omitted."""
    if "/" in name or "\\" in name:
        raise ValueError("sidecar names must be direct children")
    if not _DIR_FD_SUPPORTED:
        raise OSError("secure dir_fd sidecar operations are unavailable")
    try:
        with _open_skill_dir(skill_dir) as dir_fd:
            try:
                fd = _open_child(dir_fd, name, os.O_RDONLY)
            except FileNotFoundError:
                return None, False
            try:
                _check_single_link(fd, name)
                size = os.fstat(fd).st_size
                if size > max_bytes and not tail:
                    raise ValueError(f"sidecar exceeds {max_bytes:,} bytes")
                if tail and size > max_bytes:
                    os.lseek(fd, -max_bytes, os.SEEK_END)
                    return os.read(fd, max_bytes), True
                chunks: list[bytes] = []
                remaining = min(size, max_bytes)
                while remaining:
                    chunk = os.read(fd, remaining)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                return b"".join(chunks), False
            finally:
                os.close(fd)
    except FileNotFoundError:
        return None, False


def atomic_write_sidecar(
    skill_dir: Path,
    name: str,
    data: bytes,
    *,
    mode: int = 0o600,
) -> Path:
    """Atomically replace a direct-child sidecar and fsync its directory."""
    if "/" in name or "\\" in name:
        raise ValueError("sidecar names must be direct children")
    if not _DIR_FD_SUPPORTED:
        raise OSError("secure dir_fd sidecar operations are unavailable")
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    with _local_lock(skill_dir, name), _open_skill_dir(skill_dir) as dir_fd:
        fd = _open_child(
            dir_fd,
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            mode,
        )
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, mode)
            _write_all(fd, data)
            os.fsync(fd)
        except BaseException:
            os.close(fd)
            try:
                os.unlink(temporary, dir_fd=dir_fd)
            except OSError:
                pass
            raise
        else:
            os.close(fd)

        try:
            os.rename(temporary, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
            _fsync_dir(dir_fd)
        except BaseException:
            try:
                os.unlink(temporary, dir_fd=dir_fd)
            except OSError:
                pass
            raise
    return skill_dir / name
