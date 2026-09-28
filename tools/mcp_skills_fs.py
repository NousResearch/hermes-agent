"""Root-anchored filesystem I/O for MCP skill state and artifacts."""
from __future__ import annotations

import contextlib
import errno
import json
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)


def _descriptor_safety_available() -> bool:
    """Whether managed paths can stay anchored to unfollowed directory descriptors."""
    return (
        os.name == "posix"
        and bool(_NOFOLLOW)
        and bool(_DIRECTORY)
        and all(fn in os.supports_dir_fd for fn in (os.open, os.mkdir, os.unlink))
    )


def _require_descriptor_safety() -> None:
    if not _descriptor_safety_available():
        raise RuntimeError(
            "managed MCP skill files require POSIX root-descriptor and no-follow filesystem semantics")


def _relative_parts(relative: str | Path) -> tuple[str, ...]:
    text = Path(relative).as_posix()
    pure = PurePosixPath(text)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("managed path must be normalized and relative")
    return tuple(pure.parts)


def _root_path(root: str | Path) -> Path:
    path = Path(root).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"managed root does not exist: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"managed root is not a real directory: {path}")
    return path


def validate_managed_root(root: str | Path) -> Path:
    """Validate a configured managed root without resolving a symlink alias."""
    return _root_path(root)


def _open_parent(root: str | Path, relative: str | Path, *, create: bool) -> tuple[int, str]:
    """Open a leaf parent by walking from an unfollowed root descriptor."""
    _require_descriptor_safety()
    root_path = _root_path(root)
    parts = _relative_parts(relative)
    flags = os.O_RDONLY | _DIRECTORY | _NOFOLLOW
    fd = os.open(root_path, flags)
    try:
        for part in parts[:-1]:
            try:
                child = os.open(part, flags, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, 0o700, dir_fd=fd)
                except FileExistsError:
                    # A sibling process may have created the same managed
                    # component after our failed open; the unfollowed reopen
                    # below validates what won the race.
                    pass
                child = os.open(part, flags, dir_fd=fd)
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ValueError(f"managed path crosses a non-directory or symlink: {part}") from exc
                raise
            os.close(fd)
            fd = child
        return fd, parts[-1]
    except Exception:
        os.close(fd)
        raise


def secure_read_bytes(root: str | Path, relative: str | Path) -> bytes:
    parent_fd, leaf = _open_parent(root, relative, create=False)
    try:
        fd = os.open(leaf, os.O_RDONLY | _NOFOLLOW, dir_fd=parent_fd)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("managed file is not a regular file")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    return b"".join(chunks)
                chunks.append(chunk)
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)


def secure_read_json(root: str | Path, relative: str | Path) -> Any:
    return json.loads(secure_read_bytes(root, relative).decode("utf-8"))


def secure_atomic_bytes(root: str | Path, relative: str | Path, raw: bytes, *, mode: int = 0o600) -> Path:
    """Atomically replace a regular leaf without following any parent or leaf symlink."""
    parent_fd, leaf = _open_parent(root, relative, create=True)
    temp_name = f".{leaf}.{os.getpid()}.{os.urandom(8).hex()}"
    try:
        try:
            existing = os.open(leaf, os.O_RDONLY | _NOFOLLOW, dir_fd=parent_fd)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            try:
                if not stat.S_ISREG(os.fstat(existing).st_mode):
                    raise ValueError("managed target is not a regular file")
            finally:
                os.close(existing)
        fd = os.open(temp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW, mode, dir_fd=parent_fd)
        try:
            view = memoryview(raw)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
            os.fchmod(fd, mode)
        finally:
            os.close(fd)
        os.replace(temp_name, leaf, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
    finally:
        try:
            os.unlink(temp_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)
    return _root_path(root).joinpath(*_relative_parts(relative))


def secure_atomic_json(root: str | Path, relative: str | Path, data: Any, *, mode: int = 0o600) -> Path:
    raw = (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    return secure_atomic_bytes(root, relative, raw, mode=mode)


@contextlib.contextmanager
def secure_file_lock(root: str | Path, relative: str | Path):
    """Hold a blocking cross-process lock at an unfollowed managed path."""
    parent_fd, leaf = _open_parent(root, relative, create=True)
    try:
        target = Path(leaf) if parent_fd == -1 else leaf
        kwargs = {} if parent_fd == -1 else {"dir_fd": parent_fd}
        try:
            fd = os.open(target, os.O_RDWR | _NOFOLLOW, **kwargs)
        except FileNotFoundError:
            try:
                # O_EXCL makes creation symlink-safe without relying on the
                # macOS O_NOFOLLOW|O_CREAT behavior for an absent leaf.
                fd = os.open(target, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600, **kwargs)
            except FileExistsError:
                fd = os.open(target, os.O_RDWR | _NOFOLLOW, **kwargs)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ValueError("managed lock is a symlink or non-file") from exc
            raise
    except Exception:
        if parent_fd != -1:
            os.close(parent_fd)
        raise
    handle = os.fdopen(fd, "r+b", closefd=True)
    try:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("managed lock is not a singly-linked regular file")
        if os.name == "posix":
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        else:
            import portalocker
            portalocker.lock(handle, portalocker.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "posix":
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            else:
                portalocker.unlock(handle)
    finally:
        handle.close()
        if parent_fd != -1:
            os.close(parent_fd)
