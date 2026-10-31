"""In-process write primitives for the local POSIX backend.

``write_file`` asked the backend shell three questions per write (pre-write probe, atomic write,
sha256 verification): three ``bash`` spawns, ~100 ms on a warm host before any real work.  On the
local POSIX backend the file is on this host, so ``ShellFileOperations`` answers them here instead,
under the same gate as the native read path (``_native_read_enabled``; ``HERMES_NATIVE_FILE_READ=0``
turns both off).  Semantics mirror the shell scripts they replace; anything they cannot answer
(a FIFO, a device, an unexpected error) returns None and the caller takes the shell path.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import secrets
import stat
from typing import Optional

_UTF8_BOM = b"\xef\xbb\xbf"


def _open_regular(path: str) -> Optional[int]:
    """fd of ``path`` when it is a regular file; None when missing; raises ``_NotRegular`` otherwise.
    One lookup with O_NONBLOCK, judged by fstat: a FIFO swapped in never blocks this thread."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise _NotRegular(path)
    return fd


class _NotRegular(OSError):
    pass


def probe(path: str, want_text: bool) -> Optional[tuple[bool, Optional[bytes]]]:
    """``(has_bom, body)`` for a write target: ``body`` is the whole file when ``want_text`` else its
    first 4 KB (the line-ending sample); ``(False, None)`` when the file is missing; None = ask the shell."""
    try:
        fd = _open_regular(path)
        if fd is None:
            return False, None
        with open(fd, "rb") as fh:
            data = fh.read() if want_text else fh.read(4096)
    except OSError:
        return None
    return data.startswith(_UTF8_BOM), data


def atomic_write(path: str, data: bytes) -> Optional[str]:
    """Replace ``path`` with ``data`` atomically; None on success, else an error message.

    Same contract as the shell script: a symlink target is written through (the link survives),
    parent directories are created, the temp file lives in the target's directory so the rename is
    atomic, an existing target's mode is kept and a new one gets the umask default (``O_CREAT`` with
    0o666 applies the process umask itself), and the temp is removed on every failure.

    The temp holds its final mode before the first byte lands: an existing target's temp is created
    0o600 and ``fchmod``-ed to the target's mode, so a private file's new contents are never readable
    through a wider temp (the shell script's ``mktemp`` 0600 temp gave the same guarantee)."""
    target = os.path.realpath(path) if os.path.islink(path) else path
    parent = os.path.dirname(target) or "."
    try:
        os.makedirs(parent, exist_ok=True)
        try:
            mode: Optional[int] = stat.S_IMODE(os.stat(target).st_mode)
        except FileNotFoundError:
            mode = None
        tmp = os.path.join(parent, f".hermes-tmp.{secrets.token_hex(6)}")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                     0o666 if mode is None else 0o600)
    except OSError as e:
        return f"{e.strerror or e}: {e.filename or target}"
    try:
        if mode is not None:
            try:
                os.fchmod(fd, mode)  # before any content: the bytes never sit in a wider file
            except OSError:
                pass  # best effort, as in the shell script (the temp then stays 0o600)
        with open(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, target)
    except OSError as e:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        return f"{e.strerror or e}: {e.filename or target}"
    return None


def sha256_of(path: str) -> Optional[str]:
    """Hex sha256 of the regular file at ``path``; None when it cannot be read natively."""
    try:
        fd = _open_regular(path)
        if fd is None:
            return None
        digest = hashlib.sha256()
        with open(fd, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


__all__ = ["atomic_write", "probe", "sha256_of"]
