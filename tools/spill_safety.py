"""Symlink-safe creation helpers for spill/cache files.

Spill files (oversized terminal output, hook context, subagent summaries,
web_extract full text) are written into predictable, world-discoverable
directories under ``~/.hermes``. A plain ``open(path, "w")`` /
``Path.write_text`` there follows a pre-planted symlink, letting any local
process that can write to the spill directory redirect our write onto an
arbitrary file owned by the user (``~/.bashrc``, ``authorized_keys``, ...).

Every helper here refuses symlinks by construction:

* New files are created with ``O_CREAT | O_EXCL``, which fails on ANY
  existing path — including a dangling symlink — instead of following it.
* Overwrites first remove the existing path via ``lstat`` + ``unlink``
  (deleting a link deletes the link, never its target), then re-create
  exclusively. The check-then-create pair cannot be raced into following a
  link because creation itself is exclusive.

Two privacy tiers:

* ``private=True`` (default) also forces ``0o700`` directories and
  ``0o600`` files — for spills that may hold raw, pre-redaction secrets
  (terminal output, hook context).
* ``private=False`` keeps umask-default permissions — for cache dirs that
  are bind-mounted into remote terminal backends (Docker/Modal/SSH via
  ``credential_files._CACHE_DIRS``), where a non-root container UID must
  still be able to read them (public web content, delegation summaries).

Disk failures are the caller's concern: helpers raise ``OSError`` and the
call sites keep their existing best-effort handling.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import IO

__all__ = [
    "ensure_spill_dir",
    "open_exclusive",
    "write_text_exclusive",
]

# O_NOFOLLOW is POSIX-only; harmless to omit on Windows since O_EXCL alone
# already refuses every pre-existing path there too.
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _open_exclusive_fd(path: Path, mode: int) -> int:
    """Create ``path`` atomically without following a Windows reparse point."""
    if os.name != "nt":
        return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW, mode)

    # Windows' os.open(O_EXCL) can follow a dangling symlink and create its
    # target. CreateFileW + FILE_FLAG_OPEN_REPARSE_POINT applies CREATE_NEW to
    # the link itself, preserving the same no-follow guarantee as O_NOFOLLOW.
    import ctypes
    import msvcrt
    from ctypes import wintypes

    create_file = ctypes.windll.kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        str(path),
        0x40000000,  # GENERIC_WRITE
        0,
        None,
        1,  # CREATE_NEW
        0x00000080 | 0x00200000,  # FILE_ATTRIBUTE_NORMAL | OPEN_REPARSE_POINT
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        raise ctypes.WinError()
    try:
        return msvcrt.open_osfhandle(handle, os.O_WRONLY)
    except Exception:
        ctypes.windll.kernel32.CloseHandle(handle)
        raise


def ensure_spill_dir(path: Path, *, private: bool = True) -> Path:
    """Create ``path`` (and parents) as a directory, refusing symlinks.

    With ``private=True`` the leaf directory is created ``0o700`` and an
    already-existing leaf is tightened to ``0o700``. Raises ``OSError`` if
    the leaf exists and is not a real directory (e.g. a planted symlink).
    """
    path = Path(path)
    if private:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode):
        raise OSError(f"spill dir is not a directory (symlink?): {path}")
    if private and stat.S_IMODE(st.st_mode) != 0o700:
        os.chmod(path, 0o700)
    return path


def open_exclusive(
    path: Path,
    *,
    private: bool = True,
    overwrite: bool = False,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> IO[str]:
    """Open ``path`` for writing via exclusive create; never follows a link.

    ``overwrite=True`` first unlinks an existing path (``lstat``-checked so
    only the link itself is ever removed, and real directories are refused),
    then creates exclusively — so even the overwrite path cannot be
    redirected through a symlink.
    """
    path = Path(path)
    if overwrite:
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISDIR(st.st_mode):
                raise OSError(f"refusing to overwrite a directory: {path}")
            os.unlink(path)
    mode = 0o600 if private else 0o666  # non-private honors umask
    fd = _open_exclusive_fd(path, mode)
    try:
        return os.fdopen(fd, "w", encoding=encoding, errors=errors)
    except Exception:
        os.close(fd)
        raise


def write_text_exclusive(
    path: Path,
    text: str,
    *,
    private: bool = True,
    overwrite: bool = False,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> None:
    """``Path.write_text`` equivalent that refuses to follow symlinks."""
    with open_exclusive(
        path, private=private, overwrite=overwrite, encoding=encoding, errors=errors
    ) as fh:
        fh.write(text)
