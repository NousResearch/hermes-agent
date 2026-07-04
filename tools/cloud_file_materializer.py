"""Materialize macOS cloud-provider files before reading them.

iCloud Drive and File Provider backed folders (OneDrive, Dropbox, etc.) can
surface placeholder files that fail with ``OSError: [Errno 11] Resource
deadlock avoided`` when read from Hermes child processes.  This helper keeps
the fix centralized: for known cloud paths, make a plain local cache copy and
return that path for read-only consumers.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_CLOUD_MARKERS = (
    "/Library/Mobile Documents/",
    "/Library/CloudStorage/",
)


def is_cloud_file_path(path: str | os.PathLike[str]) -> bool:
    """Return True when *path* is under a known macOS cloud-file location."""
    try:
        raw = os.fspath(path)
    except TypeError:
        return False
    expanded = os.path.expanduser(raw)
    return any(marker in expanded for marker in _CLOUD_MARKERS)


def _cache_path_for(path: Path) -> Path:
    """Build a stable cache path for a cloud file."""
    try:
        st = path.stat()
        fingerprint = f"{path}|{st.st_mtime_ns}|{st.st_size}"
    except OSError:
        fingerprint = str(path)
    digest = hashlib.sha256(fingerprint.encode("utf-8", "surrogateescape")).hexdigest()
    suffix = path.suffix[:32]
    return get_hermes_home() / "cache" / "cloud_files" / f"{digest}{suffix}"


def _request_icloud_download(path: Path) -> None:
    """Ask iCloud Drive to download a placeholder file, if applicable."""
    if "/Library/Mobile Documents/" not in str(path):
        return
    brctl = shutil.which("brctl")
    if not brctl:
        return
    try:
        subprocess.run(
            [brctl, "download", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        logger.debug("cloud materializer: brctl download failed for %s: %s", path, exc)


def _copy_streaming(src: Path, dst: Path) -> None:
    """Copy without macOS fcopyfile/clonefile fast paths."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    try:
        with src.open("rb") as fin, tmp.open("wb") as fout:
            shutil.copyfileobj(fin, fout, length=1024 * 1024)
        try:
            shutil.copystat(src, tmp, follow_symlinks=True)
        except OSError:
            pass
        os.replace(tmp, dst)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def materialize_for_read(path: str | os.PathLike[str], *, retries: int = 3) -> Path:
    """Return a local readable copy for cloud-provider files.

    Non-cloud paths are returned unchanged.  If materialization fails, the
    original expanded path is returned so callers preserve their existing error
    behavior.
    """
    source = Path(os.path.expanduser(os.fspath(path)))
    if not is_cloud_file_path(source):
        return source

    cache_path = _cache_path_for(source)
    try:
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return cache_path
    except OSError:
        pass

    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        _request_icloud_download(source)
        try:
            _copy_streaming(source, cache_path)
            logger.info("Materialized cloud file for read: %s -> %s", source, cache_path)
            return cache_path
        except OSError as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.35 * (attempt + 1))
        except Exception as exc:
            last_error = exc
            break

    logger.warning(
        "Could not materialize cloud file %s; falling back to original path: %s",
        source,
        last_error,
    )
    return source

