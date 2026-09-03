"""Filesystem lifecycle operations for MCP OAuth storage.

The public ``HermesTokenStorage`` compatibility class inherits this mixin.
Lifecycle operations use the same per-server lock as normal OAuth writes,
validate every directory component, and publish restored files by replacement
from a staged directory rather than truncating live paths.
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger("tools.mcp_oauth")

_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_key(storage: Any) -> str:
    return os.path.normcase(os.path.abspath(str(storage._tokens_path().parent)))


@contextmanager
def lifecycle_lock(storage: Any) -> Iterator[None]:
    """Serialize all lifecycle and successful persistence writes per server."""
    key = _lock_key(storage)
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(key, threading.RLock())
    with lock:
        yield


def _is_reparse(st: os.stat_result) -> bool:
    attrs = getattr(st, "st_file_attributes", 0)
    return stat.S_ISLNK(st.st_mode) or bool(attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _existing_stat(path: Path) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None


def _validate_components(path: Path, *, include_leaf: bool) -> None:
    """Reject symlink/reparse components without resolving them."""
    current = path if include_leaf else path.parent
    components: list[Path] = []
    while True:
        components.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    for component in reversed(components):
        st = _existing_stat(component)
        if st is None:
            continue
        if _is_reparse(st):
            raise OSError(f"OAuth storage path contains a reparse point: {component}")
        if not stat.S_ISDIR(st.st_mode):
            raise OSError(f"OAuth storage component is not a directory: {component}")


def _assert_real_directory(path: Path) -> os.stat_result:
    st = _existing_stat(path)
    if st is None:
        raise FileNotFoundError(path)
    if _is_reparse(st) or not stat.S_ISDIR(st.st_mode):
        raise OSError(f"OAuth storage directory is not a real directory: {path}")
    return st


def _safe_token_dir(storage: Any, *, create: bool) -> Path:
    token_dir = storage._tokens_path().parent
    _validate_components(token_dir, include_leaf=False)
    if _existing_stat(token_dir) is None:
        if not create:
            return token_dir
        token_dir.mkdir(parents=True, exist_ok=True)
    _validate_components(token_dir, include_leaf=True)
    _assert_real_directory(token_dir)
    return token_dir


def _safe_file(path: Path) -> bool:
    """Validate a file path and return whether its leaf exists."""
    if not isinstance(path, (str, os.PathLike)):
        return False
    path = Path(path)
    _validate_components(path, include_leaf=False)
    st = _existing_stat(path)
    if st is None:
        return False
    if _is_reparse(st) or not stat.S_ISREG(st.st_mode):
        raise OSError(f"OAuth storage file is not a regular file: {path}")
    return True


def _read_file(path: Path) -> bytes | None:
    if not _safe_file(path):
        return None
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(str(path), flags)
    try:
        opened = os.fstat(fd)
        if _is_reparse(opened) or not stat.S_ISREG(opened.st_mode):
            raise OSError(f"OAuth storage file changed to a non-regular file: {path}")
        return os.read(fd, max(opened.st_size, 1) + 1)
    finally:
        os.close(fd)


def _write_exclusive(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(str(path), flags, stat.S_IRUSR | stat.S_IWUSR)
    try:
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if fd != -1:
            os.close(fd)


class OAuthStorageLifecycleMixin:
    """Provide the on-disk OAuth state lifecycle for ``HermesTokenStorage``."""

    def _lifecycle_paths(self) -> tuple[Path, ...]:
        client = self._client_info_path()
        return (
            self._tokens_path(),
            client,
            self._meta_path(),
            self._cimd_rejected_path(),
            client.with_name(client.name + ".bak"),
        )

    def _lifecycle_marker_path(self) -> Path:
        return self._tokens_path().parent / f"{self._server_name}.lifecycle-incomplete"

    def _begin_marker(self, operation: str) -> None:
        token_dir = _safe_token_dir(self, create=True)
        marker = self._lifecycle_marker_path()
        _validate_components(marker, include_leaf=False)
        if _safe_file(marker):
            return
        try:
            _write_exclusive(marker, f"operation={operation}\n".encode("ascii"))
        except FileExistsError:
            _safe_file(marker)

    def _clear_marker(self) -> None:
        marker = self._lifecycle_marker_path()
        if not _safe_file(marker):
            return
        marker.unlink()

    def _remove_unlocked(self, *, clear_marker: bool = True) -> None:
        paths = self._lifecycle_paths()
        marker = self._lifecycle_marker_path()
        present = [_safe_file(path) for path in paths]
        marker_present = _safe_file(marker)
        if not any(present) and not marker_present:
            return
        _safe_token_dir(self, create=True)
        self._begin_marker("remove")
        try:
            for path, exists in zip(paths, present):
                if exists:
                    path.unlink()
            if clear_marker:
                self._clear_marker()
        except BaseException:
            # The marker is intentionally retained. A caller must observe and
            # recover the incomplete state rather than report cleanup success.
            raise

    def remove(self) -> None:
        """Delete all stored OAuth state for this server, fail-closed."""
        with lifecycle_lock(self):
            self._remove_unlocked()

    def snapshot(self) -> dict[str, bytes]:
        """Capture primary OAuth files without following links or reparses."""
        with lifecycle_lock(self):
            snap: dict[str, bytes] = {}
            for path in (self._tokens_path(), self._client_info_path(), self._meta_path()):
                data = _read_file(path)
                if data is not None:
                    snap[path.name] = data
            return snap

    def restore(self, snapshot: dict[str, bytes], *, only_if_absent: bool = False) -> None:
        """Restore a snapshot through a staged, marked, non-truncating publish."""
        allowed = {path.name for path in (self._tokens_path(), self._client_info_path(), self._meta_path())}
        invalid = [name for name in snapshot if not isinstance(name, str) or name not in allowed]
        if invalid:
            raise ValueError(
                "Invalid OAuth snapshot filename(s): "
                + ", ".join(repr(name) for name in invalid)
            )

        with lifecycle_lock(self):
            if only_if_absent and any(_safe_file(path) for path in (
                self._tokens_path(), self._client_info_path(), self._meta_path()
            )):
                logger.info("Skipping OAuth rollback for %s because newer state exists", self._server_name)
                return
            if not snapshot:
                self._remove_unlocked()
                return

            token_dir = _safe_token_dir(self, create=True)
            directory_identity = _assert_real_directory(token_dir)
            stage = Path(tempfile.mkdtemp(prefix=f".{self._server_name}.restore-", dir=str(token_dir)))
            try:
                _assert_real_directory(stage)
                staged: list[tuple[Path, Path]] = []
                for name, data in snapshot.items():
                    staged_path = stage / name
                    _write_exclusive(staged_path, data)
                    staged.append((staged_path, token_dir / name))

                # Preserve the public remove() seam. The lock is re-entrant,
                # so normal and monkeypatched remove implementations remain
                # compatible while no writer can enter between the check and it.
                self.remove()
                self._begin_marker("restore")
                for staged_path, target in staged:
                    current_identity = _assert_real_directory(token_dir)
                    if current_identity.st_dev != directory_identity.st_dev or current_identity.st_ino != directory_identity.st_ino:
                        raise OSError("OAuth storage directory changed during restore")
                    _validate_components(target, include_leaf=False)
                    os.replace(staged_path, target)
                self._clear_marker()
            except BaseException as exc:
                try:
                    self._begin_marker("restore")
                except OSError:
                    logger.error("Could not mark incomplete OAuth restore for %s", self._server_name)
                raise exc
            finally:
                shutil.rmtree(stage, ignore_errors=True)

    def poison_client_registration(self) -> bool:
        """Discard a dead client registration while retaining token state."""
        with lifecycle_lock(self):
            client_path = self._client_info_path()
            backup = client_path.with_name(client_path.name + ".bak")
            client_exists = _safe_file(client_path)
            backup_exists = _safe_file(backup)
            meta_exists = _safe_file(self._meta_path())
            if not client_exists and not backup_exists and not meta_exists:
                return False
            _safe_token_dir(self, create=True)
            self._begin_marker("poison")
            try:
                if backup_exists:
                    backup.unlink()
                if client_exists:
                    client_path.unlink()
                if meta_exists:
                    self._meta_path().unlink()
                self._clear_marker()
            except BaseException:
                raise
            if client_exists:
                logger.warning(
                    "MCP OAuth '%s': invalid_client; removed client.json, meta.json, and legacy backup",
                    self._server_name,
                )
            return client_exists
