"""Shared file sync manager for remote execution backends.

Tracks local file changes via mtime+size, detects deletions, and
syncs to remote environments transactionally.  Used by SSH, Modal,
and Daytona.  Docker and Singularity use bind mounts (live host FS
view) and don't need this.
"""

import logging
import os
import shlex
import time
from typing import Callable

from tools.environments.base import _file_mtime_key

logger = logging.getLogger(__name__)

_SYNC_INTERVAL_SECONDS = 5.0
_FORCE_SYNC_ENV = "HERMES_FORCE_FILE_SYNC"

# Transport callbacks provided by each backend
UploadFn = Callable[[str, str], None]  # (host_path, remote_path) -> raises on failure
BulkUploadFn = Callable[[list[tuple[str, str]]], None]  # [(host_path, remote_path), ...] -> raises on failure
DeleteFn = Callable[[list[str]], None]  # (remote_paths) -> raises on failure
GetFilesFn = Callable[[], list[tuple[str, str]]]  # () -> [(host_path, remote_path), ...]


def iter_sync_files(container_base: str = "/root/.hermes") -> list[tuple[str, str]]:
    """Enumerate all files that should be synced to a remote environment.

    Combines credentials, skills, and cache into a single flat list of
    (host_path, remote_path) pairs.  Credential paths are remapped from
    the hardcoded /root/.hermes to *container_base* because the remote
    user's home may differ (e.g. /home/daytona, /home/user).
    """
    # Late import: credential_files imports agent modules that create
    # circular dependencies if loaded at file_sync module level.
    from tools.credential_files import (
        get_credential_file_mounts,
        iter_cache_files,
        iter_skills_files,
    )

    files: list[tuple[str, str]] = []
    for entry in get_credential_file_mounts():
        remote = entry["container_path"].replace(
            "/root/.hermes", container_base, 1
        )
        files.append((entry["host_path"], remote))
    for entry in iter_skills_files(container_base=container_base):
        files.append((entry["host_path"], entry["container_path"]))
    for entry in iter_cache_files(container_base=container_base):
        files.append((entry["host_path"], entry["container_path"]))
    return files


def quoted_rm_command(remote_paths: list[str]) -> str:
    """Build a shell ``rm -f`` command for a batch of remote paths."""
    return "rm -f " + " ".join(shlex.quote(p) for p in remote_paths)


class FileSyncManager:
    """Tracks local file changes and syncs to a remote environment.

    Backends instantiate this with transport callbacks (upload, delete)
    and a file-source callable.  The manager handles mtime-based change
    detection, deletion tracking, rate limiting, and transactional state.

    Not used by bind-mount backends (Docker, Singularity) — those get
    live host FS views and don't need file sync.
    """

    def __init__(
        self,
        get_files_fn: GetFilesFn,
        upload_fn: UploadFn,
        delete_fn: DeleteFn,
        sync_interval: float = _SYNC_INTERVAL_SECONDS,
        bulk_upload_fn: BulkUploadFn | None = None,
    ):
        self._get_files_fn = get_files_fn
        self._upload_fn = upload_fn
        self._bulk_upload_fn = bulk_upload_fn
        self._delete_fn = delete_fn
        self._synced_files: dict[str, tuple[float, int]] = {}  # remote_path -> (mtime, size)
        self._last_sync_time: float = 0.0  # monotonic; 0 ensures first sync runs
        self._sync_interval = sync_interval

    def sync(self, *, force: bool = False) -> None:
        """Run a sync cycle: upload changed files, delete removed files.

        Rate-limited to once per ``sync_interval`` unless *force* is True
        or ``HERMES_FORCE_FILE_SYNC=1`` is set.

        Transactional: state only committed if ALL operations succeed.
        On failure, state rolls back so the next cycle retries everything.
        """
        if not force and not os.environ.get(_FORCE_SYNC_ENV):
            now = time.monotonic()
            if now - self._last_sync_time < self._sync_interval:
                return

        current_files = self._get_files_fn()
        current_remote_paths = {remote for _, remote in current_files}

        # --- Uploads: new or changed files ---
        to_upload: list[tuple[str, str]] = []
        new_files = dict(self._synced_files)
        for host_path, remote_path in current_files:
            file_key = _file_mtime_key(host_path)
            if file_key is None:
                continue
            if self._synced_files.get(remote_path) == file_key:
                continue
            to_upload.append((host_path, remote_path))
            new_files[remote_path] = file_key

        # --- Deletes: synced paths no longer in current set ---
        to_delete = [p for p in self._synced_files if p not in current_remote_paths]

        if not to_upload and not to_delete:
            self._last_sync_time = time.monotonic()
            return

        # Snapshot for rollback (only when there's work to do)
        prev_files = dict(self._synced_files)

        if to_upload:
            logger.debug("file_sync: uploading %d file(s)", len(to_upload))
        if to_delete:
            logger.debug("file_sync: deleting %d stale remote file(s)", len(to_delete))

        try:
            if to_upload and self._bulk_upload_fn is not None:
                self._bulk_upload_fn(to_upload)
                logger.debug("file_sync: bulk-uploaded %d file(s)", len(to_upload))
            else:
                for host_path, remote_path in to_upload:
                    self._upload_fn(host_path, remote_path)
                    logger.debug("file_sync: uploaded %s -> %s", host_path, remote_path)

            if to_delete:
                self._delete_fn(to_delete)
                logger.debug("file_sync: deleted %s", to_delete)

            # --- Commit (all succeeded) ---
            for p in to_delete:
                new_files.pop(p, None)

            self._synced_files = new_files
            self._last_sync_time = time.monotonic()

        except Exception as exc:
            self._synced_files = prev_files
            self._last_sync_time = time.monotonic()
            logger.warning("file_sync: sync failed, rolled back state: %s", exc)
