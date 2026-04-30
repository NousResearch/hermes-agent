"""Shared file sync manager for remote execution backends.

Tracks local file changes via mtime+size, detects deletions, and
syncs to remote environments transactionally.  Used by SSH, Modal,
and Daytona.  Docker and Singularity use bind mounts (live host FS
view) and don't need this.
"""

import hashlib
import logging
import os
import shlex
import shutil
import signal
import tarfile
import tempfile
import threading
import time

try:
    import fcntl
except ImportError:
    fcntl = None  # Windows — file locking skipped
from pathlib import Path
from typing import Callable

from hermes_constants import get_hermes_home
from tools.environments.base import _file_mtime_key

logger = logging.getLogger(__name__)

# Keep retry sleeps patchable without mutating the shared stdlib ``time``
# module. Patching ``tools.environments.file_sync.time.sleep`` replaces
# ``time.sleep`` globally because ``time`` is the module object; under xdist
# that lets unrelated background threads inflate retry-test call counts.
_sleep = time.sleep

_SYNC_INTERVAL_SECONDS = 5.0
_FORCE_SYNC_ENV = "HERMES_FORCE_FILE_SYNC"

# Transport callbacks provided by each backend
UploadFn = Callable[[str, str], None]  # (host_path, remote_path) -> raises on failure
BulkUploadFn = Callable[[list[tuple[str, str]]], None]  # [(host_path, remote_path), ...] -> raises on failure
BulkDownloadFn = Callable[[Path], None]  # (dest_tar_path) -> writes tar archive, raises on failure
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


def quoted_mkdir_command(dirs: list[str]) -> str:
    """Build a shell ``mkdir -p`` command for a batch of directories."""
    return "mkdir -p " + " ".join(shlex.quote(d) for d in dirs)


def unique_parent_dirs(files: list[tuple[str, str]]) -> list[str]:
    """Extract sorted unique parent directories from (host, remote) pairs."""
    return sorted({str(Path(remote).parent) for _, remote in files})


def _sha256_file(path: str) -> str:
    """Return hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


_SYNC_BACK_MAX_RETRIES = 3
_SYNC_BACK_BACKOFF = (2, 4, 8)  # seconds between retries
_SYNC_BACK_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB — refuse to extract larger tars


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
        bulk_download_fn: BulkDownloadFn | None = None,
    ):
        self._get_files_fn = get_files_fn
        self._upload_fn = upload_fn
        self._bulk_upload_fn = bulk_upload_fn
        self._bulk_download_fn = bulk_download_fn
        self._delete_fn = delete_fn
        self._synced_files: dict[str, tuple[float, int]] = {}  # remote_path -> (mtime, size)
        self._pushed_hashes: dict[str, str] = {}  # remote_path -> sha256 hex digest
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
        prev_hashes = dict(self._pushed_hashes)

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
            for host_path, remote_path in to_upload:
                self._pushed_hashes[remote_path] = _sha256_file(host_path)

            for p in to_delete:
                new_files.pop(p, None)
                self._pushed_hashes.pop(p, None)

            self._synced_files = new_files
            self._last_sync_time = time.monotonic()

        except Exception as exc:
            self._synced_files = prev_files
            self._pushed_hashes = prev_hashes
            self._last_sync_time = time.monotonic()
            logger.warning("file_sync: sync failed, rolled back state: %s", exc)

    # ------------------------------------------------------------------
    # Sync-back: pull remote changes to host on teardown
    # ------------------------------------------------------------------

    def sync_back(self, hermes_home: Path | None = None) -> None:
        """Pull remote changes back to the host filesystem.

        Downloads the remote ``.hermes/`` directory as a tar archive,
        unpacks it, and applies only files that differ from what was
        originally pushed (based on SHA-256 content hashes).

        Protected against SIGINT (defers the signal until complete) and
        serialized across concurrent gateway sandboxes via file lock.
        """
        if self._bulk_download_fn is None:
            return

        # Nothing was ever committed through this manager — the initial
        # push failed or never ran. Skip sync_back to avoid retry storms
        # against an uninitialized remote .hermes/ directory.
        if not self._pushed_hashes and not self._synced_files:
            logger.debug("sync_back: no prior push state — skipping")
            return

        lock_path = (hermes_home or get_hermes_home()) / ".sync.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        last_exc: Exception | None = None
        for attempt in range(_SYNC_BACK_MAX_RETRIES):
            try:
                self._sync_back_once(lock_path)
                return
            except Exception as exc:
                last_exc = exc
                if attempt < _SYNC_BACK_MAX_RETRIES - 1:
                    delay = _SYNC_BACK_BACKOFF[attempt]
                    logger.warning(
                        "sync_back: attempt %d failed (%s), retrying in %ds",
                        attempt + 1, exc, delay,
                    )
                    _sleep(delay)

        logger.warning("sync_back: all %d attempts failed: %s", _SYNC_BACK_MAX_RETRIES, last_exc)

    def _sync_back_once(self, lock_path: Path) -> None:
        """Single sync-back attempt with SIGINT protection and file lock."""
        # signal.signal() only works from the main thread. In gateway
        # contexts cleanup() may run from a worker thread — skip SIGINT
        # deferral there rather than crashing.
        on_main_thread = threading.current_thread() is threading.main_thread()

        deferred_sigint: list[object] = []
        original_handler = None
        if on_main_thread:
            original_handler = signal.getsignal(signal.SIGINT)

            def _defer_sigint(signum, frame):
                deferred_sigint.append((signum, frame))
                logger.debug("sync_back: SIGINT deferred until sync completes")

            signal.signal(signal.SIGINT, _defer_sigint)
        try:
            self._sync_back_locked(lock_path)
        finally:
            if on_main_thread and original_handler is not None:
                signal.signal(signal.SIGINT, original_handler)
                if deferred_sigint:
                    os.kill(os.getpid(), signal.SIGINT)

    def _sync_back_locked(self, lock_path: Path) -> None:
        """Sync-back under file lock (serializes concurrent gateways)."""
        if fcntl is None:
            # Windows: no flock — run without serialization
            self._sync_back_impl()
            return
        lock_fd = open(lock_path, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            self._sync_back_impl()
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    def _sync_back_impl(self) -> None:
        """Download, diff, and apply remote changes to host."""
        if self._bulk_download_fn is None:
            raise RuntimeError("_sync_back_impl called without bulk_download_fn")

        # Cache file mapping once to avoid O(n*m) from repeated iteration
        try:
            file_mapping = list(self._get_files_fn())
        except Exception:
            file_mapping = []

        with tempfile.NamedTemporaryFile(suffix=".tar") as tf:
            self._bulk_download_fn(Path(tf.name))

            # Defensive size cap: a misbehaving sandbox could produce an
            # arbitrarily large tar. Refuse to extract if it exceeds the cap.
            try:
                tar_size = os.path.getsize(tf.name)
            except OSError:
                tar_size = 0
            if tar_size > _SYNC_BACK_MAX_BYTES:
                logger.warning(
                    "sync_back: remote tar is %d bytes (cap %d) — skipping extraction",
                    tar_size, _SYNC_BACK_MAX_BYTES,
                )
                return

            with tempfile.TemporaryDirectory(prefix="hermes-sync-back-") as staging:
                with tarfile.open(tf.name) as tar:
                    tar.extractall(staging, filter="data")

                applied = 0
                for dirpath, _dirnames, filenames in os.walk(staging):
                    for fname in filenames:
                        staged_file = os.path.join(dirpath, fname)
                        rel = os.path.relpath(staged_file, staging)
                        remote_path = "/" + rel

                        pushed_hash = self._pushed_hashes.get(remote_path)

                        # Skip hashing for files unchanged from push
                        if pushed_hash is not None:
                            remote_hash = _sha256_file(staged_file)
                            if remote_hash == pushed_hash:
                                continue
                        else:
                            remote_hash = None  # new remote file

                        # Resolve host path from cached mapping
                        host_path = self._resolve_host_path(remote_path, file_mapping)
                        if host_path is None:
                            host_path = self._infer_host_path(remote_path, file_mapping)
                            if host_path is None:
                                logger.debug(
                                    "sync_back: skipping %s (no host mapping)",
                                    remote_path,
                                )
                                continue

                        if os.path.exists(host_path) and pushed_hash is not None:
                            host_hash = _sha256_file(host_path)
                            if host_hash != pushed_hash:
                                logger.warning(
                                    "sync_back: conflict on %s — host modified "
                                    "since push, remote also changed. Applying "
                                    "remote version (last-write-wins).",
                                    remote_path,
                                )

                        os.makedirs(os.path.dirname(host_path), exist_ok=True)
                        shutil.copy2(staged_file, host_path)
                        applied += 1

                if applied:
                    logger.info("sync_back: applied %d changed file(s)", applied)
                else:
                    logger.debug("sync_back: no remote changes detected")

    def _resolve_host_path(self, remote_path: str,
                           file_mapping: list[tuple[str, str]] | None = None) -> str | None:
        """Find the host path for a known remote path from the file mapping."""
        mapping = file_mapping if file_mapping is not None else []
        for host, remote in mapping:
            if remote == remote_path:
                return host
        return None

    def _infer_host_path(self, remote_path: str,
                         file_mapping: list[tuple[str, str]] | None = None) -> str | None:
        """Infer a host path for a new remote file by matching path prefixes.

        Uses the existing file mapping to find a remote->host directory
        pair, then applies the same prefix substitution to the new file.
        For example, if the mapping has ``/root/.hermes/skills/a.md`` →
        ``~/.hermes/skills/a.md``, a new remote file at
        ``/root/.hermes/skills/b.md`` maps to ``~/.hermes/skills/b.md``.
        """
        mapping = file_mapping if file_mapping is not None else []
        for host, remote in mapping:
            remote_dir = str(Path(remote).parent)
            if remote_path.startswith(remote_dir + "/"):
                host_dir = str(Path(host).parent)
                suffix = remote_path[len(remote_dir):]
                return host_dir + suffix
        return None
