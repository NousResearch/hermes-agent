"""Shared file sync manager for remote execution backends.

Tracks local file changes via mtime+size, detects deletions, and syncs to
remote environments transactionally.  Used by SSH, Modal, and Daytona.
Docker and Singularity use bind mounts (live host FS view) and don't need this.
"""

import hashlib
import logging
import os
import posixpath
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

# Tests patch these module-level aliases instead of ``time.sleep`` /
# ``time.monotonic``: patching attributes on the shared ``time`` module object
# leaks into unrelated threads under xdist and inflates retry call counts.
_sleep = time.sleep
_monotonic = time.monotonic

_SYNC_INTERVAL_SECONDS = 5.0
_FORCE_SYNC_ENV = "HERMES_FORCE_FILE_SYNC"

# Transport callbacks provided by each backend
UploadFn = Callable[[str, str], None]  # (host_path, remote_path) -> raises on failure
BulkUploadFn = Callable[[list[tuple[str, str]]], None]  # [(host_path, remote_path), ...] -> raises on failure
BulkDownloadFn = Callable[[Path], None]  # (dest_tar_path) -> writes tar archive, raises on failure
DeleteFn = Callable[[list[str]], None]  # (remote_paths) -> raises on failure
GetFilesFn = Callable[[], list[tuple[str, str]]]  # () -> [(host_path, remote_path), ...]

_SYNC_BACK_MAX_RETRIES = 3
_SYNC_BACK_BACKOFF = (2, 4, 8)  # seconds between retries
_SYNC_BACK_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB — refuse to extract larger tars


def iter_sync_files(container_base: str = "/root/.hermes") -> list[tuple[str, str]]:
    """Enumerate all (host_path, remote_path) pairs to sync to a remote. Credential paths are
    remapped from the hardcoded /root/.hermes to *container_base* (remote home may differ)."""
    # Late import: credential_files pulls in agent modules (circular at module level).
    from tools.credential_files import get_credential_file_mounts, iter_cache_files, iter_skills_files

    files = [
        (entry["host_path"], entry["container_path"].replace("/root/.hermes", container_base, 1))
        for entry in get_credential_file_mounts()]
    files += [
        (entry["host_path"], entry["container_path"])
        for entry in (*iter_skills_files(container_base=container_base),
                      *iter_cache_files(container_base=container_base))]
    return files


def _resolve_host_path_str(host_path: str) -> str:
    """Canonical string form of a host path (``resolve()`` falling back to ``expanduser()``)."""
    try:
        return str(Path(host_path).expanduser().resolve())
    except OSError:
        return str(Path(host_path).expanduser())


def _credential_host_paths() -> set[str]:
    """Return credential files that are upload-only for remote sandboxes."""
    try:
        from tools.credential_files import get_credential_file_mounts
        mounts = get_credential_file_mounts()
    except Exception:
        return set()
    return {
        _resolve_host_path_str(entry["host_path"])
        for entry in mounts if isinstance(entry, dict) and entry.get("host_path")}


def quoted_rm_command(remote_paths: list[str]) -> str:
    """Build a shell ``rm -f`` command for a batch of remote paths."""
    return "rm -f " + " ".join(shlex.quote(p) for p in remote_paths)


def quoted_mkdir_command(dirs: list[str]) -> str:
    """Build a shell ``mkdir -p`` command for a batch of directories."""
    return "mkdir -p " + " ".join(shlex.quote(d) for d in dirs)


def unique_parent_dirs(files: list[tuple[str, str]]) -> list[str]:
    """Extract sorted unique parent directories from (host, remote) pairs."""
    return sorted({posixpath.dirname(remote) for _, remote in files})


def _sha256_file(path: str) -> str:
    """Return hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class FileSyncManager:
    """Tracks local file changes and syncs to a remote environment. Backends supply transport
    callbacks (upload, delete) and a file-source callable; the manager handles mtime-based
    change detection, deletion tracking, rate limiting, and transactional state."""

    def __init__(
        self,
        get_files_fn: GetFilesFn,
        upload_fn: UploadFn,
        delete_fn: DeleteFn,
        sync_interval: float = _SYNC_INTERVAL_SECONDS,
        bulk_upload_fn: BulkUploadFn | None = None,
        bulk_download_fn: BulkDownloadFn | None = None):
        self._get_files_fn = get_files_fn
        self._upload_fn = upload_fn
        self._bulk_upload_fn = bulk_upload_fn
        self._bulk_download_fn = bulk_download_fn
        self._delete_fn = delete_fn
        self._transaction_lock = threading.Lock()
        self._synced_files: dict[str, tuple[float, int]] = {}  # remote_path -> (mtime, size)
        self._pushed_hashes: dict[str, str] = {}  # remote_path -> sha256 hex digest
        self._upload_only_host_paths: set[str] = set()
        self._last_sync_time: float = 0.0  # monotonic; 0 ensures first sync runs
        self._sync_interval = sync_interval

    def sync(self, *, force: bool = False) -> None:
        """Run a sync cycle: upload changed files, delete removed files. Rate-limited to once
        per ``sync_interval`` unless *force* or ``HERMES_FORCE_FILE_SYNC=1``. Transactional:
        state is committed only if ALL operations succeed; on failure it rolls back so the
        next cycle retries everything."""
        with self._transaction_lock:
            self._sync_transaction(force=force)

    def _sync_transaction(self, *, force: bool = False) -> None:
        """Execute one sync cycle while holding the per-manager lock."""
        if (
            not force
            and not os.environ.get(_FORCE_SYNC_ENV)
            and _monotonic() - self._last_sync_time < self._sync_interval):
            return

        current_files = self._get_files_fn()
        self._upload_only_host_paths.update(_credential_host_paths())
        to_upload, new_files, to_delete = self._plan_sync(current_files)

        if not to_upload and not to_delete:
            self._last_sync_time = _monotonic()
            return

        prev_files = dict(self._synced_files)
        prev_hashes = dict(self._pushed_hashes)
        try:
            self._push(to_upload, to_delete)
            # Commit (all succeeded).
            for host_path, remote_path in to_upload:
                self._pushed_hashes[remote_path] = _sha256_file(host_path)
            for p in to_delete:
                new_files.pop(p, None)
                self._pushed_hashes.pop(p, None)
            self._synced_files = new_files
            self._last_sync_time = _monotonic()
        except Exception as exc:
            self._synced_files = prev_files
            self._pushed_hashes = prev_hashes
            # Do NOT advance _last_sync_time: bumping the rate-limit clock on failure would
            # suppress the retry for up to _sync_interval, contradicting the retry contract.
            logger.warning("file_sync: sync failed, rolled back state: %s", exc)

    def _plan_sync(
        self, current_files: list[tuple[str, str]]
    ) -> tuple[list[tuple[str, str]], dict[str, tuple[float, int]], list[str]]:
        """Diff *current_files* against synced state -> ``(to_upload, new_synced_state, to_delete)``."""
        to_upload: list[tuple[str, str]] = []
        new_files = dict(self._synced_files)
        for host_path, remote_path in current_files:
            file_key = _file_mtime_key(host_path)
            if file_key is None or self._synced_files.get(remote_path) == file_key:
                continue
            to_upload.append((host_path, remote_path))
            new_files[remote_path] = file_key
        current_remote_paths = {remote for _, remote in current_files}
        to_delete = [p for p in self._synced_files if p not in current_remote_paths]
        return to_upload, new_files, to_delete

    def _push(self, to_upload: list[tuple[str, str]], to_delete: list[str]) -> None:
        """Run the transport calls for one cycle (bulk upload when available)."""
        if to_upload:
            logger.debug("file_sync: uploading %d file(s)", len(to_upload))
        if to_delete:
            logger.debug("file_sync: deleting %d stale remote file(s)", len(to_delete))
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

    # --- Sync-back: pull remote changes to host on teardown ---
    def sync_back(self, hermes_home: Path | None = None) -> None:
        """Pull remote changes back to the host: download the remote ``.hermes/`` as a tar and
        apply only files whose SHA-256 differs from what was pushed. SIGINT is deferred until
        complete; concurrent gateway sandboxes are serialized via a file lock."""
        with self._transaction_lock:
            self._sync_back_transaction(hermes_home=hermes_home)

    def _sync_back_transaction(self, hermes_home: Path | None = None) -> None:
        """Execute sync-back (with retries) against a stable snapshot of manager state."""
        if self._bulk_download_fn is None:
            return

        # Nothing was ever committed (initial push failed or never ran): skip
        # to avoid retry storms against an uninitialized remote .hermes/.
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
                    logger.warning("sync_back: attempt %d failed (%s), retrying in %ds", attempt + 1, exc, delay)
                    _sleep(delay)

        logger.warning("sync_back: all %d attempts failed: %s", _SYNC_BACK_MAX_RETRIES, last_exc)

    def _sync_back_once(self, lock_path: Path) -> None:
        """Single sync-back attempt with SIGINT protection and file lock."""
        # signal.signal() only works from the main thread; gateway cleanup()
        # may run from a worker thread — skip SIGINT deferral there.
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
                    # Re-deliver the deferred Ctrl+C to the restored handler. ``os.kill(os.getpid(),
                    # SIGINT)`` is NOT graceful on Windows (routes to TerminateProcess, hard-killing
                    # the CLI); ``raise_signal`` invokes the handler everywhere.
                    signal.raise_signal(signal.SIGINT)

    def _sync_back_locked(self, lock_path: Path) -> None:
        """Sync-back under file lock (serializes concurrent gateways)."""
        if fcntl is None:
            # Windows: no flock — run without serialization
            self._sync_back_impl()
            return
        lock_fd = open(lock_path, "w", encoding="utf-8")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            self._sync_back_impl()
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except (OSError, IOError):
                pass
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

        # mkstemp + close: NamedTemporaryFile keeps an exclusive handle on Windows, so the
        # backend's open(dest, "wb") / write_bytes on the same path raised PermissionError.
        fd, tar_path = tempfile.mkstemp(suffix=".tar")
        os.close(fd)
        try:
            self._bulk_download_fn(Path(tar_path))

            # A misbehaving sandbox could produce an arbitrarily large tar.
            try:
                tar_size = os.path.getsize(tar_path)
            except OSError:
                tar_size = 0
            if tar_size > _SYNC_BACK_MAX_BYTES:
                logger.warning(
                    "sync_back: remote tar is %d bytes (cap %d) — skipping extraction",
                    tar_size, _SYNC_BACK_MAX_BYTES)
                return

            with tempfile.TemporaryDirectory(prefix="hermes-sync-back-") as staging:
                with tarfile.open(tar_path) as tar:
                    tar.extractall(staging, filter="data")

                upload_only = self._upload_only_host_paths | _credential_host_paths()
                applied = 0
                for dirpath, _dirnames, filenames in os.walk(staging):
                    for fname in filenames:
                        staged_file = os.path.join(dirpath, fname)
                        # Remote keys are POSIX; relpath uses host separators (backslashes on Windows).
                        remote_path = "/" + Path(os.path.relpath(staged_file, staging)).as_posix()
                        applied += self._apply_staged_file(staged_file, remote_path, file_mapping, upload_only)

                if applied:
                    logger.info("sync_back: applied %d changed file(s)", applied)
                else:
                    logger.debug("sync_back: no remote changes detected")
        finally:
            try:
                os.unlink(tar_path)
            except OSError:
                pass

    def _apply_staged_file(
        self, staged_file: str, remote_path: str, file_mapping: list[tuple[str, str]], upload_only_host_paths: set[str],
    ) -> int:
        """Copy one extracted remote file onto the host if it changed since push. Returns 1 if
        applied, 0 if skipped (unchanged, unmapped, or an upload-only credential). A host file
        modified since push is overwritten with the remote version (last-write-wins) with a warning."""
        pushed_hash = self._pushed_hashes.get(remote_path)
        if pushed_hash is not None and _sha256_file(staged_file) == pushed_hash:
            return 0  # unchanged from push

        host_path = self._resolve_host_path(remote_path, file_mapping)
        if host_path is None:
            host_path = self._infer_host_path(remote_path, file_mapping, upload_only_host_paths=upload_only_host_paths)
            if host_path is None:
                logger.debug("sync_back: skipping %s (no host mapping)", remote_path)
                return 0

        if self._is_upload_only_host_path(host_path, upload_only_host_paths):
            logger.debug("sync_back: skipping upload-only credential file %s", remote_path)
            return 0

        if pushed_hash is not None and os.path.exists(host_path) and _sha256_file(host_path) != pushed_hash:
            logger.warning(
                "sync_back: conflict on %s — host modified "
                "since push, remote also changed. Applying remote version (last-write-wins).",
                remote_path)

        os.makedirs(os.path.dirname(host_path), exist_ok=True)
        shutil.copy2(staged_file, host_path)
        return 1

    def _resolve_host_path(self, remote_path: str, file_mapping: list[tuple[str, str]] | None = None) -> str | None:
        """Find the host path for a known remote path from the file mapping."""
        return next((host for host, remote in file_mapping or [] if remote == remote_path), None)

    def _infer_host_path(self, remote_path: str, file_mapping: list[tuple[str, str]] | None = None, *,
                         upload_only_host_paths: set[str] | None = None) -> str | None:
        """Infer a host path for a new remote file by matching path prefixes: an existing
        remote->host pair whose parent directory prefixes *remote_path* gets the same
        substitution (``/root/.hermes/skills/b.md`` -> ``~/.hermes/skills/b.md``)."""
        upload_only_host_paths = upload_only_host_paths or set()
        for host, remote in file_mapping or []:
            if self._is_upload_only_host_path(host, upload_only_host_paths):
                continue
            remote_dir = posixpath.dirname(remote)  # remote paths are POSIX even on a Windows host
            if remote_path.startswith(remote_dir + "/"):
                return str(Path(host).parent / remote_path[len(remote_dir) + 1:])
        return None

    @staticmethod
    def _is_upload_only_host_path(host_path: str, upload_only_host_paths: set[str]) -> bool:
        return _resolve_host_path_str(host_path) in upload_only_host_paths
