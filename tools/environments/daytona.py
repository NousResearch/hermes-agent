"""Daytona cloud execution environment.

Uses the Daytona Python SDK to run commands in cloud sandboxes.
Supports persistent sandboxes: when enabled, sandboxes are stopped on cleanup
and resumed on next creation, preserving the filesystem across sessions.
"""

import hashlib
import importlib
import logging
import math
import os
import shlex
import threading
from pathlib import Path
from typing import Any

from tools.environments.base import (
    BaseEnvironment,
    _ThreadedProcessHandle,
)
from tools.environments.file_sync import (
    FileSyncManager,
    iter_sync_files,
    quoted_mkdir_command,
    quoted_rm_command,
    unique_parent_dirs,
)

logger = logging.getLogger(__name__)

_FALLBACK_SUPPORTED_LANGUAGES: tuple[str, ...] = ("python", "typescript", "javascript")


def get_supported_daytona_languages() -> tuple[str, ...]:
    """Return language values supported by the installed Daytona SDK.

    The SDK exposes these as enum values on newer releases.  Keep a narrow
    fallback matching the currently supported SDK surface so CLI docs and
    validation do not advertise languages the SDK cannot create.
    """
    try:
        from daytona import CodeLanguage

        values = tuple(
            str(getattr(member, "value", member)).strip().lower()
            for member in CodeLanguage
            if str(getattr(member, "value", member)).strip()
        )
        if values:
            return tuple(dict.fromkeys(values))
    except Exception:
        pass
    return _FALLBACK_SUPPORTED_LANGUAGES


def _derive_profile_id() -> str:
    """Derive a stable, non-secret profile identifier from get_hermes_home().

    Uses a short SHA-256 hash (first 8 hex chars) of the resolved path so
    that different profiles (e.g. ``~/.hermes`` vs
    ``~/.hermes/profiles/dev-docker``) produce different identifiers while
    keeping the sandbox name compact.  The hash is one-way; the original
    path cannot be recovered from the identifier.
    """
    from hermes_constants import get_hermes_home

    home = str(get_hermes_home().resolve())
    return hashlib.sha256(home.encode()).hexdigest()[:8]


class DaytonaEnvironment(BaseEnvironment):
    """Daytona cloud sandbox execution backend.

    Spawn-per-call via _ThreadedProcessHandle wrapping blocking SDK calls.
    cancel_fn wired to sandbox.stop() for interrupt support.
    Shell timeout wrapper preserved (SDK timeout unreliable).
    """

    _stdin_mode = "heredoc"

    def __init__(
        self,
        image: str,
        cwd: str = "/home/daytona",
        timeout: int = 60,
        cpu: int = 1,
        memory: int = 5120,
        disk: int = 10240,
        persistent_filesystem: bool = True,
        task_id: str = "default",
        # --- Daytona expansion parameters ---
        create_mode: str = "image",
        snapshot: str = "",
        language: str = "",
        name_prefix: str = "hermes",
        name_scope: str = "task",
        labels: dict | None = None,
        auto_stop_interval: int = 0,
        auto_archive_interval: int = 0,
        auto_delete_interval: int = 0,
        ephemeral: bool = False,
        env_vars: dict | None = None,
        network_block_all: bool = False,
        network_allow_list: str = "",
        volume_mounts: list | None = None,
        gpu: int = 0,
        # --- CWD sync ---
        host_cwd: str | None = None,
        # --- P7 CWD sync pilot ---
        sync_cwd: bool = False,
    ):
        requested_cwd = cwd
        super().__init__(cwd=cwd, timeout=timeout)

        try:
            from tools.lazy_deps import ensure as _lazy_ensure
            _lazy_ensure("terminal.daytona", prompt=False)
        except ImportError:
            pass
        except Exception as e:
            raise ImportError(str(e))
        daytona_mod = importlib.import_module("daytona")
        Daytona = getattr(daytona_mod, "Daytona")
        CreateSandboxFromImageParams = getattr(daytona_mod, "CreateSandboxFromImageParams")
        CreateSandboxFromSnapshotParams = getattr(
            daytona_mod,
            "CreateSandboxFromSnapshotParams",
            None,
        )
        DaytonaError = getattr(daytona_mod, "DaytonaError")
        Resources = getattr(daytona_mod, "Resources")
        SandboxState = getattr(daytona_mod, "SandboxState")

        # --- Validate create_mode BEFORE any SDK side effects ---
        valid_modes = ("image", "snapshot")
        if create_mode not in valid_modes:
            raise ValueError(
                f"Invalid daytona_create_mode={create_mode!r}; "
                f"must be one of {valid_modes}"
            )
        if create_mode == "snapshot" and not snapshot:
            raise ValueError(
                "daytona_create_mode='snapshot' requires a non-empty "
                "daytona_snapshot value; provide a snapshot name or ID "
                "via TERMINAL_DAYTONA_SNAPSHOT or the 'snapshot' parameter"
            )
        if language:
            supported_languages = get_supported_daytona_languages()
            normalized_language = language.strip().lower()
            if normalized_language not in supported_languages:
                raise ValueError(
                    f"Invalid daytona_language={language!r}; installed Daytona SDK "
                    f"supports: {', '.join(supported_languages)}"
                )
            language = normalized_language

        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._SandboxState = SandboxState
        self._daytona = Daytona()
        self._sandbox = None
        self._lock = threading.Lock()

        # --- Derive profile-scoped identifier ---
        profile_id = _derive_profile_id()

        # --- Compute sandbox name from name_prefix + name_scope ---
        # name_scope='task' (default):  '{prefix}-{task_id}'
        # name_scope='profile':         '{prefix}-{profile_id}-{task_id}'
        # name_scope='global':           '{prefix}'
        # name_scope='legacy':           exact 'hermes-{task_id}' (backward compat)
        valid_name_scopes = {"task", "profile", "global", "legacy"}
        if name_scope not in valid_name_scopes:
            raise ValueError(
                f"Invalid daytona_name_scope={name_scope!r}; "
                f"must be one of {sorted(valid_name_scopes)}"
            )
        if name_scope == "legacy":
            sandbox_name = f"hermes-{task_id}"
        elif name_scope == "global":
            sandbox_name = name_prefix
        elif name_scope == "profile":
            sandbox_name = f"{name_prefix}-{profile_id}-{task_id}"
        else:
            sandbox_name = f"{name_prefix}-{task_id}"

        # --- Merge labels ---
        # User labels are accepted, but cannot override Hermes-reserved labels
        # used for sandbox discovery/profile isolation.
        reserved_labels = {
            "hermes_task_id": task_id,
            "hermes_profile_id": profile_id,
            "hermes_backend": "daytona",
        }
        merged_labels = dict(labels or {})
        for reserved_key in reserved_labels:
            if reserved_key in merged_labels:
                logger.warning(
                    "Daytona: ignoring user-provided reserved label %s",
                    reserved_key,
                )
                merged_labels.pop(reserved_key, None)
        merged_labels.update(reserved_labels)

        # --- Compute lifecycle intervals ---
        gpu_requested = bool(gpu and gpu > 0)
        # ephemeral=True and GPU requests force auto_delete_interval to 0.
        # Daytona live validation showed GPU sandboxes are rejected unless
        # autoDeleteInterval is explicitly zero, even if a user configured a
        # non-zero auto-delete interval for normal CPU sandboxes.
        effective_auto_delete = 0 if (ephemeral or gpu_requested) else auto_delete_interval

        # --- Compute Resources ---
        memory_gib = max(1, math.ceil(memory / 1024))
        disk_gib = max(1, math.ceil(disk / 1024))
        if disk_gib > 10:
            logger.warning(
                "Daytona: requested disk (%dGB) exceeds platform limit (10GB). "
                "Capping to 10GB.", disk_gib,
            )
            disk_gib = 10

        resources_kwargs = dict(cpu=cpu, memory=memory_gib, disk=disk_gib)
        if gpu and gpu > 0:
            resources_kwargs["gpu"] = gpu

        resources = Resources(**resources_kwargs)

        if self._persistent:
            try:
                self._sandbox = self._daytona.get(sandbox_name)
                self._sandbox.start()
                logger.info("Daytona: resumed sandbox %s for task %s",
                            self._sandbox.id, task_id)
            except DaytonaError:
                self._sandbox = None
            except Exception as e:
                logger.warning("Daytona: failed to resume sandbox for task %s: %s",
                               task_id, e)
                self._sandbox = None

            if self._sandbox is None:
                try:
                    # Daytona SDK >=0.108.0 uses cursor-based pagination and
                    # list() returns an iterator. Offset-based pagination
                    # (page=1) is removed on June 10, 2026.
                    # Search by labels (includes hermes_profile_id +
                    # hermes_backend) to find existing profile-scoped sandboxes.
                    results = self._daytona.list(labels=merged_labels, limit=1)
                    existing = next(iter(results), None)
                    if existing is not None:
                        self._sandbox = existing
                        self._sandbox.start()
                        logger.info("Daytona: resumed existing sandbox %s for task %s "
                                     "(profile_id=%s)",
                                    self._sandbox.id, task_id, profile_id)
                except Exception as e:
                    logger.debug("Daytona: no existing sandbox found for task %s "
                                 "(profile_id=%s): %s",
                                 task_id, profile_id, e)
                    self._sandbox = None

        if self._sandbox is None:
            # --- Build create params based on create_mode ---
            create_params_kwargs: dict[str, Any] = dict(
                name=sandbox_name,
                labels=merged_labels,
                auto_stop_interval=auto_stop_interval,
                resources=resources,
            )

            # Forward optional lifecycle intervals
            if auto_archive_interval:
                create_params_kwargs["auto_archive_interval"] = auto_archive_interval

            # Always forward auto_delete_interval when explicitly set, when
            # ephemeral=True signals short-lived sandbox intent, or when a GPU
            # is requested.  Live Daytona validation showed GPU sandboxes are
            # rejected unless autoDeleteInterval is explicitly 0; relying on
            # the SDK/server default causes a hard API error even though the
            # requested value is the documented Hermes default.
            if effective_auto_delete or ephemeral or gpu_requested:
                create_params_kwargs["auto_delete_interval"] = effective_auto_delete
            # Forward ephemeral flag to SDK so it enables ephemeral sandbox
            # behavior (the config key is accepted but was not previously wired
            # to the SDK constructor param).
            if ephemeral:
                create_params_kwargs["ephemeral"] = True

            # Forward environment variables
            if env_vars:
                create_params_kwargs["env_vars"] = env_vars

            # Forward network parameters
            if network_block_all:
                create_params_kwargs["network_block_all"] = True
            if network_allow_list:
                create_params_kwargs["network_allow_list"] = network_allow_list

            # Forward language
            if language:
                create_params_kwargs["language"] = language

            # Forward volume mounts
            if volume_mounts:
                VolumeMount: Any = None
                for module_name in ("daytona.common.volume", "daytona_sdk.common.volume"):
                    try:
                        VolumeMount = getattr(importlib.import_module(module_name), "VolumeMount")
                        break
                    except (ImportError, AttributeError):
                        continue

                if VolumeMount is None:
                    raise ImportError(
                        "Installed Daytona SDK does not expose VolumeMount; "
                        "upgrade the daytona package or remove "
                        "daytona_volume_mounts"
                    )

                create_params_kwargs["volumes"] = [
                    VolumeMount(**vm) if isinstance(vm, dict) else vm
                    for vm in volume_mounts
                ]

            if create_mode == "snapshot":
                if CreateSandboxFromSnapshotParams is None:
                    raise ImportError(
                        "Installed Daytona SDK does not expose "
                        "CreateSandboxFromSnapshotParams; upgrade the daytona "
                        "package to use daytona_create_mode='snapshot'"
                    )
                # Snapshot mode: use CreateSandboxFromSnapshotParams.
                # The Daytona SDK (>=0.155.0) does not expose a `resources`
                # field on CreateSandboxFromSnapshotParams — snapshot-owned
                # resources take precedence. Remove `resources` from the
                # shared kwargs dict before passing to the snapshot params
                # so we don't silently pass a field the SDK ignores.
                snapshot_kwargs = {
                    k: v for k, v in create_params_kwargs.items()
                    if k != "resources"
                }
                snapshot_kwargs["snapshot"] = snapshot  # guaranteed non-empty by validation
                # Snapshot mode should not set an image
                self._sandbox = self._daytona.create(
                    CreateSandboxFromSnapshotParams(**snapshot_kwargs)
                )
            else:
                # Image mode (default): use CreateSandboxFromImageParams
                create_params_kwargs["image"] = image
                self._sandbox = self._daytona.create(
                    CreateSandboxFromImageParams(**create_params_kwargs)
                )

            logger.info("Daytona: created sandbox %s for task %s (mode=%s, profile_id=%s)",
                        self._sandbox.id, task_id, create_mode, profile_id)

        # Detect remote home dir
        self._remote_home = "/root"
        try:
            home = self._sandbox.process.exec("echo $HOME").result.strip()
            if home:
                self._remote_home = home
                if requested_cwd in {"~", "/home/daytona"}:
                    self.cwd = home
        except Exception:
            pass
        logger.info("Daytona: resolved home to %s, cwd to %s", self._remote_home, self.cwd)

        self._sync_manager = FileSyncManager(
            get_files_fn=lambda: iter_sync_files(f"{self._remote_home}/.hermes"),
            upload_fn=self._daytona_upload,
            delete_fn=self._daytona_delete,
            bulk_upload_fn=self._daytona_bulk_upload,
            bulk_download_fn=self._daytona_bulk_download,
        )
        self._sync_manager.sync(force=True)

        # --- CWD sync ---
        # When enabled (daytona_sync_cwd=True), sync the host CWD into the
        # sandbox under /workspace after the .hermes config sync completes.
        # This is explicitly opt-in because uploading host project directories
        # to cloud sandboxes has security and cost implications.
        self._sync_cwd = sync_cwd
        self._host_cwd = host_cwd
        if self._sync_cwd:
            if self._sync_cwd_to_sandbox() and requested_cwd in {"~", "/home/daytona", "/root"}:
                self.cwd = "/workspace"

        self.init_session()

    def _daytona_upload(self, host_path: str, remote_path: str) -> None:
        """Upload a single file via Daytona SDK."""
        parent = str(Path(remote_path).parent)
        self._sandbox.process.exec(f"mkdir -p {parent}")
        self._sandbox.fs.upload_file(host_path, remote_path)

    def _daytona_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        """Upload many files in a single HTTP call via Daytona SDK.

        Uses ``sandbox.fs.upload_files()`` which batches all files into one
        multipart POST, avoiding per-file TLS/HTTP overhead (~580 files
        goes from ~5 min to <2 s).
        """
        from daytona.common.filesystem import FileUpload

        if not files:
            return

        parents = unique_parent_dirs(files)
        if parents:
            self._sandbox.process.exec(quoted_mkdir_command(parents))

        uploads = [
            FileUpload(source=host_path, destination=remote_path)
            for host_path, remote_path in files
        ]
        self._sandbox.fs.upload_files(uploads)

    def _daytona_bulk_download(self, dest: Path) -> None:
        """Download remote .hermes/ as a tar archive."""
        rel_base = f"{self._remote_home}/.hermes".lstrip("/")
        # PID-suffixed remote temp path avoids collisions if sync_back fires
        # concurrently for the same sandbox (e.g. retry after partial failure).
        remote_tar = f"/tmp/.hermes_sync.{os.getpid()}.tar"
        self._sandbox.process.exec(
            f"tar cf {shlex.quote(remote_tar)} -C / {shlex.quote(rel_base)}"
        )
        self._sandbox.fs.download_file(remote_tar, str(dest))
        # Clean up remote temp file
        try:
            self._sandbox.process.exec(f"rm -f {shlex.quote(remote_tar)}")
        except Exception:
            pass  # best-effort cleanup

    def _daytona_delete(self, remote_paths: list[str]) -> None:
        """Batch-delete remote files via SDK exec."""
        self._sandbox.process.exec(quoted_rm_command(remote_paths))

    # ------------------------------------------------------------------
    # CWD sync
    # ------------------------------------------------------------------

    # Directories and file patterns that are never synced to the sandbox.
    # This protects against accidentally uploading secrets, large build
    # artifacts, or version-control internals to a cloud sandbox.
    _CWD_EXCLUDE_DIRS: tuple[str, ...] = (
        ".git",
        "__pycache__",
        "node_modules",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".venv",
        "venv",
        ".env",
        ".aws",
        ".azure",
        ".gcloud",
        ".gnupg",
        ".kube",
        ".ssh",
        ".docker",
        # Daytona-specific: avoid recursive sync
        ".daytona",
    )

    # Individual file patterns (basename matches) that are never synced.
    # Covers secrets, credentials, and large binary artifacts.
    _CWD_EXCLUDE_FILES: tuple[str, ...] = (
        ".env",
        ".env.local",
        ".env.production",
        ".env.development",
        ".envrc",
        ".npmrc",
        ".pypirc",
        ".netrc",
        ".git-credentials",
        ".dockercfg",
        "credentials.json",
        "id_rsa",
        "id_ed25519",
    )

    _CWD_EXCLUDE_EXTENSIONS: tuple[str, ...] = (
        ".pem",
        ".key",
        ".p12",
        ".pfx",
        ".crt",
        ".cert",
    )

    # Maximum total size of the CWD sync payload (bytes). Projects exceeding
    # this are not uploaded — the sandbox starts with /workspace empty and
    # the user must manage their own file transfer.
    _CWD_MAX_BYTES: int = 100 * 1024 * 1024  # 100 MiB

    @classmethod
    def _is_cwd_excluded_file(cls, fname: str) -> bool:
        """Return True when a CWD-sync basename must never be uploaded."""
        lower_fname = fname.lower()
        if lower_fname.startswith(".env"):
            return True
        if lower_fname in {p.lower() for p in cls._CWD_EXCLUDE_FILES}:
            return True
        if (
            lower_fname.startswith("secret")
            or ".secret." in lower_fname
            or lower_fname.endswith((".secret", ".secrets", ".secrets.yaml", ".secrets.yml"))
        ):
            return True
        return lower_fname.endswith(tuple(ext.lower() for ext in cls._CWD_EXCLUDE_EXTENSIONS))

    def _is_excluded_cwd_path(self, path: str, root: str, *, is_dir: bool) -> bool:
        """Return True if a CWD-sync path is excluded by basename or relative path."""
        rel = os.path.relpath(path, root).replace(os.sep, "/").lower()
        name = os.path.basename(path).lower()
        if is_dir:
            lowered_patterns = {p.lower() for p in self._CWD_EXCLUDE_DIRS}
            return rel in lowered_patterns or name in lowered_patterns
        return rel in {p.lower() for p in self._CWD_EXCLUDE_FILES} or self._is_cwd_excluded_file(name)

    def _clear_cwd_workspace(self) -> None:
        """Clear the managed CWD-sync workspace without touching .hermes sync state."""
        sandbox: Any = self._sandbox
        if sandbox is None:
            raise RuntimeError("Daytona sync_cwd: sandbox is not initialized")
        sandbox.process.exec(
            "mkdir -p /workspace && "
            "find /workspace -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +"
        )

    def _sync_cwd_to_sandbox(self) -> bool:
        """Sync the host CWD into /workspace in the Daytona sandbox.

        Only runs when daytona_sync_cwd=True. Applies exclusion rules
        (directories, files) and a total size limit before uploading.
        The host_cwd (from config) determines the source directory.
        If no host_cwd is available, the sync is skipped with a warning.
        Returns True only when /workspace was prepared from a valid explicit
        source, even if no files survived exclusions.
        """
        from hermes_constants import get_hermes_home

        # Determine the source directory. The terminal tool wires
        # TERMINAL_DAYTONA_SYNC_CWD_SOURCE into self._host_cwd; direct
        # construction can also set self._host_cwd. Do not fall back to
        # TERMINAL_CWD or os.getcwd(): gateway/run.py may default TERMINAL_CWD
        # to the operator home, and that must not trigger a home upload.
        host_cwd = getattr(self, "_host_cwd", None) or os.environ.get("TERMINAL_DAYTONA_SYNC_CWD_SOURCE")
        if host_cwd:
            host_cwd = os.path.abspath(os.path.expanduser(host_cwd))
        if not host_cwd or not os.path.isdir(host_cwd):
            logger.warning("Daytona sync_cwd: no explicit valid host CWD source (%r) — skipping sync", host_cwd)
            return False

        # Never sync the operator home or .hermes home directory. .hermes is
        # already handled by FileSyncManager above; the home guard prevents a
        # gateway-expanded TERMINAL_CWD=$HOME from becoming an upload source.
        hermes_home = str(get_hermes_home().resolve())
        resolved_host_cwd = os.path.realpath(host_cwd)
        operator_home = os.path.realpath(str(Path.home()))
        if resolved_host_cwd == operator_home:
            logger.info("Daytona sync_cwd: host CWD is operator home — skipping")
            return False
        if resolved_host_cwd == os.path.realpath(hermes_home):
            logger.info("Daytona sync_cwd: host CWD is .hermes home — skipping (already synced)")
            return False

        # Collect files, applying exclusions and size checks.
        # Perform a full size check first so we never upload a partial project.
        files_to_sync: list[tuple[str, str]] = []
        total_size = 0

        for dirpath, dirnames, filenames in os.walk(host_cwd):
            # Prune excluded directories in-place (os.walk respects mutations).
            # Also prune symlinked directories so the CWD sync never traverses
            # through links that may point outside the requested host CWD.
            dirnames[:] = [
                d for d in dirnames
                if not self._is_excluded_cwd_path(os.path.join(dirpath, d), host_cwd, is_dir=True)
                and d != ".hermes"  # .hermes is synced separately
                and not os.path.islink(os.path.join(dirpath, d))
            ]

            for fname in filenames:
                # Skip excluded file patterns
                if self._is_excluded_cwd_path(os.path.join(dirpath, fname), host_cwd, is_dir=False):
                    continue
                # Skip files with excluded extensions (case-insensitive)
                if any(fname.lower().endswith(ext) for ext in (".pem", ".key", ".p12", ".pfx")):
                    continue

                local_path = os.path.join(dirpath, fname)
                # Do not upload symlinked files. Even when the link target is
                # inside the project, uploading via the symlink path can leak
                # data outside the intended CWD boundary if the link is changed
                # between collection and upload.
                if os.path.islink(local_path):
                    continue

                resolved_local_path = os.path.realpath(local_path)
                try:
                    if os.path.commonpath([resolved_host_cwd, resolved_local_path]) != resolved_host_cwd:
                        continue
                    if not os.path.isfile(resolved_local_path):
                        continue
                    fsize = os.path.getsize(resolved_local_path)
                except (OSError, ValueError):
                    continue

                total_size += fsize
                # Map to /workspace/<relative_path> in sandbox
                rel = os.path.relpath(local_path, host_cwd)
                remote_path = f"/workspace/{rel}"
                files_to_sync.append((local_path, remote_path))

        if total_size > self._CWD_MAX_BYTES:
            logger.warning(
                "Daytona sync_cwd: CWD exceeds %d MiB size limit "
                "(%d MiB collected) — aborting sync, no files uploaded",
                self._CWD_MAX_BYTES // (1024 * 1024),
                total_size // (1024 * 1024),
            )
            return False

        # /workspace is a managed mirror of the explicit host CWD source. Clear
        # it before every successful sync so persistent sandboxes cannot retain
        # files deleted locally or files that become newly excluded secrets.
        self._clear_cwd_workspace()

        if not files_to_sync:
            logger.info("Daytona sync_cwd: no files to sync after exclusions")
            return True

        logger.info(
            "Daytona sync_cwd: uploading %d file(s) (%d KiB) to /workspace",
            len(files_to_sync),
            total_size // 1024,
        )

        self._daytona_bulk_upload(files_to_sync)
        return True

    # ------------------------------------------------------------------
    # Sandbox lifecycle
    # ------------------------------------------------------------------

    def _ensure_sandbox_ready(self) -> None:
        """Restart sandbox if it was stopped (e.g., by a previous interrupt)."""
        self._sandbox.refresh_data()
        if self._sandbox.state in {self._SandboxState.STOPPED, self._SandboxState.ARCHIVED}:
            self._sandbox.start()
            logger.info("Daytona: restarted sandbox %s", self._sandbox.id)

    def _before_execute(self) -> None:
        """Ensure sandbox is ready, then sync files via FileSyncManager."""
        with self._lock:
            self._ensure_sandbox_ready()
        self._sync_manager.sync()

    def _run_bash(self, cmd_string: str, *, login: bool = False,
                  timeout: int = 120,
                  stdin_data: str | None = None):
        """Return a _ThreadedProcessHandle wrapping a blocking Daytona SDK call."""
        sandbox = self._sandbox
        lock = self._lock

        def cancel():
            with lock:
                try:
                    sandbox.stop()
                except Exception:
                    pass

        if login:
            shell_cmd = f"bash -l -c {shlex.quote(cmd_string)}"
        else:
            shell_cmd = f"bash -c {shlex.quote(cmd_string)}"

        def exec_fn() -> tuple[str, int]:
            response = sandbox.process.exec(shell_cmd, timeout=timeout)
            return (response.result or "", response.exit_code)

        return _ThreadedProcessHandle(exec_fn, cancel_fn=cancel)

    def cleanup(self):
        with self._lock:
            if self._sandbox is None:
                return

            # Sync remote changes back to host before teardown. Running
            # inside the lock (and after the _sandbox is None guard) avoids
            # firing sync_back on an already-cleaned-up env, which would
            # trigger a 3-attempt retry storm against a nil sandbox.
            if self._sync_manager:
                logger.info("Daytona: syncing files from sandbox...")
                try:
                    self._sync_manager.sync_back()
                except Exception as e:
                    logger.warning("Daytona: sync_back failed: %s", e)

            try:
                if self._persistent:
                    self._sandbox.stop()
                    logger.info("Daytona: stopped sandbox %s (filesystem preserved)",
                                self._sandbox.id)
                else:
                    self._daytona.delete(self._sandbox)
                    logger.info("Daytona: deleted sandbox %s", self._sandbox.id)
            except Exception as e:
                logger.warning("Daytona: cleanup failed: %s", e)
            self._sandbox = None
