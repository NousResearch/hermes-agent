"""Modal cloud execution environment using the native Modal SDK directly
(``Sandbox.create()`` + ``Sandbox.exec()``) with persistent snapshots across sessions."""

import asyncio
import base64
import io
import itertools
import logging
import os
import shlex
import tarfile
import threading
from pathlib import Path
from typing import Any, Mapping, Optional

from hermes_constants import display_hermes_home, get_hermes_home
from tools.environments.base import BaseEnvironment, _load_json_store, _save_json_store
from tools.environments.base_output import _ThreadedProcessHandle
from tools.environments.file_sync import (
    FileSyncManager, iter_sync_files, quoted_mkdir_command, quoted_rm_command, unique_parent_dirs)
from tools.environments.remote_common import bash_argv, ensure_lazy_dep

logger = logging.getLogger(__name__)

def _snapshot_store() -> Path:
    # Resolved per call: the multiplexed gateway serves every profile from one process, so an
    # import-time path would keep every profile's snapshots in the launch profile's home.
    return get_hermes_home() / "modal_snapshots.json"


def _load_snapshots() -> dict:
    return _load_json_store(_snapshot_store())


def _save_snapshots(data: dict) -> None:
    _save_json_store(_snapshot_store(), data)


def _get_snapshot_restore_candidate(task_id: str) -> tuple[str | None, bool]:
    """Return (snapshot_id, from_legacy_key); the namespaced key wins over the legacy bare task id."""
    snapshots = _load_snapshots()
    for key, legacy in ((f"direct:{task_id}", False), (task_id, True)):
        snapshot_id = snapshots.get(key)
        if isinstance(snapshot_id, str) and snapshot_id:
            return snapshot_id, legacy
    return None, False


def _store_direct_snapshot(task_id: str, snapshot_id: str) -> None:
    snapshots = _load_snapshots()
    snapshots[f"direct:{task_id}"] = snapshot_id
    snapshots.pop(task_id, None)
    _save_snapshots(snapshots)


def _delete_direct_snapshot(task_id: str, snapshot_id: str | None = None) -> None:
    snapshots = _load_snapshots()
    stale = [k for k in (f"direct:{task_id}", task_id)
             if snapshots.get(k) is not None and snapshot_id in (None, snapshots[k])]
    for key in stale:
        snapshots.pop(key)
    if stale:
        _save_snapshots(snapshots)


# The tool layer's real bound on this path: tools/registry.py caps tool error
# bodies at _MAX_TOOL_ERROR_CHARS = 2048 with a hard mid-word cut plus a
# "… [truncated]" marker, after terminal_tool prefixes "Failed to execute
# command: ". Capping the SDK detail at ~200 chars (the SDK's "Token missing"
# text measures 219) keeps every branch far inside 2048 so the Fix line
# always survives; the cut keeps whole words only when that retains at least
# half the cap, otherwise it hard-cuts rather than dropping the message.
_AUTH_DETAIL_LIMIT = 200


def _diagnose_modal_auth_error(exc: BaseException, *,
                               environ: Optional[Mapping[str, str]] = None,
                               config_path: Optional[str] = None) -> str:
    """Turn a bare modal AuthError into a diagnosis naming the credential the
    SDK actually read. Hermes publishes $HERMES_HOME/.env into os.environ, so
    MODAL_TOKEN_ID/SECRET there silently beat a ~/.modal.toml profile that
    `modal token info` (a plain shell) reports as healthy (#47264). The SDK
    reads its config from MODAL_CONFIG_PATH or ~/.modal.toml, so name that
    resolved path, never a hardcoded one — and when no config file exists the
    diagnosis must not present one as the source. environ/config_path are
    injectable so every branch is testable without touching real env or files.
    Names/paths only — the tool layer caps and redacts this text; the first
    line must carry the point."""
    environ = os.environ if environ is None else environ
    if config_path is None:
        config_path = environ.get("MODAL_CONFIG_PATH") or os.path.expanduser("~/.modal.toml")
    detail = str(exc) or exc.__class__.__name__
    if len(detail) > _AUTH_DETAIL_LIMIT:
        # Whole-word trim when it keeps most of the text; hard cut otherwise:
        # a detail like "Error: " + a 200-char unbroken token would trim back
        # to "Error:…" and lose every specific, so only trust the word
        # boundary when it retains at least half the limit (no spaces at all
        # means rsplit returns the full slice, i.e. the same hard cut).
        head = detail[:_AUTH_DETAIL_LIMIT].rsplit(" ", 1)[0].rstrip()
        kept = head if len(head) >= _AUTH_DETAIL_LIMIT // 2 else detail[:_AUTH_DETAIL_LIMIT].rstrip()
        detail = kept + "…"
    # Membership test, not truthiness: the SDK decides with
    # `env_var_key in os.environ` (modal/config.py Config.get), so
    # MODAL_TOKEN_ID="" IS the credential it read — list by membership.
    env_set = [v for v in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET") if v in environ]
    if env_set:
        # The SDK consults the config file per MISSING half only, so claim
        # that only when a half is actually missing — with both vars set,
        # nothing is read from the file. Both wordings stay inside the length
        # budget pinned by the worst-case test; the no-file sub-case is the
        # longer tail, so it drops the "(plain shell)" aside to keep a
        # 60-char path plus a 200-char capped detail under 640.
        if len(env_set) == 2:
            source_note = (
                f"Source: {'/'.join(env_set)} in the environment supplies both "
                f"halves; nothing is read from {config_path}; `modal token "
                "info` (plain shell) never sees the env.")
        else:
            missing = "MODAL_TOKEN_SECRET" if env_set[0] == "MODAL_TOKEN_ID" else "MODAL_TOKEN_ID"
            # Say where the SDK LOOKED and whether anything was usable —
            # never that the half "comes from" the file. Even when the path
            # exists the SDK may read nothing usable from it (no matching
            # profile section, no token entry, an unexpanded ~ it never
            # opens), and a diagnostic path must not parse the user's TOML
            # to find out which; acquisition claims would repeat the
            # false-attribution class this diagnosis exists to remove.
            if Path(config_path).exists():
                source_note = (
                    f"Source: {env_set[0]} in the environment; {missing} is not set, "
                    "so the SDK looks for that half in "
                    f"{config_path}; `modal token info` (plain shell) never sees the env.")
            else:
                # No file at the resolved path: Config.get finds nothing for
                # the missing half, so client.py::from_env sees a falsy half,
                # builds no credentials, and raises "Token missing…".
                # "supplied nothing" is truthful for an absent, unreadable or
                # unexpanded-~ path alike — the SDK opened none of them.
                source_note = (
                    f"Source: {env_set[0]} in the environment; {missing} is not set and "
                    f"{config_path} supplied nothing, so the SDK built no complete "
                    "credential pair. `modal token info` never sees the env.")
        return (
            f"Modal authentication failed ({detail}).\n"
            f"{source_note}\n"
            f"Fix: unset MODAL_TOKEN_ID/MODAL_TOKEN_SECRET in the env Hermes runs "
            f"in (normally {display_hermes_home()}/.env), or re-run `hermes setup` → "
            "Terminal → Modal.")
    if Path(config_path).exists():
        return (
            f"Modal authentication failed ({detail}).\n"
            f"Source: the Modal profile in {config_path} (no MODAL_TOKEN_ID/MODAL_TOKEN_SECRET "
            "in the environment).\n"
            "Fix: run `modal token new` to save fresh credentials, or re-run "
            "`hermes setup` → Terminal → Modal.")
    return (
        f"Modal authentication failed ({detail}).\n"
        f"Source: no Modal credentials found — no MODAL_TOKEN_ID/MODAL_TOKEN_SECRET in the "
        f"environment, and no config file at {config_path}.\n"
        "Fix: run `hermes setup` → Terminal → Modal, or `modal token new` to register "
        "credentials.")


def _resolve_modal_image(image_spec: Any) -> Any:
    """Convert registry references or snapshot ids into Modal image objects. Registry images
    get pip repaired (ensurepip) before Modal's bootstrap; ubuntu/debian also get python3."""
    ensure_lazy_dep("modal")
    import modal as _modal

    if not isinstance(image_spec, str):
        return image_spec
    if image_spec.startswith("im-"):
        return _modal.Image.from_id(image_spec)
    setup_commands = [
        "RUN rm -rf /usr/local/lib/python*/site-packages/pip* 2>/dev/null; "
        "python -m ensurepip --upgrade --default-pip 2>/dev/null || true"]
    if any(base in image_spec.lower() for base in ("ubuntu", "debian")):
        setup_commands.insert(0,
            "RUN apt-get update -qq && apt-get install -y -qq python3 python3-venv > /dev/null 2>&1 || true")
    return _modal.Image.from_registry(image_spec, setup_dockerfile_commands=setup_commands)


async def _stream_stdin(proc, payload: str, chunk_size: int) -> None:
    """Write ``payload`` to ``proc.stdin`` in ``chunk_size`` pieces, draining after each, then EOF."""
    for offset in range(0, len(payload), chunk_size):
        proc.stdin.write(payload[offset:offset + chunk_size])
        await proc.stdin.drain.aio()
    proc.stdin.write_eof()
    await proc.stdin.drain.aio()


def _as_text(value) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


class _AsyncWorker:
    """Background thread with its own event loop for async-safe Modal calls."""

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()

    def start(self):
        def _run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._started.set()
            self._loop.run_forever()
        self._thread = threading.Thread(target=_run_loop, daemon=True)
        self._thread.start()
        self._started.wait(timeout=30)

    def run_coroutine(self, coro, timeout=600):
        from agent.async_utils import safe_schedule_threadsafe
        # safe_schedule_threadsafe closes the coroutine and returns None for a missing/closed loop.
        future = safe_schedule_threadsafe(coro, self._loop)
        if future is None:
            raise RuntimeError("AsyncWorker loop is not running")
        return future.result(timeout=timeout)

    def stop(self):
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=10)


class ModalEnvironment(BaseEnvironment):
    """Modal cloud execution via native Modal sandboxes: spawn-per-call via _ThreadedProcessHandle
    wrapping async SDK calls, cancel_fn wired to sandbox.terminate for interrupt support."""

    _stdin_mode = "heredoc"
    _snapshot_timeout = 60  # Modal cold starts can be slow
    # Modal SDK stdin buffer limit: the command-router path allows 16 MB but the legacy server
    # path caps at 2 MB, so chunks stay under 2 MB and each is flushed individually via drain().
    _STDIN_CHUNK_SIZE = 1 * 1024 * 1024

    def __init__(self, image: str, cwd: str = "/root", timeout: int = 60,
                 modal_sandbox_kwargs: Optional[dict[str, Any]] = None,
                 persistent_filesystem: bool = True, task_id: str = "default"):
        super().__init__(cwd=cwd, timeout=timeout)
        self._persistent, self._task_id = persistent_filesystem, task_id
        self._sandbox = self._app = None
        self._worker = _AsyncWorker()
        self._sync_manager: FileSyncManager | None = None  # initialized after sandbox creation
        restored_snapshot_id, restored_from_legacy_key = (
            _get_snapshot_restore_candidate(self._task_id) if self._persistent else (None, False))
        if restored_snapshot_id:
            logger.info("Modal: restoring from snapshot %s", restored_snapshot_id[:20])
        ensure_lazy_dep("modal")
        import modal as _modal
        cred_mounts = []
        try:
            from tools.credential_files import get_credential_file_mounts, iter_skills_files, iter_cache_files
            # from_iterable keeps each source lazy so a failure mid-way leaves the earlier mounts in place
            for entry in itertools.chain.from_iterable(
                    fn() for fn in (get_credential_file_mounts, iter_skills_files, iter_cache_files)):
                cred_mounts.append(
                    _modal.Mount.from_local_file(entry["host_path"], remote_path=entry["container_path"]))
        except Exception as e:
            logger.debug("Modal: could not load credential file mounts: %s", e)
        self._worker.start()

        def _create(image_spec: Any) -> None:
            async def _create_sandbox():
                app = await _modal.App.lookup.aio("hermes-agent", create_if_missing=True)
                create_kwargs = dict(modal_sandbox_kwargs or {})
                if cred_mounts:
                    create_kwargs["mounts"] = list(create_kwargs.pop("mounts", [])) + cred_mounts
                sandbox = await _modal.Sandbox.create.aio(
                    "sleep", "infinity", image=image_spec, app=app,
                    timeout=int(create_kwargs.pop("timeout", 3600)), **create_kwargs)
                return app, sandbox
            self._app, self._sandbox = self._worker.run_coroutine(_create_sandbox(), timeout=300)
        try:
            try:
                _create(_resolve_modal_image(restored_snapshot_id or image))
            except _modal.exception.AuthError:
                # Auth dies at the first RPC — before any sandbox exists — so the
                # recorded snapshot was never suspect: re-raise past the
                # delete-and-retry below (it would silently destroy restore
                # state) and let the outer AuthError arm translate (#47264).
                raise
            except Exception as exc:
                if not restored_snapshot_id:
                    raise
                logger.warning("Modal: failed to restore snapshot %s, retrying with base image: %s",
                               restored_snapshot_id[:20], exc)
                _delete_direct_snapshot(self._task_id, restored_snapshot_id)
                _create(_resolve_modal_image(image))
            else:
                if restored_snapshot_id and restored_from_legacy_key:
                    _store_direct_snapshot(self._task_id, restored_snapshot_id)
        except _modal.exception.AuthError as exc:
            # The skip of the delete-and-retry for the initial attempt is the
            # INNER `except _modal.exception.AuthError: raise` arm's doing
            # (auth dies at the first RPC, before any sandbox, so the recorded
            # snapshot is never suspect). This arm only translates the
            # failure — from the initial attempt or from the base-image retry
            # — into a diagnosed RuntimeError instead of a bare SDK AuthError,
            # and stops the worker. Clause before the generic one: AuthError
            # subclasses Exception.
            self._worker.stop()
            raise RuntimeError(_diagnose_modal_auth_error(exc)) from exc
        except Exception:
            self._worker.stop()
            raise
        logger.info("Modal: sandbox created (task=%s)", self._task_id)
        self._sync_manager = FileSyncManager(
            get_files_fn=lambda: iter_sync_files("/root/.hermes"),
            upload_fn=self._modal_upload, delete_fn=self._modal_delete,
            bulk_upload_fn=self._modal_bulk_upload, bulk_download_fn=self._modal_bulk_download)
        self._sync_manager.sync(force=True)
        self.init_session()

    def _exec(self, cmd: str, *, timeout: int, stdin: str | None = None, fail_label: str | None = None,
              capture: bool = False):
        """Run ``bash -c cmd`` in the sandbox. ``stdin`` is streamed in chunks; ``capture`` returns
        stdout; ``fail_label`` turns a non-zero exit into RuntimeError (with stderr unless capturing)."""
        async def _run():
            proc = await self._sandbox.exec.aio("bash", "-c", cmd)
            if stdin is not None:
                await _stream_stdin(proc, stdin, self._STDIN_CHUNK_SIZE)
            data = await proc.stdout.read.aio() if capture else None
            exit_code = await proc.wait.aio()
            if fail_label and exit_code != 0:
                detail = "" if capture else f": {await proc.stderr.read.aio()}"
                raise RuntimeError(f"Modal {fail_label} failed (exit {exit_code}){detail}")
            return data
        return self._worker.run_coroutine(_run(), timeout=timeout)

    def _modal_upload(self, host_path: str, remote_path: str) -> None:
        """Upload a single file via base64 piped through stdin."""
        cmd = f"mkdir -p {shlex.quote(str(Path(remote_path).parent))} && base64 -d > {shlex.quote(remote_path)}"
        self._exec(cmd, stdin=base64.b64encode(Path(host_path).read_bytes()).decode("ascii"), timeout=30)

    def _modal_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        """Upload many files as one in-memory gzipped tar streamed through stdin
        into ``base64 -d | tar xzf -``, avoiding the SDK's 64 KB exec-arg limit."""
        if not files:
            return
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for host_path, remote_path in files:
                tar.add(host_path, arcname=remote_path.lstrip("/"))
        payload = base64.b64encode(buf.getvalue()).decode("ascii")
        cmd = f"{quoted_mkdir_command(unique_parent_dirs(files))} && base64 -d | tar xzf - -C /"
        self._exec(cmd, stdin=payload, timeout=120, fail_label="bulk upload")

    def _modal_bulk_download(self, dest: Path) -> None:
        """Download remote .hermes/ as a tar archive (sandboxes run as root, so /root/.hermes)."""
        # --exclude: live sockets cannot be archived ("socket ignored") and must not fail the download.
        data = self._exec("tar cf - --exclude='*.sock' -C / root/.hermes", timeout=120, fail_label="bulk download", capture=True)
        dest.write_bytes(data.encode() if isinstance(data, str) else data)

    def _modal_delete(self, remote_paths: list[str]) -> None:
        self._exec(quoted_rm_command(remote_paths), timeout=15)

    def _before_execute(self) -> None:
        self._sync_manager.sync()  # rate-limited internally

    def _run_bash(self, cmd_string: str, *, login: bool = False, timeout: int = 120, stdin_data: str | None = None):
        sandbox, worker = self._sandbox, self._worker

        def cancel():
            worker.run_coroutine(sandbox.terminate.aio(), timeout=15)

        def exec_fn() -> tuple[str, int]:
            async def _do():
                process = await sandbox.exec.aio(*bash_argv(cmd_string, login), timeout=timeout)
                stdout = _as_text(await process.stdout.read.aio())
                stderr = _as_text(await process.stderr.read.aio())
                exit_code = await process.wait.aio()
                return "\n".join(part for part in (stdout, stderr) if part), exit_code
            return worker.run_coroutine(_do(), timeout=timeout + 30)
        return _ThreadedProcessHandle(exec_fn, cancel_fn=cancel)

    def cleanup(self):
        """Snapshot the filesystem (if persistent) then stop the sandbox."""
        if self._sandbox is None:
            return
        if self._sync_manager:
            logger.info("Modal: syncing files from sandbox...")
            self._sync_manager.sync_back()
        if self._persistent:
            async def _snapshot():
                return (await self._sandbox.snapshot_filesystem.aio()).object_id
            try:
                snapshot_id = self._worker.run_coroutine(_snapshot(), timeout=60)
            except Exception:
                snapshot_id = None  # snapshot errors are non-fatal; the sandbox is still terminated
            if snapshot_id:
                try:
                    _store_direct_snapshot(self._task_id, snapshot_id)
                    logger.info("Modal: saved filesystem snapshot %s for task %s", snapshot_id[:20], self._task_id)
                except Exception as e:
                    logger.warning("Modal: filesystem snapshot failed: %s", e)
        try:
            self._worker.run_coroutine(self._sandbox.terminate.aio(), timeout=15)
        except Exception:
            pass
        finally:
            self._worker.stop()
            self._sandbox = self._app = None
