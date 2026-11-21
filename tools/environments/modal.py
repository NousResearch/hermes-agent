"""Modal cloud execution environment using the native Modal SDK directly
(``Sandbox.create()`` + ``Sandbox.exec()``) with persistent snapshots across sessions."""

import asyncio
import base64
import functools
import io
import itertools
import logging
import shlex
import tarfile
import threading
import weakref
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_constants import get_hermes_home, hermes_home_key
from tools.environments.base import BaseEnvironment, _update_json_store
from tools.environments.base_output import _ThreadedProcessHandle
from tools.environments.file_sync import (
    FileSyncManager, iter_sync_files, quoted_mkdir_command, quoted_rm_command, unique_parent_dirs)
from tools.environments.remote_common import bash_argv, ensure_lazy_dep
from utils import read_json_or_empty

logger = logging.getLogger(__name__)

def _snapshot_store() -> Path:
    # Resolved per call: the multiplexed gateway serves every profile from one process, so an
    # import-time path would keep every profile's snapshots in the launch profile's home.
    return get_hermes_home() / "modal_snapshots.json"


def _load_snapshots() -> dict:
    return read_json_or_empty(_snapshot_store())


def _update_snapshots(update: Callable[[dict], None]) -> None:
    _update_json_store(_snapshot_store(), update)


def _get_snapshot_restore_candidate(task_id: str) -> tuple[str | None, bool]:
    """Return (snapshot_id, from_legacy_key); the namespaced key wins over the legacy bare task id."""
    snapshots = _load_snapshots()
    for key, legacy in ((f"direct:{task_id}", False), (task_id, True)):
        snapshot_id = snapshots.get(key)
        if isinstance(snapshot_id, str) and snapshot_id:
            return snapshot_id, legacy
    return None, False


def _store_direct_snapshot(task_id: str, snapshot_id: str) -> None:
    def update(snapshots: dict) -> None:
        snapshots[f"direct:{task_id}"] = snapshot_id
        snapshots.pop(task_id, None)

    _update_snapshots(update)


def _delete_direct_snapshot(task_id: str, snapshot_id: str | None = None) -> None:
    def update(snapshots: dict) -> None:
        stale = [k for k in (f"direct:{task_id}", task_id)
                 if snapshots.get(k) is not None and snapshot_id in (None, snapshots[k])]
        for key in stale:
            snapshots.pop(key)

    _update_snapshots(update)


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
    """Write byte-exact UTF-8 payload chunks to ``proc.stdin``, then EOF."""
    data = payload.encode("utf-8", "surrogateescape")
    for offset in range(0, len(data), chunk_size):
        proc.stdin.write(data[offset:offset + chunk_size])
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


# Profile home key -> (token pair, modal.Client built from it, its build's start order): one client
# per routed profile, like the routed FAL, Bedrock and Entra clients, replaced when that profile's
# pair changes. A replaced client is not closed: a sandbox created through it keeps running its RPCs
# on it.
_profile_clients: dict[str, tuple[tuple[str, str], Any, int]] = {}


@dataclass
class _ClientFlight:
    """One ``Client.from_credentials`` build in progress for a (home, pair). It runs on the loop of
    the environment that started it; concurrent first sandboxes, each on its own environment's
    loop, wait on ``done`` instead of opening a client of their own (the SDK keeps every opened
    client until process exit)."""

    loop: asyncio.AbstractEventLoop
    seq: int
    done: Future = field(default_factory=Future)
    settled: bool = False
    build: Optional[asyncio.Future] = None  # held: the loop keeps only a weak reference to a task


_client_flights: dict[tuple[str, tuple[str, str]], _ClientFlight] = {}
_client_flight_seq = itertools.count(1)
# Loops whose startup was abandoned: no client is admitted on them once their startup failed. Weak,
# so a stopped loop is not kept alive; code can only run on a loop while it is alive.
_revoked_loops: weakref.WeakSet[asyncio.AbstractEventLoop] = weakref.WeakSet()
# Guards _profile_clients, _client_flights and _revoked_loops; never held across an await.
_profile_clients_lock = threading.Lock()


def _routed_token_pair() -> tuple[str, str] | None:
    """The serving profile's own ``(MODAL_TOKEN_ID, MODAL_TOKEN_SECRET)`` when it is a routed
    profile, else None: a single-profile process keeps the SDK's own resolution (process env, then
    the ``modal`` CLI login in ``~/.modal.toml``). Without an explicit client the SDK uses one
    process-wide client built from ``os.environ``, which holds the launch profile's tokens."""
    from agent.secret_scope import get_secret, serves_routed_profile

    if not serves_routed_profile():
        return None
    token_id, token_secret = ((get_secret(name) or "").strip()
                              for name in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"))
    if not (token_id and token_secret):
        raise ValueError("MODAL_TOKEN_ID and MODAL_TOKEN_SECRET are not both set for this profile: add "
                         "them to this profile's .env (or its secret source). Another profile's Modal "
                         "tokens and the shared ~/.modal.toml login are never used for it.")
    return token_id, token_secret


async def _profile_client(modal: Any, home: str, pair: tuple[str, str]) -> Any:
    """The profile's cached client, or the one build in progress for this home and pair, started
    here when there is none. Waiters are shielded: a cancelled waiter never cancels the build the
    others wait on."""
    loop = asyncio.get_running_loop()
    with _profile_clients_lock:
        if loop in _revoked_loops:
            raise RuntimeError("Modal client not started: the sandbox startup running it was abandoned")
        cached = _profile_clients.get(home)
        if cached is not None and cached[0] == pair:
            return cached[1]
        flight = _client_flights.get((home, pair))
        starting = flight is None
        if flight is None:
            flight = _client_flights[(home, pair)] = _ClientFlight(loop, next(_client_flight_seq))
    if starting:
        try:
            build = flight.build = asyncio.ensure_future(modal.Client.from_credentials.aio(*pair))
        except Exception as exc:
            _settle_client_flight(home, pair, flight, None, exc)
            raise
        build.add_done_callback(functools.partial(_on_client_built, home, pair, flight))
    return await asyncio.shield(asyncio.wrap_future(flight.done))


def _on_client_built(home: str, pair: tuple[str, str], flight: _ClientFlight, build: asyncio.Future) -> None:
    if build.cancelled():
        _settle_client_flight(home, pair, flight, None, RuntimeError("Modal client build was cancelled"))
    elif (error := build.exception()) is not None:
        _settle_client_flight(home, pair, flight, None, error)
    else:
        _settle_client_flight(home, pair, flight, build.result(), None)


def _settle_client_flight(home: str, pair: tuple[str, str], flight: _ClientFlight, client: Any,
                          error: BaseException | None) -> None:
    """Drop the flight and hand its waiters the outcome. Only a build started after the cached
    client's replaces it, so an older pair finishing late never overwrites a newer one."""
    with _profile_clients_lock:
        if _client_flights.get((home, pair)) is flight:
            del _client_flights[(home, pair)]
        if error is None:
            cached = _profile_clients.get(home)
            if cached is None or cached[2] < flight.seq:
                _profile_clients[home] = (pair, client, flight.seq)
        notify = not flight.settled  # False once abandoned: its waiters already got the error
        flight.settled = True
    if notify:
        if error is None:
            flight.done.set_result(client)
        else:
            flight.done.set_exception(error)


def _abandon_client_flights(loop: Optional[asyncio.AbstractEventLoop]) -> None:
    """A failed or timed-out startup is about to stop ``loop``: fail and drop the client builds it
    was running, so their waiters fail now and later sandboxes start a fresh build instead of
    waiting on a stopped loop. The loop is revoked in the same critical section: a startup
    coroutine that reaches ``_profile_client`` only afterwards (the worker may run late, even past
    ``_AsyncWorker.stop``'s bounded join) is refused instead of registering a build nobody settles."""
    with _profile_clients_lock:
        if loop is not None:
            _revoked_loops.add(loop)
        abandoned = [(key, flight) for key, flight in _client_flights.items()
                     if flight.loop is loop and not flight.settled]
        for key, flight in abandoned:
            del _client_flights[key]
            flight.settled = True
    for _key, flight in abandoned:
        flight.done.set_exception(RuntimeError(
            "Modal client build abandoned: the sandbox startup running it failed or timed out"))


class _ClientBuildError(Exception):
    """Internal wrapper: routed client acquisition failed before snapshot restore began."""

    def __init__(self, cause: Exception) -> None:
        super().__init__(str(cause))
        self.cause = cause


class ModalEnvironment(BaseEnvironment):
    """Modal cloud execution via native Modal sandboxes: spawn-per-call via _ThreadedProcessHandle
    wrapping async SDK calls, cancel_fn wired to sandbox.terminate for interrupt support."""

    _stdin_mode = "payload"
    # The askpass prologue checks sudo ``-A`` support inside each command, so no capability probe
    # runs here (a timed-out probe could only be cancelled by stopping the sandbox).
    _sudo_askpass_runtime_checked = True
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
        # Read in the caller's thread: the SDK coroutines run on the worker loop, outside the turn's
        # profile scope.
        pair = _routed_token_pair()
        home = hermes_home_key() if pair else ""
        ensure_lazy_dep("modal")
        import modal as _modal
        self._worker.start()

        def _create(image_spec: Any) -> None:
            async def _create_sandbox():
                if pair:
                    try:
                        client = await _profile_client(_modal, home, pair)
                    except Exception as exc:
                        raise _ClientBuildError(exc) from exc
                else:
                    client = None
                app = await _modal.App.lookup.aio("hermes-agent", create_if_missing=True, client=client)
                create_kwargs = dict(modal_sandbox_kwargs or {})
                sandbox = await _modal.Sandbox.create.aio(
                    "sleep", "infinity", image=image_spec, app=app, client=client,
                    timeout=int(create_kwargs.pop("timeout", 3600)), **create_kwargs)
                return app, sandbox
            self._app, self._sandbox = self._worker.run_coroutine(_create_sandbox(), timeout=300)
        try:
            try:
                _create(_resolve_modal_image(restored_snapshot_id or image))
            except _ClientBuildError as exc:
                raise exc.cause from None
            except Exception as exc:
                if not restored_snapshot_id or (pair and isinstance(exc, TimeoutError)):
                    raise
                logger.warning("Modal: failed to restore snapshot %s, retrying with base image: %s",
                               restored_snapshot_id[:20], exc)
                _delete_direct_snapshot(self._task_id, restored_snapshot_id)
                _create(_resolve_modal_image(image))
            else:
                if restored_snapshot_id and restored_from_legacy_key:
                    _store_direct_snapshot(self._task_id, restored_snapshot_id)
        except BaseException as exc:
            _abandon_client_flights(self._worker._loop)
            self._worker.stop()
            if isinstance(exc, _ClientBuildError):  # the base-image retry's own client build failed
                raise exc.cause from None
            raise
        logger.info("Modal: sandbox created (task=%s)", self._task_id)
        self._sync_manager = FileSyncManager(
            get_files_fn=lambda: iter_sync_files("/root/.hermes"),
            upload_fn=self._modal_upload, delete_fn=self._modal_delete,
            bulk_upload_fn=self._modal_bulk_upload, bulk_download_fn=self._modal_bulk_download,
            get_remote_time_fn=self._modal_remote_time)
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

    def _modal_remote_time(self) -> float:
        """Read the sandbox clock before the initial file push."""
        return float(self._exec("date +%s", timeout=10, fail_label="remote clock", capture=True).strip())

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
                if stdin_data is not None:
                    await _stream_stdin(process, stdin_data, self._STDIN_CHUNK_SIZE)
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
                # ttl=None: the SDK default (30 days) would silently expire an idle
                # persistent sandbox's snapshot; Hermes owns its lifetime.
                return (await self._sandbox.snapshot_filesystem.aio(ttl=None)).object_id
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
            # Drop the handle before stopping the worker: a racing loop shutdown inside stop()
            # must not leave a terminated sandbox looking live (or raise out of cleanup).
            worker, self._sandbox, self._app = self._worker, None, None
            try:
                worker.stop()
            except Exception as e:
                logger.debug("Modal: worker stop after terminate failed: %s", e, exc_info=True)
