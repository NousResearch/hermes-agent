"""Tenki cloud sandbox execution environment."""

from __future__ import annotations

import hashlib
import inspect
import logging
import math
import os
import re
import shlex
import tarfile
import tempfile
import threading
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from tools.environments.base import (
    BaseEnvironment,
    _ThreadedProcessHandle,
    _load_json_store,
    _save_json_store,
)
from tools.environments.file_sync import (
    FileSyncManager,
    iter_sync_files,
    quoted_mkdir_command,
    quoted_rm_command,
    unique_parent_dirs,
)
from tools.tenki_config import (
    resolve_tenki_api_endpoint,
    resolve_tenki_auth_token,
    resolve_tenki_project_id,
    resolve_tenki_workspace_id,
)

logger = logging.getLogger(__name__)
_SNAPSHOT_NAMESPACE = "direct"
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _snapshot_store_path() -> Path:
    """Resolve the snapshot registry path for the *active* profile.

    Resolved per call (not frozen at import) so the multiplexing gateway,
    which overrides ``HERMES_HOME`` per turn, writes each profile's snapshot
    pointers into that profile's own home instead of whichever profile
    happened to import this module first.
    """
    return get_hermes_home() / "tenki_snapshots.json"


def _profile_token() -> str:
    """Short, stable identifier for the active Hermes profile.

    Two profiles sharing one Tenki account must get distinct sandbox
    names/metadata so they can never attach to or restore each other's
    sandbox. Prefer the canonical ``HERMES_PROFILE`` id, which is stable across
    machines and survives a home-directory move; only fall back to a
    *normalized* ``HERMES_HOME`` path when no profile id is set (the default
    profile). Resolving per call handles the multiplexing gateway's per-turn
    ``HERMES_HOME`` override (same reason as :func:`_snapshot_store_path`).
    """
    profile = os.getenv("HERMES_PROFILE", "").strip()
    if profile:
        basis = f"profile:{profile}"
    else:
        try:
            basis = str(get_hermes_home().resolve())
        except Exception:
            basis = str(get_hermes_home())
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:10]


def _load_snapshots(store_path: Path | None = None) -> dict:
    return _load_json_store(store_path or _snapshot_store_path())


def _save_snapshots(data: dict, store_path: Path | None = None) -> None:
    _save_json_store(store_path or _snapshot_store_path(), data)


def _snapshot_key(task_id: str) -> str:
    return f"{_SNAPSHOT_NAMESPACE}:{task_id}"


def _get_snapshot_restore_candidate(
    task_id: str, store_path: Path | None = None
) -> tuple[str | None, bool]:
    snapshots = _load_snapshots(store_path)
    namespaced_key = _snapshot_key(task_id)
    snapshot_id = snapshots.get(namespaced_key)
    if isinstance(snapshot_id, str) and snapshot_id:
        return snapshot_id, False
    legacy_snapshot_id = snapshots.get(task_id)
    if isinstance(legacy_snapshot_id, str) and legacy_snapshot_id:
        return legacy_snapshot_id, True
    return None, False


def _store_snapshot(task_id: str, snapshot_id: str, store_path: Path | None = None) -> None:
    snapshots = _load_snapshots(store_path)
    snapshots[_snapshot_key(task_id)] = snapshot_id
    snapshots.pop(task_id, None)
    _save_snapshots(snapshots, store_path)


def _delete_snapshot(
    task_id: str, snapshot_id: str | None = None, store_path: Path | None = None
) -> None:
    snapshots = _load_snapshots(store_path)
    updated = False
    for key in (_snapshot_key(task_id), task_id):
        value = snapshots.get(key)
        if value is None:
            continue
        if snapshot_id is None or value == snapshot_id:
            snapshots.pop(key, None)
            updated = True
    if updated:
        _save_snapshots(snapshots, store_path)


def _normalize_forward_env_names(forward_env: list[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in forward_env or []:
        if not isinstance(item, str):
            logger.warning("Ignoring non-string tenki_forward_env entry: %r", item)
            continue
        name = item.strip()
        if not name:
            continue
        if not _ENV_NAME_RE.match(name):
            logger.warning("Ignoring invalid tenki_forward_env entry: %r", item)
            continue
        if name not in seen:
            normalized.append(name)
            seen.add(name)
    return normalized


def _safe_name(value: str, *, fallback: str = "default", max_len: int = 48) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value or "").strip("-._")
    return (safe or fallback)[:max_len]


def _supports_any_kwargs(sig: inspect.Signature | None) -> bool:
    if sig is None:
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())


def _add_supported(
    kwargs: dict[str, Any],
    sig: inspect.Signature | None,
    names: tuple[str, ...],
    value: Any,
) -> None:
    if value in (None, "", [], {}):
        return
    if sig is not None:
        for name in names:
            if name in sig.parameters:
                kwargs[name] = value
                return
    if _supports_any_kwargs(sig):
        kwargs[names[0]] = value


def _result_attr(result: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if not hasattr(result, name):
            continue
        value = getattr(result, name)
        if callable(value):
            try:
                value = value()
            except TypeError:
                pass
        if value is not None:
            return value
    return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _rewrite_sudo_noninteractive(command: str) -> tuple[str, int]:
    """Add ``-n`` to real sudo invocations so Tenki never prompts."""
    from tools.terminal_tool import _looks_like_env_assignment, _read_shell_token

    out: list[str] = []
    i = 0
    n = len(command)
    command_start = True
    sudo_count = 0

    while i < n:
        ch = command[i]

        if ch.isspace():
            out.append(ch)
            if ch == "\n":
                command_start = True
            i += 1
            continue

        if ch == "#" and command_start:
            comment_end = command.find("\n", i)
            if comment_end == -1:
                out.append(command[i:])
                break
            out.append(command[i:comment_end])
            i = comment_end
            continue

        if command.startswith("&&", i) or command.startswith("||", i) or command.startswith(";;", i):
            out.append(command[i:i + 2])
            i += 2
            command_start = True
            continue

        if ch in ";|&(":
            out.append(ch)
            i += 1
            command_start = True
            continue

        if ch == ")":
            out.append(ch)
            i += 1
            command_start = False
            continue

        token, next_i = _read_shell_token(command, i)
        if command_start and token == "sudo":
            out.append("sudo -n")
            sudo_count += 1
        else:
            out.append(token)

        if command_start and _looks_like_env_assignment(token):
            command_start = True
        else:
            command_start = False
        i = next_i

    return "".join(out), sudo_count


class TenkiEnvironment(BaseEnvironment):
    """Tenki sandbox backend.

    Tenki's SDK exposes process handles inside a remote sandbox, so this adapts
    them to the normal Hermes ``ProcessHandle`` contract with
    ``_ThreadedProcessHandle``.
    """

    _stdin_mode = "pipe"
    _snapshot_timeout = 60
    _terminal_states = frozenset({"TERMINATING", "TERMINATED", "DELETED", "FAILED", "ERROR"})

    def __init__(
        self,
        image: str = "",
        cwd: str = "/home/tenki",
        timeout: int = 60,
        cpu: float = 1,
        memory: int = 5120,
        disk: int = 51200,
        persistent_filesystem: bool = False,
        task_id: str = "default",
        api_endpoint: str = "",
        workspace_id: str = "",
        project_id: str = "",
        name_prefix: str = "hermes",
        allow_inbound: bool = False,
        allow_outbound: bool = True,
        max_duration: int = 3600,
        idle_timeout: int = 0,
        pause_retention: int = 0,
        sync_hermes_home: bool = False,
        forward_env: list[str] | None = None,
    ):
        super().__init__(cwd=cwd, timeout=timeout)

        try:
            from tools.lazy_deps import ensure as _lazy_ensure

            _lazy_ensure("terminal.tenki", prompt=False)
        except ImportError:
            pass
        except Exception as exc:
            raise ImportError(str(exc))

        from tenki_sandbox import Client, Sandbox

        self._Client = Client
        self._Sandbox = Sandbox
        self._client = None
        self._sandbox = None
        self._lock = threading.Lock()
        self._persistent = persistent_filesystem
        self._sync_hermes_home = sync_hermes_home
        self._sync_manager: FileSyncManager | None = None
        self._cleanup_in_progress = False
        self._cleanup_sandbox = None
        self._task_id = task_id
        self._profile_token = _profile_token()
        # Bind the profile's snapshot-store path at construction, while the
        # correct HERMES_HOME context is active. Cleanup (and the idle-reaper
        # snapshot save) can run in a background thread that does NOT inherit
        # the per-turn HERMES_HOME contextvar, so re-resolving there would write
        # the pointer into the wrong profile's home.
        self._snapshot_store = _snapshot_store_path()
        self._snapshot_restore_id: str | None = None
        self._snapshot_restore_from_legacy_key = False
        self._image = image
        self._cpu = cpu
        self._memory = memory
        self._disk = disk
        self._api_endpoint = resolve_tenki_api_endpoint(api_endpoint)
        self._workspace_id = resolve_tenki_workspace_id(workspace_id)
        self._project_id = resolve_tenki_project_id(project_id)
        self._auth_token = resolve_tenki_auth_token()
        self._name_prefix = _safe_name(name_prefix, fallback="hermes", max_len=28)
        self._allow_inbound = allow_inbound
        self._allow_outbound = allow_outbound
        self._max_duration = max_duration
        self._idle_timeout = idle_timeout
        self._pause_retention = pause_retention
        self._forward_env = _normalize_forward_env_names(forward_env)
        self._remote_home = "/home/tenki"
        if self._persistent:
            self._snapshot_restore_id, self._snapshot_restore_from_legacy_key = (
                _get_snapshot_restore_candidate(self._task_id, self._snapshot_store)
            )

        self._ensure_sandbox()
        self._resolve_remote_home()
        if self._sync_hermes_home:
            self._sync_manager = FileSyncManager(
                get_files_fn=lambda: iter_sync_files(f"{self._remote_home}/.hermes"),
                upload_fn=self._tenki_upload,
                delete_fn=self._tenki_delete,
                bulk_upload_fn=self._tenki_bulk_upload,
                bulk_download_fn=self._tenki_bulk_download,
            )
            self._sync_manager.sync(force=True)
        self.init_session()

    def _sandbox_create_signature(self) -> inspect.Signature | None:
        try:
            return inspect.signature(self._Sandbox.create)
        except (TypeError, ValueError):
            return None

    def _create_kwargs(self) -> dict[str, Any]:
        sig = self._sandbox_create_signature()
        kwargs: dict[str, Any] = {}
        sandbox_name = self._sandbox_name()

        _add_supported(kwargs, sig, ("name",), sandbox_name)
        if self._snapshot_restore_id:
            _add_supported(kwargs, sig, ("snapshot_id",), self._snapshot_restore_id)
        else:
            _add_supported(kwargs, sig, ("image", "template"), self._image)
        cpu_cores = max(1, math.ceil(float(self._cpu))) if self._cpu else None
        _add_supported(kwargs, sig, ("cpu_cores", "cpu"), cpu_cores)
        _add_supported(kwargs, sig, ("memory_mb", "memory"), self._memory)

        if self._disk:
            disk_gb = max(1, math.ceil(float(self._disk) / 1024))
            _add_supported(kwargs, sig, ("disk_size_gb", "disk_gb", "disk"), disk_gb)

        _add_supported(kwargs, sig, ("allow_inbound",), self._allow_inbound)
        _add_supported(kwargs, sig, ("allow_outbound",), self._allow_outbound)
        _add_supported(kwargs, sig, ("max_duration",), self._max_duration)
        idle_timeout = _positive_float(self._idle_timeout)
        if idle_timeout is not None:
            idle_timeout_minutes = max(1, math.ceil(idle_timeout / 60))
            _add_supported(kwargs, sig, ("idle_timeout_minutes",), idle_timeout_minutes)
        pause_retention = _positive_float(self._pause_retention)
        if pause_retention is not None:
            _add_supported(kwargs, sig, ("pause_retention",), pause_retention)
        _add_supported(kwargs, sig, ("workspace_id",), self._workspace_id)
        _add_supported(kwargs, sig, ("project_id",), self._project_id)
        _add_supported(kwargs, sig, ("base_url", "api_endpoint"), self._api_endpoint)
        _add_supported(kwargs, sig, ("auth_token", "api_key"), self._auth_token)
        _add_supported(kwargs, sig, ("env",), self._sandbox_env())
        _add_supported(
            kwargs,
            sig,
            ("metadata",),
            {
                "hermes_task_id": self._task_id,
                "hermes_backend": "tenki",
                "hermes_profile": self._profile_token,
            },
        )
        _add_supported(kwargs, sig, ("tags",), ["hermes-agent"])
        _add_supported(kwargs, sig, ("wait",), True)
        # Do NOT emit a create-time ``timeout`` here: the SDK's Sandbox.create
        # pops ``timeout`` into the *Client* (HTTP) timeout, while Client.create
        # treats ``timeout`` as the *wait-for-ready* budget — so the same value
        # would mean two different things across the two create paths. The HTTP
        # timeout is set explicitly in _create_client(); readiness uses the
        # SDK's default wait budget.
        return kwargs

    def _sandbox_env(self) -> dict[str, str]:
        """Environment variables injected into Tenki sandbox processes.

        The supervisor's Tenki control-plane credential is used host-side to
        create and manage the sandbox (see ``_create_kwargs`` /
        ``_create_client``); it is deliberately NOT injected into the guest.
        Guest code is model-controlled and can print, exfiltrate, or reuse
        whatever is in its environment, and the sandbox is billed against the
        supervisor's account — so a leaked ``TENKI_AUTH_TOKEN`` would let guest
        code create, terminate, and bill account resources outside the
        parent's configured limits. Nested-sandbox support is still available
        as an explicit opt-in: list ``TENKI_AUTH_TOKEN`` (or ``TENKI_API_KEY``)
        in ``terminal.tenki_forward_env``.

        ``terminal.tenki_forward_env`` is the explicit allowlist for
        task-specific credentials such as GitHub tokens; the generic
        ``terminal.env_passthrough`` allowlist is also honored for skill
        variables that are not protected by Hermes' provider-secret blocklist.
        """
        env: dict[str, str] = {}
        env.update(self._resolve_forwarded_env(self._forward_env))
        env.update(self._passthrough_env())
        # If the operator explicitly opted into forwarding the control-plane
        # credential (for nested-sandbox creation), supply the already-resolved
        # token. Re-reading the env var here would miss a `tenki login`
        # credential, which lives in the Tenki CLI config, not the environment —
        # so the documented opt-in would silently forward nothing.
        if self._auth_token:
            for key in ("TENKI_AUTH_TOKEN", "TENKI_API_KEY"):
                if key in self._forward_env and not env.get(key):
                    env[key] = self._auth_token
                    logger.warning(
                        "Tenki: forwarding the control-plane credential %s into the "
                        "sandbox as requested by terminal.tenki_forward_env. Guest code "
                        "can read it and create/terminate/bill account resources. Note "
                        "that forwarded credentials are NOT profile-isolated under the "
                        "multiplexing gateway's shared terminal cache.",
                        key,
                    )
        return env

    @staticmethod
    def _resolve_forwarded_env(keys: list[str] | set[str] | tuple[str, ...]) -> dict[str, str]:
        if not keys:
            return {}
        from tools.tenki_config import _global_credential_fallback_allowed, _scoped_env

        get_env_value = None
        if _global_credential_fallback_allowed():
            try:
                from hermes_cli.config import get_env_value
            except Exception:
                get_env_value = None

        env: dict[str, str] = {}
        for key in keys:
            # Scope-aware read first: under a multiplexed profile turn this
            # resolves the active profile's value, never another profile's raw
            # os.environ. The ~/.hermes/.env fallback is consulted only when no
            # profile scope is authoritative.
            value = _scoped_env(key)
            if not value and get_env_value is not None:
                try:
                    value = get_env_value(key) or ""
                except Exception:
                    value = ""
            if value:
                env[key] = value
        return env

    @staticmethod
    def _passthrough_env() -> dict[str, str]:
        try:
            from tools.env_passthrough import get_all_passthrough

            keys = sorted(get_all_passthrough())
        except Exception:
            keys = []
        return TenkiEnvironment._resolve_forwarded_env(keys)

    def _create_client(self):
        if self._client is None:
            self._client = self._Client(
                auth_token=self._auth_token,
                base_url=self._api_endpoint,
                timeout=max(60, self.timeout),
            )
        return self._client

    def _sandbox_name(self) -> str:
        # The profile token namespaces the name so two profiles sharing one
        # Tenki account never collide on a name or reuse each other's sandbox.
        return f"{self._name_prefix}-{self._profile_token}-{_safe_name(self._task_id)}"

    @staticmethod
    def _sandbox_state(sandbox: Any) -> str:
        state = getattr(sandbox, "state", "")
        if callable(state):
            try:
                state = state()
            except TypeError:
                state = ""
        return str(state or "").upper()

    def _sandbox_matches_task(self, sandbox: Any) -> bool:
        name = getattr(sandbox, "name", "")
        info = getattr(sandbox, "info", None)
        if not name and info is not None:
            name = getattr(info, "name", "")
        if name != self._sandbox_name():
            return False
        metadata = getattr(info, "metadata", {}) if info is not None else {}
        # Never reuse another profile's sandbox: if the candidate carries a
        # profile token it must match ours (the name already encodes it, but
        # metadata is the authoritative, defense-in-depth check).
        if isinstance(metadata, dict) and metadata.get("hermes_profile"):
            if metadata.get("hermes_profile") != self._profile_token:
                return False
        if isinstance(metadata, dict) and metadata.get("hermes_task_id"):
            return metadata.get("hermes_task_id") == self._task_id
        return True

    def _find_persistent_sandbox(self):
        if not self._persistent:
            return None
        client = self._create_client()
        try:
            if self._project_id and hasattr(client, "list_project"):
                candidates = client.list_project(self._project_id, tags=["hermes-agent"])
            elif self._workspace_id and hasattr(client, "list_workspace"):
                candidates = client.list_workspace(self._workspace_id, tags=["hermes-agent"])
            else:
                candidates = client.list(tags=["hermes-agent"])
        except Exception as exc:
            logger.debug("Tenki: could not list persistent sandboxes: %s", exc)
            return None

        usable = []
        for sandbox in candidates:
            if not self._sandbox_matches_task(sandbox):
                continue
            state = self._sandbox_state(sandbox)
            if state in self._terminal_states:
                continue
            usable.append((state, sandbox))
        if not usable:
            return None
        usable.sort(key=lambda item: 0 if item[0] == "RUNNING" else 1)
        return usable[0][1]

    def _resume_persistent_sandbox(self):
        sandbox = self._find_persistent_sandbox()
        if sandbox is None:
            return None
        if not self._ensure_sandbox_ready(sandbox):
            logger.info(
                "Tenki: existing sandbox for task %s is no longer reusable; creating a fresh sandbox",
                self._task_id,
            )
            return None
        sandbox_id = getattr(sandbox, "id", None) or getattr(sandbox, "sandbox_id", None)
        logger.info("Tenki: resumed sandbox %s for task %s", sandbox_id or "<unknown>", self._task_id)
        return sandbox

    def _ensure_sandbox_ready(self, sandbox: Any) -> bool:
        refresh = getattr(sandbox, "refresh", None)
        if callable(refresh):
            try:
                refresh()
            except Exception as exc:
                logger.info("Tenki: sandbox refresh failed for task %s: %s", self._task_id, exc)
                return False

        state = self._sandbox_state(sandbox)
        if state in self._terminal_states:
            return False

        try:
            if state and state != "RUNNING":
                resume = getattr(sandbox, "resume", None)
                if callable(resume):
                    resume()
                wait_ready = getattr(sandbox, "wait_ready", None)
                if callable(wait_ready):
                    wait_ready(max(60, self.timeout))
        except Exception as exc:
            logger.info("Tenki: could not make sandbox ready for task %s: %s", self._task_id, exc)
            return False

        return self._sandbox_state(sandbox) not in self._terminal_states

    def _ensure_sandbox(self) -> None:
        with self._lock:
            if self._cleanup_in_progress:
                raise RuntimeError("Tenki cleanup is in progress")
            if self._sandbox is not None:
                if self._ensure_sandbox_ready(self._sandbox):
                    return
                self._sandbox = None
            self._sandbox = self._resume_persistent_sandbox()
            if self._sandbox is not None:
                return
            self._sandbox = self._create_sandbox_with_snapshot_fallback()
            sandbox_id = getattr(self._sandbox, "id", None) or getattr(self._sandbox, "sandbox_id", None)
            logger.info("Tenki: created sandbox %s for task %s", sandbox_id or "<unknown>", self._task_id)

    def _create_sandbox_from_kwargs(self, kwargs: dict[str, Any]):
        if self._persistent:
            client = self._create_client()
            create_kwargs = dict(kwargs)
            for key in ("auth_token", "api_key", "base_url", "api_endpoint"):
                create_kwargs.pop(key, None)
            return client.create(**create_kwargs)
        return self._Sandbox.create(**kwargs)

    # Snapshot errors that mean the recorded snapshot can never restore, so
    # dropping the pointer and booting the base image is the right recovery.
    # A *transient* failure (network, rate-limit, ambiguous) is NOT in this set:
    # it must propagate with the pointer intact so a later attempt can still
    # recover the persistent state instead of silently booting an empty sandbox.
    # Snapshot-specific error type names that mean the snapshot can never
    # restore. Note InvalidStateError is intentionally NOT here: the restore RPC
    # maps a generic FAILED_PRECONDITION (workspace/policy/etc.) to it, so it is
    # only unrecoverable when its message points at the snapshot itself
    # (handled by message inspection below); a bare InvalidStateError stays
    # transient so an unrelated precondition can't destroy a valid pointer.
    _UNRECOVERABLE_SNAPSHOT_ERRORS = frozenset({
        "SnapshotNotFoundError",          # snapshot is gone
        "RegistryArtifactNotFoundError",  # backing artifact is gone
        "SnapshotNotDurableError",        # explicitly never reached durability
    })

    @classmethod
    def _snapshot_unrecoverable(cls, exc: BaseException) -> bool:
        """True when the error confirms the snapshot can never restore.

        Covers "gone" (not-found), "explicitly non-durable", and a generic
        ``InvalidStateError`` whose message identifies the snapshot as the
        failing precondition (the SDK's restore RPC collapses a bad/non-durable
        snapshot into a generic FAILED_PRECONDITION → InvalidStateError). Only
        these justify discarding the recovery pointer and booting a base image;
        every other error (including a bare InvalidStateError, rate-limit,
        quota, auth blip, or network failure) is transient and re-raised with
        the pointer preserved.
        """
        def _is_snapshot_specific_invalid_state(e: BaseException, invalid_state_cls) -> bool:
            if invalid_state_cls is not None and not isinstance(e, invalid_state_cls):
                return False
            if invalid_state_cls is None and type(e).__name__ != "InvalidStateError":
                return False
            msg = str(e).lower()
            return "snapshot" in msg or "durable" in msg

        try:
            from tenki_sandbox import (
                InvalidStateError,
                RegistryArtifactNotFoundError,
                SnapshotNotDurableError,
                SnapshotNotFoundError,
            )

            if isinstance(
                exc,
                (SnapshotNotFoundError, RegistryArtifactNotFoundError, SnapshotNotDurableError),
            ):
                return True
            if _is_snapshot_specific_invalid_state(exc, InvalidStateError):
                return True
            return False
        except Exception:
            pass
        # Name-based fallback for SDK builds that don't export every class.
        for typ in type(exc).__mro__:
            if typ.__name__ in cls._UNRECOVERABLE_SNAPSHOT_ERRORS:
                return True
        return _is_snapshot_specific_invalid_state(exc, None)

    def _create_sandbox_with_snapshot_fallback(self):
        kwargs = self._create_kwargs()
        try:
            sandbox = self._create_sandbox_from_kwargs(kwargs)
        except Exception as exc:
            if not self._snapshot_restore_id:
                raise
            if not self._snapshot_unrecoverable(exc):
                # Ambiguous/transient failure — keep the snapshot pointer so a
                # later attempt can still recover, rather than deleting it and
                # booting a blank base image (silent loss of persistent state).
                logger.warning(
                    "Tenki: snapshot restore %s for task %s failed transiently (%s); "
                    "preserving it for retry",
                    self._snapshot_restore_id,
                    self._task_id,
                    exc,
                )
                raise
            logger.warning(
                "Tenki: snapshot %s for task %s is unrecoverable; creating from base image: %s",
                self._snapshot_restore_id,
                self._task_id,
                exc,
            )
            _delete_snapshot(self._task_id, self._snapshot_restore_id, self._snapshot_store)
            self._snapshot_restore_id = None
            self._snapshot_restore_from_legacy_key = False
            sandbox = self._create_sandbox_from_kwargs(self._create_kwargs())
        else:
            if self._snapshot_restore_id and self._snapshot_restore_from_legacy_key:
                _store_snapshot(self._task_id, self._snapshot_restore_id, self._snapshot_store)
        return sandbox

    def _remote_transfer_path(self, prefix: str) -> str:
        base = (self._remote_home or "/home/tenki").rstrip("/") or "/home/tenki"
        if base != "/home/tenki" and not base.startswith("/home/tenki/"):
            base = "/home/tenki"
        return f"{base}/{prefix}.{os.getpid()}.{self._session_id}.tar"

    def _resolve_remote_home(self) -> None:
        try:
            result = self._exec_raw("echo \"$HOME\"", timeout=15)
            home = result[0].strip() if result[1] == 0 else ""
            if home:
                self._remote_home = home
                if self.cwd in {"~", "/home/tenki"}:
                    self.cwd = home
        except Exception:
            pass

    def _tenki_upload(self, host_path: str, remote_path: str) -> None:
        self._ensure_sandbox()
        parent = str(Path(remote_path).parent)
        self._sandbox.fs.mkdir(parent, recursive=True)
        self._sandbox.fs.upload(host_path, remote_path)

    def _tenki_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        if not files:
            return

        self._ensure_sandbox()
        parents = unique_parent_dirs(files)
        if parents:
            self._exec_raw(quoted_mkdir_command(parents), timeout=30)

        remote_tar = self._remote_transfer_path(".hermes_tenki_sync")
        with tempfile.NamedTemporaryFile(suffix=".tar") as tmp:
            with tarfile.open(fileobj=tmp, mode="w") as tar:
                for host_path, remote_path in files:
                    tar.add(host_path, arcname=remote_path.lstrip("/"))
            tmp.flush()
            self._sandbox.fs.upload(tmp.name, remote_tar)

        try:
            output, exit_code = self._exec_raw(
                f"tar xf {shlex.quote(remote_tar)} -C /",
                timeout=120,
            )
            if exit_code != 0:
                raise RuntimeError(f"Tenki bulk upload failed (exit {exit_code}): {output}")
        finally:
            try:
                self._exec_raw(f"rm -f {shlex.quote(remote_tar)}", timeout=10)
            except Exception:
                pass

    def _tenki_bulk_download(self, dest: Path) -> None:
        sandbox = self._transfer_sandbox()
        remote_tar = self._remote_transfer_path(".hermes_tenki_sync_back")
        rel_base = f"{self._remote_home}/.hermes".lstrip("/")
        try:
            output, exit_code = self._exec_raw_on_sandbox(
                sandbox,
                f"tar cf {shlex.quote(remote_tar)} -C / {shlex.quote(rel_base)}",
                timeout=120,
            )
            if exit_code != 0:
                raise RuntimeError(f"Tenki bulk download failed (exit {exit_code}): {output}")
            sandbox.fs.download(remote_tar, str(dest))
        finally:
            try:
                self._exec_raw_on_sandbox(sandbox, f"rm -f {shlex.quote(remote_tar)}", timeout=10)
            except Exception:
                pass

    def _transfer_sandbox(self):
        if self._cleanup_in_progress and self._cleanup_sandbox is not None:
            return self._cleanup_sandbox
        self._ensure_sandbox()
        return self._sandbox

    def _tenki_delete(self, remote_paths: list[str]) -> None:
        if not remote_paths:
            return
        self._exec_raw(quoted_rm_command(remote_paths), timeout=30)

    def _exec_raw(self, command: str, *, login: bool = False, timeout: int = 120) -> tuple[str, int]:
        self._ensure_sandbox()
        return self._exec_raw_on_sandbox(self._sandbox, command, login=login, timeout=timeout)

    def _exec_raw_on_sandbox(
        self,
        sandbox: Any,
        command: str,
        *,
        login: bool = False,
        timeout: int = 120,
    ) -> tuple[str, int]:
        flag = "-lc" if login else "-c"
        result = sandbox.exec("bash", flag, command, timeout=timeout, env=self._sandbox_env())
        return self._result_to_output(result)

    @staticmethod
    def _result_to_output(result: Any) -> tuple[str, int]:
        stdout = _text(_result_attr(result, ("stdout_text", "stdout", "output", "result", "text")))
        stderr = _text(_result_attr(result, ("stderr_text", "stderr")))
        exit_code = _result_attr(result, ("exit_code", "returncode", "status_code"))
        if exit_code is None:
            ok = _result_attr(result, ("ok", "success"))
            exit_code = 0 if ok is True else 1
        if stdout and stderr and not stdout.endswith("\n"):
            output = stdout + "\n" + stderr
        else:
            output = stdout + stderr
        return output, int(exit_code)

    def _start_process(
        self,
        cmd_string: str,
        *,
        login: bool,
        timeout: int,
        stdin_data: str | None,
        process_ref: dict[str, Any] | None = None,
    ) -> tuple[str, int]:
        self._ensure_sandbox()
        flag = "-lc" if login else "-c"
        start = getattr(self._sandbox, "start", None)
        if not callable(start):
            kwargs: dict[str, Any] = {"timeout": timeout, "env": self._sandbox_env()}
            if stdin_data is not None:
                kwargs["input"] = stdin_data
            result = self._sandbox.exec("bash", flag, cmd_string, **kwargs)
            return self._result_to_output(result)

        process = start(
            "bash",
            flag,
            cmd_string,
            timeout=timeout,
            stdin=stdin_data,
            env=self._sandbox_env(),
        )
        if process_ref is not None:
            process_ref["process"] = process
        if stdin_data is None:
            close_stdin = getattr(process, "close_stdin", None)
            if callable(close_stdin):
                close_stdin()
        result = process.wait(timeout=timeout + 5 if timeout is not None else None)
        return self._result_to_output(result)

    def _sudo_nopasswd_works(self) -> bool:
        try:
            _output, exit_code = self._exec_raw("sudo -n true", timeout=10)
        except Exception:
            return False
        return exit_code == 0

    def _prepare_command(self, command: str | None) -> tuple[str | None, str | None]:
        if command is None:
            return None, None

        # Tenki sandboxes should rely on their own sudoers policy. Do not ask
        # the user for a host sudo password, and do not send SUDO_PASSWORD to a
        # remote cloud sandbox. The default Tenki image supports NOPASSWD sudo.
        transformed, sudo_count = _rewrite_sudo_noninteractive(command)
        if sudo_count == 0:
            return command, None
        if self._sudo_nopasswd_works():
            return command, None
        return transformed, None

    def _before_execute(self) -> None:
        self._ensure_sandbox()
        if self._sync_manager:
            self._sync_manager.sync()

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ):
        process_ref: dict[str, Any] = {}

        def cancel() -> None:
            process = process_ref.get("process")
            kill = getattr(process, "kill", None)
            if callable(kill):
                try:
                    kill()
                    return
                except Exception:
                    pass
            with self._lock:
                sandbox = self._sandbox
                # Drop our reference so the next command resumes (persistent) or
                # recreates (ephemeral) a sandbox instead of reusing a torn-down
                # one.
                self._sandbox = None
            if sandbox is None:
                return
            # For a persistent sandbox, pause (preserve the filesystem) instead of
            # terminating: an interrupted or timed-out command must not destroy
            # state the user asked to keep. The paused sandbox is re-discovered and
            # resumed on the next command via _resume_persistent_sandbox().
            if self._persistent:
                pause = getattr(sandbox, "pause", None)
                if callable(pause):
                    try:
                        pause()
                        return
                    except Exception:
                        pass  # fall through to terminate if pause is unavailable
            for method_name in ("terminate", "close"):
                method = getattr(sandbox, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        pass
                    return

        def exec_fn() -> tuple[str, int]:
            return self._start_process(
                cmd_string,
                login=login,
                timeout=timeout,
                stdin_data=stdin_data,
                process_ref=process_ref,
            )

        return _ThreadedProcessHandle(exec_fn, cancel_fn=cancel)

    def cleanup(self):
        with self._lock:
            sandbox = self._sandbox
            sync_manager = self._sync_manager
            self._sync_manager = None
            client = self._client
            self._cleanup_in_progress = True
            self._cleanup_sandbox = sandbox
        if sandbox is None:
            self._close_client(client)
            with self._lock:
                if self._client is client:
                    self._client = None
                self._cleanup_in_progress = False
                self._cleanup_sandbox = None
            return

        try:
            if sync_manager:
                logger.info("Tenki: syncing files from sandbox...")
                try:
                    sync_manager.sync_back()
                except Exception as exc:
                    logger.warning("Tenki: sync_back failed: %s", exc)

            snapshot_saved = False
            if self._persistent:
                snapshot_saved = self._save_persistent_snapshot(sandbox)

            if self._persistent and not snapshot_saved:
                # Persistent state was NOT durably snapshotted. Terminating now
                # would destroy the only copy, so prefer pause; and if pause
                # fails, still do NOT terminate — leave the sandbox live for a
                # later recovery attempt (the max-duration / idle reaper bounds
                # the cost). Terminating here would break the preservation
                # guarantee that the durability gate exists to uphold.
                pause = getattr(sandbox, "pause", None)
                if callable(pause):
                    try:
                        pause()
                        logger.info("Tenki: paused sandbox for task %s", self._task_id)
                    except Exception as exc:
                        logger.warning(
                            "Tenki: pause failed for task %s; leaving sandbox live to "
                            "preserve un-snapshotted state (not terminating): %s",
                            self._task_id, exc,
                        )
                else:
                    logger.warning(
                        "Tenki: no durable snapshot and no pause support for task %s; "
                        "leaving sandbox live to preserve state (not terminating)",
                        self._task_id,
                    )
                return

            for method_name in ("terminate", "close"):
                method = getattr(sandbox, method_name, None)
                if not callable(method):
                    continue
                try:
                    method()
                    logger.info("Tenki: terminated sandbox for task %s", self._task_id)
                except Exception as exc:
                    logger.warning("Tenki: cleanup failed: %s", exc)
                return
        finally:
            self._close_client(client)
            with self._lock:
                if self._sandbox is sandbox:
                    self._sandbox = None
                if self._client is client:
                    self._client = None
                self._cleanup_in_progress = False
                self._cleanup_sandbox = None

    def _save_persistent_snapshot(self, sandbox: Any) -> bool:
        snapshot_id: str | None = None
        try:
            snapshot = sandbox.snapshot(name=self._sandbox_name(), wait=True)
            snapshot_id = getattr(snapshot, "id", None) or getattr(snapshot, "snapshot_id", None)
        except Exception as exc:
            logger.warning("Tenki: filesystem snapshot failed: %s", exc)
            return False
        if not snapshot_id:
            logger.warning("Tenki: snapshot completed without an id; preserving paused sandbox instead")
            return False
        # snapshot(wait=True) only waits for READY; durability is a separate,
        # required gate. If durability is not confirmed the snapshot may not be
        # a safe recovery copy, so we must NOT record it as the persistent
        # pointer or let the caller terminate the live sandbox. Return False so
        # cleanup pauses the sandbox and the prior (known-durable) snapshot
        # pointer is left intact for recovery.
        if self._client is not None:
            snapshots = getattr(self._client, "snapshots", None)
            wait_durable = getattr(snapshots, "wait_durable", None)
            if callable(wait_durable):
                try:
                    wait_durable(snapshot_id, timeout=300)
                except Exception as exc:
                    logger.warning(
                        "Tenki: snapshot %s for task %s did not reach durability (%s); "
                        "preserving paused sandbox and prior snapshot instead",
                        snapshot_id, self._task_id, exc,
                    )
                    return False
        _store_snapshot(self._task_id, snapshot_id, self._snapshot_store)
        logger.info("Tenki: saved filesystem snapshot %s for task %s", snapshot_id, self._task_id)
        return True

    @staticmethod
    def _close_client(client: Any) -> None:
        if client is None:
            return
        close = getattr(client, "close", None)
        if callable(close):
            close()
