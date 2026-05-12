"""Worker-spawn runtime strategy for the kanban dispatcher.

This module isolates *how* a kanban worker process gets created from *when*
the dispatcher decides to spawn one. The dispatcher remains responsible for
claim acquisition, DAG promotion, retry policy, and PID-tracking; the
runtime is responsible only for "produce a handle to a running worker
that has the supplied env contract."

Contract: every runtime gets the same env vars in scope (HERMES_KANBAN_*,
HERMES_PROFILE, HERMES_TENANT, etc.) and must launch a process that runs
``hermes -p <profile> --skills kanban-worker chat -q "work kanban task <id>"``.

The default ``LocalRuntime`` is a thin wrapper around ``_default_spawn`` in
``hermes_cli/kanban_db.py`` so that ``worker_runtime: local`` is byte-identical
to pre-D1 behavior.

See also:
    ~/.hermes/plans/2026-05-12-d1-kanban-worker-runtime.md  (this file's plan)
    ~/.hermes/skills/software-development/mission-control-architecture/SKILL.md
"""
from __future__ import annotations

import logging
import os
import signal
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class WorkerRuntime(Protocol):
    """Strategy interface for spawning kanban workers.

    Implementations:
      - LocalRuntime — bare subprocess (today's default)
      - DockerRuntime — `docker run --rm` per-task container (D1 Task 3)
      - (future) ModalRuntime, SSHRuntime, etc.

    Attributes:
        name: short identifier matching the config key
              (`local`, `docker`, `modal`, `ssh`).

    Methods:
        spawn(task, workspace, board=None) -> handle:
            Launch a worker. Returns a *handle* (int PID for local/ssh,
            container_id string for docker, Modal call_id for modal).
            Must be string-able and stable for the worker's lifetime.
            ``None`` indicates a benign skip (e.g. dry-run or already-running).
            Raise on unrecoverable errors (image missing, daemon down, etc.).
        is_alive(handle) -> bool:
            Probe whether the worker is still running. Best-effort; the
            dispatcher's claim-TTL logic remains the authoritative
            crash detector.
        terminate(handle, reason) -> None:
            Best-effort SIGTERM (or container kill). Used by `kanban
            reclaim` and shutdown paths.
    """

    name: str

    def spawn(
        self,
        task: Any,
        workspace: str,
        *,
        board: Optional[str] = None,
    ) -> Optional[int | str]: ...

    def is_alive(self, handle: int | str) -> bool: ...

    def terminate(self, handle: int | str, reason: str = "") -> None: ...


# ---------------------------------------------------------------------------
# LocalRuntime — the regression-safe default
# ---------------------------------------------------------------------------

class LocalRuntime:
    """Today's behavior: ``hermes`` on PATH via subprocess.Popen.

    Wraps ``hermes_cli.kanban_db._default_spawn`` so ``worker_runtime: local``
    is byte-identical to pre-D1 behavior.
    """

    name = "local"

    def __init__(self, cfg: Optional[dict] = None) -> None:
        # No config consumed at v1. Reserved for future fields like
        # `start_new_session`, `cgroup_path`, etc.
        self._cfg = cfg or {}

    def spawn(
        self,
        task: Any,
        workspace: str,
        *,
        board: Optional[str] = None,
    ) -> Optional[int]:
        # Lazy import to avoid a circular dep with kanban_db (which imports
        # things from gateway/run.py at module top, and gateway/run.py may
        # in turn import this module via the runtime-loading helper).
        from hermes_cli import kanban_db
        return kanban_db._default_spawn(task, workspace, board=board)

    def is_alive(self, handle: int | str) -> bool:
        try:
            pid = int(handle)
        except (TypeError, ValueError):
            return False
        if pid <= 0:
            return False
        try:
            # Signal 0 doesn't kill; just probes existence.
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False
        except OSError:
            return False

    def terminate(self, handle: int | str, reason: str = "") -> None:
        try:
            pid = int(handle)
        except (TypeError, ValueError):
            return
        if pid <= 0:
            return
        try:
            os.kill(pid, signal.SIGTERM)
            logger.info("LocalRuntime: SIGTERM pid=%d (%s)", pid, reason)
        except ProcessLookupError:
            pass  # Already dead; not an error.
        except OSError as exc:
            logger.warning(
                "LocalRuntime: terminate failed pid=%d: %s", pid, exc
            )


# ---------------------------------------------------------------------------
# DockerRuntime — per-task isolated container (D1 Task 3)
# ---------------------------------------------------------------------------

class DockerRuntime:
    """Spawn each kanban worker in its own ``hermes-worker:latest`` container.

    Container lifecycle: ``docker run -d --rm ...`` (detached, auto-remove
    on exit). Container name is ``hermes-kanban-<task_id_short>`` so a
    human can ``docker logs`` it. The dispatcher's claim-TTL logic (15min
    default) is the authoritative crash detector; ``is_alive`` queries
    ``docker inspect`` as a faster signal.

    Config keys (all under ``kanban.docker`` in config.yaml):

        image_per_profile:
          worker: hermes-worker:latest      # primary swarm image
          default: hermes-worker:latest     # fallback for unknown profiles
          ops: hermes-ops:latest            # optional per-profile override
        bind_mounts:                        # list of `host:container[:mode]`
          - "{hermes_home}/kanban.db:/hermes/kanban.db"
          - "{hermes_home}/.env:/hermes/.env:ro"
        env_passthrough:                    # env var names (or globs)
          - OPENROUTER_API_KEY
          - HERMES_*
        mem_limit: "4g"                     # per-container RAM cap
        cpus: "2.0"                         # per-container CPU cap
        network: hermes-net                 # bridge name (compose stack)
        auto_remove: true                   # docker run --rm
    """

    name = "docker"

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg or {}
        self._images = self._cfg.get("image_per_profile") or {}
        if not self._images.get("default"):
            raise ValueError(
                "DockerRuntime requires kanban.docker.image_per_profile.default"
            )
        self._bind_mounts = self._cfg.get("bind_mounts") or []
        self._env_passthrough = self._cfg.get("env_passthrough") or []
        self._mem = self._cfg.get("mem_limit")
        self._cpus = self._cfg.get("cpus")
        self._network = self._cfg.get("network", "hermes-net")
        self._auto_remove = bool(self._cfg.get("auto_remove", True))
        # Verify docker CLI present at boot. Fails loudly.
        import shutil
        if shutil.which("docker") is None:
            raise RuntimeError(
                "DockerRuntime: `docker` CLI not found on PATH. "
                "Install Docker Engine and ensure the user is in the docker group."
            )

    def _resolve_image(self, profile: str) -> str:
        return self._images.get(profile) or self._images["default"]

    def _expand_bind(self, raw: str, hermes_home: str) -> str:
        return raw.replace("{hermes_home}", hermes_home.rstrip("/"))

    def _build_env_args(self, env: dict) -> list:
        """Emit -e KEY=VAL pairs honoring the passthrough allowlist + globs."""
        import fnmatch
        args = []
        seen = set()
        # 1) Always pass the kanban contract
        for key in (
            "HERMES_KANBAN_TASK", "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD",
            "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_WORKSPACE",
            "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_CLAIM_LOCK",
            "HERMES_PROFILE", "HERMES_TENANT",
        ):
            if key in env and key not in seen:
                args.extend(["-e", f"{key}={env[key]}"])
                seen.add(key)
        # 2) Apply allowlist (literal names + globs)
        for pattern in self._env_passthrough:
            if pattern in os.environ and pattern not in seen:
                args.extend(["-e", f"{pattern}={os.environ[pattern]}"])
                seen.add(pattern)
                continue
            # Glob matching
            for k, v in os.environ.items():
                if k in seen:
                    continue
                if fnmatch.fnmatch(k, pattern):
                    args.extend(["-e", f"{k}={v}"])
                    seen.add(k)
        return args

    def spawn(
        self,
        task: Any,
        workspace: str,
        *,
        board: Optional[str] = None,
    ) -> Optional[str]:
        # Lazy imports to avoid circular deps.
        from hermes_cli import kanban_db as kb
        from hermes_cli.profiles import normalize_profile_name
        import subprocess

        if not task.assignee:
            raise ValueError(f"task {task.id} has no assignee")

        # Apply the same swarm-as-persona resolution as LocalRuntime.
        # (kanban_db._default_spawn does this for local; we replicate here
        # so the env contract matches and only the runtime differs.)
        resolved = kb._resolve_assignee_to_profile(task.assignee)
        profile_arg = normalize_profile_name(resolved)
        image = self._resolve_image(profile_arg)

        # Build the in-container env (mirrors _default_spawn).
        env: dict = {}
        if task.tenant:
            env["HERMES_TENANT"] = task.tenant
        env["HERMES_KANBAN_TASK"] = task.id
        env["HERMES_KANBAN_WORKSPACE"] = workspace
        if task.current_run_id is not None:
            env["HERMES_KANBAN_RUN_ID"] = str(task.current_run_id)
        if task.claim_lock:
            env["HERMES_KANBAN_CLAIM_LOCK"] = task.claim_lock
        env["HERMES_KANBAN_DB"] = str(kb.kanban_db_path(board=board))
        env["HERMES_KANBAN_WORKSPACES_ROOT"] = str(kb.workspaces_root(board=board))
        resolved_board = kb._normalize_board_slug(board) or kb.get_current_board()
        env["HERMES_KANBAN_BOARD"] = resolved_board
        env["HERMES_PROFILE"] = profile_arg

        # Container name (truncated task id for readability)
        short = task.id.replace("t_", "")[:12]
        cname = f"hermes-kanban-{short}"

        # Build `docker run` argv
        cmd: list = ["docker", "run", "-d"]
        if self._auto_remove:
            cmd.append("--rm")
        cmd.extend(["--name", cname])
        cmd.extend(["--network", self._network])
        if self._mem:
            cmd.extend(["--memory", str(self._mem)])
        if self._cpus:
            cmd.extend(["--cpus", str(self._cpus)])

        # Bind mounts ({hermes_home} placeholder expansion)
        hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
        for raw in self._bind_mounts:
            cmd.extend(["-v", self._expand_bind(raw, hermes_home)])

        # Env vars
        cmd.extend(self._build_env_args(env))

        # Image
        cmd.append(image)

        # In-container command (mirrors _default_spawn's argv)
        cmd.extend([
            "hermes", "-p", profile_arg,
            "--skills", "kanban-worker",
        ])
        if task.skills:
            for sk in task.skills:
                if sk and sk != "kanban-worker":
                    cmd.extend(["--skills", sk])
        cmd.extend(["chat", "-q", f"work kanban task {task.id}"])

        # Launch
        try:
            result = subprocess.run(  # noqa: S603 -- argv is a list built above
                cmd,
                check=False,
                capture_output=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"DockerRuntime: docker CLI lost from PATH: {exc}"
            )
        if result.returncode != 0:
            stderr = (result.stderr or b"").decode("utf-8", errors="replace")
            raise RuntimeError(
                f"DockerRuntime: docker run failed (exit {result.returncode}): "
                f"{stderr.strip()}"
            )
        cid_full = (result.stdout or b"").decode("utf-8", errors="replace").strip()
        if not cid_full:
            raise RuntimeError("DockerRuntime: docker run produced no container id")
        short_cid = cid_full[:12]
        logger.info(
            "DockerRuntime spawned task=%s container=%s image=%s",
            task.id, short_cid, image,
        )
        return short_cid

    def is_alive(self, handle) -> bool:
        if not handle:
            return False
        import subprocess
        try:
            r = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Status}}", str(handle)],
                check=False, capture_output=True, timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        if r.returncode != 0:
            return False
        return (r.stdout or b"").decode("utf-8", errors="replace").strip() == "running"

    def terminate(self, handle, reason: str = "") -> None:
        if not handle:
            return
        import subprocess
        try:
            subprocess.run(
                ["docker", "kill", str(handle)],
                check=False, capture_output=True, timeout=10,
            )
            logger.info(
                "DockerRuntime: docker kill %s (%s)", handle, reason
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.warning(
                "DockerRuntime: terminate failed handle=%s: %s", handle, exc
            )


# ---------------------------------------------------------------------------
# Factory + registry
# ---------------------------------------------------------------------------

# Registry — populated below. D1 Task 3 extends with DockerRuntime;
# future deliverables add ModalRuntime / SSHRuntime via register_runtime().
_RUNTIMES: dict[str, type] = {
    "local": LocalRuntime,
    "docker": DockerRuntime,
}


def register_runtime(name: str, cls: type) -> None:
    """Register a new runtime implementation.

    Used by D1 Task 3 (DockerRuntime), future deliverables, and tests
    that need to swap in a fake runtime.

    Raises ValueError if ``name`` is already registered — re-registration
    is almost always a bug (two plugins fighting over the same key).
    """
    if name in _RUNTIMES:
        raise ValueError(
            f"runtime {name!r} already registered "
            f"(existing: {_RUNTIMES[name].__name__})"
        )
    _RUNTIMES[name] = cls


def load_runtime(name: str, cfg: Optional[dict] = None) -> WorkerRuntime:
    """Resolve a config name to a runtime instance.

    Raises ValueError on unknown names so the gateway boot fails loudly
    rather than silently fall back to local (which would mask config typos).
    """
    if name not in _RUNTIMES:
        available = ", ".join(sorted(_RUNTIMES.keys()))
        raise ValueError(
            f"unknown worker_runtime={name!r}. Available: {available}"
        )
    return _RUNTIMES[name](cfg or {})
