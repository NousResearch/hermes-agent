#!/usr/bin/env python3
"""Async delegation job manager.

The manager is intentionally file-backed so operator commands can inspect jobs
outside the parent agent turn.  Running processes are tracked in-memory when
this process started them; persisted metadata carries enough pid/path state for
CLI inspection and best-effort cancellation from a later process.
"""

from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_ACTIVE_LOCK = threading.Lock()
_ACTIVE_JOBS: Dict[str, Dict[str, Any]] = {}
_RESULT_STATUSES = {"completed", "failed", "error", "timeout", "cancelled", "interrupted"}
_DEFAULT_TAIL_CHARS = 12000


def _now() -> float:
    return time.time()


def _utc_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def jobs_root() -> Path:
    from hermes_constants import get_hermes_home

    root = get_hermes_home() / "delegation" / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def job_dir(job_id: str) -> Path:
    return jobs_root() / job_id


def _json_path(job_id: str) -> Path:
    return job_dir(job_id) / "job.json"


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _update_job(job_id: str, **updates: Any) -> Dict[str, Any]:
    path = _json_path(job_id)
    data = _read_json(path, {}) or {}
    data.update(updates)
    data["updated_at"] = _now()
    _write_json(path, data)
    return data


def _tail_file(path: Path, max_chars: int = _DEFAULT_TAIL_CHARS) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _is_pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def _load_config() -> Dict[str, Any]:
    try:
        from cli import CLI_CONFIG

        cfg = CLI_CONFIG.get("delegation", {})
        if isinstance(cfg, dict) and cfg:
            return cfg
    except Exception:
        pass
    try:
        from hermes_cli.config import load_config

        full = load_config()
        cfg = full.get("delegation", {})
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _active_profile_name() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name()
    except Exception:
        return "default"


def _resolve_profile(task: Dict[str, Any], cfg: Dict[str, Any], top_profile: Optional[str]) -> tuple[str, bool, str]:
    """Return (profile_name, explicit, hermes_home)."""
    requested = (
        task.get("profile")
        if task.get("profile") not in (None, "")
        else top_profile if top_profile not in (None, "") else cfg.get("default_profile", "")
    )
    explicit = bool(str(requested or "").strip())
    profile = str(requested).strip() if explicit else _active_profile_name()
    if profile in ("inherit", "current"):
        profile = _active_profile_name()
        explicit = False

    if profile == "custom" and not explicit:
        hermes_home = os.environ.get("HERMES_HOME", "").strip()
        if not hermes_home:
            raise ValueError("Active Hermes profile is custom but HERMES_HOME is not set.")
        return profile, False, hermes_home

    try:
        from hermes_cli.profiles import profile_exists, resolve_profile_env, validate_profile_name

        validate_profile_name(profile)
        if not profile_exists(profile):
            raise ValueError(
                f"Profile '{profile}' does not exist. Create it with: hermes profile create {profile}"
            )
        allowed = cfg.get("allowed_profiles") or []
        if isinstance(allowed, str):
            allowed = [p.strip() for p in allowed.split(",") if p.strip()]
        if allowed and profile not in set(str(p) for p in allowed):
            raise ValueError(
                f"Profile '{profile}' is not allowed for delegation. "
                "Update delegation.allowed_profiles in config.yaml."
            )
        hermes_home = resolve_profile_env(profile)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Cannot resolve delegation profile '{profile}': {exc}") from exc
    return profile, explicit, hermes_home


def _hermes_command() -> List[str]:
    raw = os.getenv("HERMES_DELEGATE_COMMAND", "").strip()
    if raw:
        import shlex

        return shlex.split(raw)
    hermes = shutil.which("hermes")
    if hermes:
        return [hermes]
    project_root = Path(__file__).resolve().parents[1]
    return [sys.executable, str(project_root / "hermes_cli" / "main.py")]


def _normalize_approval_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"allow", "allowed", "approve", "approved"}:
        return "approve"
    if mode in {"deny", "denied", "block", "blocked"}:
        return "deny"
    if mode in {"inherit", "parent", "manual", ""}:
        return "inherit" if mode else "deny"
    return "deny"


def _resolve_approval_mode(task: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    if task.get("approval_mode") not in (None, ""):
        return _normalize_approval_mode(task.get("approval_mode"))
    return _normalize_approval_mode(cfg.get("approval_mode", "deny"))


def _bridge_managed_tool_gateway_env(env: Dict[str, str]) -> None:
    """Pass managed-tool gateway auth explicitly to subprocess children.

    Children may run under a different profile/HERMES_HOME, so relying on them
    to rediscover the parent's Nous auth store is brittle.  We only bridge when
    the parent process can already prove managed tools are available.
    """
    if env.get("TOOL_GATEWAY_USER_TOKEN", "").strip():
        env["HERMES_MANAGED_TOOLS_ENABLED"] = "1"
        return

    try:
        from tools.managed_tool_gateway import read_nous_access_token
        from tools.tool_backend_helpers import managed_nous_tools_enabled

        if not managed_nous_tools_enabled():
            return
        token = read_nous_access_token()
    except Exception:
        return

    if token:
        env["TOOL_GATEWAY_USER_TOKEN"] = str(token)
        env["HERMES_MANAGED_TOOLS_ENABLED"] = "1"


def _build_subprocess_argv(
    *,
    profile: str,
    explicit_profile: bool,
    request_path: Path,
    output_path: Path,
) -> List[str]:
    argv = _hermes_command()
    # If a profile was explicitly requested, keep it explicit in argv.  For
    # inherited/default profile jobs, HERMES_HOME in the env is enough and avoids
    # surprising the sticky-profile pre-parser.
    if explicit_profile or profile != _active_profile_name():
        argv.extend(["--profile", profile])
    argv.extend(
        [
            "chat",
            "-Q",
            "--source",
            "tool",
            "--delegate-request",
            str(request_path),
            "--delegate-output",
            str(output_path),
        ]
    )
    return argv


def _normalize_status_from_output(output: Dict[str, Any], returncode: int) -> str:
    raw_status = str(output.get("status") or "")
    if returncode != 0:
        if raw_status in {"cancelled", "interrupted", "timeout"}:
            return raw_status
        return "failed"
    if raw_status in _RESULT_STATUSES:
        return raw_status
    if output.get("failed") or output.get("error"):
        return "failed"
    return "completed"


def _watch_subprocess(
    job_id: str,
    process: subprocess.Popen,
    *,
    stdout_file,
    stderr_file,
    output_path: Path,
    parent_agent=None,
    task_goal: str = "",
) -> None:
    try:
        returncode = process.wait()
    finally:
        try:
            stdout_file.close()
        except Exception:
            pass
        try:
            stderr_file.close()
        except Exception:
            pass

    output = _read_json(output_path, {}) or {}
    stdout_tail = _tail_file(job_dir(job_id) / "stdout.log")
    stderr_tail = _tail_file(job_dir(job_id) / "stderr.log")
    status = _normalize_status_from_output(output, returncode)
    finished = _now()

    result = {
        "job_id": job_id,
        "status": status,
        "returncode": returncode,
        "final_response": output.get("final_response") or stdout_tail.strip(),
        "child_session_id": output.get("session_id") or "",
        "error": output.get("error") or (stderr_tail.strip() if returncode else ""),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "finished_at": finished,
    }
    _write_json(job_dir(job_id) / "result.json", result)
    meta = _update_job(
        job_id,
        status=status,
        returncode=returncode,
        finished_at=finished,
        child_session_id=result["child_session_id"],
        result_path=str(job_dir(job_id) / "result.json"),
    )

    with _ACTIVE_LOCK:
        _ACTIVE_JOBS.pop(job_id, None)

    parent_cb = getattr(parent_agent, "tool_progress_callback", None)
    if parent_cb:
        try:
            parent_cb(
                "subagent.complete",
                preview=(result.get("final_response") or result.get("error") or "")[:160],
                status=status,
                duration_seconds=round(finished - float(meta.get("started_at") or finished), 2),
                summary=(result.get("final_response") or result.get("error") or "")[:500],
                subagent_id=job_id,
                job_id=job_id,
                profile=meta.get("profile"),
                child_session_id=result["child_session_id"],
            )
        except Exception:
            pass

    try:
        memory = getattr(parent_agent, "_memory_manager", None)
        if memory:
            memory.on_delegation(
                task=task_goal,
                result=result.get("final_response") or "",
                child_session_id=result.get("child_session_id") or "",
            )
    except Exception:
        pass


def start_jobs(
    tasks: List[Dict[str, Any]],
    *,
    parent_agent=None,
    top_profile: Optional[str] = None,
    cwd: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Start async delegation jobs and return a model-facing summary."""
    cfg = dict(cfg or _load_config())
    parent_session_id = str(getattr(parent_agent, "session_id", "") or "")
    parent_subagent_id = getattr(parent_agent, "_subagent_id", None)
    cwd = cwd or os.getenv("TERMINAL_CWD") or os.getcwd()
    job_ids: List[str] = []

    for index, task in enumerate(tasks):
        job_id = f"job-{_utc_id()}-{uuid.uuid4().hex[:8]}"
        jdir = job_dir(job_id)
        jdir.mkdir(parents=True, exist_ok=True)

        profile, explicit_profile, hermes_home = _resolve_profile(task, cfg, top_profile)
        approval_mode = _resolve_approval_mode(task, cfg)
        request_path = jdir / "request.json"
        output_path = jdir / "output.json"
        stdout_path = jdir / "stdout.log"
        stderr_path = jdir / "stderr.log"

        request = dict(task)
        request.update(
            {
                "job_id": job_id,
                "task_index": index,
                "parent_session_id": parent_session_id,
                "parent_subagent_id": parent_subagent_id,
                "profile": profile,
                "cwd": cwd,
            }
        )
        _write_json(request_path, request)

        argv = _build_subprocess_argv(
            profile=profile,
            explicit_profile=explicit_profile,
            request_path=request_path,
            output_path=output_path,
        )
        started = _now()
        meta = {
            "job_id": job_id,
            "status": "starting",
            "runner": "subprocess",
            "created_at": started,
            "started_at": started,
            "updated_at": started,
            "parent_session_id": parent_session_id,
            "parent_subagent_id": parent_subagent_id,
            "child_session_id": "",
            "profile": profile,
            "explicit_profile": explicit_profile,
            "goal": str(task.get("goal") or ""),
            "role": str(task.get("role") or "leaf"),
            "model": task.get("model") or "",
            "provider": task.get("provider") or "",
            "sandbox": task.get("sandbox") or "",
            "sandbox_mode": task.get("sandbox_mode") or "",
            "sandbox_scope": task.get("sandbox_scope") or "",
            "approval_mode": approval_mode,
            "cwd": cwd,
            "request_path": str(request_path),
            "output_path": str(output_path),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "argv": argv,
        }
        _write_json(_json_path(job_id), meta)

        env = os.environ.copy()
        env["HERMES_HOME"] = hermes_home
        env["HERMES_DELEGATE_JOB_ID"] = job_id
        if approval_mode != "inherit":
            env["HERMES_DELEGATE_APPROVAL_MODE"] = approval_mode
            if approval_mode == "deny":
                env.pop("HERMES_YOLO_MODE", None)
            elif approval_mode == "approve":
                env["HERMES_YOLO_MODE"] = "1"
        else:
            env.pop("HERMES_DELEGATE_APPROVAL_MODE", None)
        _bridge_managed_tool_gateway_env(env)
        if parent_session_id:
            env["HERMES_DELEGATE_PARENT_SESSION_ID"] = parent_session_id

        stdout_file = stdout_path.open("w", encoding="utf-8")
        stderr_file = stderr_path.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                start_new_session=True,
            )
        except Exception as exc:
            try:
                stdout_file.close()
                stderr_file.close()
            except Exception:
                pass
            failed = _update_job(job_id, status="failed", error=str(exc), finished_at=_now())
            _write_json(
                jdir / "result.json",
                {
                    "job_id": job_id,
                    "status": "failed",
                    "error": str(exc),
                    "final_response": "",
                    "child_session_id": "",
                },
            )
            job_ids.append(job_id)
            continue

        _update_job(job_id, status="running", pid=process.pid)
        with _ACTIVE_LOCK:
            _ACTIVE_JOBS[job_id] = {
                "job_id": job_id,
                "process": process,
                "parent_agent": parent_agent,
                "started_at": started,
            }

        parent_cb = getattr(parent_agent, "tool_progress_callback", None)
        if parent_cb:
            try:
                parent_cb(
                    "subagent.start",
                    preview=str(task.get("goal") or ""),
                    subagent_id=job_id,
                    job_id=job_id,
                    parent_id=parent_subagent_id,
                    depth=max(0, int(task.get("child_depth") or 1) - 1),
                    profile=profile,
                    status="running",
                    model=task.get("model") or "",
                    toolsets=list(task.get("toolsets") or []),
                    role=task.get("role") or "leaf",
                )
            except Exception:
                pass

        watcher = threading.Thread(
            target=_watch_subprocess,
            args=(job_id, process),
            kwargs={
                "stdout_file": stdout_file,
                "stderr_file": stderr_file,
                "output_path": output_path,
                "parent_agent": parent_agent,
                "task_goal": str(task.get("goal") or ""),
            },
            daemon=True,
            name=f"delegate-watch-{job_id}",
        )
        watcher.start()
        with _ACTIVE_LOCK:
            _ACTIVE_JOBS[job_id]["watcher"] = watcher
        job_ids.append(job_id)

    return {
        "status": "running",
        "job_ids": job_ids,
        "jobs": [get_job(jid) for jid in job_ids],
    }


def _refresh_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    status = str(meta.get("status") or "")
    if status in _RESULT_STATUSES:
        return meta
    job_id = str(meta.get("job_id") or "")
    result_path = Path(str(meta.get("result_path") or job_dir(job_id) / "result.json"))
    if result_path.exists():
        result = _read_json(result_path, {}) or {}
        if result.get("status"):
            meta["status"] = result.get("status")
            meta["child_session_id"] = result.get("child_session_id") or meta.get("child_session_id", "")
            _write_json(_json_path(job_id), meta)
            return meta
    pid = meta.get("pid")
    if status in {"running", "starting", "cancelling"} and pid and not _is_pid_alive(int(pid)):
        meta["status"] = "failed" if status != "cancelling" else "cancelled"
        meta["finished_at"] = meta.get("finished_at") or _now()
        meta["error"] = meta.get("error") or "Subprocess exited before writing a result."
        _write_json(_json_path(job_id), meta)
    return meta


def get_job(job_id: str) -> Dict[str, Any]:
    meta = _read_json(_json_path(job_id), {}) or {}
    if meta:
        meta = _refresh_meta(meta)
    return meta


def list_jobs(
    *,
    status: Optional[str] = None,
    profile: Optional[str] = None,
    parent_session_id: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    metas: List[Dict[str, Any]] = []
    root = jobs_root()
    for path in sorted(root.glob("job-*/job.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        meta = _read_json(path, {}) or {}
        if not meta:
            continue
        meta = _refresh_meta(meta)
        if status and meta.get("status") != status:
            continue
        if profile and meta.get("profile") != profile:
            continue
        if parent_session_id and meta.get("parent_session_id") != parent_session_id:
            continue
        metas.append(meta)
        if len(metas) >= limit:
            break
    return metas


def wait_jobs(job_ids: Iterable[str], timeout: float = 0) -> Dict[str, Any]:
    deadline = time.monotonic() + max(0.0, float(timeout or 0))
    wanted = [str(j) for j in job_ids if str(j).strip()]
    completed: List[Dict[str, Any]] = []
    running: List[str] = []

    while True:
        completed.clear()
        running.clear()
        for jid in wanted:
            meta = get_job(jid)
            if not meta:
                completed.append({"job_id": jid, "status": "missing", "error": "Job not found"})
            elif meta.get("status") in _RESULT_STATUSES:
                completed.append(get_result(jid))
            else:
                running.append(jid)
        if not running or timeout == 0 or time.monotonic() >= deadline:
            break
        time.sleep(0.25)

    return {"completed": completed, "running": running}


def get_result(job_id: str, tail_chars: int = _DEFAULT_TAIL_CHARS) -> Dict[str, Any]:
    meta = get_job(job_id)
    if not meta:
        return {"job_id": job_id, "status": "missing", "error": "Job not found"}
    result = _read_json(job_dir(job_id) / "result.json", {}) or {}
    stdout_path = Path(str(meta.get("stdout_path") or job_dir(job_id) / "stdout.log"))
    stderr_path = Path(str(meta.get("stderr_path") or job_dir(job_id) / "stderr.log"))
    return {
        "job": meta,
        "result": result,
        "stdout_tail": _tail_file(stdout_path, tail_chars),
        "stderr_tail": _tail_file(stderr_path, tail_chars),
    }


def cancel_jobs(job_ids: Iterable[str], grace_seconds: float = 2.0) -> Dict[str, Any]:
    cancelled: List[str] = []
    missing: List[str] = []
    errors: Dict[str, str] = {}
    for jid in [str(j) for j in job_ids if str(j).strip()]:
        meta = get_job(jid)
        if not meta:
            missing.append(jid)
            continue
        with _ACTIVE_LOCK:
            runtime = _ACTIVE_JOBS.get(jid)
        proc = runtime.get("process") if runtime else None
        pid = int(meta.get("pid") or 0)
        _update_job(jid, status="cancelling")
        try:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=grace_seconds)
                except subprocess.TimeoutExpired:
                    proc.kill()
            elif pid:
                try:
                    os.killpg(pid, signal.SIGTERM)
                except Exception:
                    os.kill(pid, signal.SIGTERM)
            _update_job(jid, status="cancelled", finished_at=_now())
            _write_json(
                job_dir(jid) / "result.json",
                {
                    "job_id": jid,
                    "status": "cancelled",
                    "error": "Delegation job cancelled.",
                    "final_response": "",
                    "child_session_id": meta.get("child_session_id") or "",
                },
            )
            cancelled.append(jid)
        except Exception as exc:
            errors[jid] = str(exc)
            _update_job(jid, status="failed", error=str(exc), finished_at=_now())
    return {"cancelled": cancelled, "missing": missing, "errors": errors}


def list_active_jobs_as_subagents() -> List[Dict[str, Any]]:
    active: List[Dict[str, Any]] = []
    with _ACTIVE_LOCK:
        ids = list(_ACTIVE_JOBS.keys())
    for jid in ids:
        meta = get_job(jid)
        if not meta or meta.get("status") not in {"starting", "running", "cancelling"}:
            continue
        active.append(
            {
                "subagent_id": jid,
                "job_id": jid,
                "parent_id": meta.get("parent_subagent_id"),
                "depth": max(0, int(meta.get("child_depth") or 1) - 1),
                "goal": meta.get("goal") or "",
                "model": meta.get("model") or None,
                "profile": meta.get("profile") or None,
                "started_at": meta.get("started_at"),
                "status": meta.get("status"),
                "tool_count": 0,
                "runner": "subprocess",
            }
        )
    return active


def interrupt_job(job_id: str) -> bool:
    result = cancel_jobs([job_id])
    return job_id in result.get("cancelled", [])


def prune_old_jobs(retention_hours: Optional[float] = None) -> int:
    cfg = _load_config()
    if retention_hours is None:
        try:
            retention_hours = float(cfg.get("job_retention_hours", 24))
        except (TypeError, ValueError):
            retention_hours = 24.0
    cutoff = _now() - max(0.0, retention_hours) * 3600
    count = 0
    for meta in list_jobs(limit=10000):
        if meta.get("status") not in _RESULT_STATUSES:
            continue
        finished = float(meta.get("finished_at") or meta.get("updated_at") or 0)
        if finished and finished < cutoff:
            shutil.rmtree(job_dir(meta["job_id"]), ignore_errors=True)
            count += 1
    return count
