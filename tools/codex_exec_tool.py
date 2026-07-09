#!/usr/bin/env python3
"""Codex execution bridge for Hermes.

This tool intentionally treats Codex as a separate programming runtime instead
of another Hermes model provider. Hermes decides when to delegate; Codex keeps
its own auth, sandboxing, repo instructions, session traces, and JSONL events.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error, tool_result


_MAX_TASK_CHARS = 32_000
_DEFAULT_TIMEOUT_SECONDS = 20 * 60
_MAX_TIMEOUT_SECONDS = 60 * 60
_SAFE_SANDBOXES = {"read-only", "workspace-write"}
_ALL_SANDBOXES = _SAFE_SANDBOXES | {"danger-full-access"}
_SECRET_ENV_RE = re.compile(r"(API[_-]?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|PRIVATE[_-]?KEY)", re.I)
_ALWAYS_KEEP_ENV = {
    "CODEX_HOME",
    "HOME",
    "LANG",
    "LC_ALL",
    "LOGNAME",
    "PATH",
    "SHELL",
    "SSH_AUTH_SOCK",
    "TERM",
    "TMPDIR",
    "USER",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_codex_exec_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
        section = cfg.get("codex_exec") or {}
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _configured_allowed_roots(config: Optional[Dict[str, Any]] = None) -> List[Path]:
    cfg = config if config is not None else _load_codex_exec_config()
    roots: List[str] = []
    roots.extend(_as_list(cfg.get("allowed_roots")))
    roots.extend(_as_list(cfg.get("allowed_root")))
    env_roots = os.getenv("HERMES_CODEX_EXEC_ALLOWED_ROOTS", "")
    if env_roots.strip():
        roots.extend(part for part in env_roots.split(os.pathsep) if part.strip())

    if not roots:
        roots = [str(_repo_root())]

    resolved: List[Path] = []
    for root in roots:
        try:
            path = Path(root).expanduser().resolve()
        except Exception:
            continue
        if path.exists() and path.is_dir() and path not in resolved:
            resolved.append(path)
    return resolved


def _configured_default_sandbox(config: Optional[Dict[str, Any]] = None) -> str:
    cfg = config if config is not None else _load_codex_exec_config()
    value = str(cfg.get("default_sandbox") or "read-only").strip()
    return value if value in _SAFE_SANDBOXES else "read-only"


def _configured_max_timeout(config: Optional[Dict[str, Any]] = None) -> int:
    cfg = config if config is not None else _load_codex_exec_config()
    try:
        value = int(cfg.get("max_timeout_seconds") or _MAX_TIMEOUT_SECONDS)
    except (TypeError, ValueError):
        value = _MAX_TIMEOUT_SECONDS
    return max(10, min(value, _MAX_TIMEOUT_SECONDS))


def _resolve_cwd(cwd: Optional[str], allowed_roots: Optional[Iterable[Path]] = None) -> Path:
    raw = (cwd or str(_repo_root())).strip()
    if not raw:
        raw = str(_repo_root())
    path = Path(raw).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise ValueError(f"cwd must be an existing directory: {raw}")

    roots = list(allowed_roots) if allowed_roots is not None else _configured_allowed_roots()
    for root in roots:
        try:
            path.relative_to(root)
            return path
        except ValueError:
            continue
    allowed = ", ".join(str(root) for root in roots) or "(none)"
    raise PermissionError(f"cwd is outside codex_exec.allowed_roots. cwd={path}; allowed={allowed}")


def _artifact_root() -> Path:
    return Path(get_hermes_home()) / "codex-runs"


def _make_artifact_dir() -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = _artifact_root() / f"{stamp}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _scrub_env(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    source = dict(env or os.environ)
    clean: Dict[str, str] = {}
    for key, value in source.items():
        if key in _ALWAYS_KEEP_ENV:
            clean[key] = value
            continue
        if _SECRET_ENV_RE.search(key):
            continue
        if key.startswith(("CODEX_", "GIT_", "HERMES_CODEX_EXEC_")):
            clean[key] = value
    return clean


def _codex_home(env: Optional[Dict[str, str]] = None) -> Path:
    source = env or os.environ
    raw = str(source.get("CODEX_HOME") or "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".codex"


def _codex_auth_available(env: Optional[Dict[str, str]] = None) -> bool:
    return (_codex_home(env) / "auth.json").is_file()


def _git_status(cwd: Path) -> Optional[str]:
    if not (cwd / ".git").exists():
        try:
            subprocess.run(
                ["git", "-C", str(cwd), "rev-parse", "--is-inside-work-tree"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except Exception:
            return None
    try:
        proc = subprocess.run(
            ["git", "-C", str(cwd), "status", "--short"],
            check=False,
            text=True,
            capture_output=True,
            timeout=10,
        )
        return proc.stdout
    except Exception:
        return None


def _parse_jsonl(stdout: str) -> Dict[str, Any]:
    final_messages: List[str] = []
    command_count = 0
    tool_call_count = 0
    thread_id = None
    usage = None
    errors: List[Any] = []

    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_type = event.get("type")
        if event_type == "thread.started":
            thread_id = event.get("thread_id") or thread_id
        elif event_type == "turn.completed":
            usage = event.get("usage") or usage
        elif event_type == "error":
            errors.append(event)

        item = event.get("item") if isinstance(event.get("item"), dict) else {}
        item_type = item.get("type")
        if item_type == "agent_message":
            text = item.get("text")
            if isinstance(text, str):
                final_messages.append(text)
        elif item_type == "command_execution":
            command_count += 1
        elif item_type == "mcp_tool_call":
            tool_call_count += 1

    final_message = final_messages[-1] if final_messages else ""
    return {
        "thread_id": thread_id,
        "usage": usage,
        "command_count": command_count,
        "tool_call_count": tool_call_count,
        "final_message": final_message,
        "errors": errors,
    }


def _build_prompt(task: str) -> str:
    return (
        "You are Codex being invoked by Hermes through the codex_exec tool.\n"
        "Respect the target repository instructions, keep changes scoped, avoid "
        "printing secrets, and finish with a concise summary of what you did and "
        "how you verified it.\n\n"
        f"Task:\n{task}"
    )


def check_codex_exec_requirements() -> bool:
    return shutil.which("codex") is not None and _codex_auth_available()


def codex_exec_tool(
    task: str,
    cwd: Optional[str] = None,
    sandbox: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
    model: Optional[str] = None,
    profile: Optional[str] = None,
    output_schema: Optional[str] = None,
    allow_danger: bool = False,
    ephemeral: bool = True,
    skip_git_repo_check: bool = False,
) -> str:
    """Run Codex non-interactively and return a compact audited summary."""
    if not shutil.which("codex"):
        return tool_error("codex CLI is not installed or not on PATH.")
    if not _codex_auth_available():
        return tool_error("Codex auth is unavailable. Run `codex login` and retry; codex_exec does not fall back.")

    if not isinstance(task, str) or not task.strip():
        return tool_error("task is required.")
    task = task.strip()
    if len(task) > _MAX_TASK_CHARS:
        return tool_error(f"task is too large ({len(task)} chars; max {_MAX_TASK_CHARS}).")

    cfg = _load_codex_exec_config()
    sandbox = (sandbox or _configured_default_sandbox(cfg)).strip()
    if sandbox not in _ALL_SANDBOXES:
        return tool_error(f"invalid sandbox '{sandbox}'. Use read-only, workspace-write, or danger-full-access.")
    if sandbox == "danger-full-access" and not allow_danger:
        return tool_error("danger-full-access requires allow_danger=true.")

    try:
        resolved_cwd = _resolve_cwd(cwd)
    except Exception as exc:
        return tool_error(str(exc))

    try:
        timeout = int(timeout_seconds or _DEFAULT_TIMEOUT_SECONDS)
    except (TypeError, ValueError):
        return tool_error("timeout_seconds must be an integer.")
    timeout = max(10, min(timeout, _configured_max_timeout(cfg)))

    schema_path = None
    if output_schema:
        try:
            schema_path = Path(output_schema).expanduser().resolve()
            schema_path.relative_to(resolved_cwd)
        except Exception:
            return tool_error("output_schema must resolve inside cwd.")
        if not schema_path.exists() or not schema_path.is_file():
            return tool_error(f"output_schema does not exist: {schema_path}")

    artifact_dir = _make_artifact_dir()
    before_status = _git_status(resolved_cwd)

    cmd = ["codex", "exec", "--json", "--sandbox", sandbox, "--cd", str(resolved_cwd)]
    if ephemeral:
        cmd.append("--ephemeral")
    if skip_git_repo_check:
        cmd.append("--skip-git-repo-check")
    if model:
        cmd.extend(["--model", str(model)])
    if profile:
        cmd.extend(["--profile", str(profile)])
    if schema_path:
        cmd.extend(["--output-schema", str(schema_path)])
    cmd.append(_build_prompt(task))

    started = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            cwd=str(resolved_cwd),
            env=_scrub_env(),
            timeout=timeout,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", "replace")
        returncode = 124

    duration = round(time.monotonic() - started, 2)
    parsed = _parse_jsonl(stdout)
    after_status = _git_status(resolved_cwd)

    (artifact_dir / "stdout.jsonl").write_text(stdout, encoding="utf-8")
    (artifact_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
    if parsed.get("final_message"):
        (artifact_dir / "final_message.md").write_text(parsed["final_message"], encoding="utf-8")
    metadata = {
        "cmd": [part if part != cmd[-1] else "<prompt>" for part in cmd],
        "cwd": str(resolved_cwd),
        "sandbox": sandbox,
        "timeout_seconds": timeout,
        "duration_seconds": duration,
        "returncode": returncode,
        "timed_out": timed_out,
        "thread_id": parsed.get("thread_id"),
        "usage": parsed.get("usage"),
        "before_git_status": before_status,
        "after_git_status": after_status,
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return tool_result(
        success=(returncode == 0 and not timed_out),
        returncode=returncode,
        timed_out=timed_out,
        duration_seconds=duration,
        cwd=str(resolved_cwd),
        sandbox=sandbox,
        artifact_dir=str(artifact_dir),
        thread_id=parsed.get("thread_id"),
        usage=parsed.get("usage"),
        command_count=parsed.get("command_count"),
        tool_call_count=parsed.get("tool_call_count"),
        before_git_status=before_status,
        after_git_status=after_status,
        final_message=parsed.get("final_message", "")[-6000:],
        stderr_tail=stderr[-4000:],
        errors=parsed.get("errors", [])[-5:],
    )


CODEX_EXEC_SCHEMA = {
    "name": "codex_exec",
    "description": (
        "Delegate coding, repository analysis, debugging, or test-fix loops to "
        "OpenAI Codex CLI using Codex auth. Hermes remains the orchestrator; "
        "Codex runs as a separate audited programming engine with JSONL traces "
        "saved under ~/.hermes/codex-runs. Use read-only for diagnosis and "
        "workspace-write for scoped implementation. Do not use for sending "
        "messages, deploys, DB migrations, or secret handling without explicit "
        "human approval."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The coding/repo task for Codex. Include expected tests or constraints.",
            },
            "cwd": {
                "type": "string",
                "description": "Target repository/workspace directory. Must be inside codex_exec.allowed_roots.",
            },
            "sandbox": {
                "type": "string",
                "enum": ["read-only", "workspace-write", "danger-full-access"],
                "default": "read-only",
                "description": "Codex sandbox. Use read-only for audits, workspace-write for edits.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 10,
                "maximum": _MAX_TIMEOUT_SECONDS,
                "default": _DEFAULT_TIMEOUT_SECONDS,
            },
            "model": {"type": "string", "description": "Optional Codex model override."},
            "profile": {"type": "string", "description": "Optional Codex config profile."},
            "output_schema": {
                "type": "string",
                "description": "Optional JSON schema path inside cwd for Codex final response.",
            },
            "allow_danger": {
                "type": "boolean",
                "default": False,
                "description": "Required true to permit danger-full-access.",
            },
            "ephemeral": {
                "type": "boolean",
                "default": True,
                "description": "Use codex --ephemeral; Hermes still stores artifacts.",
            },
            "skip_git_repo_check": {
                "type": "boolean",
                "default": False,
                "description": "Pass --skip-git-repo-check for controlled non-git workspaces.",
            },
        },
        "required": ["task"],
    },
}


registry.register(
    name="codex_exec",
    toolset="codex_exec",
    schema=CODEX_EXEC_SCHEMA,
    handler=lambda args, **kw: codex_exec_tool(
        task=args.get("task", ""),
        cwd=args.get("cwd"),
        sandbox=args.get("sandbox", "read-only"),
        timeout_seconds=args.get("timeout_seconds"),
        model=args.get("model"),
        profile=args.get("profile"),
        output_schema=args.get("output_schema"),
        allow_danger=bool(args.get("allow_danger", False)),
        ephemeral=bool(args.get("ephemeral", True)),
        skip_git_repo_check=bool(args.get("skip_git_repo_check", False)),
    ),
    check_fn=check_codex_exec_requirements,
    emoji="🧑‍💻",
    max_result_size_chars=20_000,
)
