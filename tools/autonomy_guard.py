"""Core autonomy policy loading and enforcement helpers."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from hermes_constants import get_hermes_home


POLICY_PATH = Path(__file__).resolve().parent.parent / "autonomy_policy.yaml"


class AutonomyPolicyError(RuntimeError):
    """Raised when the autonomy policy is unavailable or malformed."""


@lru_cache(maxsize=1)
def load_autonomy_policy() -> Dict[str, Any]:
    """Load the repo-local autonomy policy."""
    try:
        with POLICY_PATH.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except FileNotFoundError as exc:
        raise AutonomyPolicyError(f"Autonomy policy is missing at {POLICY_PATH}.") from exc
    except yaml.YAMLError as exc:
        raise AutonomyPolicyError(f"Autonomy policy at {POLICY_PATH} is malformed: {exc}") from exc
    if not isinstance(data, dict):
        raise AutonomyPolicyError(f"Autonomy policy at {POLICY_PATH} must be a mapping.")
    return data


def _iter_pattern_entries(entries: Iterable[Dict[str, Any]]) -> Iterable[tuple[re.Pattern[str], str]]:
    for entry in entries:
        pattern = entry.get("pattern", "")
        description = entry.get("description", pattern or "policy rule")
        if not isinstance(pattern, str) or not pattern.strip():
            continue
        yield re.compile(pattern, re.IGNORECASE), str(description)


def _git(args: list[str], cwd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def get_git_context(start_path: str) -> Dict[str, Optional[str]]:
    """Return git repo metadata for the given path, if any."""
    path = Path(start_path).expanduser()
    base = path if path.is_dir() else path.parent
    base = base.resolve()
    repo_root_proc = _git(["rev-parse", "--show-toplevel"], str(base))
    if repo_root_proc.returncode != 0:
        return {"repo_root": None, "branch": None}

    repo_root = repo_root_proc.stdout.strip()
    branch_proc = _git(["branch", "--show-current"], repo_root)
    branch = branch_proc.stdout.strip() if branch_proc.returncode == 0 else ""
    return {"repo_root": repo_root or None, "branch": branch or None}


def _protected_branches() -> set[str]:
    policy = load_autonomy_policy()
    branches = policy.get("branch_protection", {}).get("protected_branches", [])
    return {str(branch).strip() for branch in branches if str(branch).strip()}


def enforce_write_policy(action: str, target_path: str) -> Dict[str, Any]:
    """Block mutating writes on protected branches."""
    try:
        protected_branches = _protected_branches()
    except AutonomyPolicyError as exc:
        return {
            "allowed": False,
            "status": "blocked",
            "branch": None,
            "repo_root": None,
            "description": "autonomy policy unavailable",
            "message": f"BLOCKED: {exc}",
        }
    git_ctx = get_git_context(target_path)
    branch = git_ctx.get("branch")
    repo_root = git_ctx.get("repo_root")
    if repo_root and branch in protected_branches:
        return {
            "allowed": False,
            "status": "blocked",
            "branch": branch,
            "repo_root": repo_root,
            "message": (
                f"BLOCKED: {action} is not allowed on protected branch '{branch}'. "
                "Create or switch to a feature branch before writing."
            ),
        }
    return {
        "allowed": True,
        "status": "ok",
        "branch": branch,
        "repo_root": repo_root,
    }


def command_mutates_filesystem(command: str) -> bool:
    """Return True when the command is likely to change files or git state."""
    policy = load_autonomy_policy()
    entries = policy.get("terminal_writes", {}).get("patterns", [])
    for regex, _description in _iter_pattern_entries(entries):
        if regex.search(command):
            return True
    return False


def evaluate_terminal_command(command: str, *, workdir: Optional[str], force: bool = False) -> Dict[str, Any]:
    """Apply autonomy policy to terminal execution."""
    try:
        policy = load_autonomy_policy()
    except AutonomyPolicyError as exc:
        return {
            "allowed": False,
            "status": "blocked",
            "description": "autonomy policy unavailable",
            "message": f"BLOCKED: {exc}",
        }
    approval_cfg = policy.get("approval", {})

    for fragment in approval_cfg.get("forbidden_command_fragments", []):
        text = str(fragment).strip()
        if text and text in command:
            return {
                "allowed": False,
                "status": "blocked",
                "description": f"forbidden autonomy bypass: {text}",
                "message": (
                    f"BLOCKED: autonomy policy forbids `{text}` in terminal commands."
                ),
            }

    cwd = workdir or os.getenv("TERMINAL_CWD", os.getcwd())
    if command_mutates_filesystem(command):
        write_decision = enforce_write_policy("terminal command", cwd)
        if not write_decision["allowed"]:
            write_decision["description"] = "write on protected branch"
            return write_decision

    for regex, description in _iter_pattern_entries(
        approval_cfg.get("require_explicit_approval_patterns", [])
    ):
        if regex.search(command):
            if force:
                return {
                    "allowed": True,
                    "status": "ok",
                    "description": description,
                    "approved_via_force": True,
                }
            return {
                "allowed": False,
                "status": "approval_required",
                "description": description,
                "message": (
                    f"Human approval required by autonomy policy: {description}.\n\n"
                    f"Command:\n```\n{command}\n```"
                ),
            }

    return {"allowed": True, "status": "ok"}


def _has_explicit_runtime_credentials(explicit_api_key: Optional[str], explicit_base_url: Optional[str]) -> bool:
    return bool((explicit_api_key or "").strip() or (explicit_base_url or "").strip())


def run_bootstrap_preflight(
    *,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
    requested_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate machine prerequisites before an autonomous turn runs."""
    try:
        policy = load_autonomy_policy()
    except AutonomyPolicyError as exc:
        return {
            "ok": False,
            "provider": None,
            "diagnostics": [str(exc)],
            "message": f"Autonomy bootstrap preflight failed:\n- {exc}",
        }
    bootstrap = policy.get("bootstrap", {})
    diagnostics: list[str] = []

    missing_commands = [
        cmd for cmd in bootstrap.get("required_commands", [])
        if not shutil.which(str(cmd))
    ]
    if missing_commands:
        diagnostics.append(
            "Missing required commands: " + ", ".join(sorted(missing_commands))
        )

    hermes_home = get_hermes_home()
    config_path = hermes_home / "config.yaml"
    explicit_creds = _has_explicit_runtime_credentials(explicit_api_key, explicit_base_url)

    if (
        bootstrap.get("require_hermes_config_without_explicit_credentials", True)
        and not explicit_creds
        and not config_path.exists()
    ):
        diagnostics.append(
            f"Missing Hermes config at {config_path}. Run `hermes model` or `hermes setup` first."
        )

    provider = None
    if bootstrap.get("require_provider_resolution", True):
        try:
            from hermes_cli.auth import AuthError, get_auth_status, resolve_provider

            provider = resolve_provider(
                requested=requested_provider,
                explicit_api_key=explicit_api_key,
                explicit_base_url=explicit_base_url,
            )

            if bootstrap.get("require_provider_auth", True) and not explicit_creds:
                if provider == "openrouter":
                    has_openrouter_key = bool(os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"))
                    if not has_openrouter_key:
                        diagnostics.append(
                            "Resolved provider 'openrouter' but no OPENROUTER_API_KEY or OPENAI_API_KEY is available."
                        )
                else:
                    status = get_auth_status(provider)
                    if not status.get("logged_in"):
                        msg = status.get("error") or f"Provider '{provider}' is not authenticated."
                        diagnostics.append(str(msg))
        except AuthError as exc:
            diagnostics.append(str(exc))
        except Exception as exc:  # pragma: no cover - defensive path
            diagnostics.append(f"Autonomy bootstrap preflight failed: {exc}")

    if diagnostics:
        return {
            "ok": False,
            "provider": provider,
            "diagnostics": diagnostics,
            "message": "Autonomy bootstrap preflight failed:\n- " + "\n- ".join(diagnostics),
        }

    return {
        "ok": True,
        "provider": provider,
        "diagnostics": [],
        "message": "Autonomy bootstrap preflight passed.",
    }


def create_proof_state(*, session_id: str, task_id: str, user_message: str, model: str, provider: str) -> Dict[str, Any]:
    """Return a fresh in-memory proof-of-done state object."""
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "task_id": task_id,
        "user_message": user_message,
        "model": model,
        "provider": provider,
        "tool_events": [],
        "files_touched": [],
        "approval_events": [],
        "commands_run": [],
        "gates_run": [],
        "preflight": None,
    }


def extract_patch_paths(patch: str) -> list[str]:
    return re.findall(r"^\*\*\*\s+(?:Update|Add|Delete)\s+File:\s*(.+)$", patch or "", re.MULTILINE)


def record_tool_event(proof_state: Dict[str, Any], tool_name: str, args: Dict[str, Any], result: str) -> None:
    """Record tool activity for the final proof artifact."""
    preview_limit = int(load_autonomy_policy().get("proof_of_done", {}).get("max_command_preview_chars", 240))
    event: Dict[str, Any] = {"tool": tool_name}

    if tool_name == "terminal":
        command = str(args.get("command", ""))
        event["command"] = command[:preview_limit]
        proof_state["commands_run"].append(command[:preview_limit])
    if tool_name == "write_file":
        path = str(args.get("path", ""))
        if path:
            proof_state["files_touched"].append(path)
            event["path"] = path
    if tool_name == "patch":
        paths: list[str] = []
        if args.get("path"):
            paths.append(str(args["path"]))
        if args.get("mode") == "patch":
            paths.extend(extract_patch_paths(str(args.get("patch", ""))))
        if paths:
            proof_state["files_touched"].extend(paths)
            event["paths"] = paths

    parsed: Optional[Dict[str, Any]] = None
    try:
        parsed = json.loads(result)
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        status = parsed.get("status")
        error = parsed.get("error")
        exit_code = parsed.get("exit_code")
        if status:
            event["status"] = status
        if error:
            event["error"] = error
        if isinstance(exit_code, int):
            event["exit_code"] = exit_code
        if tool_name == "terminal" and "command" in event:
            gate_name = _classify_gate_command(event["command"])
            if gate_name:
                proof_state["gates_run"].append(
                    {
                        "name": gate_name,
                        "command": event["command"],
                        "outcome": _gate_outcome(parsed),
                    }
                )
        if status == "approval_required" or parsed.get("approved") is False:
            proof_state["approval_events"].append(
                {
                    "tool": tool_name,
                    "status": status or "blocked",
                    "description": parsed.get("description", ""),
                    "error": error,
                }
            )
    proof_state["tool_events"].append(event)


def _classify_gate_command(command: str) -> Optional[str]:
    lowered = command.lower()
    if "scripts/run_readiness.py" in lowered:
        return "run_readiness"
    if "scripts/smoke_autonomy.py" in lowered:
        return "smoke_autonomy"
    if "pytest" in lowered:
        return "pytest"
    return None


def _gate_outcome(result: Dict[str, Any]) -> str:
    status = str(result.get("status") or "")
    if status == "approval_required":
        return "approval-required"
    if status == "blocked":
        return "blocked"
    exit_code = result.get("exit_code")
    if isinstance(exit_code, int):
        return "passed" if exit_code == 0 else "failed"
    if result.get("error"):
        return "failed"
    return "passed"


def _derive_artifact_status(proof_state: Dict[str, Any], *, completed: bool, interrupted: bool) -> str:
    if interrupted:
        return "failed"
    preflight = proof_state.get("preflight") or {}
    if isinstance(preflight, dict) and preflight.get("ok") is False:
        return "failed"

    tool_events = proof_state.get("tool_events", [])
    statuses = [str(event.get("status", "")) for event in tool_events if isinstance(event, dict)]
    if "approval_required" in statuses:
        return "approval-required"
    if "blocked" in statuses:
        return "blocked"

    for event in tool_events:
        if not isinstance(event, dict):
            continue
        if event.get("error"):
            return "failed"
        if isinstance(event.get("exit_code"), int) and event["exit_code"] != 0:
            return "failed"
    return "passed" if completed else "failed"


def _next_required_human_action(status: str) -> Optional[str]:
    if status == "approval-required":
        return "Review the requested action and explicitly approve or reject it."
    if status == "blocked":
        return "Move the work onto a feature branch or change the requested action so it satisfies policy."
    if status == "failed":
        return "Inspect the diagnostics, fix the failure, then re-run the task."
    return None


def write_proof_artifact(
    proof_state: Dict[str, Any],
    *,
    completed: bool,
    interrupted: bool,
    final_response: Optional[str],
    api_calls: int,
    stop_reason: Optional[str] = None,
    repo_hint: Optional[str] = None,
) -> str:
    """Persist a proof-of-done artifact and return its path."""
    proof_cfg: Dict[str, Any] = {}
    try:
        policy = load_autonomy_policy()
        proof_cfg = policy.get("proof_of_done", {})
    except AutonomyPolicyError:
        proof_cfg = {}
    out_dir = get_hermes_home() / str(proof_cfg.get("output_dir", "proof_of_done"))
    out_dir.mkdir(parents=True, exist_ok=True)

    git_ctx = get_git_context(repo_hint or os.getcwd())
    status = _derive_artifact_status(proof_state, completed=completed, interrupted=interrupted)
    artifact = {
        **proof_state,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "completed": completed,
        "interrupted": interrupted,
        "api_calls": api_calls,
        "stop_reason": stop_reason,
        "final_response_present": bool(final_response),
        "final_response_length": len(final_response or ""),
        "files_touched": sorted(set(proof_state.get("files_touched", []))),
        "commands_run": proof_state.get("commands_run", []),
        "gates_run": proof_state.get("gates_run", []),
        "next_step_requires_human_approval": bool(proof_state.get("approval_events")),
        "next_required_human_action": _next_required_human_action(status),
        "git": git_ctx,
    }

    session_stub = proof_state.get("session_id") or proof_state.get("task_id") or "session"
    safe_stub = re.sub(r"[^A-Za-z0-9._-]+", "-", str(session_stub)).strip("-") or "session"
    artifact_path = out_dir / f"{safe_stub}.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(artifact_path)
