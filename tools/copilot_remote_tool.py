"""Copilot remote job tool for semantic Hermes delegation.

This is the model-facing equivalent of ``hermes copilot``/``/copilot_remote``:
it lets a normal agent turn launch a tracked GitHub Copilot remote job
without asking the model to improvise shell commands or ACP provider usage.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, Optional

from agent.redact import redact_sensitive_text
from copilot_remote.models import RepoEntry
from hermes_state import SessionDB
from tools.registry import registry


logger = logging.getLogger(__name__)


COPILOT_REMOTE_SCHEMA = {
    "name": "copilot_remote",
    "description": (
        "Launch, list, or inspect tracked GitHub Copilot remote jobs. This is "
        "Hermes' default implementation tool for code-writing, file-editing, "
        "website-building, docs-writing, scripting, refactoring, testing, and "
        "repository change requests. Use this "
        "when the user asks Hermes to hand work off to Copilot, have Copilot "
        "build or edit something in a repository, start a Copilot remote/session, "
        "or otherwise delegate coding, site, docs, build, or file-editing work "
        "as an unattended implementation job. This launches the existing detached "
        "`copilot -i --remote` job flow. Do not run terminal Copilot probes, "
        "Copilot smoke tests, or Copilot ACP for these requests; call this tool "
        "directly and let it report launch errors."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["launch", "list", "show"],
                "description": "Operation to perform. Defaults to launch.",
            },
            "prompt": {
                "type": "string",
                "description": (
                    "Full task prompt to give the Copilot remote job. Include "
                    "all relevant user requirements and constraints. Required "
                    "for action=launch."
                ),
            },
            "repo": {
                "type": "string",
                "description": (
                    "Optional repository slug, such as static-pages or "
                    "fridai-backend. If omitted, Hermes routes from the prompt."
                ),
            },
            "repo_path": {
                "type": "string",
                "description": (
                    "Optional absolute repository path visible to Hermes. If "
                    "repo is provided without repo_path, Hermes tries to find "
                    "the path from HERMES_WORKSPACE_PATH/repos."
                ),
            },
            "model": {
                "type": "string",
                "description": "Optional Copilot CLI model hint for the launched job.",
            },
            "job_id": {
                "type": "string",
                "description": "Tracked Copilot remote ID for action=show.",
            },
            "state": {
                "type": "string",
                "enum": ["running", "done", "failed"],
                "description": "Optional state filter for action=list.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum jobs to return for action=list. Defaults to 20.",
            },
            "dry_run": {
                "type": "boolean",
                "description": "Test the launch path without spawning Copilot.",
            },
            "signal_source": {
                "type": "string",
                "description": (
                    "Optional origin label for the launch (e.g. 'slack', "
                    "'cli', 'webhook'). Stored as job metadata; does not "
                    "affect Copilot reconnect."
                ),
            },
            "signal_ref": {
                "type": "string",
                "description": (
                    "Optional external reference for the launch (e.g. a "
                    "Jira ticket ID or Slack message ts). Stored as job "
                    "metadata only; the Copilot reconnect handle is tracked "
                    "separately and is not affected by this value."
                ),
            },
            "hermes_session_id": {
                "type": "string",
                "description": (
                    "Optional Hermes session ID to link this Copilot remote "
                    "job to. Stored as a foreign key into ``sessions``; does "
                    "not affect Copilot reconnect."
                ),
            },
        },
        "required": [],
    },
}


def _get_db() -> SessionDB:
    return SessionDB()


def _error(message: str) -> str:
    return json.dumps({"success": False, "error": redact_sensitive_text(message)})


def _job_handle(job: Dict[str, Any]) -> str:
    """Return the Copilot reconnect handle (never the metadata ``signal_ref``)."""
    return str(job.get("connect_handle") or job.get("id") or "")


def _serialize_job(job: Dict[str, Any]) -> Dict[str, Any]:
    handle = _job_handle(job)
    serialized = {
        "job_id": job.get("id"),
        "state": job.get("state"),
        "repo": job.get("repo_slug"),
        "repo_path": job.get("repo_path"),
        "created_at": job.get("created_at"),
        "finished_at": job.get("finished_at"),
        "exit_code": job.get("exit_code"),
        "error_text": job.get("error_text"),
        "connect_handle": handle,
        "connect_command": f"copilot --connect={handle}" if handle else None,
        "resume_command": f"copilot --resume={handle}" if handle else None,
    }
    prompt = job.get("prompt")
    if prompt:
        serialized["prompt_preview"] = prompt[:160] + ("..." if len(prompt) > 160 else "")
    return serialized


def _discover_repos() -> list[RepoEntry]:
    from copilot_remote.router import _discover_repos as discover

    return discover()


def _route_repo(prompt: str) -> Optional[RepoEntry]:
    from copilot_remote.router import route_repo

    return route_repo(prompt)


def _resolve_repo(prompt: str, repo: str = "", repo_path: str = "") -> tuple[Optional[RepoEntry], Optional[str]]:
    repo = (repo or "").strip()
    repo_path = (repo_path or "").strip()

    if repo and repo_path:
        return RepoEntry(slug=repo, path=repo_path), None

    entries: list[RepoEntry] = []
    discover_error: Optional[str] = None
    if repo:
        try:
            entries = _discover_repos()
        except Exception as exc:
            # Surface the failure reason — swallowing it here makes routing
            # bugs (permission errors, missing HERMES_WORKSPACE_PATH, etc.)
            # impossible to diagnose from the error message alone. The text
            # is sanitized via redact_sensitive_text() before going back to
            # the caller in case workspace paths embed credentials.
            from agent.redact import redact_sensitive_text

            logger.warning(
                "copilot_remote: repo discovery failed for slug=%r: %s",
                repo,
                redact_sensitive_text(repr(exc)),
            )
            discover_error = redact_sensitive_text(str(exc) or exc.__class__.__name__)
            entries = []

    if repo:
        for entry in entries:
            if entry.slug.lower() == repo.lower():
                return RepoEntry(slug=entry.slug, path=repo_path or entry.path), None
        suffix = f" (discovery error: {discover_error})" if discover_error else ""
        return None, (
            f"Could not find repo slug '{repo}'. Provide repo_path or use a "
            f"slug under HERMES_WORKSPACE_PATH/repos.{suffix}"
        )

    if repo_path:
        slug = repo_path.rstrip("/").rsplit("/", 1)[-1] or "repo"
        return RepoEntry(slug=slug, path=repo_path), None

    routed = _route_repo(prompt)
    if routed:
        return routed, None

    return None, (
        "Could not determine target repo from the prompt. Provide repo or "
        "repo_path, or make sure HERMES_WORKSPACE_PATH/repos is available."
    )


def _finish_job(job_id: str, exit_code: int) -> None:
    db = _get_db()
    try:
        db.finish_copilot_remote(
            job_id,
            state="done" if exit_code == 0 else "failed",
            exit_code=exit_code,
        )
    finally:
        db.close()


def _launch(args: Dict[str, Any]) -> str:
    prompt = str(args.get("prompt") or "").strip()
    if not prompt:
        return _error("prompt is required when launching a Copilot remote job")

    repo_entry, repo_error = _resolve_repo(
        prompt,
        repo=str(args.get("repo") or ""),
        repo_path=str(args.get("repo_path") or ""),
    )
    if repo_error or repo_entry is None:
        return _error(repo_error or "Could not resolve target repo")

    job_id = str(uuid.uuid4())
    db = _get_db()
    try:
        db.create_copilot_remote(
            job_id=job_id,
            repo_slug=repo_entry.slug,
            repo_path=repo_entry.path,
            prompt=prompt,
            signal_source=str(args.get("signal_source") or "tool"),
            signal_ref=str(args.get("signal_ref") or "") or None,
            hermes_session_id=str(args.get("hermes_session_id") or "") or None,
        )

        from copilot_remote.launcher import launch_copilot

        result = launch_copilot(
            repo_entry,
            prompt,
            session_id=job_id,
            model=str(args.get("model") or "") or None,
            dry_run=bool(args.get("dry_run", False)),
            on_complete=_finish_job,
        )

        connect_handle = result.get("connect_id") or job_id
        if connect_handle != job_id:
            db.update_copilot_remote_connect_handle(job_id, str(connect_handle))

        job = db.get_copilot_remote(job_id) or {
            "id": job_id,
            "state": "done" if args.get("dry_run") else "running",
            "repo_slug": repo_entry.slug,
            "repo_path": repo_entry.path,
            "prompt": prompt,
            "connect_handle": connect_handle if connect_handle != job_id else None,
        }
        payload = {
            "success": True,
            "action": "launch",
            "job": _serialize_job(job),
            "prompt_delivery_status": result.get("prompt_delivery_status"),
            "prompt_delivery_warning": result.get("prompt_delivery_warning"),
        }
        return json.dumps(payload, ensure_ascii=False)
    except Exception as exc:
        try:
            db.finish_copilot_remote(job_id, state="failed", error_text=redact_sensitive_text(str(exc)))
        except Exception:
            pass
        return _error(f"Failed to launch Copilot remote job: {exc}")
    finally:
        db.close()


def _list(args: Dict[str, Any]) -> str:
    state = str(args.get("state") or "").strip() or None
    try:
        limit = int(args.get("limit") or 20)
    except (TypeError, ValueError):
        limit = 20
    limit = max(1, min(limit, 100))

    db = _get_db()
    try:
        jobs = [_serialize_job(job) for job in db.list_copilot_remote(state=state, limit=limit)]
        return json.dumps({"success": True, "action": "list", "jobs": jobs}, ensure_ascii=False)
    finally:
        db.close()


def _show(args: Dict[str, Any]) -> str:
    job_id = str(args.get("job_id") or "").strip()
    if not job_id:
        return _error("job_id is required for action=show")

    db = _get_db()
    try:
        job = db.get_copilot_remote(job_id)
        if not job:
            return _error(f"Copilot remote not found: {job_id}")
        return json.dumps({"success": True, "action": "show", "job": _serialize_job(job)}, ensure_ascii=False)
    finally:
        db.close()


def copilot_remote(args: Dict[str, Any], **kwargs) -> str:
    """Launch, list, or inspect tracked Copilot remote jobs."""
    args = args or {}
    action = str(args.get("action") or "launch").strip().lower()
    if action == "launch":
        return _launch(args)
    if action == "list":
        return _list(args)
    if action == "show":
        return _show(args)
    return _error(f"Unknown copilot_remote action: {action}")


registry.register(
    name="copilot_remote",
    toolset="copilot",
    schema=COPILOT_REMOTE_SCHEMA,
    handler=copilot_remote,
    emoji="C",
    max_result_size_chars=20_000,
)
