#!/usr/bin/env python3
"""Delegate coding tasks to an opencode ``serve`` backend.

Integrates Hermes with a headless opencode server (``opencode serve`` /
``opencode web``) running on an always-on host. The profile sends a coding
task; opencode's full agent loop (read, edit, bash, git, LSP) executes on the
remote host; Hermes receives a text summary plus the per-file diff stats.

HTTP surface used (all documented at ``<server>/doc``):

- ``GET  /global/health``                     — reachability + auth check
- ``POST /session``                            — create a session
- ``POST /session/:id/message``                — blocking run (waits for reply)
- ``POST /session/:id/prompt_async``           — fire and forget (204)
- ``GET  /session/status``                     — per-session status map
- ``GET  /session/:id/message?limit=1``        — last message fallback
- ``GET  /session/:id/diff?messageID=<id>``    — per-file diff stats

Sessions are reused per project (persisted under ``HERMES_HOME/data/``) so
context accumulates across turns like a real pairing session. ``new_session``
forces a reset. Note: naive JSON store — concurrent tool calls racing on the
same project are unsupported; the profile should serialize runs per project.

Configuration (profile/gateway env, NOT child envs — ``PASSWORD`` is scrubbed
from delegated child environments on purpose):

- ``OPENCODE_SERVER_URL``      e.g. ``http://<vm-tailscale-ip>:4096`` (required)
- ``OPENCODE_SERVER_USERNAME`` basic-auth user (default ``opencode``)
- ``OPENCODE_SERVER_PASSWORD`` basic-auth password (unset = no auth)

The tool is inactive unless ``OPENCODE_SERVER_URL`` is set (``requires_env``).
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error
from utils import atomic_write_text

MAX_SUMMARY_CHARS = 8000


def _session_store() -> Path:
    return get_hermes_home() / "data" / "opencode_sessions.json"


def _config() -> tuple[str, str, str] | None:
    url = (os.environ.get("OPENCODE_SERVER_URL") or "").strip().rstrip("/")
    if not url:
        return None
    username = (os.environ.get("OPENCODE_SERVER_USERNAME") or "opencode").strip()
    password = os.environ.get("OPENCODE_SERVER_PASSWORD") or ""
    return url, username, password


def _load_sessions() -> dict[str, str]:
    try:
        data = json.loads(_session_store().read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_sessions(sessions: dict[str, str]) -> None:
    store = _session_store()
    store.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(store, json.dumps(sessions, indent=2))


def _ensure_session(client: httpx.Client, project: str, title: str) -> str | None:
    sessions = _load_sessions()
    existing = sessions.get(project)
    if existing:
        try:
            client.get(f"/session/{existing}")
            return existing
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
    resp = client.post("/session", json={"title": title})
    resp.raise_for_status()
    session_id = resp.json()["id"]
    sessions[project] = session_id
    _save_sessions(sessions)
    return session_id


def _text_from_parts(parts: list[dict]) -> str:
    chunks = [p.get("text", "") for p in parts if p.get("type") == "text" and p.get("text")]
    text = "\n".join(chunks).strip()
    if len(text) > MAX_SUMMARY_CHARS:
        text = text[:MAX_SUMMARY_CHARS] + "\n… [truncated; use opencode_status for the rest]"
    return text


def _diff_summary(client: httpx.Client, session_id: str, message_id: str | None) -> str:
    if not message_id:
        return ""
    try:
        resp = client.get(f"/session/{session_id}/diff", params={"messageID": message_id})
        resp.raise_for_status()
    except httpx.HTTPError:
        return ""
    diffs = resp.json() or []
    if not diffs:
        return ""
    lines = ["", "Files changed:"]
    for d in diffs:
        path = d.get("path") or "?"
        additions = int(d.get("additions") or 0)
        deletions = int(d.get("deletions") or 0)
        lines.append(f"  - {path} (+{additions} -{deletions})")
    return "\n".join(lines)


def opencode_run(
    task: str,
    project: str = "",
    new_session: bool = False,
    background: bool = False,
    timeout_minutes: int = 30,
) -> str:
    cfg = _config()
    if cfg is None:
        return tool_error(
            "OPENCODE_SERVER_URL is not set — the opencode toolset is inactive. "
            "Set it to the opencode serve URL (e.g. http://<host>:4096) and set "
            "OPENCODE_SERVER_PASSWORD for basic auth."
        )
    url, username, password = cfg
    task = (task or "").strip()
    if not task:
        return tool_error("task is required.")
    project = (project or "").strip().rstrip("/") or os.environ.get("OPENCODE_DEFAULT_PROJECT", "")
    if not project:
        return tool_error("project is required: the project directory path ON THE opencode SERVER host.")

    timeout = httpx.Timeout(
        connect=10.0,
        read=float(max(60, timeout_minutes * 60)),
        write=30.0,
        pool=10.0,
    )
    try:
        with httpx.Client(base_url=url, auth=(username, password) if password else None, timeout=timeout) as client:
            health = client.get("/global/health")
            health.raise_for_status()

            if new_session:
                sessions = _load_sessions()
                sessions.pop(project, None)
                _save_sessions(sessions)

            session_id = _ensure_session(client, project, task[:80])

            parts = [{"type": "text", "text": task, "synthetic": False}]
            if background:
                client.post(f"/session/{session_id}/prompt_async", json={"parts": parts})
                return (
                    f"Dispatched to opencode session {session_id} in background.\n"
                    "Poll with opencode_status(project=…) until the session is idle."
                )

            started = time.monotonic()
            resp = client.post(f"/session/{session_id}/message", json={"parts": parts})
            resp.raise_for_status()
            elapsed = int(time.monotonic() - started)
            body = resp.json()
            message_id = (body.get("info") or {}).get("id")
            summary = _text_from_parts(body.get("parts") or [])
            if not summary:
                summary = "(no text in response parts — check opencode_status)"
            summary += _diff_summary(client, session_id, message_id)
            return (
                f"[session {session_id}, {elapsed}s]\n{summary}\n\n"
                "Note: all edits happened on the opencode server host; "
                "the working tree lives under the project path given above."
            )
    except httpx.ReadTimeout:
        return tool_error(
            "opencode run exceeded the wait timeout. The task is still executing "
            "server-side — poll with opencode_status(project=…) or re-run with "
            "background=true."
        )
    except httpx.ConnectError as exc:
        return tool_error(f"cannot reach opencode server at {url}: {exc}")
    except httpx.HTTPStatusError as exc:
        return tool_error(
            f"opencode server error {exc.response.status_code}: "
            f"{exc.response.text[:400] or exc}"
        )


def opencode_status(project: str = "") -> str:
    cfg = _config()
    if cfg is None:
        return tool_error("OPENCODE_SERVER_URL is not set — opencode toolset inactive.")
    url, username, password = cfg
    project = (project or "").strip().rstrip("/") or os.environ.get("OPENCODE_DEFAULT_PROJECT", "")
    if not project:
        return tool_error("project is required: the project directory path ON THE opencode SERVER host.")

    sessions = _load_sessions()
    session_id = sessions.get(project)
    if not session_id:
        return f"No opencode session for project {project} yet — run opencode_run first."

    try:
        with httpx.Client(base_url=url, auth=(username, password) if password else None, timeout=15.0) as client:
            try:
                status_resp = client.get("/session/status")
                status_resp.raise_for_status()
                status_map = status_resp.json() or {}
                entry = status_map.get(session_id)
                if isinstance(entry, dict):
                    return (
                        f"Session {session_id} status: {json.dumps(entry, default=str)[:2000]}"
                    )
            except httpx.HTTPError:
                pass

            resp = client.get(f"/session/{session_id}/message", params={"limit": 1})
            resp.raise_for_status()
            messages = resp.json() or []
            if not messages:
                return f"Session {session_id} exists but has no messages yet."
            last = messages[-1]
            role = (last.get("info") or {}).get("role", "unknown")
            text = _text_from_parts(last.get("parts") or [])
            return f"Session {session_id} — last message role={role}:\n{text[:3000]}"
    except httpx.ConnectError as exc:
        return tool_error(f"cannot reach opencode server at {url}: {exc}")
    except httpx.HTTPStatusError as exc:
        return tool_error(
            f"opencode server error {exc.response.status_code}: "
            f"{exc.response.text[:400] or exc}"
        )


OPENCODE_RUN_SCHEMA = {
    "name": "opencode_run",
    "description": (
        "Delegate a coding task to the opencode serve backend — a full coding "
        "agent loop (read, edit, bash, git, LSP) running on an always-on remote "
        "host. Pass a detailed task; optionally pass project (the directory ON "
        "THE SERVER HOST to work in — defaults to the last project for that "
        "session store). Sessions persist per project: subsequent calls continue "
        "the same session context unless new_session=true. Set background=true "
        "for long tasks and poll with opencode_status. Blocking calls return a "
        "summary plus per-file diff stats; edits happen server-side, never on "
        "the machine Hermes runs on."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The coding task for the opencode agent (be specific: what to change, where, acceptance criteria).",
            },
            "project": {
                "type": "string",
                "description": "Absolute project path on the opencode server host (e.g. /home/steve/repos/myapp). Defaults to the session's last project.",
            },
            "new_session": {
                "type": "boolean",
                "description": "Start a fresh session instead of continuing the project's existing one (default false).",
            },
            "background": {
                "type": "boolean",
                "description": "Dispatch asynchronously (prompt_async) and return immediately; poll with opencode_status (default false).",
            },
            "timeout_minutes": {
                "type": "integer",
                "description": "Maximum minutes to wait in blocking mode before returning a timeout notice (the task keeps running server-side). Default 30 — note the blocking wait counts against the agent's own tool-call deadline, so prefer background=true for long tasks.",
            },
        },
        "required": ["task"],
    },
}

OPENCODE_STATUS_SCHEMA = {
    "name": "opencode_status",
    "description": (
        "Check the opencode session for a project on the serve backend: server "
        "reported status when available, otherwise the last message and a text "
        "tail. Use after opencode_run(background=true) or a timed-out blocking "
        "run to see whether the agent finished and what it produced."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "project": {
                "type": "string",
                "description": "Absolute project path on the opencode server host. Defaults to the session's last project.",
            },
        },
        "required": [],
    },
}


registry.register(
    name="opencode_run",
    toolset="opencode",
    schema=OPENCODE_RUN_SCHEMA,
    handler=lambda args, **kw: opencode_run(
        task=args.get("task", ""),
        project=args.get("project", ""),
        new_session=bool(args.get("new_session", False)),
        background=bool(args.get("background", False)),
        timeout_minutes=int(args.get("timeout_minutes", 30) or 30),
    ),
    requires_env=["OPENCODE_SERVER_URL"],
)

registry.register(
    name="opencode_status",
    toolset="opencode",
    schema=OPENCODE_STATUS_SCHEMA,
    handler=lambda args, **kw: opencode_status(project=args.get("project", "")),
    requires_env=["OPENCODE_SERVER_URL"],
)
