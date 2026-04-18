"""
kart_task_tool.py — Hermes tool adapter for the Willow Kart task queue.

Exposes Willow's Kart task queue as a Hermes-compatible tool.
Agents can submit shell commands, Python scripts, and bash blocks
to Kart for sandboxed execution, then poll for results.

Registration (in your Hermes config or tools/__init__.py):
    from tools.kart_task_tool import register_kart_tool
    register_kart_tool(registry)

Requirements:
    - Willow running locally with Postgres on Unix socket
    - WILLOW_PG_DB env var (default: willow)
    - WILLOW_PG_USER env var (default: $USER)

b17: HKT1W
ΔΣ=42
"""

import json
import os
import uuid
from typing import Optional

try:
    import psycopg2
    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False

_pg_conn = None


def _get_pg():
    global _pg_conn
    try:
        if _pg_conn is None or _pg_conn.closed:
            _pg_conn = psycopg2.connect(
                dbname=os.environ.get("WILLOW_PG_DB", "willow"),
                user=os.environ.get("WILLOW_PG_USER", os.environ.get("USER", "")),
            )
            _pg_conn.autocommit = True
        _pg_conn.cursor().execute("SELECT 1")
        return _pg_conn
    except Exception:
        _pg_conn = None
        return None


def check_kart_requirements() -> bool:
    if not _PG_AVAILABLE:
        return False
    return _get_pg() is not None


def kart_task_tool(
    action: str,
    task: Optional[str] = None,
    task_id: Optional[str] = None,
    agent: str = "kart",
    submitted_by: str = "hermes",
    limit: int = 10,
    **kwargs,
) -> str:
    """
    Interact with the Willow Kart task queue.

    Actions:
      submit  — queue a task for sandboxed execution. Returns task_id.
      status  — check status of a task by task_id.
      list    — list pending tasks.

    Task format (for submit):
      Plain shell: "cp /src /dst"
      Python block: ```python\\ncode here\\n```
      Bash block:   ```bash\\nscript here\\n```
    """
    pg = _get_pg()
    if not pg:
        return json.dumps({"error": "Willow Postgres not available"})

    cur = pg.cursor()

    if action == "submit":
        if not task:
            return json.dumps({"error": "task is required for submit"})
        task_id = "".join(__import__("random").choices(
            "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789", k=8
        ))
        cur.execute(
            "INSERT INTO kart_task_queue (task_id, submitted_by, agent, task) "
            "VALUES (%s, %s, %s, %s)",
            (task_id, submitted_by, agent, task),
        )
        cur.close()
        return json.dumps({"task_id": task_id, "status": "pending",
                           "message": f"Task queued. Poll with action=status task_id={task_id}"})

    if action == "status":
        if not task_id:
            return json.dumps({"error": "task_id is required for status"})
        cur.execute(
            "SELECT task_id, status, result, steps, created_at, completed_at "
            "FROM kart_task_queue WHERE task_id = %s",
            (task_id,),
        )
        row = cur.fetchone()
        cur.close()
        if not row:
            return json.dumps({"error": "task not found", "task_id": task_id})
        return json.dumps({
            "task_id": row[0], "status": row[1], "result": row[2],
            "steps": row[3], "created_at": str(row[4]), "completed_at": str(row[5]),
        })

    if action == "list":
        cur.execute(
            "SELECT task_id, task, submitted_by, created_at, status "
            "FROM kart_task_queue WHERE agent = %s "
            "ORDER BY created_at DESC LIMIT %s",
            (agent, limit),
        )
        rows = cur.fetchall()
        cur.close()
        return json.dumps({"tasks": [
            {"task_id": r[0], "task": str(r[1])[:80],
             "submitted_by": r[2], "created_at": str(r[3]), "status": r[4]}
            for r in rows
        ]})

    cur.close()
    return json.dumps({"error": f"unknown action: {action}. Use submit|status|list"})


_SCHEMA = {
    "name": "kart_task",
    "description": (
        "Submit tasks to the Willow Kart execution queue for sandboxed shell/Python execution. "
        "Use action=submit to queue work, action=status to poll results, action=list to see queue."
    ),
    "parameters": {
        "type": "object",
        "required": ["action"],
        "properties": {
            "action": {
                "type": "string",
                "enum": ["submit", "status", "list"],
                "description": "submit a task, check status, or list tasks",
            },
            "task": {
                "type": "string",
                "description": "Task to execute (required for submit). Supports shell commands, ```python blocks, ```bash blocks.",
            },
            "task_id": {
                "type": "string",
                "description": "Task ID to check (required for status).",
            },
            "agent": {
                "type": "string",
                "default": "kart",
                "description": "Target worker agent.",
            },
            "limit": {
                "type": "integer",
                "default": 10,
                "description": "Max tasks to return for list.",
            },
        },
    },
}


def register_kart_tool(registry) -> None:
    """Register the Kart tool with a Hermes tool registry."""
    registry.register(
        name="kart_task",
        toolset="willow",
        emoji="⚙️",
        handler=lambda **kwargs: kart_task_tool(**kwargs),
        schema=_SCHEMA,
        check_requirements=check_kart_requirements,
    )
