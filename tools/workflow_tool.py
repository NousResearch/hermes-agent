"""Declarative workflow orchestration tool for Hermes.

This is intentionally smaller and safer than Claude Code's JavaScript workflow
runtime: the model supplies a JSON-compatible phase definition, while Hermes
owns persistence, phase execution, and delegation boundaries.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error


_MAX_PHASES = 20
_MAX_TASKS_PER_PHASE = 25
_DEFAULT_MAX_TOTAL_TASKS = 100
_ALLOWED_PHASE_TYPES = {"fanout", "pipeline", "review", "gate", "synthesize"}
_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


def _workflow_home() -> Path:
    return Path(get_hermes_home()) / "workflows"


def _runs_db_path() -> Path:
    return _workflow_home() / "runs.db"


def _connect() -> sqlite3.Connection:
    path = _runs_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS workflow_runs (
            run_id TEXT PRIMARY KEY,
            workflow_name TEXT NOT NULL,
            objective TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            workflow_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS workflow_phase_results (
            run_id TEXT NOT NULL,
            phase_index INTEGER NOT NULL,
            phase_title TEXT NOT NULL,
            phase_type TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at REAL NOT NULL,
            finished_at REAL,
            result_json TEXT,
            PRIMARY KEY (run_id, phase_index)
        )
        """
    )
    conn.commit()
    return conn


def _safe_workflow_name(name: str) -> str:
    name = _SAFE_NAME_RE.sub("-", (name or "workflow").strip()).strip(".-_").lower()
    return name[:80] or "workflow"


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return str(value)


def _parse_workflow_arg(raw: Any, *, name: str | None = None, objective: str | None = None, phases: list | None = None) -> dict[str, Any]:
    if isinstance(raw, str) and raw.strip():
        try:
            workflow = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"workflow must be a JSON object or object-like value: {exc}") from exc
    elif isinstance(raw, dict):
        workflow = dict(raw)
    elif raw in (None, ""):
        workflow = {}
    else:
        raise ValueError("workflow must be an object or JSON string")

    if name and not workflow.get("name"):
        workflow["name"] = name
    if objective and not workflow.get("objective"):
        workflow["objective"] = objective
    if phases is not None and not workflow.get("phases"):
        workflow["phases"] = phases
    return workflow


def _normalise_task(task: Any, phase: dict[str, Any], inherited_context: str) -> dict[str, Any]:
    if isinstance(task, str):
        goal = task
        context = inherited_context
        toolsets = phase.get("toolsets")
    elif isinstance(task, dict):
        goal = task.get("goal") or task.get("task") or task.get("prompt")
        context_parts = [inherited_context]
        if task.get("context"):
            context_parts.append(str(task.get("context")))
        context = "\n\n".join(part for part in context_parts if part)
        toolsets = task.get("toolsets", phase.get("toolsets"))
    else:
        raise ValueError("phase tasks must be strings or objects")
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("each task needs a non-empty goal")
    out: dict[str, Any] = {"goal": goal.strip()}
    if context:
        out["context"] = context
    if toolsets:
        if not isinstance(toolsets, list) or not all(isinstance(item, str) for item in toolsets):
            raise ValueError("task toolsets must be a list of strings")
        out["toolsets"] = toolsets
    return out


def _validate_workflow(workflow: dict[str, Any], *, max_total_tasks: int = _DEFAULT_MAX_TOTAL_TASKS) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(workflow, dict):
        raise ValueError("workflow must be an object")
    name = _safe_workflow_name(str(workflow.get("name") or "workflow"))
    objective = str(workflow.get("objective") or workflow.get("description") or "").strip()
    if not objective:
        raise ValueError("workflow.objective is required")
    phases = workflow.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("workflow.phases must be a non-empty array")
    if len(phases) > _MAX_PHASES:
        raise ValueError(f"workflow has {len(phases)} phases; max is {_MAX_PHASES}")

    normalised_phases: list[dict[str, Any]] = []
    warnings: list[str] = []
    total_tasks = 0
    inherited_context = str(workflow.get("context") or "")

    for index, phase_raw in enumerate(phases):
        if not isinstance(phase_raw, dict):
            raise ValueError(f"phase {index + 1} must be an object")
        phase_type = str(phase_raw.get("type") or phase_raw.get("pattern") or "fanout").strip().lower()
        if phase_type not in _ALLOWED_PHASE_TYPES:
            raise ValueError(
                f"phase {index + 1} type '{phase_type}' is invalid; use one of {sorted(_ALLOWED_PHASE_TYPES)}"
            )
        title = str(phase_raw.get("title") or phase_raw.get("name") or f"Phase {index + 1}").strip()
        tasks_raw = phase_raw.get("tasks")
        prompt = phase_raw.get("prompt")
        if tasks_raw is None and prompt:
            tasks_raw = [str(prompt)]
        if phase_type == "synthesize" and tasks_raw is None:
            tasks_raw = [
                "Synthesize the prior workflow phase results into a concise final answer. "
                "Separate facts, assumptions, risks, and recommended next action."
            ]
        if not isinstance(tasks_raw, list) or not tasks_raw:
            raise ValueError(f"phase {index + 1} needs a non-empty tasks array or prompt")
        if len(tasks_raw) > _MAX_TASKS_PER_PHASE:
            raise ValueError(f"phase {index + 1} has {len(tasks_raw)} tasks; max is {_MAX_TASKS_PER_PHASE}")
        total_tasks += len(tasks_raw)
        if total_tasks > max_total_tasks:
            raise ValueError(f"workflow has {total_tasks} tasks; max_total_tasks is {max_total_tasks}")

        phase_context = "\n\n".join(
            part
            for part in [inherited_context, str(phase_raw.get("context") or "")]
            if part
        )
        tasks = [_normalise_task(task, phase_raw, phase_context) for task in tasks_raw]
        if phase_type in {"gate", "review"} and len(tasks) > 10:
            warnings.append(f"Phase {index + 1} is {phase_type} with {len(tasks)} tasks; consider a smaller gate.")
        normalised_phases.append(
            {
                "title": title,
                "type": phase_type,
                "tasks": tasks,
                "description": str(phase_raw.get("description") or ""),
            }
        )

    return {
        "name": name,
        "objective": objective,
        "context": inherited_context,
        "phases": normalised_phases,
    }, warnings


def _saved_workflow_path(name: str) -> Path:
    return _workflow_home() / f"{_safe_workflow_name(name)}.json"


def _save_workflow(workflow: dict[str, Any]) -> str:
    path = _saved_workflow_path(workflow["name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def _load_saved_workflow(name: str) -> dict[str, Any]:
    path = _saved_workflow_path(name)
    if not path.exists():
        raise ValueError(f"saved workflow not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _completed_phase_result(conn: sqlite3.Connection, run_id: str, phase_index: int) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT result_json FROM workflow_phase_results WHERE run_id=? AND phase_index=? AND status='completed'",
        (run_id, phase_index),
    ).fetchone()
    if not row or not row[0]:
        return None
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return {"raw": row[0]}


def _phase_context(objective: str, prior_results: list[dict[str, Any]]) -> str:
    if not prior_results:
        return f"Workflow objective:\n{objective}"
    compact = json.dumps(prior_results, ensure_ascii=False)
    if len(compact) > 24000:
        compact = compact[:24000] + "\n...[truncated prior phase results]"
    return f"Workflow objective:\n{objective}\n\nPrior phase results JSON:\n{compact}"


def _run_phase(phase: dict[str, Any], *, objective: str, prior_results: list[dict[str, Any]], parent_agent: Any) -> dict[str, Any]:
    from tools.delegate_tool import delegate_task

    tasks = []
    prior_context = _phase_context(objective, prior_results)
    for task in phase["tasks"]:
        item = dict(task)
        item["context"] = "\n\n".join(part for part in [prior_context, item.get("context", "")] if part)
        tasks.append(item)

    if len(tasks) == 1:
        raw = delegate_task(
            goal=tasks[0]["goal"],
            context=tasks[0].get("context"),
            toolsets=tasks[0].get("toolsets"),
            parent_agent=parent_agent,
        )
    else:
        raw = delegate_task(tasks=tasks, parent_agent=parent_agent)
    return {
        "title": phase["title"],
        "type": phase["type"],
        "task_count": len(tasks),
        "result": _jsonable(raw),
    }


def workflow_tool(
    *,
    action: str = "run",
    workflow: Any = None,
    name: str | None = None,
    objective: str | None = None,
    phases: list | None = None,
    save: bool = False,
    run_id: str | None = None,
    resume: bool = False,
    max_total_tasks: int = _DEFAULT_MAX_TOTAL_TASKS,
    parent_agent: Any = None,
) -> str:
    """Validate, save, or execute a declarative Hermes workflow."""
    try:
        action = (action or "run").strip().lower()
        if action not in {"validate", "save", "run", "resume", "list"}:
            return tool_error(f"Unsupported workflow action: {action}")

        conn = _connect()
        if action == "list":
            rows = conn.execute(
                "SELECT run_id, workflow_name, objective, status, created_at, updated_at "
                "FROM workflow_runs ORDER BY updated_at DESC LIMIT 20"
            ).fetchall()
            saved = sorted(path.name for path in _workflow_home().glob("*.json")) if _workflow_home().exists() else []
            return json.dumps(
                {
                    "saved_workflows": saved,
                    "recent_runs": [
                        {
                            "run_id": row[0],
                            "name": row[1],
                            "objective": row[2],
                            "status": row[3],
                            "created_at": row[4],
                            "updated_at": row[5],
                        }
                        for row in rows
                    ],
                },
                ensure_ascii=False,
            )

        if action == "resume":
            resume = True
            if not run_id:
                return tool_error("resume requires run_id")
            row = conn.execute("SELECT workflow_json FROM workflow_runs WHERE run_id=?", (run_id,)).fetchone()
            if not row:
                return tool_error(f"workflow run not found: {run_id}")
            workflow_obj = json.loads(row[0])
        elif workflow is None and name and not phases and not objective and action in {"run", "validate", "save"}:
            workflow_obj = _load_saved_workflow(name)
        else:
            workflow_obj = _parse_workflow_arg(workflow, name=name, objective=objective, phases=phases)

        normalised, warnings = _validate_workflow(workflow_obj, max_total_tasks=max_total_tasks)

        if action == "validate":
            return json.dumps({"ok": True, "workflow": normalised, "warnings": warnings}, ensure_ascii=False)

        saved_path = None
        if save or action == "save":
            saved_path = _save_workflow(normalised)
        if action == "save":
            return json.dumps({"ok": True, "saved_path": saved_path, "workflow": normalised, "warnings": warnings}, ensure_ascii=False)

        if parent_agent is None:
            return tool_error("workflow run requires an active Hermes agent parent context")

        now = time.time()
        run_id = run_id or f"wf_{uuid.uuid4().hex[:12]}"
        conn.execute(
            "INSERT OR REPLACE INTO workflow_runs(run_id, workflow_name, objective, status, created_at, updated_at, workflow_json) "
            "VALUES (?, ?, ?, ?, COALESCE((SELECT created_at FROM workflow_runs WHERE run_id=?), ?), ?, ?)",
            (
                run_id,
                normalised["name"],
                normalised["objective"],
                "running",
                run_id,
                now,
                now,
                json.dumps(normalised, ensure_ascii=False),
            ),
        )
        conn.commit()

        phase_results: list[dict[str, Any]] = []
        for index, phase in enumerate(normalised["phases"]):
            if resume:
                completed = _completed_phase_result(conn, run_id, index)
                if completed is not None:
                    phase_results.append(completed)
                    continue
            started = time.time()
            conn.execute(
                "INSERT OR REPLACE INTO workflow_phase_results(run_id, phase_index, phase_title, phase_type, status, started_at, finished_at, result_json) "
                "VALUES (?, ?, ?, ?, 'running', ?, NULL, NULL)",
                (run_id, index, phase["title"], phase["type"], started),
            )
            conn.commit()
            try:
                result = _run_phase(
                    phase,
                    objective=normalised["objective"],
                    prior_results=phase_results,
                    parent_agent=parent_agent,
                )
            except Exception as exc:
                finished = time.time()
                error_payload = {"error": str(exc), "phase": phase["title"], "type": phase["type"]}
                conn.execute(
                    "UPDATE workflow_phase_results SET status='failed', finished_at=?, result_json=? "
                    "WHERE run_id=? AND phase_index=?",
                    (finished, json.dumps(error_payload, ensure_ascii=False), run_id, index),
                )
                conn.execute("UPDATE workflow_runs SET status='failed', updated_at=? WHERE run_id=?", (finished, run_id))
                conn.commit()
                return json.dumps(
                    {
                        "ok": False,
                        "run_id": run_id,
                        "failed_phase": index + 1,
                        "error": str(exc),
                        "completed_phases": phase_results,
                        "warnings": warnings,
                    },
                    ensure_ascii=False,
                )
            finished = time.time()
            phase_results.append(result)
            conn.execute(
                "UPDATE workflow_phase_results SET status='completed', finished_at=?, result_json=? "
                "WHERE run_id=? AND phase_index=?",
                (finished, json.dumps(result, ensure_ascii=False), run_id, index),
            )
            conn.execute("UPDATE workflow_runs SET updated_at=? WHERE run_id=?", (finished, run_id))
            conn.commit()

        finished = time.time()
        conn.execute("UPDATE workflow_runs SET status='completed', updated_at=? WHERE run_id=?", (finished, run_id))
        conn.commit()
        return json.dumps(
            {
                "ok": True,
                "run_id": run_id,
                "workflow_name": normalised["name"],
                "objective": normalised["objective"],
                "saved_path": saved_path,
                "phase_results": phase_results,
                "warnings": warnings,
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return tool_error(str(exc))


WORKFLOW_SCHEMA = {
    "type": "function",
    "function": {
        "name": "workflow",
        "description": (
            "Run a declarative, Claude-Code-style dynamic workflow in Hermes. "
            "The model supplies JSON-compatible phases; Hermes persists run state "
            "and executes each phase via isolated delegate_task subagents. Use for "
            "fan-out research, audits, adversarial verification, tournaments, and "
            "phase-gated multi-agent work."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["validate", "save", "run", "resume", "list"],
                    "description": "validate only, save to ~/.hermes/workflows, run now, resume a run_id, or list saved/recent workflows.",
                },
                "workflow": {
                    "type": "object",
                    "description": (
                        "Workflow object: {name, objective, context?, phases:[{title,type,tasks,toolsets?,context?}]}. "
                        "Phase type is fanout, pipeline, review, gate, or synthesize. Tasks are strings or objects with goal/context/toolsets."
                    ),
                },
                "name": {"type": "string", "description": "Workflow name, or saved workflow name for run/validate/save."},
                "objective": {"type": "string", "description": "Workflow objective when not embedded in workflow."},
                "phases": {"type": "array", "description": "Workflow phases when not embedded in workflow."},
                "save": {"type": "boolean", "description": "Save the normalized workflow before running."},
                "run_id": {"type": "string", "description": "Run id for resume or explicit continuation."},
                "resume": {"type": "boolean", "description": "Skip completed phases for this run_id and continue remaining phases."},
                "max_total_tasks": {
                    "type": "integer",
                    "description": "Safety cap on total delegated tasks in this run. Default 100.",
                },
            },
            "required": [],
        },
    },
}


registry.register(
    name="workflow",
    toolset="workflow",
    schema=WORKFLOW_SCHEMA,
    handler=lambda args, **kw: workflow_tool(
        action=args.get("action", "run"),
        workflow=args.get("workflow"),
        name=args.get("name"),
        objective=args.get("objective"),
        phases=args.get("phases"),
        save=bool(args.get("save", False)),
        run_id=args.get("run_id"),
        resume=bool(args.get("resume", False)),
        max_total_tasks=int(args.get("max_total_tasks") or _DEFAULT_MAX_TOTAL_TASKS),
        parent_agent=kw.get("parent_agent"),
    ),
    emoji="🧭",
)
