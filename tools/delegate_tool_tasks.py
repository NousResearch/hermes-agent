"""delegate_task input validation: tasks=[...] / legacy goal normalisation and per-task output schemas."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

# Placeholder shapes for batch goal validation: bare 'TODO' / 'task N' labels, or unexpanded template markers. The
# marker regex is deliberately NARROW — only snake_case / space-separated placeholder identifiers (`<feature_name>`,
# `{file path}`, `<FEATURE-NAME>`), the shape LLM templates leave behind. Bare single-word brackets must never be
# rejected: legitimate goals are full of generics (`Vec<T>`), HTML tags (`<div>`), dict snippets (`{"key": 1}`), glob
# braces (`{a,b}`) and f-string style (`{i}`).
# See #81141.
_PLACEHOLDER_GOAL_RE = re.compile(r"^(todo|task\s*\d+)$", re.IGNORECASE)
_TEMPLATE_MARKER_RE = re.compile(
    r"<[A-Za-z][A-Za-z0-9]*(?:[ _-][A-Za-z0-9]+)+>|\{[A-Za-z][A-Za-z0-9]*(?:[ _-][A-Za-z0-9]+)+\}"
)
_MIN_BATCH_GOAL_LEN = 10

def _recover_tasks_from_json_string(tasks: Any) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """``(parsed_list, None)`` for a JSON-array string, ``(None, error)`` for a bad string, ``(None, None)`` otherwise."""
    if not isinstance(tasks, str):
        return None, None
    raw = tasks.strip()
    if not raw:
        return None, "Provide either 'goal' (single task) or 'tasks' (batch)."
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"tasks must be a JSON array of task objects; received a string that could not be parsed as JSON ({exc.msg})."
    if not isinstance(parsed, list):
        return None, f"tasks must be a JSON array of task objects; parsed {type(parsed).__name__} instead."
    return parsed, None

def _validate_batch_tasks(task_list: List[Dict[str, Any]]) -> Optional[str]:
    """Batch-only quality gate beyond per-task goal presence; actionable error or None. No minimum count: a one-entry
    array is the canonical single-task shape (legacy top-level `goal` is wrapped into one). Duplicate goals are
    deliberately NOT rejected — identical-goal fan-outs (best-of-N / ensemble sampling) are legitimate and blocking
    them broke real workflows. The too-short check applies only to multi-task fan-outs (terse goals there are
    usually unexpanded templates); a SINGLE task legitimately uses short goals ("Fix the tests").

    See #81141.
    """
    for i, task in enumerate(task_list):
        goal = str(task.get("goal", "")).strip()
        if _PLACEHOLDER_GOAL_RE.match(" ".join(goal.lower().split())):
            return (
                f"Task {i} has a placeholder goal ({goal!r}). Replace it "
                "with a specific, self-contained description of what the subagent should accomplish."
            )
        marker = _TEMPLATE_MARKER_RE.search(goal)
        if marker:
            return (
                f"Task {i} goal contains an unexpanded template marker "
                f"({marker.group(0)!r}). Substitute the real value before "
                "calling delegate_task — subagents cannot resolve placeholders."
            )
        if len(goal) < _MIN_BATCH_GOAL_LEN and len(task_list) >= 2:
            return (
                f"Task {i} goal is too short ({goal!r}). Write a specific, "
                f"self-contained goal of at least {_MIN_BATCH_GOAL_LEN} characters so the subagent knows "
                "exactly what to do."
            )
    return None

def _normalize_task_list(
    goal, context, tasks, output_schema, top_role: str, max_children: int
) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """``(task_list, None)`` from ``tasks=[...]`` or the legacy single ``goal``, else ``(None, error)``."""
    recovered_tasks, tasks_error = _recover_tasks_from_json_string(tasks)
    if tasks_error:
        return None, tasks_error
    if recovered_tasks is not None:
        tasks = recovered_tasks
    # Small models emit tasks=[] alongside a single goal: treat as "no batch".
    if isinstance(tasks, list) and not tasks:
        tasks = None

    if tasks and isinstance(tasks, list):
        if len(tasks) > max_children:
            return None, (
                f"Too many tasks: {len(tasks)} provided, but max_concurrent_children is {max_children}. "
                f"Either reduce the task count, split into multiple delegate_task calls, or increase "
                f"delegation.max_concurrent_children in config.yaml."
            )
        task_list = tasks
    elif goal and isinstance(goal, str) and goal.strip():
        task_list = [{"goal": goal, "context": context, "role": top_role}]
        if output_schema is not None:
            task_list[0]["output_schema"] = output_schema
    else:
        return None, (
            "No tasks provided. Pass tasks=[{goal: '...', context: '...'}, "
            "...] — one entry per subagent (a single task is a one-entry array)."
        )

    for i, task in enumerate(task_list):
        if not isinstance(task, dict):
            return None, f"Task {i} must be an object, got {type(task).__name__}."
        if not task.get("goal", "").strip():
            return None, f"Task {i} is missing a 'goal'."
    # The single-goal form is exempt from the batch gate (short goals are valid there).
    batch_error = _validate_batch_tasks(task_list) if isinstance(tasks, list) else None
    return (None, batch_error) if batch_error else (task_list, None)

def _coerce_task_schemas(
    task_list: List[Dict[str, Any]], output_schema: Optional[Dict[str, Any]]
) -> tuple[List[Optional[Dict[str, Any]]], Optional[str]]:
    """Per-task coerced output schemas. A malformed output_schema fails the whole call before any child spawns;
    schema-less tasks resolve to None and take no new code paths downstream."""
    from tools.delegation_output_schema import coerce_output_schema
    task_schemas: List[Optional[Dict[str, Any]]] = []
    for i, task in enumerate(task_list):
        raw_schema = task.get("output_schema")
        if raw_schema is None and len(task_list) == 1 and output_schema is not None:
            raw_schema = output_schema
        coerced_schema, schema_err = coerce_output_schema(raw_schema)
        if schema_err:
            return [], f"Task {i} output_schema invalid: {schema_err}"
        task_schemas.append(coerced_schema)
    return task_schemas, None
