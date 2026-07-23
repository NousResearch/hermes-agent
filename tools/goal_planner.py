"""Goal Planner - decompose big goals into actionable tasks.

Tools: goal_create, goal_track, goal_list.
Stores goals as JSON in the Hermes state directory.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home
from tools.registry import registry

logger = logging.getLogger(__name__)

_GOALS_DIR = get_hermes_home() / "goals"


def _ensure_dir():
    _GOALS_DIR.mkdir(parents=True, exist_ok=True)


def _goal_path(goal_id: str) -> Path:
    return _GOALS_DIR / f"{goal_id}.json"


def goal_create(title: str, description: str = "",
                milestones: list = None, parent_goal_id: str = "",
                task_id: str = None) -> str:
    """Create a new goal with optional milestones."""
    _ensure_dir()
    import uuid

    gid = parent_goal_id or f"goal_{uuid.uuid4().hex[:8]}"
    goal = {
        "id": gid,
        "title": title,
        "description": description,
        "milestones": [],
        "status": "active",
        "progress": 0.0,
        "parent_goal_id": parent_goal_id or None,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    if milestones:
        for i, m in enumerate(milestones):
            goal["milestones"].append({
                "id": f"{gid}_m{i}",
                "title": m if isinstance(m, str) else m.get("title", str(m)),
                "status": "pending",
            })
    with open(_goal_path(gid), 'w') as f:
        json.dump(goal, f, indent=2, ensure_ascii=False)
    return json.dumps({"ok": True, "goal": goal}, ensure_ascii=False)


def goal_track(goal_id: str, milestone_id: str = "",
               status: str = "", task_id: str = None) -> str:
    """Update goal or milestone progress."""
    fp = _goal_path(goal_id)
    if not fp.exists():
        return json.dumps({"ok": False, "error": f"Goal not found: {goal_id}"})
    with open(fp) as f:
        goal = json.load(f)

    if milestone_id:
        for m in goal.get("milestones", []):
            if m["id"] == milestone_id:
                m["status"] = status or "completed"
                break
        # Recalculate progress
        ms = goal.get("milestones", [])
        if ms:
            done = sum(1 for m in ms if m["status"] == "completed")
            goal["progress"] = round(done / len(ms) * 100, 1)
            if done == len(ms):
                goal["status"] = "completed"
    elif status:
        goal["status"] = status

    goal["updated_at"] = time.time()
    with open(fp, 'w') as f:
        json.dump(goal, f, indent=2, ensure_ascii=False)
    return json.dumps({"ok": True, "goal": goal}, ensure_ascii=False)


def goal_list(status: str = "", task_id: str = None) -> str:
    """List goals, optionally filtered by status."""
    _ensure_dir()
    goals = []
    for fp in sorted(_GOALS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        with open(fp) as f:
            g = json.load(f)
        if status and g.get("status") != status:
            continue
        goals.append({"id": g["id"], "title": g["title"],
                       "status": g["status"], "progress": g["progress"]})
    return json.dumps({"ok": True, "count": len(goals), "goals": goals[:30]}, ensure_ascii=False)


# ═══ Register ═══

registry.register(
    name="goal_create",
    toolset="planning",
    schema={
        "name": "goal_create",
        "description": "Create a new goal with optional milestones. Break big goals into trackable sub-tasks.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Goal title"},
                "description": {"type": "string", "description": "Detailed description"},
                "milestones": {"type": "array", "items": {"type": "string"},
                               "description": "List of milestone titles"},
                "parent_goal_id": {"type": "string", "description": "Parent goal ID if this is a sub-goal"},
            },
            "required": ["title"],
        },
    },
    handler=lambda args, **kw: goal_create(
        title=args["title"],
        description=args.get("description", ""),
        milestones=args.get("milestones"),
        parent_goal_id=args.get("parent_goal_id", ""),
        **kw,
    ),
)

registry.register(
    name="goal_track",
    toolset="planning",
    schema={
        "name": "goal_track",
        "description": "Update goal or milestone progress. Mark milestones complete or update goal status.",
        "parameters": {
            "type": "object",
            "properties": {
                "goal_id": {"type": "string", "description": "Goal ID to update"},
                "milestone_id": {"type": "string", "description": "Specific milestone ID"},
                "status": {"type": "string", "enum": ["active", "completed", "paused", "cancelled"]},
            },
            "required": ["goal_id"],
        },
    },
    handler=lambda args, **kw: goal_track(
        goal_id=args["goal_id"],
        milestone_id=args.get("milestone_id", ""),
        status=args.get("status", ""),
        **kw,
    ),
)

registry.register(
    name="goal_list",
    toolset="planning",
    schema={
        "name": "goal_list",
        "description": "List goals, optionally filtered by status. Shows progress for each goal.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "Filter: active, completed, paused, cancelled"},
            },
        },
    },
    handler=lambda args, **kw: goal_list(status=args.get("status", ""), **kw),
)