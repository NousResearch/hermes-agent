from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


@dataclass
class TodoSnapshot:
    source_path: Path
    updated_at: str | None
    pending: list[str]
    active: list[str]
    resolved_recent: list[str]


def get_today_todo_path() -> Path:
    return get_hermes_home() / "hermes-daily-state" / "todo-state.json"


def _titles(items: list[Any] | None) -> list[str]:
    titles: list[str] = []
    for item in items or []:
        if isinstance(item, dict):
            text = str(item.get("title") or "").strip()
        else:
            text = str(item or "").strip()
        if text:
            titles.append(text)
    return titles


def load_today_todo_snapshot(path: str | Path | None = None) -> TodoSnapshot:
    todo_path = Path(path) if path is not None else get_today_todo_path()
    if todo_path.exists():
        raw = json.loads(todo_path.read_text(encoding="utf-8"))
    else:
        raw = {
            "updated_at": None,
            "pending": [],
            "active": [],
            "resolved_recent": [],
        }
    return TodoSnapshot(
        source_path=todo_path,
        updated_at=raw.get("updated_at"),
        pending=_titles(raw.get("pending")),
        active=_titles(raw.get("active")),
        resolved_recent=_titles(raw.get("resolved_recent")),
    )


def _append_bucket(lines: list[str], title: str, items: list[str], *, limit: int | None = None) -> None:
    shown = items if limit is None else items[:limit]
    lines.append(f"{title} ({len(items)}):")
    if shown:
        for item in shown:
            lines.append(f"- {item}")
        remaining = len(items) - len(shown)
        if remaining > 0:
            lines.append(f"- … +{remaining} more")
    else:
        lines.append("- (none)")


def render_today_todo_text(snapshot: TodoSnapshot) -> str:
    lines = ["Today Todo"]
    if snapshot.updated_at:
        lines.append(f"Updated: {snapshot.updated_at}")
    _append_bucket(lines, "Active", snapshot.active)
    _append_bucket(lines, "Pending", snapshot.pending, limit=8)
    _append_bucket(lines, "Resolved recent", snapshot.resolved_recent, limit=5)
    lines.append("")
    lines.append("Queued into next turn context for this session.")
    return "\n".join(lines)


def build_today_todo_note(snapshot: TodoSnapshot) -> str:
    lines = [
        "[Note: today todo snapshot was just synced from the local canonical state. "
        "Treat it as the current todo context for later turns in this session.]",
    ]
    if snapshot.updated_at:
        lines.append(f"Updated: {snapshot.updated_at}")
    _append_bucket(lines, "Active", snapshot.active)
    _append_bucket(lines, "Pending", snapshot.pending)
    _append_bucket(lines, "Resolved recent", snapshot.resolved_recent, limit=10)
    return "\n".join(lines)
