"""Ideas tools — markdown idea drafts scoped by Kanban board slug.

Ideas live under ``~/.hermes/ideas`` (SQLite index + ``.md`` bodies). They are
**not** Kanban tasks — use ``ideas_convert`` when you want to promote an idea
into the task board.

Included in the default ``hermes-cli`` toolset (via ``includes: [ideas]``).
Humans can also use ``hermes ideas …`` or the dashboard Ideas tab.
"""

from __future__ import annotations

import json
from typing import Any

from hermes_cli import ideas_db as db
from tools.registry import registry, tool_error


def _ok(**fields: Any) -> str:
    return json.dumps({"ok": True, **fields})


def _idea_dict(idea: dict[str, Any], *, include_body: bool = True) -> dict[str, Any]:
    out = {
        "id": idea["id"],
        "board": idea["board"],
        "title": idea["title"],
        "summary": idea.get("summary"),
        "status": idea["status"],
        "tags": idea.get("tags") or [],
        "task_id": idea.get("task_id"),
        "file_path": idea.get("file_path"),
        "created_at": idea.get("created_at"),
        "updated_at": idea.get("updated_at"),
    }
    if include_body:
        out["body"] = idea.get("body", "")
    return out


def _parse_bool_arg(args: dict, name: str, *, default: bool = False) -> bool:
    value = args.get(name)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return default


def _handle_list(args: dict, **kw) -> str:
    all_boards = _parse_bool_arg(args, "all_boards")
    try:
        if all_boards:
            result = db.list_ideas_all_boards(
                status=args.get("status"),
                q=args.get("q"),
                tag=args.get("tag"),
                include_archived=bool(args.get("include_archived")),
            )
            ideas = [_idea_dict(i, include_body=False) for i in result["ideas"]]
            return _ok(
                all_boards=True,
                ideas=ideas,
                count=result["count"],
                boards=result["boards"],
            )
        result = db.list_ideas(
            board=args.get("board"),
            status=args.get("status"),
            q=args.get("q"),
            tag=args.get("tag"),
            include_archived=bool(args.get("include_archived")),
        )
    except db.IdeasError as exc:
        return tool_error(str(exc))
    ideas = [_idea_dict(i, include_body=False) for i in result["ideas"]]
    return _ok(board=result["board"], ideas=ideas, count=len(ideas))


def _handle_boards(args: dict, **kw) -> str:
    try:
        result = db.list_boards()
    except db.IdeasError as exc:
        return tool_error(str(exc))
    boards = [
        {
            "slug": b.get("slug"),
            "name": b.get("name"),
            "idea_count": int(b.get("idea_count") or 0),
            "is_current": bool(b.get("is_current")),
        }
        for b in result.get("boards") or []
    ]
    return _ok(boards=boards, current=result.get("current"), count=len(boards))


def _handle_show(args: dict, **kw) -> str:
    idea_id = (args.get("idea_id") or "").strip()
    if not idea_id:
        return tool_error("idea_id is required")
    try:
        idea = db.get_idea(idea_id)
    except db.IdeaNotFoundError:
        return tool_error(f"idea {idea_id} not found")
    except db.IdeasError as exc:
        return tool_error(str(exc))
    return _ok(idea=_idea_dict(idea))


def _handle_create(args: dict, **kw) -> str:
    title = (args.get("title") or "").strip()
    if not title:
        return tool_error("title is required")
    try:
        idea = db.create_idea(
            title=title,
            body=args.get("body") or "",
            summary=args.get("summary"),
            status=args.get("status") or db.DEFAULT_STATUS,
            tags=args.get("tags"),
            board=args.get("board"),
        )
    except db.IdeasError as exc:
        return tool_error(str(exc))
    return _ok(idea=_idea_dict(idea))


def _handle_update(args: dict, **kw) -> str:
    idea_id = (args.get("idea_id") or "").strip()
    if not idea_id:
        return tool_error("idea_id is required")
    try:
        idea = db.update_idea(
            idea_id,
            title=args.get("title"),
            body=args.get("body"),
            summary=args.get("summary"),
            status=args.get("status"),
            tags=args.get("tags"),
            task_id=args.get("task_id"),
        )
    except db.IdeaNotFoundError:
        return tool_error(f"idea {idea_id} not found")
    except db.IdeasError as exc:
        return tool_error(str(exc))
    return _ok(idea=_idea_dict(idea))


def _handle_delete(args: dict, **kw) -> str:
    idea_id = (args.get("idea_id") or "").strip()
    if not idea_id:
        return tool_error("idea_id is required")
    delete_file = args.get("delete_file")
    if delete_file is None:
        keep_file = False
    else:
        keep_file = not bool(delete_file)
    try:
        db.delete_idea(idea_id, delete_file=not keep_file)
    except db.IdeaNotFoundError:
        return tool_error(f"idea {idea_id} not found")
    return _ok(deleted=idea_id)


def _handle_convert(args: dict, **kw) -> str:
    idea_id = (args.get("idea_id") or "").strip()
    if not idea_id:
        return tool_error("idea_id is required")
    triage = args.get("triage")
    if triage is None:
        triage_flag = True
    else:
        triage_flag = bool(triage)
    try:
        result = db.convert_to_task(
            idea_id,
            assignee=args.get("assignee"),
            priority=int(args.get("priority") or 0),
            triage=triage_flag,
            tenant=args.get("tenant"),
        )
    except db.IdeaNotFoundError:
        return tool_error(f"idea {idea_id} not found")
    except db.IdeasError as exc:
        return tool_error(str(exc))
    return _ok(
        task_id=result["task_id"],
        board=result["board"],
        idea=_idea_dict(result["idea"]),
    )


_BOARD_PROP = {
    "type": "string",
    "description": (
        "Kanban board slug (project boundary). Omit for the default board. "
        "Ignored when all_boards is true."
    ),
}


IDEAS_LIST_SCHEMA = {
    "name": "ideas_list",
    "description": (
        "List markdown ideas. Ideas are lightweight drafts in ~/.hermes/ideas "
        "— not Kanban tasks. Set all_boards=true to list every board in one "
        "call (preferred over reading .md files or shelling out to hermes ideas)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "all_boards": {
                "type": "boolean",
                "description": (
                    "If true, return ideas from every board in one response. "
                    "Use this when the user asks for ideas across all projects."
                ),
            },
            "board": _BOARD_PROP,
            "status": {
                "type": "string",
                "enum": sorted(db.IDEA_STATUSES),
                "description": "Filter by idea status.",
            },
            "q": {"type": "string", "description": "Search title/summary."},
            "tag": {"type": "string", "description": "Filter by tag (without #)."},
            "include_archived": {
                "type": "boolean",
                "description": "Include archived ideas.",
            },
        },
    },
}

IDEAS_BOARDS_SCHEMA = {
    "name": "ideas_boards",
    "description": (
        "List Kanban boards with per-board idea counts. Use before ideas_list "
        "when you only need board names, or to see which board is current."
    ),
    "parameters": {"type": "object", "properties": {}},
}

IDEAS_SHOW_SCHEMA = {
    "name": "ideas_show",
    "description": "Read one idea including its markdown body.",
    "parameters": {
        "type": "object",
        "properties": {
            "idea_id": {"type": "string", "description": "Idea id (i_…)."},
        },
        "required": ["idea_id"],
    },
}

IDEAS_CREATE_SCHEMA = {
    "name": "ideas_create",
    "description": (
        "Create a markdown idea on a board. Use this for rough drafts "
        "before promoting to Kanban via ideas_convert."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short title."},
            "body": {"type": "string", "description": "Markdown draft body."},
            "summary": {"type": "string", "description": "One-line summary."},
            "status": {
                "type": "string",
                "enum": sorted(db.IDEA_STATUSES),
                "description": "Defaults to draft.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags.",
            },
            "board": _BOARD_PROP,
        },
        "required": ["title"],
    },
}

IDEAS_UPDATE_SCHEMA = {
    "name": "ideas_update",
    "description": "Update idea metadata and/or markdown body.",
    "parameters": {
        "type": "object",
        "properties": {
            "idea_id": {"type": "string"},
            "title": {"type": "string"},
            "body": {"type": "string"},
            "summary": {"type": "string"},
            "status": {"type": "string", "enum": sorted(db.IDEA_STATUSES)},
            "tags": {"type": "array", "items": {"type": "string"}},
            "task_id": {"type": "string", "description": "Linked Kanban task id."},
        },
        "required": ["idea_id"],
    },
}

IDEAS_DELETE_SCHEMA = {
    "name": "ideas_delete",
    "description": "Delete an idea from the index (and its .md file by default).",
    "parameters": {
        "type": "object",
        "properties": {
            "idea_id": {"type": "string"},
            "delete_file": {
                "type": "boolean",
                "description": "Remove the markdown file (default true).",
            },
        },
        "required": ["idea_id"],
    },
}

IDEAS_CONVERT_SCHEMA = {
    "name": "ideas_convert",
    "description": (
        "Promote an idea to a Kanban task on the same board. Marks the "
        "idea converted and links task_id."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "idea_id": {"type": "string"},
            "assignee": {"type": "string", "description": "Profile to assign."},
            "priority": {"type": "integer"},
            "triage": {
                "type": "boolean",
                "description": "Land in triage column (default true).",
            },
            "tenant": {"type": "string"},
        },
        "required": ["idea_id"],
    },
}


registry.register(
    name="ideas_list",
    toolset="ideas",
    schema=IDEAS_LIST_SCHEMA,
    handler=_handle_list,
    emoji="💡",
)

registry.register(
    name="ideas_boards",
    toolset="ideas",
    schema=IDEAS_BOARDS_SCHEMA,
    handler=_handle_boards,
    emoji="💡",
)

registry.register(
    name="ideas_show",
    toolset="ideas",
    schema=IDEAS_SHOW_SCHEMA,
    handler=_handle_show,
    emoji="💡",
)

registry.register(
    name="ideas_create",
    toolset="ideas",
    schema=IDEAS_CREATE_SCHEMA,
    handler=_handle_create,
    emoji="💡",
)

registry.register(
    name="ideas_update",
    toolset="ideas",
    schema=IDEAS_UPDATE_SCHEMA,
    handler=_handle_update,
    emoji="💡",
)

registry.register(
    name="ideas_delete",
    toolset="ideas",
    schema=IDEAS_DELETE_SCHEMA,
    handler=_handle_delete,
    emoji="💡",
)

registry.register(
    name="ideas_convert",
    toolset="ideas",
    schema=IDEAS_CONVERT_SCHEMA,
    handler=_handle_convert,
    emoji="💡",
)
