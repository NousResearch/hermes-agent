from __future__ import annotations

import json
import re
from typing import Any

from tools.registry import registry


FIXED_TOOL_IDS = {
    "search_files": 101,
    "read_file": 102,
    "patch": 201,
    "terminal": 202,
    "web_search": 301,
    "web_extract": 302,
    "browser_navigate": 401,
    "browser_snapshot": 402,
    "browser_vision": 403,
}

TOOL_SHORT_DESCRIPTIONS = {
    "search_files": "search names/content",
    "read_file": "read paged text",
    "patch": "apply file patch",
    "terminal": "run shell command",
    "web_search": "search web",
    "web_extract": "extract web page",
    "browser_navigate": "open page",
    "browser_snapshot": "inspect page snapshot",
    "browser_vision": "inspect screenshot",
}

IGNORED_TOOLS = {
    "tool_backpack",
    "skill_backpack",
    "clarify",
    "delegate_task",
    "memory",
    "session_search",
    "skill_view",
    "skills_list",
    "skill_manage",
    "todo",
}


def _response(**payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _all_index_tool_names() -> list[str]:
    candidates = sorted(name for name in registry.get_all_tool_names() if name not in IGNORED_TOOLS)
    available_defs = registry.get_definitions(set(candidates), quiet=True)
    return sorted(
        tool["function"]["name"]
        for tool in available_defs
        if tool.get("function", {}).get("name") in candidates
    )


def _stable_tool_id(tool_name: str) -> int:
    if tool_name in FIXED_TOOL_IDS:
        return FIXED_TOOL_IDS[tool_name]
    return 1000 + sum((index + 1) * ord(char) for index, char in enumerate(tool_name)) % 9000


def _tool_name_by_id(tool_id: int) -> str | None:
    for name in _all_index_tool_names():
        if _stable_tool_id(name) == tool_id:
            return name
    return None


def build_tool_prompt_index(available_names: set[str] | None = None) -> str:
    names = _all_index_tool_names()
    if available_names is not None:
        names = [name for name in names if name in available_names]
    if not names:
        return ""
    lines = ["Tool Backpack index:"]
    for name in names:
        description = TOOL_SHORT_DESCRIPTIONS.get(name, "tool")
        lines.append(f"{_stable_tool_id(name)}: {name} - {description}")
    lines.append(
        "Select all anticipated tools in one call by calling tool_backpack with: "
        "select <id|tool_name>[,<id|tool_name>...]."
    )
    return "\n".join(lines)


def _selected_tool_names(request: str) -> list[str] | None:
    match = re.match(r"^\s*select\s+(.+?)\s*$", request.lower())
    explicit_select = match is not None
    raw_selection = match.group(1).strip() if match else request.strip().lower()
    if not explicit_select and not re.fullmatch(r"[a-z0-9_]+(?:[,\s]+[a-z0-9_]+)*", raw_selection):
        return None

    tokens = [token for token in re.split(r"[,\s]+", raw_selection) if token]
    if not tokens:
        return []

    all_names = set(_all_index_tool_names())
    if not explicit_select and any(not token.isdigit() and token not in all_names for token in tokens):
        return None

    selected = []
    for token in tokens:
        if token.isdigit():
            tool_name = _tool_name_by_id(int(token))
        elif re.match(r"^[a-z0-9_]+$", token) and token in all_names:
            tool_name = token
        else:
            tool_name = None
        if tool_name is None:
            return []
        if tool_name not in selected:
            selected.append(tool_name)
    return selected


def tool_backpack(args: dict[str, Any], **_kwargs: Any) -> str:
    request = args.get("request")
    if not isinstance(request, str) or not request.strip():
        return _response(
            status="blocked",
            decision="blocked",
            message="tool_backpack requires select <id|tool_name>[,<id|tool_name>...].",
        )

    selected_tool_names = _selected_tool_names(request)
    if selected_tool_names is None:
        return _response(
            status="blocked",
            decision="blocked",
            message="Call tool_backpack only with select <id|tool_name>[,<id|tool_name>...].",
        )
    if not selected_tool_names:
        return _response(status="blocked", decision="blocked", message="No installed tool matched this selection.")
    if len(selected_tool_names) == 1:
        tool_name = selected_tool_names[0]
        return _response(
            status="ok",
            decision="select_tool",
            d="selected",
            id=_stable_tool_id(tool_name),
            tool=tool_name,
            next=f"call {tool_name}",
            message="Tool selected; call the named tool directly.",
        )
    return _response(
        status="ok",
        decision="select_tools",
        d="selected",
        tools=selected_tool_names,
        next="call selected tools",
        message="Tools selected; call any selected tool directly as needed for the task.",
    )


TOOL_BACKPACK_SCHEMA = {
    "name": "tool_backpack",
    "description": "Tool gateway.",
    "parameters": {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": "Use select <id|tool_name>[,<id|tool_name>...] only.",
            }
        },
        "required": ["request"],
        "additionalProperties": False,
    },
}


registry.register(
    name="tool_backpack",
    toolset="tool_backpack",
    schema=TOOL_BACKPACK_SCHEMA,
    handler=tool_backpack,
    emoji="🎒",
    max_result_size_chars=100_000,
)
