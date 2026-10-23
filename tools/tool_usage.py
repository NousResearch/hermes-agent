"""Read-only tool usage analytics over Hermes' canonical session history."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

PRUNE_THRESHOLD = 10


def _messages(db: Any) -> list[dict[str, Any]]:
    rows = db._conn.execute(
        "SELECT session_id, role, tool_name, content FROM messages WHERE active = 1 ORDER BY id"
    ).fetchall()
    return [dict(row) for row in rows]


def _usage(db: Any) -> dict[str, dict[str, int]]:
    calls, failures, tokens, sessions = Counter(), Counter(), Counter(), defaultdict(set)
    for row in _messages(db):
        name = row["tool_name"]
        if row["role"] != "tool" or not name:
            continue
        content = row.get("content") or ""
        calls[name] += 1
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            payload = None
        failures[name] += int(content.startswith("[TOOL_ERROR]") or (isinstance(payload, dict) and (bool(payload.get("error")) or payload.get("success") is False)))
        tokens[name] += len(content) // 4
        if row.get("session_id"):
            sessions[name].add(row["session_id"])
    return {name: {"calls": count, "failures": failures[name], "estimated_output_tokens": tokens[name], "sessions": len(sessions[name])} for name, count in calls.items()}


def tool_summary(db: Any) -> dict[str, Any]:
    tools = _usage(db)
    return {"total_calls": sum(v["calls"] for v in tools.values()), "tools": [{"tool": n, **v} for n, v in sorted(tools.items(), key=lambda x: (-x[1]["calls"], x[0]))]}


def _toolsets() -> tuple[str, dict[str, set[str]]]:
    from hermes_cli.config import load_config_readonly
    from hermes_cli.tools_config import _get_platform_tools
    from toolsets import TOOLSETS, resolve_toolset
    enabled = _get_platform_tools(load_config_readonly(), "cli")
    return "cli", {name: set(resolve_toolset(name)) for name in enabled if name in TOOLSETS}


def generate_prune_diff(db: Any) -> dict[str, Any]:
    platform, groups = _toolsets()
    usage = _usage(db)
    universe = set().union(*groups.values()) if groups else set()
    cold = [name for name, members in sorted(groups.items()) if members and all(usage.get(tool, {}).get("sessions", 0) < PRUNE_THRESHOLD for tool in members)]
    suggestions = [{"tool": t, "sessions": usage.get(t, {}).get("sessions", 0)} for t in sorted(universe) if usage.get(t, {}).get("sessions", 0) < PRUNE_THRESHOLD]
    return {"platform": platform, "suggestions": suggestions, "toolsets": cold,
            "diff_hint": "Suggestions only; no configuration was changed.\n"
            f"platform_toolsets:\n  {platform}:\n" + "".join(f"    - {name}\n" for name in groups) +
            ("\n# Cold toolsets to consider removing:\n" + "".join(f"#   - {name}\n" for name in cold) if cold else "")}


def _open_db():
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    path = get_hermes_home() / "state.db"
    return SessionDB(db_path=path, read_only=True) if path.exists() else None


def cmd_analytics(args=None) -> None:
    from hermes_cli.config import load_config_readonly
    if not (load_config_readonly().get("tools") or {}).get("analytics", False):
        print("Tool analytics are disabled. Set tools.analytics: true in config.yaml.")
        return
    db = _open_db()
    if db is None:
        print("No session history found.")
        return
    try:
        report = tool_summary(db)
        print(f"Tool usage from existing session history ({report['total_calls']} calls)")
        for item in report["tools"]:
            print(f"  {item['tool']}: {item['calls']} calls, {item['failures']} failures, ~{item['estimated_output_tokens']} estimated output tokens, {item['sessions']} sessions")
    finally:
        db.close()


def cmd_prune(args=None) -> None:
    db = _open_db()
    if db is None:
        print("No session history found; no configuration changed.")
        return
    try:
        report = generate_prune_diff(db)
        print(report["diff_hint"])
        print("Cold toolsets: " + (", ".join(report["toolsets"]) or "none"))
    finally:
        db.close()
