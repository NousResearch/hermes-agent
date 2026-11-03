"""Read-only tool usage analytics over Hermes' canonical session history."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterator
from typing import Any

PRUNE_THRESHOLD = 10


def _source_scope(db: Any, platform: str) -> tuple[str, tuple[str, ...]]:
    # Match InsightsEngine's local/created_source contract, including old read-only DBs.
    columns = {row[1] for row in db._conn.execute("PRAGMA table_info(sessions)")}
    if "created_source" in columns:
        return "(s.source = ? OR (s.source = 'local' AND s.created_source = ?))", (platform, platform)
    return "s.source = ?", (platform,)


def _messages(db: Any, platform: str | None = None) -> Iterator[dict[str, Any]]:
    query = (
        "SELECT m.session_id, m.role, m.tool_name, m.content, m.tool_calls, m.tool_call_id "
        "FROM messages m JOIN sessions s ON s.id = m.session_id "
        "WHERE m.role IN ('assistant', 'tool')"
    )
    params: tuple[str, ...] = ()
    if platform:
        clause, params = _source_scope(db, platform)
        query += " AND " + clause
    current_session = None
    names: dict[str, str] = {}
    # Include inactive/compressed rows. Ordering bounds name resolution to one session.
    for raw in db._conn.execute(query + " ORDER BY m.session_id, m.id", params):
        row = dict(raw)
        if row["session_id"] != current_session:
            names.clear()
            current_session = row["session_id"]
        if row["role"] == "assistant":
            try:
                calls = json.loads(row["tool_calls"] or "[]")
            except (json.JSONDecodeError, TypeError):
                continue
            for call in calls if isinstance(calls, list) else []:
                function = call.get("function") if isinstance(call, dict) else None
                name = function.get("name") if isinstance(function, dict) else None
                if not isinstance(name, str) or not name:
                    continue
                if isinstance(call.get("id"), str):
                    names[call["id"]] = name
                yield {**row, "tool_name": name, "content": ""}
        else:
            name = row["tool_name"] or names.get(row["tool_call_id"])
            if name:
                yield {**row, "tool_name": name}


def _usage(db: Any, platform: str | None = None) -> dict[str, dict[str, int]]:
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    failures, tokens, sessions = Counter(), Counter(), defaultdict(set)
    for row in _messages(db, platform):
        name, sid = row["tool_name"], row["session_id"]
        counts[(sid, name)][row["role"]] += 1
        sessions[name].add(sid)
        if row["role"] != "tool":
            continue
        content = row.get("content") or ""
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            payload = None
        failed = content.startswith("[TOOL_ERROR]") or (
            isinstance(payload, dict) and (bool(payload.get("error")) or payload.get("success") is False)
        )
        failures[name] += int(failed)
        tokens[name] += len(content) // 4
    calls = Counter()
    # Same calls have two persisted representations. Reconcile PER session, as InsightsEngine does.
    for (_, name), roles in counts.items():
        calls[name] += max(roles["assistant"], roles["tool"])
    return {
        name: {"calls": count, "failures": failures[name], "estimated_output_tokens": tokens[name],
               "sessions": len(sessions[name])}
        for name, count in calls.items()
    }


def tool_summary(db: Any) -> dict[str, Any]:
    tools = _usage(db)
    return {
        "total_calls": sum(v["calls"] for v in tools.values()),
        "tools": [{"tool": n, **v} for n, v in sorted(tools.items(), key=lambda x: (-x[1]["calls"], x[0]))],
    }


def _toolsets() -> tuple[str, dict[str, set[str]]]:
    from hermes_cli.config import load_config_readonly
    from hermes_cli.tools_config import _get_platform_tools
    from toolsets import TOOLSETS, resolve_toolset

    enabled = _get_platform_tools(load_config_readonly(), "cli")
    return "cli", {name: set(resolve_toolset(name)) if name in TOOLSETS else set() for name in sorted(enabled)}


def generate_prune_diff(db: Any) -> dict[str, Any]:
    platform, groups = _toolsets()
    clause, params = _source_scope(db, platform)
    observed = db._conn.execute("SELECT COUNT(*) FROM sessions s WHERE " + clause, params).fetchone()[0]
    if observed < PRUNE_THRESHOLD:
        return {
            "platform": platform, "suggestions": [], "toolsets": [],
            "diff_hint": f"Not enough {platform} session history for pruning ({observed}/{PRUNE_THRESHOLD}). No configuration changed.",
        }
    usage = _usage(db, platform)
    universe = set().union(*groups.values()) if groups else set()
    cold = [name for name, members in sorted(groups.items()) if members and all(
        usage.get(tool, {}).get("sessions", 0) < PRUNE_THRESHOLD for tool in members)]
    suggestions = [{"tool": t, "sessions": usage.get(t, {}).get("sessions", 0)} for t in sorted(universe)
                   if usage.get(t, {}).get("sessions", 0) < PRUNE_THRESHOLD]
    return {
        "platform": platform, "suggestions": suggestions, "toolsets": cold,
        "diff_hint": "Suggestions only; no configuration was changed.\n"
        f"platform_toolsets:\n  {platform}:\n" + "".join(f"    - {name}\n" for name in groups) +
        ("\n# Cold toolsets to consider removing:\n" + "".join(f"#   - {name}\n" for name in cold) if cold else ""),
    }


def _open_db():
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    path = get_hermes_home() / "state.db"
    return SessionDB(db_path=path, read_only=True) if path.exists() else None


def is_enabled() -> bool:
    from hermes_cli.config import load_config_readonly

    tools = load_config_readonly().get("tools")
    return isinstance(tools, dict) and tools.get("analytics") is True


def cmd_analytics(args=None) -> None:
    if not is_enabled():
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
            print(f"  {item['tool']}: {item['calls']} calls, {item['failures']} failures, "
                  f"~{item['estimated_output_tokens']} estimated output tokens, {item['sessions']} sessions")
    finally:
        db.close()


def cmd_prune(args=None) -> None:
    if not is_enabled():
        print("Tool analytics are disabled. Set tools.analytics: true in config.yaml.")
        return
    db = _open_db()
    if db is None:
        print("No session history found; no configuration changed.")
        return
    try:
        report = generate_prune_diff(db)
        for item in report["suggestions"]:
            print(f"  {item['tool']}: used in {item['sessions']} recorded sessions")
        print(report["diff_hint"])
        print("Cold toolsets: " + (", ".join(report["toolsets"]) or "none"))
    finally:
        db.close()
