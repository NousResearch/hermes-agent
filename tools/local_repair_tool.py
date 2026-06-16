#!/usr/bin/env python3
"""Small deterministic local repair helpers for known Hermes ops tasks."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import re
import sqlite3
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from tools.registry import registry


LOCAL_REPAIR_SCHEMA = {
    "name": "local_repair",
    "description": (
        "Run a deterministic, low-token local repair for known Hermes Feishu ops "
        "issues. For morning briefing / operation summary file matching issues, "
        "and daily token usage reports, use this before broad execute_code/search_files "
        "exploration."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_type": {
                "type": "string",
                "enum": ["morning_briefing_operation_summary", "today_token_usage_report"],
                "description": "Known local repair flow to run.",
            },
            "target_date": {
                "type": "string",
                "description": "Optional operation summary date to verify, in YYYY-MM-DD form.",
            },
            "title_hint": {
                "type": "string",
                "description": "Optional expected operation summary title substring.",
            },
            "dry_run": {
                "type": "boolean",
                "description": "When true, inspect and report the intended fix without writing files.",
                "default": False,
            },
            "scope": {
                "type": "string",
                "enum": ["all", "feishu"],
                "description": "For today_token_usage_report, include all Hermes sessions or only Feishu sessions.",
                "default": "all",
            },
            "top_n": {
                "type": "integer",
                "description": "For today_token_usage_report, number of highest-input sessions to return.",
                "default": 3,
            },
        },
        "required": ["task_type"],
        "additionalProperties": False,
    },
}


FIXED_OP_TITLES_FOR = '''def op_titles_for(day: dt.date) -> list[str]:
    prefix = day.strftime("%Y%m%d")
    prefix_dash = day.strftime("%Y-%m-%d")
    titles: list[str] = []
    if not OPS_DIR.exists():
        return titles
    paths = list(OPS_DIR.glob(f"{prefix}*.html")) + list(OPS_DIR.glob(f"{prefix_dash}_*.html"))
    for path in sorted(paths):
        stem = path.stem
        title = re.sub(r"^(\\d+|\\d{4}-\\d{2}-\\d{2})_", "", stem).replace("_", " ")
        titles.append(title)
    return titles[:10]
'''


def _default_script_path() -> Path:
    return Path.home() / ".hermes" / "scripts" / "morning-briefing-report.py"


def _default_docs_dir() -> Path:
    return Path.home() / "Documents" / "Hermes"


def _default_state_db_path() -> Path:
    return Path.home() / ".hermes" / "state.db"


def _parse_target_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def _date_from_filename(path: Path) -> dt.date | None:
    stem = path.stem
    match = re.match(r"^(\d{8}|\d{4}-\d{2}-\d{2})", stem)
    if not match:
        return None
    raw = match.group(1)
    try:
        if "-" in raw:
            return dt.date.fromisoformat(raw)
        return dt.datetime.strptime(raw, "%Y%m%d").date()
    except ValueError:
        return None


def _wide_titles_for(ops_dir: Path, day: dt.date) -> list[str]:
    prefix = day.strftime("%Y%m%d")
    prefix_dash = day.strftime("%Y-%m-%d")
    paths = list(ops_dir.glob(f"{prefix}*.html")) + list(ops_dir.glob(f"{prefix_dash}_*.html"))
    titles: list[str] = []
    for path in sorted(set(paths)):
        stem = path.stem
        title = re.sub(r"^(\d+|\d{4}-\d{2}-\d{2})_", "", stem).replace("_", " ")
        titles.append(title)
    return titles[:10]


def _replace_op_titles_for(source: str) -> tuple[str, bool]:
    lines = source.splitlines(keepends=True)
    start: int | None = None
    end = len(lines)
    for idx, line in enumerate(lines):
        if line.startswith("def op_titles_for("):
            start = idx
            break
    if start is None:
        return source, False
    for idx in range(start + 1, len(lines)):
        line = lines[idx]
        if line.startswith("def ") or line.startswith("class "):
            end = idx
            break
    replacement = FIXED_OP_TITLES_FOR
    if end < len(lines) and not replacement.endswith("\n\n"):
        replacement += "\n"
    new_source = "".join(lines[:start]) + replacement + "".join(lines[end:])
    return new_source, new_source != source


def _load_script_module(script_path: Path):
    name = f"_hermes_morning_briefing_report_{abs(hash(str(script_path)))}"
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _script_titles_for(script_path: Path, day: dt.date) -> list[str]:
    module = _load_script_module(script_path)
    titles = module.op_titles_for(day)
    return list(titles) if isinstance(titles, list) else []


def _candidate_dates(ops_dir: Path) -> list[dt.date]:
    dates = {
        day
        for path in ops_dir.glob("*.html")
        for day in [_date_from_filename(path)]
        if day is not None
    }
    return sorted(dates, reverse=True)


def _select_target_date(
    script_path: Path,
    ops_dir: Path,
    requested: dt.date | None,
    title_hint: str,
) -> tuple[dt.date | None, list[str], list[str]]:
    if requested is not None:
        wide = _wide_titles_for(ops_dir, requested)
        try:
            current = _script_titles_for(script_path, requested)
        except Exception:
            current = []
        return requested, current, wide

    for day in _candidate_dates(ops_dir):
        wide = _wide_titles_for(ops_dir, day)
        if not wide:
            continue
        try:
            current = _script_titles_for(script_path, day)
        except Exception:
            current = []
        missing_hint = bool(title_hint) and not any(title_hint in title for title in current)
        if current != wide or missing_hint:
            return day, current, wide

    dates = _candidate_dates(ops_dir)
    if not dates:
        return None, [], []
    day = dates[0]
    try:
        current = _script_titles_for(script_path, day)
    except Exception:
        current = []
    return day, current, _wide_titles_for(ops_dir, day)


def repair_morning_briefing_operation_summary(
    *,
    script_path: Path | None = None,
    docs_dir: Path | None = None,
    target_date: str | dt.date | None = None,
    title_hint: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Repair the known operation-summary glob mismatch in the briefing script."""
    script_path = script_path or _default_script_path()
    docs_dir = docs_dir or _default_docs_dir()
    ops_dir = docs_dir / "operation summary"
    if isinstance(target_date, dt.date):
        requested_date = target_date
    else:
        requested_date = _parse_target_date(target_date)

    result: dict[str, Any] = {
        "task_type": "morning_briefing_operation_summary",
        "script_path": str(script_path),
        "ops_dir": str(ops_dir),
        "dry_run": bool(dry_run),
    }

    if not script_path.exists():
        return {**result, "success": False, "error": "morning briefing script not found"}
    if not ops_dir.exists():
        return {**result, "success": False, "error": "operation summary directory not found"}

    before_source = script_path.read_text(encoding="utf-8")
    selected_date, current_titles, wide_titles = _select_target_date(
        script_path,
        ops_dir,
        requested_date,
        title_hint.strip(),
    )
    result.update(
        {
            "target_date": selected_date.isoformat() if selected_date else None,
            "before_titles": current_titles[:5],
            "expected_titles": wide_titles[:5],
        }
    )
    if selected_date is None:
        return {**result, "success": False, "error": "no operation summary html candidates found"}

    already_fixed = 'OPS_DIR.glob(f"{prefix}*.html")' in before_source and "prefix_dash" in before_source
    new_source, replaced = _replace_op_titles_for(before_source)
    if not replaced:
        return {**result, "success": False, "error": "op_titles_for function not found"}

    changed = new_source != before_source and not already_fixed
    if changed and not dry_run:
        script_path.write_text(new_source, encoding="utf-8")

    verify_titles = _script_titles_for(script_path, selected_date) if not dry_run else wide_titles
    expected_found = bool(wide_titles) and all(title in verify_titles for title in wide_titles[:1])
    if title_hint:
        expected_found = expected_found and any(title_hint in title for title in verify_titles)

    return {
        **result,
        "success": bool(expected_found),
        "changed": bool(changed and not dry_run),
        "would_change": bool(changed),
        "already_fixed": bool(already_fixed),
        "root_cause": (
            "op_titles_for used a narrow YYYYMMDD_*.html glob and missed "
            "operation summary files named like YYYYMMDDHHMM_title.html."
        ),
        "fix": "Use YYYYMMDD*.html plus YYYY-MM-DD_*.html and strip either numeric or dashed date prefixes.",
        "verify_titles": verify_titles[:5],
    }


def _clamp_top_n(value: int | str | None) -> int:
    try:
        top_n = int(value if value is not None else 3)
    except (TypeError, ValueError):
        top_n = 3
    return max(1, min(top_n, 10))


def _session_row_to_item(row: sqlite3.Row, tz: ZoneInfo) -> dict[str, Any]:
    started_at = float(row["started_at"] or 0.0)
    started_local = dt.datetime.fromtimestamp(started_at, tz).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "session_id": row["id"],
        "title": row["title"] or "(untitled)",
        "source": row["source"] or "",
        "model": row["model"] or "",
        "started_at": started_at,
        "started_local": started_local,
        "end_reason": row["end_reason"] or "",
        "api_call_count": row["api_call_count"] or 0,
        "tool_call_count": row["tool_call_count"] or 0,
        "message_count": row["message_count"] or 0,
        "input_tokens": row["input_tokens"] or 0,
        "output_tokens": row["output_tokens"] or 0,
        "cache_read_tokens": row["cache_read_tokens"] or 0,
        "estimated_cost_usd": row["estimated_cost_usd"],
        "cost_status": row["cost_status"] or "",
    }


def collect_today_token_usage_report(
    *,
    db_path: Path | None = None,
    scope: str = "all",
    top_n: int = 3,
    now: dt.datetime | None = None,
    timezone: str = "Asia/Shanghai",
) -> dict[str, Any]:
    """Collect today's Hermes token usage from state.db with one bounded query."""
    db_path = db_path or _default_state_db_path()
    tz = ZoneInfo(timezone)
    now_local = now.astimezone(tz) if now is not None else dt.datetime.now(tz)
    start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    start_ts = start_local.timestamp()
    end_ts = now_local.timestamp()
    scope = "feishu" if str(scope or "").lower() == "feishu" else "all"
    top_n = _clamp_top_n(top_n)

    result: dict[str, Any] = {
        "task_type": "today_token_usage_report",
        "success": False,
        "db_path": str(db_path),
        "scope": scope,
        "timezone": timezone,
        "window_start": start_local.strftime("%Y-%m-%d %H:%M:%S"),
        "window_end": now_local.strftime("%Y-%m-%d %H:%M:%S"),
        "top_n": top_n,
    }
    if not db_path.exists():
        return {**result, "error": "state.db not found"}

    where = "started_at >= ? AND started_at <= ?"
    params: list[Any] = [start_ts, end_ts]
    if scope == "feishu":
        where += " AND source = ?"
        params.append("feishu")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        totals = conn.execute(
            f"""
            SELECT COUNT(*) AS session_count,
                   COALESCE(SUM(input_tokens), 0) AS input_tokens,
                   COALESCE(SUM(output_tokens), 0) AS output_tokens,
                   COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens,
                   COALESCE(SUM(cache_write_tokens), 0) AS cache_write_tokens,
                   COALESCE(SUM(api_call_count), 0) AS api_call_count,
                   COALESCE(SUM(tool_call_count), 0) AS tool_call_count,
                   COALESCE(SUM(estimated_cost_usd), 0.0) AS estimated_cost_usd
            FROM sessions
            WHERE {where}
            """,
            params,
        ).fetchone()
        rows = conn.execute(
            f"""
            SELECT id, title, source, model, started_at, ended_at, end_reason,
                   message_count, tool_call_count, api_call_count, input_tokens,
                   output_tokens, cache_read_tokens, estimated_cost_usd, cost_status
            FROM sessions
            WHERE {where}
            ORDER BY input_tokens DESC, started_at DESC
            LIMIT ?
            """,
            [*params, top_n],
        ).fetchall()
    finally:
        conn.close()

    return {
        **result,
        "success": True,
        "session_count": totals["session_count"] or 0,
        "input_tokens": totals["input_tokens"] or 0,
        "output_tokens": totals["output_tokens"] or 0,
        "cache_read_tokens": totals["cache_read_tokens"] or 0,
        "cache_write_tokens": totals["cache_write_tokens"] or 0,
        "api_call_count": totals["api_call_count"] or 0,
        "tool_call_count": totals["tool_call_count"] or 0,
        "estimated_cost_usd": totals["estimated_cost_usd"] or 0.0,
        "top_sessions": [_session_row_to_item(row, tz) for row in rows],
    }


def format_today_token_usage_report(report: dict[str, Any]) -> str:
    """Render a compact Chinese report suitable for Feishu text delivery."""
    if not report.get("success"):
        return f"今日 Token 统计失败：{report.get('error') or 'unknown error'}"

    scope_label = "飞书会话" if report.get("scope") == "feishu" else "Hermes 全部会话"
    lines = [
        f"今日 Token 消耗统计（{scope_label}）",
        f"时间范围：{report['window_start']} 至 {report['window_end']}（{report['timezone']}）",
        (
            f"整体：{report['session_count']} 个 session，"
            f"输入 {report['input_tokens']:,}，输出 {report['output_tokens']:,}，"
            f"cache read {report['cache_read_tokens']:,}，"
            f"API 调用 {report['api_call_count']:,}，工具调用 {report['tool_call_count']:,}"
        ),
        "",
        f"输入 Token Top {report['top_n']}：",
    ]
    top_sessions = report.get("top_sessions") or []
    if not top_sessions:
        lines.append("无记录。")
    for idx, item in enumerate(top_sessions, 1):
        title = str(item.get("title") or "(untitled)")
        if len(title) > 60:
            title = title[:57] + "..."
        lines.append(
            f"{idx}. {title} [{item.get('source') or 'unknown'}] "
            f"输入 {item['input_tokens']:,}，输出 {item['output_tokens']:,}，"
            f"API {item['api_call_count']}，工具 {item['tool_call_count']}，"
            f"开始 {item['started_local']}"
        )
    return "\n".join(lines)


def render_today_token_usage_report(
    *,
    db_path: Path | None = None,
    scope: str = "all",
    top_n: int = 3,
    now: dt.datetime | None = None,
    timezone: str = "Asia/Shanghai",
) -> str:
    return format_today_token_usage_report(
        collect_today_token_usage_report(
            db_path=db_path,
            scope=scope,
            top_n=top_n,
            now=now,
            timezone=timezone,
        )
    )


def _handle_local_repair(args: dict, **kw) -> str:
    task_type = args.get("task_type")
    if task_type == "today_token_usage_report":
        result = collect_today_token_usage_report(
            scope=args.get("scope") or "all",
            top_n=_clamp_top_n(args.get("top_n")),
        )
        result["text"] = format_today_token_usage_report(result)
        return json.dumps(result, ensure_ascii=False)

    if task_type != "morning_briefing_operation_summary":
        return json.dumps(
            {
                "success": False,
                "error": f"Unsupported local_repair task_type: {task_type}",
                "supported": ["morning_briefing_operation_summary", "today_token_usage_report"],
            },
            ensure_ascii=False,
        )
    result = repair_morning_briefing_operation_summary(
        target_date=args.get("target_date"),
        title_hint=args.get("title_hint") or "",
        dry_run=bool(args.get("dry_run", False)),
    )
    return json.dumps(result, ensure_ascii=False)


registry.register(
    name="local_repair",
    toolset="file",
    schema=LOCAL_REPAIR_SCHEMA,
    handler=_handle_local_repair,
    emoji="🛠️",
    max_result_size_chars=12_000,
)
