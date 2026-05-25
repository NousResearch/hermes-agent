"""Local-first trace extraction and feedback eval helpers for Hermes fleet work.

This module intentionally reads existing Hermes state instead of instrumenting the
agent loop. It is the MVP substrate for WQ-021: extract trace-shaped records from
``state.db``, run deterministic behavioral evals, and render a feedback-to-plan
report that can be copied into gbrain/hermes reports.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Sequence


CANONICAL_GBRAIN_PREFIXES = (
    "entities/",
    "people/",
    "projects/",
    "wiki/",
    "daily/",
    "media/",
    "tech/",
    "companies/",
    "concepts/",
    "originals/",
    "writing/",
)
HERMES_GBRAIN_PREFIX = "gbrain/hermes/"
TOOL_REQUIRED_PROMPT_RE = re.compile(
    r"\b(what time|what date|current|latest|today|yesterday|weather|version|hash|checksum|file|git|port|process|disk|memory|cpu)\b",
    re.IGNORECASE,
)
PATH_KEYS = {"path", "file_path", "slug", "page_slug", "repo", "workdir"}
CORRECTION_RE = re.compile(
    r"\b(wrong|missed|missing|not quite|did you test|gap|wasn't|was not|incorrect|failed)\b",
    re.IGNORECASE,
)
WRITE_TOOL_NAMES = {
    "write_file",
    "patch",
    "mcp_gbrain_put_page",
    "mcp_gbrain_put_raw_data",
    "mcp_gbrain_add_tag",
    "mcp_gbrain_add_link",
    "mcp_gbrain_add_timeline_entry",
}
READ_TOOL_NAMES = {
    "read_file",
    "search_files",
    "mcp_gbrain_get_page",
    "mcp_gbrain_query",
    "mcp_gbrain_search",
}


@dataclass(frozen=True)
class EvalResult:
    """One deterministic behavioral eval result."""

    rule_id: str
    trace_id: str
    status: str
    detail: str
    suggested_delta: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def extract_traces(db_path: str | Path, since_ts: float | None = None) -> list[dict[str, Any]]:
    """Extract trace-shaped records from a Hermes ``state.db`` SQLite file."""

    path = Path(db_path).expanduser()
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        session_rows = _fetch_sessions(con, since_ts)
        return [_session_to_trace(con, row) for row in session_rows]
    finally:
        con.close()


def evaluate_traces(traces: Iterable[dict[str, Any]]) -> list[EvalResult]:
    """Run the MVP deterministic eval suite over trace dictionaries."""

    results: list[EvalResult] = []
    for trace in traces:
        results.append(_eval_tool_required_question_used_tools(trace))
        results.append(_eval_no_canonical_gbrain_writes(trace))
        results.append(_eval_citation_quality_for_gbrain_claims(trace))
        results.append(_eval_gbrain_vs_filesystem_routing(trace))
        results.append(_eval_telegram_formatting(trace))
        results.append(_eval_domain_handoff_to_dobby(trace))
        results.append(_eval_cross_profile_write_guard(trace))
    return results


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write rows as UTF-8 JSONL, creating parent directories."""

    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def run_pipeline(
    db_path: str | Path,
    workspace_dir: str | Path,
    date: str | None = None,
    since_ts: float | None = None,
) -> dict[str, Path]:
    """Extract traces, run evals, and write MVP feedback artifacts."""

    run_date = date or datetime.now(timezone.utc).date().isoformat()
    workspace = Path(workspace_dir).expanduser()
    trace_path = workspace / "traces" / f"{run_date}.jsonl"
    eval_path = workspace / "evals" / "wq-021" / "results" / f"{run_date}.jsonl"
    report_path = workspace / "reports" / "feedback-to-plan" / f"{run_date}.md"

    traces = extract_traces(db_path, since_ts=since_ts)
    results = evaluate_traces(traces)
    write_jsonl(trace_path, traces)
    write_jsonl(eval_path, [result.to_dict() for result in results])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_feedback_report(results, date=run_date), encoding="utf-8")
    return {"trace_path": trace_path, "eval_path": eval_path, "report_path": report_path}


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for manual WQ-021 trace/eval/feedback runs."""

    parser = argparse.ArgumentParser(description="Run Hermes trace/eval/feedback MVP pipeline.")
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "state.db"), help="Path to Hermes state.db")
    parser.add_argument("--workspace", default=str(Path.home() / ".hermes" / "workspace"), help="Output workspace directory")
    parser.add_argument("--date", default=None, help="Run date YYYY-MM-DD; defaults to today UTC")
    parser.add_argument("--since-ts", type=float, default=None, help="Only include sessions started at or after this Unix timestamp")
    args = parser.parse_args(argv)

    outputs = run_pipeline(args.db, args.workspace, date=args.date, since_ts=args.since_ts)
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0


def render_feedback_report(results: Iterable[EvalResult], date: str | None = None) -> str:
    """Render a compact Markdown feedback-to-plan report."""

    report_date = date or datetime.now(timezone.utc).date().isoformat()
    result_list = list(results)
    failures = [result for result in result_list if result.status == "fail"]
    passes = [result for result in result_list if result.status == "pass"]

    lines = [
        f"# Feedback-to-Plan Report — {report_date}",
        "",
        "## Summary",
        "",
        f"- Eval results: {len(result_list)}",
        f"- Passing: {len(passes)}",
        f"- Failing: {len(failures)}",
        "",
        "## New failures",
        "",
    ]
    grouped_failures = _group_failures(failures)
    if grouped_failures:
        for group in grouped_failures:
            lines.extend(
                [
                    f"- Rule: `{group['rule_id']}`",
                    f"  - Count: {group['count']}",
                    f"  - Traces: {group['traces']}",
                    f"  - Example detail: {group['detail']}",
                    f"  - Suggested delta: {group['suggested_delta'] or 'Investigate and convert into a concrete task.'}",
                ]
            )
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Suggested GitHub issues / skill updates",
            "",
        ]
    )
    if grouped_failures:
        for group in grouped_failures:
            lines.append(f"- `{group['rule_id']}` ({group['count']}): {group['suggested_delta'] or group['detail']}")
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Blocked items needing user action",
            "",
            "- None detected by deterministic evals.",
            "",
        ]
    )
    return "\n".join(lines)


def _group_failures(failures: list[EvalResult]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str], list[EvalResult]] = defaultdict(list)
    for failure in failures:
        groups[(failure.rule_id, failure.suggested_delta or failure.detail)].append(failure)

    rendered: list[dict[str, str | int]] = []
    for (rule_id, suggested_delta), items in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0][0])):
        rendered.append(
            {
                "rule_id": rule_id,
                "count": len(items),
                "traces": ", ".join(f"`{item.trace_id}`" for item in items),
                "detail": items[0].detail,
                "suggested_delta": suggested_delta,
            }
        )
    return rendered


def _fetch_sessions(con: sqlite3.Connection, since_ts: float | None) -> list[sqlite3.Row]:
    sql = "SELECT * FROM sessions"
    params: tuple[float, ...] = ()
    if since_ts is not None:
        sql += " WHERE started_at >= ?"
        params = (since_ts,)
    sql += " ORDER BY started_at ASC"
    return list(con.execute(sql, params))


def _session_to_trace(con: sqlite3.Connection, session: sqlite3.Row) -> dict[str, Any]:
    messages = list(
        con.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC, id ASC",
            (session["id"],),
        )
    )
    first_user = _first_message_content(messages, "user")
    last_assistant = _last_message_content(messages, "assistant")
    parsed_tool_calls = _extract_tool_calls(messages)
    tool_names = [call["name"] for call in parsed_tool_calls]
    errors = _extract_errors(messages)
    paths_read, paths_written = _extract_paths(parsed_tool_calls)
    retry_count, retry_tools = _extract_retries(parsed_tool_calls, errors)

    return {
        "trace_id": session["id"],
        "session_id": session["id"],
        "source": session["source"],
        "created_at": _iso_from_timestamp(session["started_at"]),
        "user_prompt_summary": _summarize(first_user),
        "model_provider": _model_provider(session),
        "model": session["model"],
        "selected_skills": _extract_selected_skills(parsed_tool_calls),
        "tool_calls": tool_names,
        "tool_call_details": parsed_tool_calls,
        "tool_call_count": len(tool_names) or int(session["tool_call_count"] or 0),
        "paths_read": sorted(paths_read),
        "paths_written": sorted(paths_written),
        "errors": errors,
        "retry_count": retry_count,
        "retry_tools": retry_tools,
        "final_response_summary": _summarize(last_assistant),
        "user_followup_or_correction": _extract_user_correction(messages),
        "risk_flags": _risk_flags(paths_written, errors),
    }


def _first_message_content(messages: list[sqlite3.Row], role: str) -> str:
    for message in messages:
        if message["role"] == role and message["content"]:
            return str(message["content"])
    return ""


def _last_message_content(messages: list[sqlite3.Row], role: str) -> str:
    for message in reversed(messages):
        if message["role"] == role and message["content"]:
            return str(message["content"])
    return ""


def _extract_tool_calls(messages: list[sqlite3.Row]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    saw_structured_calls = False
    for message in messages:
        tool_calls_text = message["tool_calls"]
        if tool_calls_text:
            parsed_calls = _parse_tool_calls(tool_calls_text)
            if parsed_calls:
                saw_structured_calls = True
                calls.extend(parsed_calls)
        elif not saw_structured_calls and message["role"] == "tool" and message["tool_name"]:
            calls.append({"name": message["tool_name"], "arguments": {}})
    return calls


def _parse_tool_calls(raw: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []

    calls: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        raw_function = item.get("function")
        function = raw_function if isinstance(raw_function, dict) else {}
        name = item.get("name") or function.get("name") or item.get("tool_name")
        arguments = item.get("arguments", function.get("arguments", {}))
        calls.append({"name": str(name or "unknown"), "arguments": _coerce_arguments(arguments)})
    return calls


def _coerce_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"raw": arguments}
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    return {}


def _extract_errors(messages: list[sqlite3.Row]) -> list[str]:
    errors: list[str] = []
    for message in messages:
        content = str(message["content"] or "")
        if re.search(r"\b(error|traceback|exception|failed|permission_denied)\b", content, re.IGNORECASE):
            errors.append(_summarize(content, limit=200))
    return errors


def _extract_paths(calls: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    paths_read: set[str] = set()
    paths_written: set[str] = set()
    for call in calls:
        name = str(call.get("name", ""))
        paths = set(_walk_path_values(call.get("arguments", {})))
        if name in WRITE_TOOL_NAMES:
            paths_written.update(paths)
        elif name in READ_TOOL_NAMES:
            paths_read.update(paths)
        else:
            paths_read.update(path for path in paths if path.startswith("/") or "/" in path)
    return paths_read, paths_written


def _extract_selected_skills(calls: list[dict[str, Any]]) -> list[str]:
    skills: list[str] = []
    for call in calls:
        if call.get("name") != "skill_view":
            continue
        raw_arguments = call.get("arguments")
        arguments = raw_arguments if isinstance(raw_arguments, dict) else {}
        skill_name = str(arguments.get("name") or "")
        if skill_name and skill_name not in skills:
            skills.append(skill_name)
    return skills


def _extract_retries(calls: list[dict[str, Any]], errors: list[str]) -> tuple[int, list[str]]:
    if not errors:
        return 0, []
    retry_tools: list[str] = []
    retry_count = 0
    previous_signature: tuple[str, str] | None = None
    for call in calls:
        name = str(call.get("name") or "")
        signature = (name, json.dumps(call.get("arguments", {}), sort_keys=True))
        if signature == previous_signature:
            retry_count += 1
            if name and name not in retry_tools:
                retry_tools.append(name)
        previous_signature = signature
    return retry_count, retry_tools


def _extract_user_correction(messages: list[sqlite3.Row]) -> str:
    seen_assistant = False
    for message in messages:
        if message["role"] == "assistant" and message["content"]:
            seen_assistant = True
            continue
        if seen_assistant and message["role"] == "user" and message["content"]:
            content = str(message["content"])
            if CORRECTION_RE.search(content):
                return _summarize(content)
    return ""


def _walk_path_values(value: Any, key: str | None = None) -> Iterable[str]:
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            yield from _walk_path_values(child_value, str(child_key))
    elif isinstance(value, list):
        for item in value:
            yield from _walk_path_values(item, key)
    elif isinstance(value, str) and key in PATH_KEYS and value:
        yield value


def _model_provider(session: sqlite3.Row) -> str:
    if session["billing_provider"]:
        return str(session["billing_provider"])
    raw_config = session["model_config"]
    if not raw_config:
        return ""
    try:
        config = json.loads(raw_config)
    except json.JSONDecodeError:
        return ""
    return str(config.get("provider") or config.get("provider_name") or "")


def _risk_flags(paths_written: set[str], errors: list[str]) -> list[str]:
    flags: list[str] = []
    if any(_is_canonical_gbrain_path(path) for path in paths_written):
        flags.append("canonical_gbrain_write")
    if errors:
        flags.append("errors_detected")
    return flags


def _eval_tool_required_question_used_tools(trace: dict[str, Any]) -> EvalResult:
    prompt = str(trace.get("user_prompt_summary") or "")
    tool_count = int(trace.get("tool_call_count") or 0)
    if not TOOL_REQUIRED_PROMPT_RE.search(prompt):
        return EvalResult(
            rule_id="tool_required_question_used_tools",
            trace_id=str(trace.get("trace_id", "unknown")),
            status="pass",
            detail="Prompt did not require live/tool-backed lookup.",
        )
    if tool_count > 0:
        return EvalResult(
            rule_id="tool_required_question_used_tools",
            trace_id=str(trace.get("trace_id", "unknown")),
            status="pass",
            detail=f"Tool-required prompt used {tool_count} tool call(s).",
        )
    return EvalResult(
        rule_id="tool_required_question_used_tools",
        trace_id=str(trace.get("trace_id", "unknown")),
        status="fail",
        detail="Prompt appears to require live/tool-backed lookup but no tool calls were recorded.",
        suggested_delta="Add or tighten eval coverage for mandatory tool-use questions.",
    )


def _eval_no_canonical_gbrain_writes(trace: dict[str, Any]) -> EvalResult:
    written = [str(path) for path in trace.get("paths_written", [])]
    violations = [path for path in written if _is_canonical_gbrain_path(path)]
    if not violations:
        return EvalResult(
            rule_id="no_canonical_gbrain_writes_by_hermes",
            trace_id=str(trace.get("trace_id", "unknown")),
            status="pass",
            detail="No canonical gbrain writes detected.",
        )
    return EvalResult(
        rule_id="no_canonical_gbrain_writes_by_hermes",
        trace_id=str(trace.get("trace_id", "unknown")),
        status="fail",
        detail="canonical write: " + ", ".join(violations),
        suggested_delta="Route writes to gbrain/hermes/* or Dobby.",
    )


def _eval_citation_quality_for_gbrain_claims(trace: dict[str, Any]) -> EvalResult:
    output = str(trace.get("final_response_summary") or "")
    used_gbrain = any(str(name).startswith("mcp_gbrain_") for name in trace.get("tool_calls", []))
    touched_gbrain_path = any("/kb/" in str(path) or _looks_like_gbrain_slug(str(path)) for path in trace.get("paths_read", []))
    if not used_gbrain and not touched_gbrain_path:
        return EvalResult(
            rule_id="citation_quality_for_gbrain_claims",
            trace_id=_trace_id(trace),
            status="pass",
            detail="No gbrain-backed answer detected.",
        )
    if _has_gbrain_citation(output):
        return EvalResult(
            rule_id="citation_quality_for_gbrain_claims",
            trace_id=_trace_id(trace),
            status="pass",
            detail="gbrain-backed answer includes a slug-like citation.",
        )
    return EvalResult(
        rule_id="citation_quality_for_gbrain_claims",
        trace_id=_trace_id(trace),
        status="fail",
        detail="gbrain-backed answer did not include a visible slug citation.",
        suggested_delta="Require gbrain-backed answers to cite the relevant page slug inline.",
    )


def _eval_gbrain_vs_filesystem_routing(trace: dict[str, Any]) -> EvalResult:
    paths_read = [str(path) for path in trace.get("paths_read", [])]
    tool_calls = [str(name) for name in trace.get("tool_calls", [])]
    kb_markdown_reads = [path for path in paths_read if "/kb/" in path and path.endswith(".md")]
    used_gbrain_tool = any(name.startswith("mcp_gbrain_") for name in tool_calls)
    if kb_markdown_reads and not used_gbrain_tool:
        return EvalResult(
            rule_id="gbrain_vs_filesystem_routing",
            trace_id=_trace_id(trace),
            status="fail",
            detail="Read mirrored KB markdown directly without using gbrain MCP: " + ", ".join(kb_markdown_reads),
            suggested_delta="Use gbrain MCP first for gbrain-backed factual lookup; filesystem mirror is fallback/verification.",
        )
    return EvalResult(
        rule_id="gbrain_vs_filesystem_routing",
        trace_id=_trace_id(trace),
        status="pass",
        detail="No improper KB mirror routing detected.",
    )


def _eval_telegram_formatting(trace: dict[str, Any]) -> EvalResult:
    if str(trace.get("source") or "") != "telegram":
        return EvalResult(
            rule_id="telegram_formatting",
            trace_id=_trace_id(trace),
            status="pass",
            detail="Trace is not from Telegram.",
        )
    output = str(trace.get("final_response_summary") or "")
    if _looks_like_markdown_table(output):
        return EvalResult(
            rule_id="telegram_formatting",
            trace_id=_trace_id(trace),
            status="fail",
            detail="Telegram response appears to contain a markdown table.",
            suggested_delta="Use bullets/key-value lines instead of pipe tables for Telegram output.",
        )
    return EvalResult(
        rule_id="telegram_formatting",
        trace_id=_trace_id(trace),
        status="pass",
        detail="No Telegram table formatting issue detected.",
    )


def _eval_domain_handoff_to_dobby(trace: dict[str, Any]) -> EvalResult:
    prompt = str(trace.get("user_prompt_summary") or "")
    output = str(trace.get("final_response_summary") or "")
    life_lane = re.search(r"\b(family|calendar|health|doctor|medical|wife|kid|child|personal)\b", prompt, re.IGNORECASE)
    if not life_lane:
        return EvalResult(
            rule_id="domain_handoff_to_dobby",
            trace_id=_trace_id(trace),
            status="pass",
            detail="Prompt does not look like Dobby life-lane work.",
        )
    if "@RickClevBot" in output or "Dobby" in output:
        return EvalResult(
            rule_id="domain_handoff_to_dobby",
            trace_id=_trace_id(trace),
            status="pass",
            detail="Life-lane prompt appears to hand off to Dobby.",
        )
    return EvalResult(
        rule_id="domain_handoff_to_dobby",
        trace_id=_trace_id(trace),
        status="fail",
        detail="Life-lane prompt did not visibly hand off to Dobby.",
        suggested_delta="For family/calendar/health/personal requests, acknowledge and route to @RickClevBot.",
    )


def _eval_cross_profile_write_guard(trace: dict[str, Any]) -> EvalResult:
    details = trace.get("tool_call_details", [])
    violations: list[str] = []
    for call in details if isinstance(details, list) else []:
        if not isinstance(call, dict):
            continue
        raw_arguments = call.get("arguments")
        arguments = raw_arguments if isinstance(raw_arguments, dict) else {}
        path = str(arguments.get("path") or arguments.get("file_path") or "")
        if arguments.get("cross_profile") is True or "/.hermes/profiles/" in path:
            violations.append(path or str(call.get("name") or "unknown"))
    if violations:
        return EvalResult(
            rule_id="cross_profile_write_guard",
            trace_id=_trace_id(trace),
            status="fail",
            detail="Cross-profile write or override detected: " + ", ".join(violations),
            suggested_delta="Do not modify another Hermes profile unless the user explicitly directed that exact cross-profile edit.",
        )
    return EvalResult(
        rule_id="cross_profile_write_guard",
        trace_id=_trace_id(trace),
        status="pass",
        detail="No cross-profile write override detected.",
    )


def _trace_id(trace: dict[str, Any]) -> str:
    return str(trace.get("trace_id") or trace.get("session_id") or "unknown")


def _looks_like_gbrain_slug(value: str) -> bool:
    normalized = value.lstrip("/")
    return normalized.startswith((HERMES_GBRAIN_PREFIX, *CANONICAL_GBRAIN_PREFIXES))


def _has_gbrain_citation(output: str) -> bool:
    return bool(re.search(r"`?(gbrain/hermes/|entities/|people/|projects/|wiki/|daily/|media/|tech/)[^`\s),.;:]+`?", output))


def _looks_like_markdown_table(output: str) -> bool:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if any(line.startswith("|") and line.endswith("|") for line in lines):
        return True
    return output.count("|") >= 2


def _is_canonical_gbrain_path(path: str) -> bool:
    normalized = path.lstrip("/")
    if normalized.startswith(HERMES_GBRAIN_PREFIX):
        return False
    return normalized.startswith(CANONICAL_GBRAIN_PREFIXES)


def _iso_from_timestamp(timestamp: float | int | None) -> str:
    if timestamp is None:
        return ""
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat()


def _summarize(text: str, limit: int = 240) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


if __name__ == "__main__":
    raise SystemExit(main())
