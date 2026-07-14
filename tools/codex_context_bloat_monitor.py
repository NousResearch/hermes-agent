#!/usr/bin/env python3
"""Scan local Codex threads for context-bloat incidents.

The monitor is intentionally read-only. It combines the Codex thread index
(``state_5.sqlite``) with rollout JSONL files and reports threads that exceed
the guardrails used for Hermes/Codex UI verification work.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CODEX_HOME = Path.home() / ".codex"
DEFAULT_THRESHOLDS = {
    "tokens_used": 800_000,
    "last_input": 60_000,
    "computer_use_inline_chars": 8_000,
    "image_base64_chars": 1_500,
}

_DATA_IMAGE_RE = re.compile(r"data:image/[A-Za-z0-9.+-]+;base64,([A-Za-z0-9+/=_-]+)")
_MAX_TITLE_CHARS = 160
_COMPUTER_USE_TOOLS = {"get_app_state", "computer_use", "click", "type_text", "list_apps"}
_THREAD_READ_TOOLS = {"read_thread", "list_threads"}
_SHELL_TOOLS = {"exec_command", "write_stdin"}


def _short_text(text: Any, max_chars: int = _MAX_TITLE_CHARS) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "..."


@dataclass
class ThreadBloatMetrics:
    thread_id: str
    title: str
    tokens_used: int
    rollout_path: str
    last_input_max: int = 0
    last_input_avg: int = 0
    token_count_events: int = 0
    total_input_tokens: int = 0
    total_cached_input_tokens: int = 0
    total_tokens_used: int = 0
    tool_call_count: int = 0
    tool_output_count: int = 0
    tool_output_max_chars: int = 0
    computer_use_output_max_chars: int = 0
    thread_read_output_max_chars: int = 0
    shell_output_max_chars: int = 0
    image_base64_chars: int = 0

    def incidents(self, thresholds: dict[str, int] | None = None) -> list[str]:
        t = thresholds or DEFAULT_THRESHOLDS
        out: list[str] = []
        if max(self.tokens_used, self.total_tokens_used) > t["tokens_used"]:
            out.append("tokens_used")
        if self.last_input_max > t["last_input"]:
            out.append("last_input")
        if self.computer_use_output_max_chars > t["computer_use_inline_chars"]:
            out.append("computer_use_inline_chars")
        if self.image_base64_chars > t["image_base64_chars"]:
            out.append("image_base64_chars")
        return out


def _iter_thread_rows(db_path: Path, *, limit: int) -> Iterable[dict[str, Any]]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, title, tokens_used, rollout_path
            FROM threads
            WHERE archived = 0
            ORDER BY updated_at_ms DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _payload_item(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return {}
    item = payload.get("item")
    if isinstance(item, dict):
        return item
    return payload


def _event_item_type(event: dict[str, Any]) -> str:
    item = _payload_item(event)
    return str(item.get("type") or "")


def _event_tool_name(event: dict[str, Any]) -> str:
    item = _payload_item(event)
    return str(item.get("name") or "")


def _event_call_id(event: dict[str, Any]) -> str:
    item = _payload_item(event)
    return str(item.get("call_id") or "")


def _event_output(event: dict[str, Any]) -> str:
    item = _payload_item(event)
    output = item.get("output")
    if isinstance(output, str):
        return output
    return _json_text(output)


def _extract_token_usage(event: dict[str, Any]) -> tuple[int, int, int, int] | None:
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None

    if event.get("type") == "event_msg" and payload.get("type") == "token_count":
        info = payload.get("info")
        if not isinstance(info, dict):
            return None
        last = info.get("last_token_usage")
        total = info.get("total_token_usage")
        if not isinstance(last, dict) or not isinstance(total, dict):
            return None
        return (
            int(last.get("input_tokens") or 0),
            int(total.get("input_tokens") or 0),
            int(total.get("cached_input_tokens") or 0),
            int(total.get("total_tokens") or 0),
        )

    # Legacy/best-effort support for older synthetic fixtures. Do not scan for
    # generic input_tokens here, because that confuses cumulative usage with a
    # single-turn input size in current Codex rollouts.
    if "last_input" in payload:
        last_input = int(payload.get("last_input") or 0)
        return (last_input, 0, 0, 0)
    if "last_input" in event:
        last_input = int(event.get("last_input") or 0)
        return (last_input, 0, 0, 0)
    return None


def _scan_rollout(path: Path) -> dict[str, int]:
    call_names: dict[str, str] = {}
    metrics = {
        "last_input_max": 0,
        "last_input_sum": 0,
        "token_count_events": 0,
        "total_input_tokens": 0,
        "total_cached_input_tokens": 0,
        "total_tokens_used": 0,
        "tool_call_count": 0,
        "tool_output_count": 0,
        "tool_output_max_chars": 0,
        "computer_use_output_max_chars": 0,
        "thread_read_output_max_chars": 0,
        "shell_output_max_chars": 0,
        "image_base64_chars": 0,
    }
    last_input_max = 0
    if not path.exists():
        return metrics

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            metrics["image_base64_chars"] += sum(
                len(m.group(1)) for m in _DATA_IMAGE_RE.finditer(line)
            )
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            usage = _extract_token_usage(event)
            if usage is not None:
                last_input, total_input, cached_input, total_tokens = usage
                metrics["token_count_events"] += 1
                metrics["last_input_sum"] += last_input
                metrics["last_input_max"] = max(metrics["last_input_max"], last_input)
                metrics["total_input_tokens"] = max(metrics["total_input_tokens"], total_input)
                metrics["total_cached_input_tokens"] = max(
                    metrics["total_cached_input_tokens"], cached_input
                )
                metrics["total_tokens_used"] = max(metrics["total_tokens_used"], total_tokens)

            if event.get("type") != "response_item":
                continue

            item_type = _event_item_type(event)
            if item_type in {"function_call", "custom_tool_call"}:
                call_id = _event_call_id(event)
                if call_id:
                    call_names[call_id] = _event_tool_name(event)
                metrics["tool_call_count"] += 1
                continue

            if item_type not in {"function_call_output", "custom_tool_call_output"}:
                continue

            output_len = len(_event_output(event))
            tool_name = call_names.get(_event_call_id(event), "")
            metrics["tool_output_count"] += 1
            metrics["tool_output_max_chars"] = max(metrics["tool_output_max_chars"], output_len)
            if tool_name in _COMPUTER_USE_TOOLS:
                metrics["computer_use_output_max_chars"] = max(
                    metrics["computer_use_output_max_chars"], output_len
                )
            elif tool_name in _THREAD_READ_TOOLS:
                metrics["thread_read_output_max_chars"] = max(
                    metrics["thread_read_output_max_chars"], output_len
                )
            elif tool_name in _SHELL_TOOLS:
                metrics["shell_output_max_chars"] = max(
                    metrics["shell_output_max_chars"], output_len
                )

    return metrics


def collect_metrics(
    codex_home: Path = DEFAULT_CODEX_HOME,
    *,
    limit: int = 50,
    thread_id: str | None = None,
    rollout_path: Path | None = None,
) -> list[ThreadBloatMetrics]:
    if rollout_path is not None:
        rollout_metrics = _scan_rollout(rollout_path)
        events = rollout_metrics["token_count_events"]
        return [
            ThreadBloatMetrics(
                thread_id=thread_id or "",
                title="rollout",
                tokens_used=rollout_metrics["total_tokens_used"],
                rollout_path=str(rollout_path),
                last_input_max=rollout_metrics["last_input_max"],
                last_input_avg=(
                    rollout_metrics["last_input_sum"] // events if events else 0
                ),
                token_count_events=events,
                total_input_tokens=rollout_metrics["total_input_tokens"],
                total_cached_input_tokens=rollout_metrics["total_cached_input_tokens"],
                total_tokens_used=rollout_metrics["total_tokens_used"],
                tool_call_count=rollout_metrics["tool_call_count"],
                tool_output_count=rollout_metrics["tool_output_count"],
                tool_output_max_chars=rollout_metrics["tool_output_max_chars"],
                computer_use_output_max_chars=rollout_metrics[
                    "computer_use_output_max_chars"
                ],
                thread_read_output_max_chars=rollout_metrics[
                    "thread_read_output_max_chars"
                ],
                shell_output_max_chars=rollout_metrics["shell_output_max_chars"],
                image_base64_chars=rollout_metrics["image_base64_chars"],
            )
        ]

    db_path = codex_home / "state_5.sqlite"
    metrics: list[ThreadBloatMetrics] = []
    for row in _iter_thread_rows(db_path, limit=limit):
        if thread_id and row.get("id") != thread_id:
            continue
        rollout = Path(row.get("rollout_path") or "")
        rollout_metrics = _scan_rollout(rollout)
        events = rollout_metrics["token_count_events"]
        metrics.append(
            ThreadBloatMetrics(
                thread_id=str(row.get("id") or ""),
                title=_short_text(row.get("title") or ""),
                tokens_used=int(row.get("tokens_used") or 0),
                rollout_path=str(rollout),
                last_input_max=rollout_metrics["last_input_max"],
                last_input_avg=(
                    rollout_metrics["last_input_sum"] // events if events else 0
                ),
                token_count_events=events,
                total_input_tokens=rollout_metrics["total_input_tokens"],
                total_cached_input_tokens=rollout_metrics["total_cached_input_tokens"],
                total_tokens_used=rollout_metrics["total_tokens_used"],
                tool_call_count=rollout_metrics["tool_call_count"],
                tool_output_count=rollout_metrics["tool_output_count"],
                tool_output_max_chars=rollout_metrics["tool_output_max_chars"],
                computer_use_output_max_chars=rollout_metrics[
                    "computer_use_output_max_chars"
                ],
                thread_read_output_max_chars=rollout_metrics[
                    "thread_read_output_max_chars"
                ],
                shell_output_max_chars=rollout_metrics["shell_output_max_chars"],
                image_base64_chars=rollout_metrics["image_base64_chars"],
            )
        )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codex-home", type=Path, default=DEFAULT_CODEX_HOME)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--thread-id")
    parser.add_argument("--rollout-path", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    metrics = collect_metrics(
        args.codex_home,
        limit=args.limit,
        thread_id=args.thread_id,
        rollout_path=args.rollout_path,
    )
    incidents = [
        {**asdict(item), "incident_reasons": item.incidents()}
        for item in metrics
        if item.incidents()
    ]
    if args.json:
        print(json.dumps({"incidents": incidents}, ensure_ascii=False, indent=2))
    else:
        for item in incidents:
            print(
                f"context-bloat incident thread={item['thread_id']} "
                f"tokens={item['tokens_used']} reasons={','.join(item['incident_reasons'])} "
                f"rollout={item['rollout_path']}"
            )
    return 1 if incidents else 0


if __name__ == "__main__":
    raise SystemExit(main())
