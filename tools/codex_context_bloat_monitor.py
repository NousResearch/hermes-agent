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
_COMPUTER_USE_RE = re.compile(r"\b(get_app_state|computer_use|click|type_text)\b", re.I)
_MAX_TITLE_CHARS = 160


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
    computer_use_output_max_chars: int = 0
    image_base64_chars: int = 0

    def incidents(self, thresholds: dict[str, int] | None = None) -> list[str]:
        t = thresholds or DEFAULT_THRESHOLDS
        out: list[str] = []
        if self.tokens_used > t["tokens_used"]:
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


def _extract_last_input(event: dict[str, Any], raw_line: str) -> int:
    text = _json_text(event)
    candidates = []
    for pattern in (
        r'"last_input"\s*:\s*(\d+)',
        r'last_input[=: ]+(\d+)',
        r'"input_tokens"\s*:\s*(\d+)',
    ):
        candidates.extend(int(m.group(1)) for m in re.finditer(pattern, text))
    if not candidates:
        candidates.extend(int(m.group(1)) for m in re.finditer(r"last_input[=: ]+(\d+)", raw_line))
    return max(candidates or [0])


def _scan_rollout(path: Path) -> tuple[int, int, int]:
    last_input_max = 0
    computer_use_output_max = 0
    image_base64_chars = 0
    if not path.exists():
        return (0, 0, 0)

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            image_base64_chars += sum(len(m.group(1)) for m in _DATA_IMAGE_RE.finditer(line))
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                event = {}
            last_input_max = max(last_input_max, _extract_last_input(event, line))
            if _COMPUTER_USE_RE.search(line):
                computer_use_output_max = max(computer_use_output_max, len(line))
    return (last_input_max, computer_use_output_max, image_base64_chars)


def collect_metrics(codex_home: Path = DEFAULT_CODEX_HOME, *, limit: int = 50) -> list[ThreadBloatMetrics]:
    db_path = codex_home / "state_5.sqlite"
    metrics: list[ThreadBloatMetrics] = []
    for row in _iter_thread_rows(db_path, limit=limit):
        rollout = Path(row.get("rollout_path") or "")
        last_input, computer_use_chars, image_chars = _scan_rollout(rollout)
        metrics.append(
            ThreadBloatMetrics(
                thread_id=str(row.get("id") or ""),
                title=_short_text(row.get("title") or ""),
                tokens_used=int(row.get("tokens_used") or 0),
                rollout_path=str(rollout),
                last_input_max=last_input,
                computer_use_output_max_chars=computer_use_chars,
                image_base64_chars=image_chars,
            )
        )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codex-home", type=Path, default=DEFAULT_CODEX_HOME)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    metrics = collect_metrics(args.codex_home, limit=args.limit)
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
