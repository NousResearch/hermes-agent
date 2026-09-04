"""Parse live subagent transcripts into structured event streams.

The live transcript format (see tools/delegation_live_log.py) is one
`HH:MM:SS role| content` line per event. ``action="tail"`` in
``delegate_task`` returns the raw lines; ``action="stream"`` returns the
same data parsed into typed event records the parent can dispatch on
(grep "ERROR", filter by tool name, count iterations, etc.) without a
regex.

This module is intentionally pure:
- No I/O. Pass an iterable of lines in, get a list of events back.
- No state. Each call returns an independent list.
- No reliance on agent internals — the parser only knows the live-log
  format. Safe to import from CLI, tests, or any UI consumer.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional


# Lines look like:  "14:22:08 tool    | -> terminal({\"command\": ...})"
# The 9-char role field is left-padded to align the pipe.
_LINE_RE = re.compile(
    r"^(?P<ts>\d{2}:\d{2}:\d{2})\s+(?P<role>\S+)\s*\|\s*(?P<content>.*)$"
)

# Recognised roles (live-log vocabulary — keep in sync with delegation_live_log.py).
_ROLES = frozenset({"user", "assistant", "think", "tool", "result", "final", "start"})

# "-> name(args)"  — emitted by tool_start()
_TOOL_START_RE = re.compile(r"^->\s*(?P<name>[^(\\s]+)(?:\((?P<args>.*)\))?\s*$")

# "name status duration: preview"  — emitted by tool_result()
# duration is optional; status is "ok" / "ERROR"
_TOOL_RESULT_RE = re.compile(
    r"^(?P<name>[^\s]+)\s+(?P<status>ok|ERROR)(?:\s+(?P<duration>[0-9.]+)s)?:\s*(?P<result>.*)$"
)


@dataclass
class StreamEvent:
    """One parsed event from the live transcript.

    Fields are intentionally wide so every role can be represented; most
    consumers will switch on ``kind`` and ignore the rest.
    """

    index: int                          # 0-based position from start of file
    ts: Optional[str]                   # "HH:MM:SS" or None for unparseable
    role: str                           # raw role label from the log
    kind: str                           # normalised: kickoff/assistant/thinking/tool_start/tool_result/marker/raw
    text: str                           # the raw line content (sans role prefix)
    tool_name: Optional[str] = None     # for tool_start / tool_result
    tool_args: Optional[str] = None     # for tool_start, args preview
    tool_status: Optional[str] = None   # "ok" / "ERROR" for tool_result
    tool_duration_seconds: Optional[float] = None
    tool_result_preview: Optional[str] = None
    is_error: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_event(index: int, raw_line: str) -> StreamEvent:
    """Parse one log line. Never raises; unparseable lines become kind='raw'."""
    line = raw_line.rstrip("\n").rstrip("\r")
    m = _LINE_RE.match(line)
    if not m:
        # Header line, blank line, or malformed input — keep the text but mark raw.
        return StreamEvent(
            index=index,
            ts=None,
            role="",
            kind="raw",
            text=line,
        )
    ts = m.group("ts")
    role = m.group("role")
    content = m.group("content")
    kind = _normalise_role(role)
    ev = StreamEvent(index=index, ts=ts, role=role, kind=kind, text=content)

    if kind == "tool_start":
        mm = _TOOL_START_RE.match(content)
        if mm:
            ev.tool_name = mm.group("name")
            ev.tool_args = mm.group("args") or None
    elif kind == "tool_result":
        mm = _TOOL_RESULT_RE.match(content)
        if mm:
            ev.tool_name = mm.group("name")
            ev.tool_status = mm.group("status")
            ev.is_error = mm.group("status") == "ERROR"
            dur = mm.group("duration")
            if dur:
                try:
                    ev.tool_duration_seconds = float(dur)
                except ValueError:
                    pass
            ev.tool_result_preview = mm.group("result")

    return ev


def _normalise_role(role: str) -> str:
    """Map the live-log role to a stable ``kind`` for downstream switching."""
    r = (role or "").strip().lower()
    if r == "user":
        return "kickoff"
    if r == "assistant":
        return "assistant"
    if r == "think":
        return "thinking"
    if r == "tool":
        return "tool_start"
    if r == "result":
        return "tool_result"
    if r in {"final", "start"}:
        return "marker"
    return "raw"


def parse_lines(lines: Iterable[str]) -> List[StreamEvent]:
    """Parse an iterable of lines (typically the tail of a log file)."""
    out: List[StreamEvent] = []
    for i, ln in enumerate(lines):
        out.append(parse_event(i, ln))
    return out


def filter_events(
    events: List[StreamEvent],
    *,
    kinds: Optional[List[str]] = None,
    tool_name: Optional[str] = None,
    errors_only: bool = False,
) -> List[StreamEvent]:
    """Convenience: subset events by kind / tool / error filter."""
    out = events
    if kinds is not None:
        wanted = set(kinds)
        out = [e for e in out if e.kind in wanted]
    if tool_name is not None:
        out = [e for e in out if e.tool_name == tool_name]
    if errors_only:
        out = [e for e in out if e.is_error]
    return out


def summarise(events: List[StreamEvent]) -> Dict[str, Any]:
    """Aggregate stats over an event stream."""
    tool_calls: Dict[str, int] = {}
    tool_errors: Dict[str, int] = {}
    total_duration: float = 0.0
    for e in events:
        if e.kind == "tool_start" and e.tool_name:
            tool_calls[e.tool_name] = tool_calls.get(e.tool_name, 0) + 1
        if e.kind == "tool_result" and e.tool_name:
            if e.is_error:
                tool_errors[e.tool_name] = tool_errors.get(e.tool_name, 0) + 1
            if e.tool_duration_seconds is not None:
                total_duration += e.tool_duration_seconds
    return {
        "event_count": len(events),
        "tool_call_count": sum(tool_calls.values()),
        "tool_error_count": sum(tool_errors.values()),
        "tool_call_breakdown": tool_calls,
        "tool_error_breakdown": tool_errors,
        "total_tool_duration_seconds": round(total_duration, 3),
    }
