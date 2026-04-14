"""
Session-scoped taint tracking for tool execution.

Marks a session as tainted when tool results originate from
untrusted external sources (web, inbound messages, MCP servers).
Taint propagates forward: once set in a session, all subsequent
writes to persistent agent state (skills, memory) require approval.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class TaintSource(str, Enum):
    WEB_FETCH    = "web_fetch"
    WEB_SEARCH   = "web_search"
    INBOUND_MSG  = "inbound_message"
    MCP_RESULT   = "mcp_result"
    SUBAGENT     = "subagent"

@dataclass
class TaintState:
    tainted: bool = False
    sources: list[TaintSource] = field(default_factory=list)
    first_taint_tool: Optional[str] = None   # tool call that introduced taint

    def mark(self, source: TaintSource, tool_name: str) -> None:
        if not self.tainted:
            self.tainted = True
            self.first_taint_tool = tool_name
        if source not in self.sources:
            self.sources.append(source)

    def summary(self) -> str:
        if not self.tainted:
            return "clean"
        srcs = ", ".join(s.value for s in self.sources)
        return f"tainted (sources: {srcs}, first at: {self.first_taint_tool})"


# Per-session taint state, keyed by session_id.
# Thread-safe: each session_id maps to its own TaintState.
_lock = threading.Lock()
_session_taint: dict[str, TaintState] = {}


def get_taint(session_id: str) -> TaintState:
    with _lock:
        if session_id not in _session_taint:
            _session_taint[session_id] = TaintState()
        return _session_taint[session_id]


def mark_tainted(session_id: str, source: TaintSource, tool_name: str) -> None:
    get_taint(session_id).mark(source, tool_name)


def is_tainted(session_id: str) -> bool:
    return get_taint(session_id).tainted


def clear_taint(session_id: str) -> None:
    """Called on /new or /reset."""
    with _lock:
        _session_taint.pop(session_id, None)
