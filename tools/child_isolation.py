#!/usr/bin/env python3
"""
Child Agent Isolation — Structured error types for delegation.

Provides typed error taxonomy and user-facing formatting used by
_run_single_child() in delegate_tool.py.
"""

import enum
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


class ChildErrorType(str, enum.Enum):
    TIMEOUT = "timeout"
    CRASH = "crash"
    INTERRUPTED = "interrupted"
    INTERNAL_ERROR = "internal_error"
    DEPTH_LIMIT = "depth_limit"
    PAUSED = "paused"


@dataclass
class ChildResult:
    """Structured result from a child agent execution."""

    success: bool
    task_index: int = 0
    status: str = "completed"
    summary: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    api_calls: int = 0
    duration_seconds: float = 0.0
    child_role: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def format_child_error(result: ChildResult) -> str:
    """Format a child error into a user-facing message.

    Args:
        result: ChildResult with error information

    Returns:
        Formatted error string
    """
    if result.success:
        return ""

    messages = {
        ChildErrorType.TIMEOUT.value: (
            f"Subagent timed out after {result.duration_seconds:.0f}s. "
            f"Increase delegation.child_timeout_seconds in config.yaml "
            f"(current: {result.duration_seconds:.0f}s) if tasks consistently need more time."
        ),
        ChildErrorType.CRASH.value: (
            f"Subagent crashed: {result.error or 'Unknown error'}. "
            f"The parent agent was not affected."
        ),
        ChildErrorType.INTERRUPTED.value: (
            f"Subagent was interrupted by parent."
        ),
        ChildErrorType.DEPTH_LIMIT.value: (
            f"Delegation depth limit reached. Increase delegation.max_spawn_depth in config.yaml."
        ),
    }

    return messages.get(result.error_type or "", result.error or "Unknown error")
