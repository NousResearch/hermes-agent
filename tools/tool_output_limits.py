"""Configurable tool-output truncation limits.

Ported from anomalyco/opencode PR #23770 (``feat(truncate): allow
configuring tool output truncation limits``).

OpenCode hardcoded ``MAX_LINES = 2000`` and ``MAX_BYTES = 50 * 1024``
as tool-output truncation thresholds. Hermes-agent had the same
hardcoded constants in two places:

* ``tools/terminal_tool.py`` — ``MAX_OUTPUT_CHARS = 50000`` (terminal
  stdout/stderr cap)
* ``tools/file_operations.py`` — ``MAX_LINES = 2000`` /
  ``MAX_LINE_LENGTH = 2000`` (read_file pagination cap + per-line cap)

This module centralises those values behind a single config section
(``tool_output`` in ``config.yaml``) so power users can tune them
without patching the source. The existing hardcoded numbers remain as
defaults, so behaviour is unchanged when the config key is absent.

Example ``config.yaml``::

    tool_output:
      max_bytes: 100000        # terminal output cap (chars)
      max_lines: 5000          # read_file pagination + truncation cap
      max_line_length: 2000    # per-line length cap before '... [truncated]'

The limits reader is defensive: any error (missing config file, invalid
value type, etc.) falls back to the built-in defaults so tools never
fail because of a malformed config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Hardcoded defaults — these match the pre-existing values, so adding
# this module is behaviour-preserving for users who don't set
# ``tool_output`` in config.yaml.
DEFAULT_MAX_BYTES = 50_000       # terminal_tool.MAX_OUTPUT_CHARS
DEFAULT_MAX_LINES = 2000         # file_operations.MAX_LINES
DEFAULT_MAX_LINE_LENGTH = 2000   # file_operations.MAX_LINE_LENGTH
DEFAULT_SEARCH_RESULTS = 50

COMPACT_MAX_BYTES = 12_000
COMPACT_MAX_LINES = 200
COMPACT_SEARCH_RESULTS = 25
SYNTHESIZER_MAX_BYTES = 8_000
SYNTHESIZER_MAX_LINES = 160
SYNTHESIZER_SEARCH_RESULTS = 20
MONITOR_MAX_BYTES = 6_000
MONITOR_MAX_LINES = 120
MONITOR_SEARCH_RESULTS = 20

# Module-level cache — populated on first call.
# Avoids repeated config file I/O on every tool call.
_cached_limits: dict | None = None


@dataclass(frozen=True)
class ToolOutputPolicy:
    mode: str
    terminal_max_chars: int
    read_max_lines: int
    search_max_results: int
    compact_terminal_output: bool = False
    require_narrow_reads: bool = False


_MODE_ALIASES = {
    "build": "builder",
    "builder": "builder",
    "worker": "builder",
    "review": "reviewer",
    "reviewer": "reviewer",
    "guardian": "reviewer",
    "audit": "reviewer",
    "synth": "synthesizer",
    "synthesis": "synthesizer",
    "synthesizer": "synthesizer",
    "summary": "synthesizer",
    "summarizer": "synthesizer",
    "humanizer": "synthesizer",
    "monitor": "monitor",
    "status": "monitor",
    "observer": "monitor",
}


def _coerce_positive_int(value: Any, default: int) -> int:
    """Return ``value`` as a positive int, or ``default`` on any issue."""
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return default
    if iv <= 0:
        return default
    return iv


def get_tool_output_limits() -> Dict[str, int]:
    """Return resolved tool-output limits, reading ``tool_output`` from config.

    Keys: ``max_bytes``, ``max_lines``, ``max_line_length``. Missing or
    invalid entries fall through to the ``DEFAULT_*`` constants. This
    function NEVER raises.

    Result is cached for the process lifetime to avoid repeated disk I/O
    on every tool call. Call ``_reset_tool_output_limits_cache()`` in
    tests that need a fresh read after config changes.
    """
    global _cached_limits
    if _cached_limits is not None:
        return _cached_limits
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        section = cfg.get("tool_output") if isinstance(cfg, dict) else None
        if not isinstance(section, dict):
            section = {}
    except Exception:
        section = {}

    _cached_limits = {
        "max_bytes": _coerce_positive_int(section.get("max_bytes"), DEFAULT_MAX_BYTES),
        "max_lines": _coerce_positive_int(section.get("max_lines"), DEFAULT_MAX_LINES),
        "max_line_length": _coerce_positive_int(
            section.get("max_line_length"), DEFAULT_MAX_LINE_LENGTH
        ),
    }
    return _cached_limits


def resolve_tool_output_mode(mode: Optional[str] = None) -> str:
    """Return the canonical output policy mode for this process."""
    candidates = [
        mode,
        os.environ.get("HERMES_TOOL_OUTPUT_MODE"),
        os.environ.get("HERMES_AGENT_ROLE"),
        os.environ.get("HERMES_KANBAN_ROLE"),
        os.environ.get("HERMES_PROFILE"),
    ]
    for raw in candidates:
        text = str(raw or "").strip().lower()
        if not text:
            continue
        if text in _MODE_ALIASES:
            return _MODE_ALIASES[text]
        if "synth" in text or "summary" in text or "humanizer" in text:
            return "synthesizer"
        if "review" in text or "guardian" in text or "audit" in text:
            return "reviewer"
        if "monitor" in text or "observer" in text or "status" in text:
            return "monitor"
    return "builder"


def get_mode_output_policy(
    mode: Optional[str] = None,
    *,
    usage_guard_active: bool = False,
) -> ToolOutputPolicy:
    """Return assistant-facing output caps for a worker/reviewer mode."""
    limits = get_tool_output_limits()
    canonical = resolve_tool_output_mode(mode)
    if canonical == "synthesizer":
        return ToolOutputPolicy(
            mode=canonical,
            terminal_max_chars=min(limits["max_bytes"], SYNTHESIZER_MAX_BYTES),
            read_max_lines=min(limits["max_lines"], SYNTHESIZER_MAX_LINES),
            search_max_results=SYNTHESIZER_SEARCH_RESULTS,
            compact_terminal_output=True,
            require_narrow_reads=True,
        )
    if canonical == "monitor":
        return ToolOutputPolicy(
            mode=canonical,
            terminal_max_chars=min(limits["max_bytes"], MONITOR_MAX_BYTES),
            read_max_lines=min(limits["max_lines"], MONITOR_MAX_LINES),
            search_max_results=MONITOR_SEARCH_RESULTS,
            compact_terminal_output=True,
            require_narrow_reads=True,
        )
    if usage_guard_active:
        return ToolOutputPolicy(
            mode=canonical,
            terminal_max_chars=min(limits["max_bytes"], COMPACT_MAX_BYTES),
            read_max_lines=min(limits["max_lines"], COMPACT_MAX_LINES),
            search_max_results=COMPACT_SEARCH_RESULTS,
            compact_terminal_output=True,
            require_narrow_reads=True,
        )
    return ToolOutputPolicy(
        mode=canonical,
        terminal_max_chars=limits["max_bytes"],
        read_max_lines=limits["max_lines"],
        search_max_results=DEFAULT_SEARCH_RESULTS,
    )


def _reset_tool_output_limits_cache() -> None:
    """Reset the cached limits — for tests or after config hot-reload."""
    global _cached_limits
    _cached_limits = None


def get_max_bytes() -> int:
    """Shortcut for terminal-tool callers that only need the byte cap."""
    return get_tool_output_limits()["max_bytes"]


def get_max_lines() -> int:
    """Shortcut for file-ops callers that only need the line cap."""
    return get_tool_output_limits()["max_lines"]


def get_max_line_length() -> int:
    """Shortcut for file-ops callers that only need the per-line cap."""
    return get_tool_output_limits()["max_line_length"]
