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

Automation can override those values for one process with
``HERMES_TOOL_OUTPUT_MAX_BYTES``, ``HERMES_TOOL_OUTPUT_MAX_LINES``, and
``HERMES_TOOL_OUTPUT_MAX_LINE_LENGTH``. Environment values take precedence
over config; invalid values fail safe to the built-in defaults.

The limits reader is defensive: any error (missing config file, invalid
value type, etc.) falls back to the built-in defaults so tools never
fail because of a malformed config.
"""

from __future__ import annotations

import os
from typing import Any, Dict

# Hardcoded defaults — these match the pre-existing values, so adding
# this module is behaviour-preserving for users who don't set
# ``tool_output`` in config.yaml.
DEFAULT_MAX_BYTES = 50_000       # terminal_tool.MAX_OUTPUT_CHARS
DEFAULT_MAX_LINES = 2000         # file_operations.MAX_LINES
DEFAULT_MAX_LINE_LENGTH = 2000   # file_operations.MAX_LINE_LENGTH

# Module-level cache — populated on first call.
# Avoids repeated config file I/O on every tool call.
_cached_limits: dict | None = None


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
        "max_bytes": _coerce_positive_int(
            os.getenv("HERMES_TOOL_OUTPUT_MAX_BYTES", section.get("max_bytes")),
            DEFAULT_MAX_BYTES,
        ),
        "max_lines": _coerce_positive_int(
            os.getenv("HERMES_TOOL_OUTPUT_MAX_LINES", section.get("max_lines")),
            DEFAULT_MAX_LINES,
        ),
        "max_line_length": _coerce_positive_int(
            os.getenv(
                "HERMES_TOOL_OUTPUT_MAX_LINE_LENGTH",
                section.get("max_line_length"),
            ),
            DEFAULT_MAX_LINE_LENGTH,
        ),
    }
    return _cached_limits


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
