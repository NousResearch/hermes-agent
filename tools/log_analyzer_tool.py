"""Log analyzer tool.

Reads Hermes log files from the profile-aware logs directory and returns
a structured summary. No external dependencies; uses stdlib plus the local
`tools.validation_helper` envelope helpers.

Success envelope contains:
* recent_error_counts: per-severity tallies for recent lines
* top_error_warning_patterns: most common normalized error/warning message buckets
* recent_log_tail: capped slice of recent representative log lines with timestamps

Failure envelope uses code ``log_analyzer_error`` when logs are unavailable
or unreadable.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.validation_helper import relay_failure, relay_success

logger = logging.getLogger(__name__)

_LOG_FILES = ("agent.log", "errors.log", "gateway.log")
_ERROR_CODE = "log_analyzer_error"
_RECENT_TAIL_CAP = 100
_TOP_PATTERN_LIMIT = 10

_LOG_TS_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[\.,]\d+)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?\s*"
)
_LOG_BRACKET_RE = re.compile(r"^\[[^\]]*\]\s*")
_LOG_LEVEL_RE = re.compile(
    r"^(?:ERROR|WARNING|WARN|INFO|DEBUG|CRITICAL)\b[:\s]*", re.IGNORECASE
)
_MULTI_WS_RE = re.compile(r"\s+")
_SEVERITY_RE = re.compile(
    r"\b(error|warn(?:ing)?|critical|fatal)\b", re.IGNORECASE
)


def _default_logs_dir() -> Optional[Path]:
    """Return the default Hermes logs directory, or None on resolution failure."""
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home()).resolve() / "logs"
    except (OSError, ImportError, AttributeError) as exc:
        logger.debug("Failed to resolve default logs directory: %s", exc)
        return None


def _read_lines(path: Path) -> List[str]:
    """Read all lines from a single file, preserving raw text."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.debug("Failed to read log file %s: %s", path, exc)
        return []
    return text.splitlines()


def _collect_lines(logs_dir: Path) -> List[str]:
    """Read all known log files and concatenate their lines in file order."""
    lines: List[str] = []
    for name in _LOG_FILES:
        candidate = logs_dir / name
        if candidate.is_file():
            lines.extend(_read_lines(candidate))
    return lines


def _bucket_pattern(line: str) -> str:
    """Normalize a log line to a string bucket for grouping."""
    s = line
    s = _LOG_TS_RE.sub("", s)
    s = _LOG_BRACKET_RE.sub("", s)
    s = _LOG_LEVEL_RE.sub("", s)
    s = s.strip()
    s = _MULTI_WS_RE.sub(" ", s)
    return s.lower()


def _severity_counts(lines: List[str]) -> Dict[str, int]:
    """Return a simple severity tally from raw log lines."""
    counts: Dict[str, int] = {}
    for line in lines:
        match = _SEVERITY_RE.search(line)
        if not match:
            continue
        key = match.group(1).lower()
        if key in {"warn", "warning"}:
            key = "warning"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _top_error_warning_patterns(lines: List[str]) -> List[Dict[str, Any]]:
    """Return the most common error/warning message patterns."""
    buckets: Counter = Counter()
    for line in lines:
        if not _SEVERITY_RE.search(line):
            continue
        severity_match = _SEVERITY_RE.search(line)
        if severity_match:
            sev = severity_match.group(1).lower()
            if sev in {"warn", "warning", "error", "critical", "fatal"}:
                buckets[_bucket_pattern(line)] += 1
    top = buckets.most_common(_TOP_PATTERN_LIMIT)
    return [{"pattern": pattern, "count": count} for pattern, count in top]


def _build_success_payload(lines: List[str]) -> Dict[str, Any]:
    tail = lines[-_RECENT_TAIL_CAP:] if lines else []
    return {
        "recent_error_counts": _severity_counts(lines),
        "top_error_warning_patterns": _top_error_warning_patterns(lines),
        "recent_log_tail": tail,
    }


def summarize_logs() -> Dict[str, Any]:
    """Read Hermes logs and return a structured summary.

    Returns a zodern:relay-style success envelope on success, or a failure
    envelope when logs are unavailable or unreadable.
    """
    logs_dir = _default_logs_dir()
    if logs_dir is None or not logs_dir.is_dir():
        return relay_failure(
            "Hermes logs directory is unavailable or unreadable",
            code=_ERROR_CODE,
        )

    try:
        lines = _collect_lines(logs_dir)
    except OSError as exc:
        return relay_failure(
            f"Hermes logs could not be read ({exc})",
            code=_ERROR_CODE,
        )
    except Exception as exc:
        logger.debug("Unexpected failure reading Hermes logs: %s", exc)
        return relay_failure(
            f"Hermes logs could not be read ({exc})",
            code=_ERROR_CODE,
        )

    return relay_success(_build_success_payload(lines))


def check_log_analyzer_requirements() -> bool:
    """Return False when logs are not available for analysis."""
    logs_dir = _default_logs_dir()
    if logs_dir is None or not logs_dir.is_dir():
        return False
    try:
        for name in _LOG_FILES:
            candidate = logs_dir / name
            if candidate.exists():
                candidate.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return True


def log_analyzer_tool(_args: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Tool handler wrapping :func:`summarize_logs`."""
    try:
        return summarize_logs()
    except Exception as exc:
        logger.debug("log_analyzer tool failed unexpectedly: %s", exc)
        return relay_failure(
            f"Unexpected log analyzer failure: {exc}",
            code=_ERROR_CODE,
        )


LOG_ANALYZER_SCHEMA: Dict[str, Any] = {
    "name": "log_analyzer",
    "description": (
        "Return a structured summary of recent Hermes logs: per-severity counts, "
        "top error/warning patterns, and a capped tail of recent log lines with timestamps. "
        "No external APIs are used."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


def register(registry):  # pragma: no cover - registration side-effect
    registry.register(
        name="log_analyzer",
        toolset="ops",
        schema=LOG_ANALYZER_SCHEMA,
        handler=log_analyzer_tool,
        check_fn=check_log_analyzer_requirements,
        emoji="📄",
    )
