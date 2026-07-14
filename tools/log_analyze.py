#!/usr/bin/env python3
"""
Log Analyze Tool - Analyze log files

Provides log file analysis: filter by level, search patterns, and aggregate statistics.
"""

import json
import os
import re
from typing import Dict, List, Optional


LOG_LEVELS = ["DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL", "FATAL", "ALL"]


def _parse_log_line(line: str) -> Dict[str, str]:
    """Parse a log line to extract level and message."""
    line = line.strip()
    if not line:
        return {"raw": "", "level": None, "message": ""}

    for level in LOG_LEVELS:
        if level in line.upper():
            parts = line.split(level, 1)
            return {
                "raw": line,
                "level": level,
                "message": parts[1].strip() if len(parts) > 1 else "",
            }

    return {"raw": line, "level": None, "message": line}


def log_analyze(
    file_path: str,
    level: str = "ALL",
    search_pattern: Optional[str] = None,
    limit: int = 100,
    statistics: bool = False,
    task_id: Optional[str] = None,
) -> str:  # noqa: D205
    """
    Analyze log files.

    Args:
        file_path: Path to log file
        level: Filter by log level (DEBUG, INFO, WARN, ERROR, ALL)
        search_pattern: Search pattern (regex)
        limit: Limit number of results
        statistics: Return log level statistics

    Returns:
        JSON string with analysis results
    """
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB limit

    if not os.path.exists(file_path):
        return json.dumps({
            "success": False,
            "error": f"File not found: {file_path}",
        })

    if not os.path.isfile(file_path):
        return json.dumps({
            "success": False,
            "error": f"Not a file: {file_path}",
        })

    abs_path = os.path.abspath(file_path)
    if not abs_path:
        return json.dumps({
            "success": False,
            "error": "Invalid file path",
        })

    try:
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE:
            return json.dumps({
                "success": False,
                "error": f"File too large: {file_size / (1024*1024):.1f}MB (max 50MB)",
            })

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        parsed_lines = [_parse_log_line(line) for line in lines]

        level_upper = level.upper()
        filtered = parsed_lines

        if level_upper != "ALL":
            filtered = [l for l in parsed_lines if l["level"] and l["level"].upper() == level_upper]

        if search_pattern:
            try:
                regex = re.compile(search_pattern, re.IGNORECASE)
                filtered = [l for l in filtered if regex.search(l.get("message", "") or l.get("raw", ""))]
            except re.error as e:
                return json.dumps({
                    "success": False,
                    "error": f"Invalid regex pattern: {e}",
                })

        results = filtered[:limit]

        result = {
            "success": True,
            "file": file_path,
            "total_lines": len(lines),
            "filtered_count": len(filtered),
            "returned_count": len(results),
            "lines": [r["raw"] for r in results] if results else [],
        }

        if statistics:
            level_counts: Dict[str, int] = {}
            for line in parsed_lines:
                lvl = line.get("level") or "UNKNOWN"
                level_counts[lvl] = level_counts.get(lvl, 0) + 1

            result["statistics"] = {
                "total": len(parsed_lines),
                "by_level": dict(sorted(level_counts.items())),
            }

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
        })


def check_log_analyze_requirements() -> bool:
    """Log analyze tool has no external requirements."""
    return True


LOG_ANALYZE_SCHEMA = {
    "name": "log_analyze",
    "description": (
        "Analyze log files: filter by level, search patterns, and aggregate statistics.\n\n"
        "Parameters:\n"
        "- level: Filter by log level (DEBUG, INFO, WARN, ERROR, ALL)\n"
        "- search_pattern: Regex pattern to search in log messages\n"
        "- limit: Maximum number of lines to return\n"
        "- statistics: Include level distribution counts"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to log file",
            },
            "level": {
                "type": "string",
                "description": "Filter by log level",
                "enum": ["DEBUG", "INFO", "WARN", "ERROR", "ALL"],
                "default": "ALL",
            },
            "search_pattern": {
                "type": "string",
                "description": "Search pattern (regex)",
            },
            "limit": {
                "type": "integer",
                "description": "Limit number of results",
                "default": 100,
            },
            "statistics": {
                "type": "boolean",
                "description": "Return log level statistics",
                "default": False,
            },
            "task_id": {
                "type": "string",
                "description": "Optional task ID for tracking",
            },
        },
        "required": ["file_path"],
    },
}


from tools.registry import registry

registry.register(
    name="log_analyze",
    toolset="monitoring",
    schema=LOG_ANALYZE_SCHEMA,
    handler=lambda args, **kw: log_analyze(
        file_path=args.get("file_path", ""),
        level=args.get("level", "ALL"),
        search_pattern=args.get("search_pattern"),
        limit=args.get("limit", 100),
        statistics=args.get("statistics", False),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_log_analyze_requirements,
    emoji="📋",
)
