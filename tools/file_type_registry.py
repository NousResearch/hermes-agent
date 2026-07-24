"""
File type registry — detects structured data formats and produces
intelligent summaries instead of raw content injection.

Provides:
- Extension-based format detection
- Per-format handler functions that return human-readable summaries
- A registry for adding new format handlers

Usage:
    from tools.file_type_registry import detect_and_summarize, register_format
    
    summary = detect_and_summarize("/path/to/file.har", file_content, file_size)
    if summary:
        print(summary)  # Short, structured summary
"""

import json
import os
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Handler signature: (path: str, content: str, file_size: int) -> str | None
# Returns a human-readable summary string, or None if the file can't be parsed.
HandlerFunc = Callable[[str, str, int], Optional[str]]

# Registry: extension → (handler, description)
_FORMAT_REGISTRY: dict[str, tuple[HandlerFunc, str]] = {}


def register_format(
    extension: str,
    handler: HandlerFunc,
    description: str = "",
) -> None:
    """Register a handler for a specific file extension.

    Args:
        extension: File extension including the dot (e.g. '.har', '.json')
        handler: Function that takes (path, content, file_size) and returns a summary
        description: Human-readable description of the format
    """
    ext = extension.lower()
    if not ext.startswith("."):
        ext = f".{ext}"
    _FORMAT_REGISTRY[ext] = (handler, description)


def known_format_extensions() -> frozenset[str]:
    """Return the set of registered format extensions."""
    return frozenset(_FORMAT_REGISTRY.keys())


def detect_format(path: str) -> Optional[str]:
    """Detect format extension by file extension.

    Returns the matching extension (e.g. '.har'), or None if no format matches.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in _FORMAT_REGISTRY:
        return ext
    return None


def detect_and_summarize(path: str, content: str, file_size: int) -> Optional[str]:
    """Detect format and produce a summary for *path*.

    Args:
        path: Absolute file path (used for extension detection)
        content: Full file content as a string
        file_size: File size in bytes

    Returns:
        A human-readable summary string, or None if no registered format matches.
    """
    ext = os.path.splitext(path)[1].lower()
    entry = _FORMAT_REGISTRY.get(ext)
    if entry is None:
        return None
    handler, _ = entry
    try:
        return handler(path, content, file_size)
    except Exception as exc:
        logger.warning("Format handler for %s failed: %s", ext, exc)
        return None


# =========================================================================
# Built-in format handlers
# =========================================================================


def _har_handler(path: str, content: str, file_size: int) -> Optional[str]:
    """Parse HAR (HTTP Archive) file and return a network trace summary."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        return f"HAR file detected but is not valid JSON: {e}"

    log = data["log"] if isinstance(data, dict) and "log" in data else data
    entries = log.get("entries", []) if isinstance(log, dict) else data

    if not isinstance(entries, list):
        return "HAR file detected but format is unrecognised (expected 'log.entries' array)"

    total = len(entries)
    if total == 0:
        return "HAR file detected — empty (no network requests recorded)."

    domains: dict[str, int] = {}
    status_codes: dict[int, int] = {}
    methods: dict[str, int] = {}
    content_types: dict[str, int] = {}
    total_transfer = 0
    total_body = 0
    timeline_start: Optional[float] = None
    timeline_end: Optional[float] = None
    errors = 0

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        req = entry.get("request", {})
        res = entry.get("response", {})

        url = req.get("url", "")
        if "//" in url:
            domain = url.split("/")[2] if len(url.split("/")) > 2 else "unknown"
        else:
            domain = "unknown"
        domains[domain] = domains.get(domain, 0) + 1

        method = req.get("method", "GET")
        methods[method] = methods.get(method, 0) + 1

        status = res.get("status", 0)
        if isinstance(status, int):
            status_codes[status] = status_codes.get(status, 0) + 1
            if status >= 400:
                errors += 1

        # Transfer size (wire bytes)
        transfer = res.get("_transferSize", 0) or 0
        total_transfer += transfer

        # Body size
        body_size = res.get("content", {}).get("size", 0) or 0
        total_body += body_size

        # Content type
        ct = (res.get("content", {}).get("mimeType", "") or "").split(";")[0]
        if ct:
            content_types[ct] = content_types.get(ct, 0) + 1

        # Timeline
        started = entry.get("startedDateTime")
        if started:
            try:
                from datetime import datetime
                t = datetime.fromisoformat(started.replace("Z", "+00:00")).timestamp()
                if timeline_start is None or t < timeline_start:
                    timeline_start = t
                if timeline_end is None or t > timeline_end:
                    timeline_end = t
            except (ValueError, TypeError):
                pass

    # Build summary
    lines = [
        f"HAR Network Trace: {total} requests",
        f"  File size:       {_format_size(file_size)}",
        f"  Transfer size:   {_format_size(total_transfer)} (wire)",
        f"  Body size:       {_format_size(total_body)} (decompressed)",
        f"  Unique domains:  {len(domains)}",
        f"  Error responses: {errors}",
    ]

    if timeline_start is not None and timeline_end is not None:
        duration = timeline_end - timeline_start
        if duration < 120:
            lines.append(f"  Time span:       {duration:.1f}s")
        else:
            lines.append(f"  Time span:       {duration / 60:.1f} min")

    # Top domains
    if domains:
        lines.append("")
        lines.append("Top domains by request count:")
        for domain, count in sorted(domains.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  {domain}: {count}")

    # Methods
    if methods:
        lines.append("")
        lines.append("HTTP methods:")
        for method, count in sorted(methods.items(), key=lambda x: -x[1]):
            lines.append(f"  {method}: {count}")

    # Status code distribution (grouped)
    if status_codes:
        lines.append("")
        lines.append("Status code distribution:")
        for code in sorted(status_codes.keys()):
            label = _status_label(code)
            count = status_codes[code]
            bar = "█" * min(count, 40)
            lines.append(f"  {code} {label}: {count} {bar}")

    # Top content types
    if content_types:
        lines.append("")
        lines.append("Content types:")
        for ct, count in sorted(content_types.items(), key=lambda x: -x[1])[:8]:
            lines.append(f"  {ct}: {count}")

    return "\n".join(lines)


def _json_structure_handler(path: str, content: str, file_size: int) -> Optional[str]:
    """Analyze JSON structure and return a summary."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        return f"JSON file detected but is not valid: {e}"

    if isinstance(data, dict):
        keys = list(data.keys())
        lines = [
            f"JSON Object: {len(keys)} top-level key{'s' if len(keys) != 1 else ''}",
            f"  File size: {_format_size(file_size)}",
        ]
        if keys:
            lines.append("")
            lines.append("Top-level keys:")
            for key in keys[:20]:
                val = data[key]
                if isinstance(val, dict):
                    lines.append(f"  {key}: object ({len(val)} keys)")
                elif isinstance(val, list):
                    lines.append(f"  {key}: array ({len(val)} items)")
                elif isinstance(val, str):
                    lines.append(f"  {key}: string ({len(val)} chars)")
                elif val is None:
                    lines.append(f"  {key}: null")
                else:
                    lines.append(f"  {key}: {type(val).__name__} ({val!r})")
            if len(keys) > 20:
                lines.append(f"  ... and {len(keys) - 20} more keys")
        return "\n".join(lines)

    elif isinstance(data, list):
        lines = [
            f"JSON Array: {len(data)} items",
            f"  File size: {_format_size(file_size)}",
        ]
        if data:
            lines.append("")
            lines.append("First 5 items:")
            for i, item in enumerate(data[:5]):
                if isinstance(item, (dict, list)):
                    preview = json.dumps(item, ensure_ascii=False)[:200]
                else:
                    preview = repr(item)[:200]
                lines.append(f"  [{i}]: {preview}")
            if len(data) > 5:
                lines.append(f"  ... and {len(data) - 5} more items")
        return "\n".join(lines)

    else:
        preview = json.dumps(data, ensure_ascii=False)[:500]
        return f"JSON value: {preview}"


def _csv_handler(path: str, content: str, file_size: int) -> Optional[str]:
    """Preview CSV/TSV file with structure info."""
    import csv as csv_module
    import io

    delimiter = _detect_csv_delimiter(content)
    try:
        reader = csv_module.reader(io.StringIO(content), delimiter=delimiter)
        rows = list(reader)
    except Exception as e:
        return f"CSV file detected but could not parse: {e}"

    if not rows:
        return "CSV file detected — empty."

    header = rows[0] if rows else []
    data_rows = rows[1:] if len(rows) > 1 else []
    total_rows = len(data_rows)

    lines = [
        f"CSV File: {total_rows} data rows, {len(header)} columns",
        f"  File size: {_format_size(file_size)}",
        f"  Delimiter: {repr(delimiter)}",
    ]

    if header:
        lines.append("")
        lines.append("Columns:")
        for i, col in enumerate(header):
            sample_vals = []
            for row in data_rows[:5]:
                if i < len(row) and row[i].strip():
                    sample_vals.append(row[i][:60])
                    break
            sample = f' (e.g. {sample_vals[0]!r})' if sample_vals else ''
            lines.append(f"  [{i}] {col}{sample}")

        if total_rows > 0:
            lines.append("")
            lines.append("First 5 rows:")
            for idx, row in enumerate(data_rows[:5]):
                vals = [row[i] if i < len(row) else "" for i in range(min(len(header), 8))]
                lines.append(f"  [{idx}]: {' | '.join(v[:40] for v in vals)}")
                if len(header) > 8:
                    lines.append(f"        ... and {len(header) - 8} more columns")
            if total_rows > 5:
                lines.append(f"  ... and {total_rows - 5} more rows")

    return "\n".join(lines)


def _log_file_handler(path: str, content: str, file_size: int) -> Optional[str]:
    """Summarise a log file: count lines, detect common patterns."""
    lines = content.splitlines()
    total = len(lines)

    # Detect severity levels
    import re
    severity_patterns = {
        "ERROR": re.compile(r"\bERROR\b", re.IGNORECASE),
        "WARN": re.compile(r"\bWARN(?:ING)?\b", re.IGNORECASE),
        "INFO": re.compile(r"\bINFO\b", re.IGNORECASE),
        "DEBUG": re.compile(r"\bDEBUG\b", re.IGNORECASE),
        "TRACE": re.compile(r"\bTRACE\b", re.IGNORECASE),
        "FATAL": re.compile(r"\bFATAL\b", re.IGNORECASE),
        "CRITICAL": re.compile(r"\bCRITICAL\b", re.IGNORECASE),
    }
    severity_counts: dict[str, int] = {}
    for line in lines:
        for level, pattern in severity_patterns.items():
            if pattern.search(line):
                severity_counts[level] = severity_counts.get(level, 0) + 1

    summary = [
        f"Log File: {total} lines",
        f"  File size: {_format_size(file_size)}",
    ]
    if severity_counts:
        summary.append("")
        summary.append("Severity distribution:")
        for level in ["FATAL", "CRITICAL", "ERROR", "WARN", "INFO", "DEBUG", "TRACE"]:
            if level in severity_counts:
                summary.append(f"  {level}: {severity_counts[level]}")
    summary.append("")
    summary.append(f"First 3 lines:")
    summary.extend(f"  {l}" for l in lines[:3])
    summary.append(f"Last 3 lines:")
    summary.extend(f"  {l}" for l in lines[-3:])

    return "\n".join(summary)


# =========================================================================
# Helpers
# =========================================================================

_MAX_STRUCTURED_FILE_SIZE = 50 * 1024 * 1024  # 50 MB — max file to process


def max_structured_file_size() -> int:
    """Return the maximum file size (bytes) allowed for structured processing."""
    return _MAX_STRUCTURED_FILE_SIZE


def _format_size(size_bytes: int) -> str:
    """Format byte size for human display."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def _detect_csv_delimiter(content: str) -> str:
    """Detect CSV delimiter by examining the first line."""
    first_line = content.splitlines()[0] if content else ""
    if not first_line:
        return ","
    # Count occurrences of potential delimiters
    counts = {d: first_line.count(d) for d in ",;\t|"}
    if not counts:
        return ","
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _status_label(code: int) -> str:
    """Return a short label for an HTTP status code."""
    if 100 <= code < 200:
        return "Informational"
    elif 200 <= code < 300:
        return "Success"
    elif 300 <= code < 400:
        return "Redirect"
    elif 400 <= code < 500:
        return "Client Error"
    elif 500 <= code < 600:
        return "Server Error"
    return "Unknown"


# =========================================================================
# Register built-in handlers
# =========================================================================

register_format(".har", _har_handler, "HTTP Archive (network trace)")
register_format(".json", _json_structure_handler, "JSON data file")
register_format(".csv", _csv_handler, "Comma-separated values")
register_format(".tsv", _csv_handler, "Tab-separated values")
register_format(".log", _log_file_handler, "Log file")
register_format(".txt", _log_file_handler, "Text log file")
