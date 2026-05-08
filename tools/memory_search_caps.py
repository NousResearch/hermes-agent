"""Shared cap for raw memory-search style tool outputs."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_SEARCH_RESULT_CHAR_LIMIT = 10_000
MIN_MEMORY_SEARCH_RESULT_CHAR_LIMIT = 1_000
MAX_MEMORY_SEARCH_RESULT_CHAR_LIMIT = 1_000_000


def get_memory_search_result_char_limit(default: int = DEFAULT_MEMORY_SEARCH_RESULT_CHAR_LIMIT) -> int:
    """Read memory.search_result_char_limit with sane bounds."""
    try:
        from hermes_cli.config import load_config

        config = load_config()
    except Exception:
        return default

    memory_config = config.get("memory", {}) if isinstance(config, dict) else {}
    raw = memory_config.get("search_result_char_limit", default) if isinstance(memory_config, dict) else default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(MIN_MEMORY_SEARCH_RESULT_CHAR_LIMIT, min(value, MAX_MEMORY_SEARCH_RESULT_CHAR_LIMIT))


def format_memory_search_truncation_notice(limit: int) -> str:
    """Return the visible truncation notice appended to capped outputs."""
    if limit % 1000 == 0:
        label = f"{limit // 1000}K"
    else:
        label = f"{limit}"
    return (
        f"[Result truncated at {label} chars. Use a more specific query, narrower date range, "
        "or recall: filter to get focused results.]"
    )


def cap_memory_search_result(raw: str, *, limit: int | None = None) -> str:
    """Cap a raw memory/search tool result and append a visible truncation notice.

    The final returned string stays at or below the configured limit, so callers
    cannot accidentally inject oversized recall payloads into the agent context.
    """
    if raw is None:
        raw = ""
    if not isinstance(raw, str):
        raw = str(raw)

    effective_limit = get_memory_search_result_char_limit() if limit is None else int(limit)
    effective_limit = max(MIN_MEMORY_SEARCH_RESULT_CHAR_LIMIT, min(effective_limit, MAX_MEMORY_SEARCH_RESULT_CHAR_LIMIT))
    if len(raw) <= effective_limit:
        return raw

    notice = format_memory_search_truncation_notice(effective_limit)
    suffix = "\n" + notice
    keep = max(0, effective_limit - len(suffix))
    logger.info(
        "Truncated memory/search tool result from %d to %d chars",
        len(raw),
        effective_limit,
    )
    return raw[:keep].rstrip() + suffix
