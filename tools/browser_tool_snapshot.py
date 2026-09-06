"""Snapshot post-processing for the browser tools: truncate-and-store of oversized
accessibility trees, model-boundary secret redaction, screenshot-path recovery.

Facade-owned state is read through ``_bt`` (``tools.browser_tool``, resolved per call) — no import cycle.
"""

import re
from typing import Any, Optional
from tools.browser_tool_origin import origin as _bt


_SCREENSHOT_PATH_PATTERNS = (
    r"Screenshot saved to ['\"](?P<path>/[^'\"]+?\.png)['\"]",
    r"Screenshot saved to (?P<path>/\S+?\.png)(?:\s|$)",
    r"(?P<path>/\S+?\.png)(?:\s|$)",
)


def _extract_screenshot_path_from_text(text: str) -> Optional[str]:
    """Extract a screenshot file path from agent-browser human-readable output."""
    if not text:
        return None

    for pattern in _SCREENSHOT_PATH_PATTERNS:
        match = re.search(pattern, text)
        if match:
            path = match.group("path").strip().strip("'\"")
            if path:
                return path
    return None


def _store_full_snapshot(snapshot_text: str) -> Optional[str]:
    """Write a full snapshot to cache/web and return its path (None on failure — best-effort).

    Mirrors ``web_tools._store_full_text``: cache/web is mounted read-only into
    remote backends, so read_file can page through the complete tree on any
    backend. The stored copy is force-redacted (page-rendered secrets must not
    hit disk unmasked) and named by content hash so identical snapshots dedupe.
    """
    try:
        import hashlib
        from hermes_constants import get_hermes_dir
        from agent.redact import redact_sensitive_text

        content = redact_sensitive_text(snapshot_text, force=True)
        if len(content) > _bt.MAX_STORED_SNAPSHOT_CHARS:
            content = (
                content[:_bt.MAX_STORED_SNAPSHOT_CHARS]
                + f"\n\n[... stored copy truncated at {_bt.MAX_STORED_SNAPSHOT_CHARS:,} chars "
                f"of {len(content):,} ...]"
            )
        from tools.spill_safety import ensure_spill_dir, write_text_exclusive

        cache_dir = get_hermes_dir("cache/web", "web_cache")
        ensure_spill_dir(cache_dir, private=False)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:10]
        path = cache_dir / f"browser-snapshot-{digest}.txt"
        # Deterministic filename in a well-known dir: refuse symlinks (lstat-unlink +
        # exclusive create); same-content re-snapshots legitimately overwrite. Not
        # private: cache/web is bind-mounted into remote backends' container UID.
        write_text_exclusive(path, content, private=False, overwrite=True)
        return str(path)
    except Exception as exc:  # noqa: BLE001
        _bt.logger.debug("Failed to store full browser snapshot: %s", exc)
        return None


def _truncate_snapshot(snapshot_text: str, max_chars: Optional[int] = None) -> str:
    """Truncate a snapshot at line boundaries (never mid-element) to ``max_chars``.

    Defaults to ``browser.snapshot_threshold``. The full snapshot is stored to
    cache/web and the appended note tells the agent how to page through it
    with read_file — element refs beyond the cut are in the file, not lost.
    """
    if max_chars is None:
        max_chars = _bt.get_browser_snapshot_threshold()
    if len(snapshot_text) <= max_chars:
        return snapshot_text

    stored_path = _store_full_snapshot(snapshot_text)

    lines = snapshot_text.split('\n')
    result: list[str] = []
    chars = 0
    # Reserve space for the truncation note (the stored-path variant is the
    # longer of the two). Clamp so tiny max_chars values still keep content.
    reserve = min(110 + len(stored_path or ""), max_chars // 2)
    for line in lines:
        if chars + len(line) + 1 > max_chars - reserve:
            break
        result.append(line)
        chars += len(line) + 1
    remaining = len(lines) - len(result)
    if remaining > 0:
        if stored_path:
            next_line = len(result) + 1
            result.append(
                f'\n[... {remaining} more lines truncated — full snapshot: '
                f'read_file path="{stored_path}" offset={next_line} limit=200]'
            )
        else:
            result.append(f'\n[... {remaining} more lines truncated, use browser_snapshot for full content]')
    return '\n'.join(result)


def _redact_browser_output(value: Any) -> Any:
    """Force-redact secrets in browser-originated data (snapshots, console, eval
    results can carry page-rendered keys/cookies/tokens). Tool output is a model
    boundary, so this applies even when global log redaction is disabled."""
    from agent.redact import redact_sensitive_text

    if isinstance(value, str):
        return redact_sensitive_text(value, force=True)
    if isinstance(value, list):
        return [_redact_browser_output(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_browser_output(item) for item in value)
    if isinstance(value, dict):
        return {key: _redact_browser_output(item) for key, item in value.items()}
    return value
