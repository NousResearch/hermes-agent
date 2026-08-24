"""Truncate-and-store pipeline for web_extract (no LLM).

Pages at or under the char budget are returned whole; larger pages become a head+tail window plus a
footer that says how much is shown, where the full text is stored (cache/web) and the exact read_file
call that pages the omitted middle. Inline base64 images become ``[IMAGE: alt]`` placeholders. Logs under the
origin (tools.web_tools) logger.
"""

import logging
import re
from typing import Any, List, Optional

logger = logging.getLogger("tools.web_tools")

# Per-page char budget sent to the model (override: web.extract_char_limit); larger pages are head+tail
# truncated, full text stored on disk.
# Per-vendor client construction, request helpers and response normalizers live in
# plugins.web.<vendor>.provider (parallel / tavily / firecrawl) since PR #25182.
DEFAULT_EXTRACT_CHAR_LIMIT = 15000
# Ceiling on the full-text file written to cache/web so a multi-MB page can't write unbounded bytes on
# every extract; the model only ever sees char_limit.
MAX_STORED_TEXT_CHARS = 2_000_000

_CHAR_LIMIT_FLOOR, _CHAR_LIMIT_CEILING = 2000, 500_000


def _clamp_char_limit(value: Any) -> int:
    """Clamp to [2k, 500k] (below 2k the truncation footer dominates; a config typo must not blow up context);
    raises TypeError/ValueError for non-numeric input."""
    return max(_CHAR_LIMIT_FLOOR, min(int(value), _CHAR_LIMIT_CEILING))


def _clamp_or_default(value: Any) -> int:
    """``_clamp_char_limit(value)``; ``None`` or non-numeric input falls back to the default."""
    try:
        return DEFAULT_EXTRACT_CHAR_LIMIT if value is None else _clamp_char_limit(value)
    except (TypeError, ValueError):
        return DEFAULT_EXTRACT_CHAR_LIMIT


def _get_extract_char_limit() -> int:
    """``web.extract_char_limit`` clamped to a sane range, else the default."""
    from tools.web_tools import _load_web_config  # lazy: tests patch tools.web_tools._load_web_config
    return _clamp_or_default(_load_web_config().get("extract_char_limit"))


def convert_base64_images_to_links(text: str) -> str:
    """Replace inline base64 image blobs (token bombs) with ``[IMAGE: alt]`` placeholders: markdown images
    (alt kept), parenthesised blobs, and bare ``data:image/...;base64,`` payloads. Real http(s) markdown
    image links are left untouched so the agent can ``web_extract`` / ``vision_analyze`` them."""
    def _md_repl(m: "re.Match[str]") -> str:
        return f"[IMAGE: {alt}]" if (alt := (m.group("alt") or "").strip()) else "[IMAGE]"

    out = re.sub(r"!\[(?P<alt>[^\]]*)\]\(\s*data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+\)", _md_repl, text)
    out = re.sub(r"\(\s*data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+\)", "[IMAGE]", out)
    return re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "[IMAGE]", out)


def _store_full_text(url: str, content: str) -> Optional[str]:
    """Write the full page to cache/web; absolute path or None (best-effort: the truncated content is still
    returned). cache/web is mounted read-only into remote backends (credential_files _CACHE_DIRS) so
    read_file can page the complete text on any backend."""
    try:
        import hashlib
        from hermes_constants import get_hermes_dir
        from tools.web_result_cache import _host_slug
        cache_dir = get_hermes_dir("cache/web", "web_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{_host_slug(url)}-{hashlib.sha256(url.encode('utf-8')).hexdigest()[:10]}.md"
        if len(content) > MAX_STORED_TEXT_CHARS:
            content = content[:MAX_STORED_TEXT_CHARS] + (
                f"\n\n[... stored copy truncated at {MAX_STORED_TEXT_CHARS:,} chars "
                f"of {len(content):,}; re-extract a more specific URL for the rest ...]"
            )
        from tools.spill_safety import write_text_exclusive
        # Deterministic name in a well-known dir: refuse symlinks (lstat-unlink + exclusive create);
        # same-URL re-extraction legitimately overwrites. Not private: cache/web is bind-mounted
        # into remote backends' container UID.
        write_text_exclusive(path, content, private=False, overwrite=True)
        return str(path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to store full web_extract text for %s: %s", url, exc)
        return None


def _truncate_with_footer(content: str, url: str, char_limit: int) -> tuple[str, bool]:
    """Return (model_text, was_truncated). Pages over ``char_limit`` become a ~75% head / ~25% tail window cut
    on line boundaries, plus a footer saying how much is shown, where the full text is stored, and the
    read_file call that pages the omitted middle. Deterministic."""
    if len(content) <= char_limit:
        return content, False
    head_budget = int(char_limit * 0.75)
    tail_budget = char_limit - head_budget
    head, tail = content[:head_budget], content[-tail_budget:]
    # Snap both cuts to line boundaries (head back, tail forward) so we never slice mid-line.
    if (nl := head.rfind("\n")) > head_budget * 0.5:
        head = head[:nl]
    if 0 <= (nl := tail.find("\n")) < tail_budget * 0.5:
        tail = tail[nl + 1:]

    stored_path = _store_full_text(url, content)
    if stored_path:
        # The footer is read by the AGENT, whose read_file runs inside the active backend: render the
        # path where docker/modal/ssh/... see the mounted cache, not the host path (#72389, #81984).
        from tools.credential_files import to_agent_visible_cache_path
        stored_path = to_agent_visible_cache_path(stored_path)
    footer_lines = [
        "", "─" * 8 + " [TRUNCATED] " + "─" * 8,
        f"Showing {len(head):,} chars (head) + {len(tail):,} chars (tail) "
        f"of {len(content):,} total clean characters.",
    ]
    if stored_path:
        # read_file is 1-indexed; +2 lands on the first line after the shown head.
        middle_start_line = head.count("\n") + 2
        footer_lines += [
            f"Full text saved to: {stored_path}",
            f'To read the omitted middle: read_file path="{stored_path}" '
            f"offset={middle_start_line} limit=200  (the file is the complete page; "
            f"raise/lower offset to page through it).",
        ]
    else:
        footer_lines.append(
            "Full text could not be stored; re-run web_extract on a more "
            "specific URL or use browser_navigate for the complete page."
        )
    footer_lines.append("─" * 29)
    model_text = head + "\n\n[... middle omitted — see footer ...]\n\n" + tail
    return model_text + "\n" + "\n".join(footer_lines), True


def _effective_char_limit(char_limit: Optional[int]) -> int:
    """Caller's ``char_limit`` (else config) clamped; non-numeric input falls back to the default."""
    return _clamp_or_default(char_limit) if char_limit is not None else _get_extract_char_limit()


def _truncate_results(results: List[dict], char_limit: int, debug_call_data: dict) -> None:
    """In place: replace each successful entry's content with its base64-cleaned, budgeted text;
    per-page truncation metrics go into ``debug_call_data``."""
    for result in results:
        url = result.get("url", "")
        raw_content = result.get("raw_content", "") or result.get("content", "")
        if result.get("error") or not raw_content:
            continue
        clean = convert_base64_images_to_links(raw_content)
        model_text, truncated = _truncate_with_footer(clean, url, char_limit)
        result["content"] = model_text
        if truncated:
            debug_call_data["pages_truncated"] += 1
            debug_call_data["truncation_metrics"].append(
                {"url": url, "original_size": len(clean), "sent_size": len(model_text)}
            )
            logger.info("%s (truncated %d -> %d chars)", url, len(clean), len(model_text))
        else:
            logger.info("%s (%d chars, whole)", url, len(clean))


def _trim_results(results: List[dict]) -> List[dict]:
    """Keep only url/title/content/error per entry (+ blocked_by_policy when present)."""
    return [
        {
            "url": r.get("url", ""), "title": r.get("title", ""), "content": r.get("content", ""),
            "error": r.get("error"),
            **({"blocked_by_policy": r["blocked_by_policy"]} if "blocked_by_policy" in r else {}),
        }
        for r in results
    ]
