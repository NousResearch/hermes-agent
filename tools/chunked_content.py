"""Chunk selection and source-search helpers for browser/web tools.

Public browser_snapshot/web_extract pagination is source-oriented: cache raw
input chunks, render only the requested chunk, and never synthesize across all
chunks for the tool caller.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_SOURCE_CHUNK_SIZE = 80_000


def compile_chunk_find_query(query: str, *, regex: bool, case_sensitive: bool):
    flags = 0 if case_sensitive else re.IGNORECASE
    if regex:
        return re.compile(query, flags)
    return re.compile(re.escape(query), flags)


def chunk_find_snippet(lines: List[str], line_index: int, *, radius: int = 2) -> str:
    start = max(0, line_index - radius)
    end = min(len(lines), line_index + radius + 1)
    return "\n".join(lines[start:end]).strip()


_SEMANTIC_BOUNDARY_LINE_RE = re.compile(
    r"^(?:#{1,6}\s+|-\s+(?:banner|navigation|main|article|section|region|contentinfo|complementary|form|search|dialog|alertdialog|heading|paragraph|list|table|grid|treegrid|link|button|textbox|searchbox|combobox|checkbox|radio)\b)"
)
_ACCESSIBILITY_CONTAINER_LINE_RE = re.compile(r"^\s*-\s+'?(?:row|cell)\s+\"")
_URL_LINE_RE = re.compile(r"^\s*-\s+/url:")


def _line_indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _is_accessibility_container_line(line: str) -> bool:
    return bool(_ACCESSIBILITY_CONTAINER_LINE_RE.match(line or ""))


def _is_url_line(line: str) -> bool:
    return bool(_URL_LINE_RE.match(line or ""))


def _collapse_accessibility_duplicate_results(candidates: List[Dict[str, Any]], lines_by_chunk: Dict[int, List[str]]) -> List[Dict[str, Any]]:
    """Drop noisy ancestor/url matches when a nearby specific line matched too."""
    indexed: List[Tuple[int, Dict[str, Any], str, bool, bool, int]] = []
    for idx, result in enumerate(candidates):
        chunk_idx = int(result.get("chunk_index", 0))
        line_no = int(result.get("line", 1))
        lines = lines_by_chunk.get(chunk_idx, [])
        line = lines[line_no - 1] if 0 <= line_no - 1 < len(lines) else ""
        indexed.append((idx, result, line, _is_accessibility_container_line(line), _is_url_line(line), _line_indent(line)))

    keep: List[Dict[str, Any]] = []
    kept_container_clusters: set[Tuple[int, int, str]] = set()
    for idx, result, line, is_container, is_url, indent in indexed:
        chunk_idx = int(result.get("chunk_index", 0))
        line_no = int(result.get("line", 1))
        match_key = str(result.get("match_text", "")).casefold()
        nearby = [
            other
            for other in indexed
            if other[0] != idx
            and int(other[1].get("chunk_index", 0)) == chunk_idx
            and str(other[1].get("match_text", "")).casefold() == match_key
            and abs(int(other[1].get("line", 1)) - line_no) <= 30
        ]
        descendant_specific = [other for other in nearby if not other[3] and not other[4] and other[5] > indent]
        if is_container and descendant_specific:
            continue
        if is_url and any(not other[3] and not other[4] and abs(int(other[1].get("line", 1)) - line_no) <= 3 for other in nearby):
            continue
        if is_container:
            cluster_key = (chunk_idx, line_no // 30, match_key)
            if cluster_key in kept_container_clusters:
                continue
            kept_container_clusters.add(cluster_key)
        keep.append(result)
    return keep


def _semantic_split_after(source: str, *, start: int, hard_end: int, size: int) -> int:
    lower_bound = start + max(1, int(size * 0.55))
    best = -1
    pos = source.find("\n", lower_bound, hard_end + 1)
    while pos != -1:
        next_start = pos + 1
        next_end = source.find("\n", next_start, min(len(source), next_start + 240))
        if next_end == -1:
            next_end = min(len(source), next_start + 240)
        if _SEMANTIC_BOUNDARY_LINE_RE.match(source[next_start:next_end]):
            best = pos
        pos = source.find("\n", pos + 1, hard_end + 1)
    return best


def split_source_chunks(text: str, chunk_size: int = DEFAULT_SOURCE_CHUNK_SIZE) -> List[str]:
    source = text or ""
    try:
        size = int(chunk_size or DEFAULT_SOURCE_CHUNK_SIZE)
    except (TypeError, ValueError):
        size = DEFAULT_SOURCE_CHUNK_SIZE
    if size <= 0:
        size = DEFAULT_SOURCE_CHUNK_SIZE
    if not source:
        return [""]

    chunks: List[str] = []
    start = 0
    source_len = len(source)
    while start < source_len:
        hard_end = min(start + size, source_len)
        if hard_end >= source_len:
            chunks.append(source[start:source_len])
            break

        split_at = _semantic_split_after(source, start=start, hard_end=hard_end, size=size)
        if split_at <= start:
            split_at = source.rfind("\n", start + 1, hard_end + 1)
        if split_at <= start:
            split_at = hard_end
        else:
            split_at += 1
        chunks.append(source[start:split_at])
        start = split_at
    return chunks


def normalize_chunk_index(chunk_index: Any = 0) -> int:
    try:
        parsed = int(chunk_index or 0)
    except (TypeError, ValueError):
        parsed = 0
    return max(parsed, 0)


def legacy_offset_to_chunk_index(offset: Any, chunk_size: int = DEFAULT_SOURCE_CHUNK_SIZE) -> int:
    try:
        off = int(offset or 0)
    except (TypeError, ValueError):
        off = 0
    if off <= 0:
        return 0
    return off // max(int(chunk_size or DEFAULT_SOURCE_CHUNK_SIZE), 1)


def chunk_nav(chunks: List[str], chunk_index: Any = 0) -> Dict[str, Any]:
    idx = normalize_chunk_index(chunk_index)
    count = len(chunks or [])
    out_of_range = idx >= count if count else idx > 0
    selected = "" if out_of_range or count == 0 else (chunks[idx] or "")
    return {
        "chunk_index": idx,
        "chunk_count": count,
        "next_chunk": (idx + 1) if count and idx + 1 < count else None,
        "has_more": bool(count and idx + 1 < count),
        "out_of_range": bool(out_of_range),
        "selected": selected,
        "selected_chars": len(selected),
    }


def tool_json_debug_enabled() -> bool:
    raw = os.getenv("HERMES_TOOL_DEBUG_JSON", "") or os.getenv("HERMES_DEBUG_TOOL_JSON", "")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def build_chunked_result(
    result: Dict[str, Any],
    *,
    chunk_index: Any = 0,
    cache_hit: bool = False,
    content_key: str = "content",
    output_key: str = "content",
    chunks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    source = result.get(content_key, "") or ""
    source_chunks = chunks if chunks is not None else split_source_chunks(source)
    nav = chunk_nav(source_chunks, chunk_index)
    base = {
        "url": result.get("url", ""),
        "title": result.get("title", ""),
        **({"blocked_by_policy": result["blocked_by_policy"]} if "blocked_by_policy" in result else {}),
    }
    existing_error = result.get("error")
    if existing_error:
        base["error"] = existing_error
    elif nav["out_of_range"]:
        base["error"] = f"chunk_index {nav['chunk_index']} is out of range for {nav['chunk_count']} chunks"

    content = "" if nav["out_of_range"] else nav["selected"]
    base.update({
        output_key: content,
        "chunk_index": nav["chunk_index"],
        "chunk_count": nav["chunk_count"],
        "next_chunk": nav["next_chunk"],
        "has_more": nav["has_more"],
    })
    if tool_json_debug_enabled():
        base["debug"] = {
            "returned_chars": len(content or ""),
            "source_chunk_chars": nav["selected_chars"],
            "total_chars": sum(len(chunk or "") for chunk in source_chunks),
            "input_cache_hit": bool(cache_hit),
            "cache_hit": bool(cache_hit),
            "chunk_out_of_range": nav["out_of_range"],
        }
    return base


def search_source_chunks(
    chunks: List[str],
    *,
    matcher,
    include_context: bool = True,
    max_results: int = 10,
    collapse_accessibility_duplicates: bool = False,
) -> List[Dict[str, Any]]:
    limit = max(1, int(max_results or 10))
    results: List[Dict[str, Any]] = []
    lines_by_chunk: Dict[int, List[str]] = {}
    for chunk_idx, chunk in enumerate(chunks):
        lines = chunk.splitlines()
        lines_by_chunk[chunk_idx] = lines
        line_starts: List[int] = []
        pos = 0
        for line in lines:
            line_starts.append(pos)
            pos += len(line) + 1
        for match in matcher.finditer(chunk):
            line_index = 0
            for i, start in enumerate(line_starts):
                if start <= match.start():
                    line_index = i
                else:
                    break
            result: Dict[str, Any] = {
                "chunk_index": chunk_idx,
                "line": line_index + 1,
                "match_text": match.group(0),
            }
            if include_context:
                result["snippet"] = chunk_find_snippet(lines, line_index)
            results.append(result)
            if not collapse_accessibility_duplicates and len(results) >= limit:
                return results
    if collapse_accessibility_duplicates:
        results = _collapse_accessibility_duplicate_results(results, lines_by_chunk)
    return results[:limit]
