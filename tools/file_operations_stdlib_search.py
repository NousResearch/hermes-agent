"""Host-filesystem search when rg, grep, and find are unavailable.

LocalEnvironment only. Remote sandboxes do not share the agent process
filesystem, so callers must resolve a host path before using these helpers.
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Iterator, Optional

from tools.binary_extensions import BINARY_EXTENSIONS
from tools.file_operations_common import SearchMatch, SearchResult

_STDLIB_SEARCH_MAX_FILE_BYTES = 256 * 1024
_STDLIB_SEARCH_MAX_FILES_SCANNED = 80_000
_STDLIB_SEARCH_MAX_MATCHES = 50_000


def _search_tool_rg_grep_exits_ok(exit_code: int) -> bool:
    """True when rg or grep completed normally: 0 = matches, 1 = no matches."""
    return exit_code in (0, 1)


def _stdlib_binary_sample(sample: bytes, path: str) -> bool:
    """Return True if *sample* looks like binary (extension + byte heuristic)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in BINARY_EXTENSIONS:
        return True
    if not sample:
        return False
    chunk = sample[:1000]
    non_printable = sum(1 for b in chunk if b < 32 and b not in (9, 10, 13))
    return non_printable / max(len(chunk), 1) > 0.30


def _stdlib_walk_files(
    root: Path, max_files: int
) -> tuple[Iterator[Path], dict[str, bool]]:
    """Walk *root* skipping hidden directories (ripgrep default).

    The returned state dict has ``truncated`` set when the walk stops early.
    """
    state: dict[str, bool] = {"truncated": False}

    def gen():
        count = 0
        if root.is_file():
            yield root
            return
        if not root.is_dir():
            return
        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fn in filenames:
                if count >= max_files:
                    state["truncated"] = True
                    return
                count += 1
                yield Path(dirpath) / fn

    return gen(), state


def stdlib_search_files(
    root: str, pattern: str, limit: int, offset: int
) -> SearchResult:
    """Find files under *root* whose basename matches *pattern* (find -name rules)."""
    if not pattern.startswith("**/") and "/" not in pattern:
        search_pattern = pattern
    else:
        search_pattern = pattern.split("/")[-1]

    root_p = Path(root)
    gen, walk_state = _stdlib_walk_files(root_p, _STDLIB_SEARCH_MAX_FILES_SCANNED)
    hits: list[tuple[float, str]] = []
    for fpath in gen:
        if not fnmatch.fnmatch(fpath.name, search_pattern):
            continue
        try:
            mtime = fpath.stat().st_mtime
        except OSError:
            continue
        hits.append((mtime, str(fpath)))

    hits.sort(key=lambda item: -item[0])
    paths = [hit[1] for hit in hits]
    page = paths[offset:offset + limit]
    return SearchResult(
        files=page, total_count=len(paths), truncated=walk_state["truncated"]
    )


def stdlib_search_content(
    root: str,
    pattern: str,
    file_glob: Optional[str],
    limit: int,
    offset: int,
    output_mode: str,
    context: int,
) -> SearchResult:
    """Regex search file contents under *root* without rg/grep."""
    try:
        cre = re.compile(pattern)
    except re.error as exc:
        return SearchResult(error=f"Invalid regex pattern: {exc}", total_count=0)

    gen, walk_state = _stdlib_walk_files(Path(root), _STDLIB_SEARCH_MAX_FILES_SCANNED)
    matches: list[SearchMatch] = []
    counts: dict[str, int] = {}
    files_only_order: list[str] = []
    files_only_seen: set = set()
    truncated_matches = False
    max_collect = min(
        _STDLIB_SEARCH_MAX_MATCHES,
        offset + limit + 200,
    )

    for fpath in gen:
        fstr = str(fpath)
        if file_glob and not fnmatch.fnmatch(fpath.name, file_glob):
            continue
        try:
            data = fpath.read_bytes()
        except OSError:
            continue
        if len(data) > _STDLIB_SEARCH_MAX_FILE_BYTES:
            data = data[:_STDLIB_SEARCH_MAX_FILE_BYTES]
        if _stdlib_binary_sample(data[:2048], fstr):
            continue
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        n = len(lines)

        if output_mode == "count":
            file_hits = sum(1 for i in range(n) if cre.search(lines[i]))
            if file_hits:
                counts[fstr] = file_hits
            continue

        if output_mode == "files_only":
            for i in range(n):
                if cre.search(lines[i]):
                    if fstr not in files_only_seen:
                        files_only_seen.add(fstr)
                        files_only_order.append(fstr)
                    break
            continue

        if context <= 0:
            for i in range(n):
                if cre.search(lines[i]):
                    matches.append(
                        SearchMatch(path=fstr, line_number=i + 1, content=lines[i][:500])
                    )
                    if len(matches) >= max_collect:
                        truncated_matches = True
                        break
            if truncated_matches:
                break
            continue

        i = 0
        while i < n:
            if cre.search(lines[i]):
                lo = max(0, i - context)
                hi = min(n, i + context + 1)
                for j in range(lo, hi):
                    matches.append(
                        SearchMatch(path=fstr, line_number=j + 1, content=lines[j][:500])
                    )
                    if len(matches) >= max_collect:
                        truncated_matches = True
                        break
                if truncated_matches:
                    break
            i += 1
        if truncated_matches:
            break

    walk_truncated = walk_state["truncated"]
    if output_mode == "files_only":
        total = len(files_only_order)
        return SearchResult(
            files=files_only_order[offset:offset + limit],
            total_count=total,
            truncated=walk_truncated,
        )
    if output_mode == "count":
        return SearchResult(
            counts=counts,
            total_count=sum(counts.values()),
            truncated=walk_truncated,
        )
    total = len(matches)
    return SearchResult(
        matches=matches[offset:offset + limit],
        total_count=total,
        truncated=truncated_matches or walk_truncated or total > offset + limit,
    )
