#!/usr/bin/env python3
"""Search Cache & Trigram Index for accelerating repeated searches.

Two layers of caching for the search_files tool:

1. Result Cache (④): Caches raw ripgrep/grep results keyed by
   (pattern, path, file_glob, output_mode, context). Invalidated when
   any file is written or patched via Hermes tools. Cheap, immediate win
   for agents that re-search the same pattern.

2. Trigram Index (⑤): In-memory inverted index mapping trigrams to file
   paths. Built lazily on first content search by scanning the working
   tree. Subsequent searches decompose the regex into trigrams, intersect
   posting lists to get candidate files, and only run ripgrep on those.
   Dies with the session — no staleness across sessions.

Both layers are per-task_id (matching ShellFileOperations scoping).
"""

import hashlib
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result Cache (④)
# ---------------------------------------------------------------------------

@dataclass
class CachedResult:
    """A cached search result with timestamp."""
    result_json: str
    timestamp: float
    pattern: str
    path: str


class ResultCache:
    """LRU-ish cache of search results, invalidated on writes."""

    MAX_ENTRIES = 64

    def __init__(self):
        self._cache: Dict[str, CachedResult] = {}
        self._lock = threading.Lock()
        self._dirty_files: Set[str] = set()  # files modified since last search

    def _make_key(self, pattern: str, path: str, file_glob: Optional[str],
                  output_mode: str, context: int, target: str,
                  limit: int, offset: int) -> str:
        """Deterministic cache key from search parameters."""
        raw = f"{pattern}|{path}|{file_glob or ''}|{output_mode}|{context}|{target}|{limit}|{offset}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, pattern: str, path: str, file_glob: Optional[str],
            output_mode: str, context: int, target: str,
            limit: int, offset: int) -> Optional[str]:
        """Return cached result JSON if available and not invalidated."""
        key = self._make_key(pattern, path, file_glob, output_mode,
                             context, target, limit, offset)
        with self._lock:
            if self._dirty_files:
                # Any file write invalidates the whole cache — simple and correct.
                # A smarter approach would check if dirty files intersect with
                # the search path, but that adds complexity for minimal gain.
                self._cache.clear()
                self._dirty_files.clear()
                return None
            entry = self._cache.get(key)
            if entry is None:
                return None
            # Expire after 120s even without writes (external changes)
            if time.time() - entry.timestamp > 120:
                del self._cache[key]
                return None
            return entry.result_json

    def put(self, pattern: str, path: str, file_glob: Optional[str],
            output_mode: str, context: int, target: str,
            limit: int, offset: int, result_json: str) -> None:
        """Store a search result."""
        key = self._make_key(pattern, path, file_glob, output_mode,
                             context, target, limit, offset)
        with self._lock:
            if len(self._cache) >= self.MAX_ENTRIES:
                # Evict oldest entry
                oldest_key = min(self._cache, key=lambda k: self._cache[k].timestamp)
                del self._cache[oldest_key]
            self._cache[key] = CachedResult(
                result_json=result_json,
                timestamp=time.time(),
                pattern=pattern,
                path=path,
            )

    def invalidate(self, file_path: str) -> None:
        """Mark a file as dirty — next get() will flush the cache."""
        with self._lock:
            self._dirty_files.add(file_path)

    def clear(self) -> None:
        """Drop everything."""
        with self._lock:
            self._cache.clear()
            self._dirty_files.clear()

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._cache),
                "dirty_files": len(self._dirty_files),
            }


# ---------------------------------------------------------------------------
# Trigram Index (⑤)
# ---------------------------------------------------------------------------

def _extract_trigrams(text: str) -> Set[str]:
    """Extract all overlapping trigrams from text (lowercased)."""
    text = text.lower()
    trigrams = set()
    for i in range(len(text) - 2):
        trigrams.add(text[i:i+3])
    return trigrams


def _extract_trigrams_from_regex(pattern: str) -> Optional[Set[str]]:
    """Decompose a regex pattern into required trigrams.

    Returns None if the pattern can't be meaningfully decomposed
    (too many wildcards, alternations we can't handle, etc.).

    This is a conservative decomposition — it only extracts trigrams
    from literal runs in the pattern. A regex like 'foo.*bar' yields
    {'foo', 'bar'} trigrams (from the literals). A regex like '.*'
    yields None (can't narrow candidates).

    Strategy: extract literal runs, pick the LONGEST ones, and only
    use a few high-value trigrams rather than all of them. This avoids
    the problem where common trigrams like 'the' or 'ing' balloon the
    candidate set.
    """
    # Strip anchors
    p = pattern.strip('^$')

    # Extract literal segments: contiguous runs of non-special characters.
    # We treat anything that's not a regex metacharacter as literal.
    # This is intentionally conservative — we'd rather miss some trigrams
    # than generate wrong ones.
    literal_runs = re.findall(r'[A-Za-z0-9_\-\./ ]{3,}', p)

    if not literal_runs:
        return None

    # Sort runs by length descending — longer literals are more selective
    literal_runs.sort(key=len, reverse=True)

    trigrams = set()
    for run in literal_runs:
        run_trigrams = _extract_trigrams(run)
        trigrams.update(run_trigrams)
        # Cap at 8 trigrams — more than that just adds lookup cost
        # without meaningfully narrowing candidates
        if len(trigrams) >= 8:
            break

    if not trigrams:
        return None

    return trigrams


class TrigramIndex:
    """In-memory trigram inverted index for candidate file filtering.

    Built lazily by scanning the working tree on first use. Maps each
    trigram to the set of file paths containing it. When a search comes
    in, we decompose the pattern into trigrams, intersect posting lists,
    and return candidate files for ripgrep to scan (instead of everything).
    """

    # Don't index files larger than this (they'll be caught by full rg anyway)
    MAX_FILE_SIZE = 512 * 1024  # 512KB

    # Don't build index for trees with more files than this
    MAX_FILES = 50_000

    # Minimum tree size where indexing pays off (below this, rg is fast enough)
    MIN_FILES = 500

    def __init__(self):
        self._index: Dict[str, Set[str]] = {}  # trigram -> set of file paths
        self._indexed_files: Set[str] = set()   # all files in the index
        self._lock = threading.Lock()
        self._built = False
        self._building = False
        self._build_time: float = 0
        self._root: Optional[str] = None
        self._file_count: int = 0
        self._skipped = False  # True if tree too small or too large

    def build(self, file_ops, root: str = ".") -> bool:
        """Build the trigram index by scanning the tree.

        Uses the terminal backend to list and read files, so it works
        across all environments (local, docker, ssh, etc.).

        Returns True if the index was built, False if skipped.
        """
        with self._lock:
            if self._built or self._building:
                return self._built
            self._building = True

        start = time.time()
        try:
            # Get file list via rg --files (respects .gitignore)
            result = file_ops._exec(
                f"rg --files {file_ops._escape_shell_arg(root)} 2>/dev/null | head -n {self.MAX_FILES + 1}",
                timeout=30
            )
            if not result.stdout.strip():
                # Fallback: try find
                result = file_ops._exec(
                    f"find {file_ops._escape_shell_arg(root)} -type f -not -path '*/.*' 2>/dev/null | head -n {self.MAX_FILES + 1}",
                    timeout=30
                )

            files = [f for f in result.stdout.strip().split('\n') if f]

            if len(files) < self.MIN_FILES:
                logger.debug("Trigram index: only %d files, skipping (rg is fast enough)", len(files))
                with self._lock:
                    self._skipped = True
                    self._building = False
                return False

            if len(files) > self.MAX_FILES:
                logger.warning("Trigram index: %d+ files exceeds limit, skipping", self.MAX_FILES)
                with self._lock:
                    self._skipped = True
                    self._building = False
                return False

            index: Dict[str, Set[str]] = {}
            indexed: Set[str] = set()
            batch_size = 50

            for i in range(0, len(files), batch_size):
                batch = files[i:i+batch_size]
                for fpath in batch:
                    fpath = fpath.strip()
                    if not fpath:
                        continue

                    # Skip binary-looking extensions
                    ext = os.path.splitext(fpath)[1].lower()
                    from tools.file_operations import BINARY_EXTENSIONS
                    if ext in BINARY_EXTENSIONS:
                        continue

                    # Read file content — use head to cap size
                    read_result = file_ops._exec(
                        f"head -c {self.MAX_FILE_SIZE} {file_ops._escape_shell_arg(fpath)} 2>/dev/null",
                        timeout=5
                    )
                    content = read_result.stdout
                    if not content:
                        continue

                    # Check for binary content (null bytes)
                    if '\x00' in content:
                        continue

                    trigrams = _extract_trigrams(content)
                    for tri in trigrams:
                        if tri not in index:
                            index[tri] = set()
                        index[tri].add(fpath)
                    indexed.add(fpath)

            with self._lock:
                self._index = index
                self._indexed_files = indexed
                self._file_count = len(indexed)
                self._root = root
                self._built = True
                self._building = False
                self._build_time = time.time() - start

            logger.info(
                "Trigram index built: %d files, %d unique trigrams, %.1fs",
                len(indexed), len(index), self._build_time
            )
            return True

        except Exception as e:
            logger.warning("Trigram index build failed: %s", e)
            with self._lock:
                self._building = False
                self._skipped = True
            return False

    def get_candidates(self, pattern: str) -> Optional[List[str]]:
        """Get candidate files that might match the pattern.

        Returns None if:
        - Index isn't built
        - Pattern can't be decomposed into trigrams
        - Candidates would be most of the indexed files (not worth filtering)

        Returns a list of file paths if we can narrow down candidates.
        """
        with self._lock:
            if not self._built:
                return None

        trigrams = _extract_trigrams_from_regex(pattern)
        if trigrams is None:
            return None

        with self._lock:
            # Intersect posting lists for all required trigrams
            candidate_sets = []
            for tri in trigrams:
                posting = self._index.get(tri)
                if posting is None:
                    # Trigram not in index at all — no files can match
                    return []
                candidate_sets.append(posting)

            if not candidate_sets:
                return None

            # Start with smallest posting list for efficiency
            candidate_sets.sort(key=len)
            candidates = candidate_sets[0].copy()
            for posting in candidate_sets[1:]:
                candidates &= posting
                if not candidates:
                    return []

            # If candidates are > 60% of all files, not worth filtering
            if len(candidates) > 0.6 * self._file_count:
                return None

            return list(candidates)

    def invalidate_file(self, file_path: str) -> None:
        """Remove a file from the index (after write/patch).

        We don't re-index it — the file will be picked up by the
        fallback rg scan since it won't be in the candidate list.
        Actually, we add it to every future candidate set so it's
        always included in searches after modification.
        """
        with self._lock:
            if not self._built:
                return
            # Remove from all posting lists
            for tri_set in self._index.values():
                tri_set.discard(file_path)
            # Mark as "always include" — modified files should always
            # be searched since we don't know their new trigrams
            self._indexed_files.discard(file_path)

    def add_always_include(self, file_path: str) -> None:
        """Mark a file to always be included in candidate lists.

        Used for files that have been written/patched — since we don't
        re-index them, they need to be in every search result.
        """
        # We handle this in get_candidates by checking _dirty_files
        pass

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._built

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "built": self._built,
                "building": self._building,
                "skipped": self._skipped,
                "file_count": self._file_count,
                "trigram_count": len(self._index),
                "build_time_s": round(self._build_time, 2),
                "root": self._root,
            }


# ---------------------------------------------------------------------------
# Per-task cache/index manager
# ---------------------------------------------------------------------------

class SearchAccelerator:
    """Manages both the result cache and trigram index for a task.

    One instance per task_id. Created lazily and stored in a module-level dict.
    """

    def __init__(self):
        self.result_cache = ResultCache()
        self.trigram_index = TrigramIndex()
        self._dirty_files: Set[str] = set()  # files modified, always include in searches

    def on_file_written(self, file_path: str) -> None:
        """Called when a file is written or patched."""
        self.result_cache.invalidate(file_path)
        self.trigram_index.invalidate_file(file_path)
        self._dirty_files.add(file_path)

    def get_candidates_for_search(self, pattern: str) -> Optional[List[str]]:
        """Get candidate files from trigram index, including dirty files."""
        candidates = self.trigram_index.get_candidates(pattern)
        if candidates is None:
            return None

        # Always include dirty files (modified since index was built)
        if self._dirty_files:
            candidate_set = set(candidates)
            candidate_set.update(self._dirty_files)
            return list(candidate_set)

        return candidates

    def clear(self) -> None:
        self.result_cache.clear()
        self._dirty_files.clear()
        # trigram index is not clearable — it's rebuilt from scratch

    @property
    def stats(self) -> dict:
        return {
            "result_cache": self.result_cache.stats,
            "trigram_index": self.trigram_index.stats,
            "dirty_files": len(self._dirty_files),
        }


# ---------------------------------------------------------------------------
# Module-level registry
# ---------------------------------------------------------------------------

_accelerators: Dict[str, SearchAccelerator] = {}
_accelerators_lock = threading.Lock()


def get_accelerator(task_id: str = "default") -> SearchAccelerator:
    """Get or create a SearchAccelerator for a task."""
    with _accelerators_lock:
        if task_id not in _accelerators:
            _accelerators[task_id] = SearchAccelerator()
        return _accelerators[task_id]


def clear_accelerator(task_id: str = None) -> None:
    """Clear accelerators for a task (or all)."""
    with _accelerators_lock:
        if task_id:
            acc = _accelerators.pop(task_id, None)
            if acc:
                acc.clear()
        else:
            for acc in _accelerators.values():
                acc.clear()
            _accelerators.clear()


def notify_file_written(file_path: str, task_id: str = "default") -> None:
    """Convenience: notify the accelerator that a file was modified."""
    acc = get_accelerator(task_id)
    acc.on_file_written(file_path)
