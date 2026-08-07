"""tools/claude_session/output_buffer.py — Ring buffer for tmux output."""

import hashlib
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OutputLine:
    """A single line of captured output."""
    text: str
    timestamp: float
    index: int = 0  # Global monotonic index

    def content_hash(self) -> str:
        return hashlib.md5(self.text.encode()).hexdigest()


class OutputBuffer:
    """Thread-safe ring buffer that stores deduplicated tmux output lines.

    Features:
      - Fixed capacity with automatic eviction of oldest lines
      - Content-hash deduplication for repeated capture-pane output
      - Marker-based range queries (for output_since_send)
      - Pagination via offset/limit
    """

    _HASH_GC_THRESHOLD = 100  # rebuild hash set every N appends to prevent unbounded growth

    def __init__(self, max_lines: int = 1000):
        self._max_lines = max_lines
        self._lines: deque = deque(maxlen=max_lines)
        self._counter = 0
        self._lock = threading.Lock()
        self._last_seen_hashes: set = set()
        self._append_since_gc = 0

    def _maybe_gc_hashes(self) -> None:
        """Rebuild hash set from current deque contents if threshold exceeded."""
        if self._append_since_gc < self._HASH_GC_THRESHOLD:
            return
        self._last_seen_hashes = {line.content_hash() for line in self._lines}
        self._append_since_gc = 0

    def append(self, text: str) -> int:
        """Append a single line. Returns its global index (marker)."""
        with self._lock:
            self._counter += 1
            ol = OutputLine(
                text=text,
                timestamp=time.monotonic(),
                index=self._counter,
            )
            self._lines.append(ol)
            # Incremental: track hash of new line only
            self._last_seen_hashes.add(ol.content_hash())
            self._append_since_gc += 1
            self._maybe_gc_hashes()
            return self._counter

    def append_batch(self, texts: List[str]) -> int:
        """Append multiple lines, skipping duplicates. Returns count of new lines added."""
        added = 0
        with self._lock:
            for text in texts:
                h = hashlib.md5(text.encode()).hexdigest()
                if h not in self._last_seen_hashes:
                    self._counter += 1
                    self._lines.append(OutputLine(
                        text=text,
                        timestamp=time.monotonic(),
                        index=self._counter,
                    ))
                    self._last_seen_hashes.add(h)
                    added += 1
                    self._append_since_gc += 1
                    # Inline GC check during large batches to prevent unbounded growth
                    if self._append_since_gc >= self._HASH_GC_THRESHOLD:
                        self._maybe_gc_hashes()
        return added

    def read(self, offset: int = 0, limit: Optional[int] = None) -> List[OutputLine]:
        """Read lines with offset/limit pagination."""
        with self._lock:
            total = len(self._lines)
            if total == 0:
                return []
            start = min(offset, total)
            end = total if limit is None else min(start + limit, total)
            return list(self._lines)[start:end]

    def total_count(self) -> int:
        """Total lines ever appended (may exceed current buffer size)."""
        with self._lock:
            return self._counter

    def since(self, marker: int) -> List[OutputLine]:
        """Return all lines appended after the given marker index.

        If the marker has been evicted from the ring, returns all current lines.
        """
        with self._lock:
            result = [line for line in self._lines if line.index > marker]
            if not result and any(line.index <= marker for line in self._lines):
                # Marker was evicted, return everything we have
                return list(self._lines)
            return result

    def last_n_chars(self, n: int) -> str:
        """Return up to n characters from the tail of the buffer."""
        with self._lock:
            text = "\n".join(line.text for line in self._lines)
            return text[-n:] if len(text) > n else text

    def clear(self) -> None:
        """Clear all stored lines. Counter is preserved to keep markers valid."""
        with self._lock:
            self._lines.clear()
            self._last_seen_hashes.clear()
            self._append_since_gc = 0
