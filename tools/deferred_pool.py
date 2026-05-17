"""Deferred tool pool for ToolSearch (MCP context-bloat mitigation).

When ``mcp.tool_search.enabled: true``, MCP tool schemas are removed from
the per-call ``self.tools`` array and parked here. The model sees only a
compact list of (name, 1-line summary) in the system message and uses the
``tool_search`` tool to fetch full schemas on demand. On a successful
select, the schema is promoted back into ``self.tools`` / ``valid_tool_names``
so subsequent calls dispatch normally.

Thread-safe; designed for the long-lived Gateway process where MCP refresh
notifications can mutate the pool while another agent thread is reading.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Set, Tuple


class DeferredToolPool:
    __slots__ = ("_entries", "_promoted", "_lock", "_generation")

    def __init__(self) -> None:
        self._entries: Dict[str, Dict[str, Any]] = {}
        # Names the model has fetched via tool_search and should stay visible
        # in self.tools across repopulation passes.
        self._promoted: Set[str] = set()
        self._lock = threading.RLock()
        self._generation: int = 0

    def put(self, name: str, schema: Dict[str, Any], summary: str, toolset: str) -> None:
        with self._lock:
            self._entries[name] = {
                "name": name,
                "schema": schema,
                "summary": summary,
                "toolset": toolset,
            }
            self._generation += 1

    def remove(self, name: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._entries.pop(name, None)
            if entry is not None:
                self._promoted.add(name)
                self._generation += 1
            return entry

    def is_promoted(self, name: str) -> bool:
        with self._lock:
            return name in self._promoted

    def promoted_names(self) -> Set[str]:
        with self._lock:
            return set(self._promoted)

    def reset_promotions(self) -> None:
        """Clear the promoted set — call when MCP servers refresh and the
        per-session promotion state should be discarded."""
        with self._lock:
            if self._promoted:
                self._promoted.clear()
                self._generation += 1

    def remove_by_toolset(self, toolset: str) -> List[str]:
        with self._lock:
            removed = [n for n, e in self._entries.items() if e["toolset"] == toolset]
            for n in removed:
                self._entries.pop(n, None)
            if removed:
                self._generation += 1
            return removed

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._entries.get(name)

    def names(self) -> List[str]:
        with self._lock:
            return list(self._entries.keys())

    def items(self) -> List[Tuple[str, Dict[str, Any]]]:
        with self._lock:
            return list(self._entries.items())

    def summaries(self) -> List[Tuple[str, str, str]]:
        """Return (name, toolset, summary) for system-message rendering."""
        with self._lock:
            return [(n, e["toolset"], e["summary"]) for n, e in self._entries.items()]

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._entries

    @property
    def generation(self) -> int:
        return self._generation

    def clear(self) -> None:
        """Clear pool entries only — preserves promotion history.

        Use ``reset_promotions()`` separately to forget which tools the model
        has already fetched (e.g. after an MCP server schema refresh).
        """
        with self._lock:
            self._entries.clear()
            self._generation += 1


_pool = DeferredToolPool()


def get_pool() -> DeferredToolPool:
    return _pool
