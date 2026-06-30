"""Memory relevance diagnostics tools for the agent.

Registered via the tool autodiscovery system. Delegates to
``agent.memory.diagnostics`` for formatting logic.
"""
from __future__ import annotations

from tools.registry import registry, tool_error

from agent.memory.diagnostics import (
    memory_relevance_debug as _debug_handler,
    memory_relevance_stats as _stats_handler,
    RELEVANCE_DEBUG_SCHEMA,
    RELEVANCE_STATS_SCHEMA,
)


def _check() -> bool:
    """Always available — returns gracefully when no data exists."""
    return True


# Register debug tool
registry.register(
    name="memory_relevance_debug",
    toolset="memory",
    schema=RELEVANCE_DEBUG_SCHEMA,
    handler=lambda args, **kw: _debug_handler(
        turns=args.get("turns", 5),
    ),
    check_fn=_check,
    emoji="📊",
)

# Register stats tool
registry.register(
    name="memory_relevance_stats",
    toolset="memory",
    schema=RELEVANCE_STATS_SCHEMA,
    handler=lambda args, **kw: _stats_handler(),
    check_fn=_check,
    emoji="📈",
)
