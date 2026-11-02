"""Token counters of a parent's finished subagents.

A ``delegate_task`` child is its own ``AIAgent`` with its own ``session_*_tokens``. When it is torn
down, its counters (and those of its own subagents) are added to the parent's subagent totals, so a
caller reporting the parent's usage can include the children. The parent's ``session_*`` counters
are left alone: the session token budget checks read them, and the children's spend must not stop
the parent early.

Folding happens in the child's teardown (``_ChildRun.cleanup``), which runs for completed and
interrupted children alike. An interrupted parallel batch does not join its workers, so a child can
be folded after the parent's turn has ended; readers that report per turn must use the totals they
saw at their previous report as the base, not a snapshot taken when the turn started.
"""
from __future__ import annotations

from typing import Any

from tools.delegate_tool_results import _parent_finalization_lock

TOKEN_COUNTERS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens")

_TOTALS_ATTR = "_subagent_token_totals"
_FOLDED_ATTR = "_subagent_tokens_folded"


def _count(value: Any) -> int:
    return value if type(value) is int and value > 0 else 0


def subagent_tokens(agent: Any) -> dict[str, int]:
    """The tokens of the agent's folded subagents, all counters present (zeros when none)."""
    totals = getattr(agent, _TOTALS_ATTR, None)
    if not isinstance(totals, dict):
        totals = {}
    return {name: _count(totals.get(name)) for name in TOKEN_COUNTERS}


def fold_child_tokens(parent_agent: Any, child: Any) -> None:
    """Add ``child``'s tokens and its own subagents' to ``parent_agent``'s subagent totals, once.

    Siblings of a parallel batch tear down on their own worker threads, so the read-add-write runs
    under the parent's finalization lock; teardown is idempotent, so a child is folded only once.
    """
    if parent_agent is None or child is None:
        return
    with _parent_finalization_lock(parent_agent):
        if getattr(child, _FOLDED_ATTR, False) is True:
            return
        totals = subagent_tokens(parent_agent)
        nested = subagent_tokens(child)
        for name in TOKEN_COUNTERS:
            totals[name] += _count(getattr(child, f"session_{name}", 0)) + nested[name]
        setattr(parent_agent, _TOTALS_ATTR, totals)
        setattr(child, _FOLDED_ATTR, True)
