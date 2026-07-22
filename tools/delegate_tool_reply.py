"""``delegate_tool_reply`` — explicit delivery channel for subagent results.

A leaf-subagent-only tool that hands the deliverable back to the parent agent
through a structured tool call instead of relying on the trailing
``final_response`` prose. This closes the information-loss bug where a
subagent's real result (emitted on a turn that also called a housekeeping /
cleanup tool like ``terminal``) gets overwritten by a short closing comment
on a later turn.

Why a tool and not just ``final_response``?

The agent loop treats content emitted on a tool-calling turn as mid-task
narration (``_last_content_with_tools`` in ``conversation_loop.py``), so it
is **not** promoted to ``final_response`` when a later turn produces text.
For a subagent whose job is to *produce a deliverable*, the trailing prose is
a fragile proxy for the real result — a one-line "done, cleaned up" closer
clobbers the six-block report. ``delegate_tool_reply`` gives the subagent an
explicit, in-band channel: the deliverable is the tool's ``content`` arg,
and the parent's extraction layer reads that arg directly.

Visibility (zero core footprint):

This tool lives in the ``delegation_reply`` toolset, which is **not** in
``_HERMES_CORE_TOOLS`` (so no platform bundle auto-includes it) and **not** in
``CONFIGURABLE_TOOLSETS`` (so it never appears in ``hermes tools`` / ``/tools``).
Only ``_build_child_agent`` in ``delegate_tool.py`` adds ``delegation_reply``
to a child's toolset, so the tool is only ever visible to subagents spawned by
``delegate_task``. Ordinary conversations never see the schema.

Execution model:

The handler runs in the agent's Python process (like ``todo`` / ``memory``),
not in the terminal sandbox — so the spill file lands on the agent host and is
reachable by the parent's ``read_file`` regardless of whether the child's
terminal points at a remote Docker / SSH / Modal backend.

Compression resilience:

The deliverable is recorded in **two** places at call time:

1. **Agent-instance state** (``child._delegate_reply_chunks``) — a list the
   handler appends to at execution time. ``_run_single_child`` reads this list
   after the child finishes; it is never touched by context compression,
   which only mutates the ``messages`` transcript. This is the authoritative
   source.
2. **Spill file** (``cache/delegation/delegate_reply_*.txt``) — a backup on
   disk, mirroring ``_spill_summary_to_file``. Useful if the agent instance
   is somehow lost (e.g. timeout where the future result is unavailable but
   the child object still exists).

Because the agent-instance state lives outside the ``messages`` list, context
compression — which replaces the middle transcript with a summary
(``context_compressor.py`` Phase 4) — cannot destroy the recorded delivery.
The earlier approach of scanning ``result["messages"]`` for tool-call args +
spill paths was vulnerable: a delivery call that fell into the compacted
window lost both its args *and* its tool-result spill path.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

_TOOL_NAME = "delegate_tool_reply"

DELEGATE_TOOL_REPLY_SCHEMA = {
    "name": _TOOL_NAME,
    "description": (
        "Hand back your final result to the parent agent. Call this with the "
        "complete deliverable text as `content`. You may call it multiple "
        "times to deliver in chunks — every call's content is appended in "
        "order to form the final result. This does NOT stop you; finish any "
        "cleanup afterward. Always deliver your real result through this "
        "tool, not as a trailing prose comment."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The complete deliverable text for the parent agent.",
            },
        },
        "required": ["content"],
    },
}


def _spill_reply_to_file(content: str, subagent_id: Optional[str]) -> Optional[str]:
    """Write the deliverable to the delegation cache and return the abs path.

    Mirrors ``_spill_summary_to_file`` in ``delegate_tool.py``: the file lands
    in ``cache/delegation`` which is mounted read-only into remote backends
    (Docker/Modal/SSH) via ``credential_files._CACHE_DIRS``, so the parent's
    ``terminal`` / ``read_file`` tools can page through the complete text on any
    backend. Returns the absolute path, or ``None`` on failure (best-effort).
    """
    try:
        from hermes_constants import get_hermes_dir
        import datetime as _dt

        cache_dir = get_hermes_dir("cache/delegation", "delegation_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        sid = subagent_id or "unknown"
        # Sanitize the subagent id for use in a filename.
        safe_sid = "".join(c if c.isalnum() or c in "-_" else "_" for c in sid)[:64]
        path = cache_dir / f"delegate_reply-{safe_sid}-{ts}.txt"
        path.write_text(content, encoding="utf-8")
        return str(path)
    except Exception as exc:
        logger.debug("Failed to spill delegate_tool_reply content to file: %s", exc)
        return None


def delegate_tool_reply(content: str, parent_agent=None, **kw) -> str:
    """Record a subagent deliverable and spill it to disk.

    Appends ``content`` to ``parent_agent._delegate_reply_chunks`` (the
    authoritative store) and writes a spill-file backup. Does **not** terminate
    the subagent loop — the child keeps running (e.g. cleanup) to natural end.
    ``_run_single_child`` in ``delegate_tool.py`` reads
    ``child._delegate_reply_chunks`` after the child completes.

    Args:
        content: the full deliverable text (one chunk).
        parent_agent: the child AIAgent instance (threaded in by the registry
            via ``kw["parent_agent"]``).

    Returns:
        JSON string ``{"acknowledged": true, "path": <abs path or null>}``.
    """
    if not isinstance(content, str):
        content = str(content) if content is not None else ""

    # Record in agent-instance state — compression-safe (not in messages[]).
    if parent_agent is not None:
        chunks = getattr(parent_agent, "_delegate_reply_chunks", None)
        if chunks is None:
            chunks = []
            setattr(parent_agent, "_delegate_reply_chunks", chunks)
        chunks.append(content)

    # Spill to disk as a backup (best-effort).
    subagent_id = getattr(parent_agent, "_subagent_id", None) if parent_agent is not None else None
    spill_path = _spill_reply_to_file(content, subagent_id)
    return json.dumps(
        {"acknowledged": True, "path": spill_path},
        ensure_ascii=False,
    )


def _handle_delegate_tool_reply(args, **kw):
    content = args.get("content", "")
    if not isinstance(content, str):
        content = str(content) if content is not None else ""
    return delegate_tool_reply(content=content, parent_agent=kw.get("parent_agent"))


def check_delegate_reply_requirements() -> bool:
    """No external requirements — always available when the toolset is enabled.

    Mirrors ``check_delegate_requirements`` in ``delegate_tool.py``: visibility
    is governed purely by toolset membership (the ``delegation_reply``
    toolset is only added to child agents by ``_build_child_agent``).
    """
    return True


registry.register(
    name=_TOOL_NAME,
    toolset="delegation_reply",
    schema=DELEGATE_TOOL_REPLY_SCHEMA,
    handler=_handle_delegate_tool_reply,
    check_fn=check_delegate_reply_requirements,
    emoji="📨",
)
