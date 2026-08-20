"""``delegate_tool_reply`` — explicit delivery channel for subagent results.

A subagent-only tool that hands the deliverable back to the parent agent
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

Compression resilience:

The deliverable is recorded in ``child._delegate_reply_chunks`` at execution
time. ``_run_single_child`` reads this list after the child finishes; it is
never touched by context compression, which only mutates the ``messages``
transcript. Oversized final summaries still use the existing bounded summary
spill path after extraction, so every call does not duplicate potentially
sensitive output on disk.

Because the agent-instance state lives outside the ``messages`` list, context
compression — which replaces the middle transcript with a summary
(``context_compressor.py`` Phase 4) — cannot destroy the recorded delivery.
The earlier approach of scanning ``result["messages"]`` for tool-call args +
spill paths was vulnerable: a delivery call that fell into the compacted
window lost both its args *and* its tool-result spill path.
"""
from __future__ import annotations

import json
from tools.registry import registry

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


def delegate_tool_reply(content: str, parent_agent=None, **kw) -> str:
    """Record a subagent deliverable on the executing child agent.

    Appends ``content`` to ``parent_agent._delegate_reply_chunks`` (the
    authoritative store). Does **not** terminate the subagent loop — the child
    keeps running (e.g. cleanup) to natural end. ``_run_single_child`` in
    ``delegate_tool.py`` reads ``child._delegate_reply_chunks`` after the child
    completes.

    Args:
        content: the full deliverable text (one chunk).
        parent_agent: the child AIAgent instance (threaded in by the registry
            via ``kw["parent_agent"]``).

    Returns:
        JSON acknowledgement. Missing agent context fails closed because an
        acknowledgement without recording would silently lose the result.
    """
    if not isinstance(content, str):
        content = str(content) if content is not None else ""

    if parent_agent is None:
        return json.dumps(
            {
                "acknowledged": False,
                "error": "delegate_tool_reply requires the executing subagent context",
            }
        )

    # Agent-instance state is compression-safe because it is not in messages[].
    chunks = getattr(parent_agent, "_delegate_reply_chunks", None)
    if chunks is None:
        chunks = []
        setattr(parent_agent, "_delegate_reply_chunks", chunks)
    chunks.append(content)

    return json.dumps({"acknowledged": True})


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
