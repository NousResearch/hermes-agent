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

If a long-running subagent triggers context compression, the compressor
(``context_compressor.py``) truncates ``tool_call`` args > 500 chars for
assistant messages outside the protected tail. To survive that, the handler
spills the full ``content`` to ``cache/delegation/delegate_reply_*.txt`` and
returns the absolute path in its result; the extraction layer prefers the
spill file over the (possibly truncated) args.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

_TOOL_NAME = "delegate_tool_reply"
_TRUNCATED_MARKER = "...[truncated]"

DELEGATE_TOOL_REPLY_SCHEMA = {
    "name": _TOOL_NAME,
    "description": (
        "Hand back your final result to the parent agent. Call this with the "
        "complete deliverable text as `content`. You may call it multiple "
        "times to deliver in chunks (they are concatenated in order), or "
        "update it by calling again — the last value per chunk is used. This "
        "does NOT stop you; finish any cleanup afterward. Always deliver your "
        "real result through this tool, not as a trailing prose comment."
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
    backend. Returns the absolute path, or ``None`` on failure (best-effort:
    extraction then falls back to the in-memory args).
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
    """Acknowledge a subagent deliverable and spill it to disk.

    No side effects beyond writing the spill file. Does **not** terminate the
    subagent loop — the child keeps running (e.g. cleanup) to natural end.
    The parent's extraction layer (``delegate_tool.py``) reads this call's
    args/result after the child completes.

    Args:
        content: the full deliverable text.
        parent_agent: the child AIAgent instance (threaded in by the registry
            via ``kw["parent_agent"]``); used only for the subagent id when
            naming the spill file.

    Returns:
        JSON string ``{"acknowledged": true, "path": <abs path or null>}``.
    """
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