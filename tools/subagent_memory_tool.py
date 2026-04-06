"""
Subagent Memory Write Tool

Provides append-only, rate-limited memory persistence for subagents.

When the parent enables write_memory=True on delegate_task, the subagent
gets this tool. Writes route through whatever memory provider the parent
has active -- builtin MEMORY.md, mem0, Honcho, or any other registered
provider. If the parent has no memory store, the tool returns a clear error
instead of silently discarding the write.

Constraints (enforced here):
  - Append-only: no read, replace, or delete
  - 3 writes per subagent run (shared rate limit across tool calls)
  - 400 chars max per entry
  - Content tagged [subagent] prefix so the parent can identify the source
"""

import json
import logging
import threading
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

MAX_WRITES = 3
MAX_CHARS = 400


def make_subagent_memory_writer(parent_agent) -> Optional[Callable[[str], Dict[str, Any]]]:
    """Create a rate-limited, append-only write callback for a subagent.

    Returns None if the parent has no memory store (graceful degradation).
    The returned callable is thread-safe.

    parent_agent must be the PARENT (the agent that spawned this subagent),
    not the child. The callback closes over parent_agent._memory_store so
    writes go to the parent's backing store, not the child's (which has
    skip_memory=True and no store).
    """
    store = getattr(parent_agent, "_memory_store", None)
    if store is None:
        return None

    _lock = threading.Lock()
    _write_count = [0]  # mutable list so the closure can mutate the counter

    def _write(content: str) -> Dict[str, Any]:
        with _lock:
            if _write_count[0] >= MAX_WRITES:
                return {
                    "success": False,
                    "error": (
                        f"Write limit reached. Subagents may write at most "
                        f"{MAX_WRITES} entries per run."
                    ),
                    "writes_used": _write_count[0],
                    "writes_max": MAX_WRITES,
                }

            content = content.strip()
            if not content:
                return {"success": False, "error": "Content cannot be empty."}
            if len(content) > MAX_CHARS:
                return {
                    "success": False,
                    "error": (
                        f"Content too long ({len(content)} chars). "
                        f"Max is {MAX_CHARS} chars per entry."
                    ),
                }

            tagged = f"[subagent] {content}"
            result = store.add("memory", tagged)

            if result.get("success"):
                _write_count[0] += 1
                # Notify external providers (mem0, Honcho, etc.)
                mm = getattr(parent_agent, "_memory_manager", None)
                if mm:
                    try:
                        mm.on_memory_write("add", "memory", tagged)
                    except Exception as e:
                        logger.debug("on_memory_write notification failed: %s", e)

                result["writes_used"] = _write_count[0]
                result["writes_remaining"] = MAX_WRITES - _write_count[0]

            return result

    return _write


def subagent_memory_write(content: str, writer: Optional[Callable] = None) -> str:
    """Write a finding to the parent agent's memory.

    Handler called by run_agent.py when the model invokes subagent_memory_write.
    writer is the closure created by make_subagent_memory_writer() and stored on
    the child agent as _subagent_memory_writer.
    """
    if writer is None:
        return json.dumps({
            "success": False,
            "error": (
                "Memory write not available. Either write_memory was not "
                "enabled on delegate_task, or the parent agent has no "
                "memory provider configured."
            ),
        }, ensure_ascii=False)

    result = writer(content)
    return json.dumps(result, ensure_ascii=False)


# NOTE: This tool intentionally does NOT call registry.register().
# The standard registry pattern assumes a static, globally-available handler,
# but subagent_memory_write requires a per-subagent writer closure that is
# created at delegation time and bound to a specific parent agent's _memory_store.
# Instead, delegate_tool.py injects the schema + writer directly onto child.tools
# and child.valid_tool_names, and run_agent.py dispatches via a dedicated elif
# branch in _invoke_tool and _execute_tool_calls_sequential.
SUBAGENT_MEMORY_WRITE_SCHEMA = {
    "name": "subagent_memory_write",
    "description": (
        "Write a key finding to the parent agent's persistent memory. "
        "Use this for facts that should outlast this subagent run -- a root "
        "cause, a confirmed configuration value, a working procedure, or a "
        "constraint discovered during investigation. Not for task progress "
        "or interim notes.\n\n"
        "Constraints:\n"
        f"- Append-only (no read, replace, or delete)\n"
        f"- Max {MAX_WRITES} writes per subagent run\n"
        f"- Max {MAX_CHARS} chars per entry\n"
        "- Entries are tagged [subagent] so the parent can identify them\n\n"
        "When to call this:\n"
        "- You found the root cause of a bug or outage\n"
        "- You confirmed a fact that required significant investigation\n"
        "- You discovered a working procedure or workaround\n"
        "- The finding would require re-running this investigation if lost\n\n"
        "Write one targeted entry per finding. The parent already receives "
        "your full summary -- this is for what needs to survive future sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": (
                    f"The finding to persist. Max {MAX_CHARS} chars. "
                    "Include enough context (names, values, environment) "
                    "that the entry is useful without surrounding context."
                ),
            },
        },
        "required": ["content"],
    },
}
