"""
Tool: headroom_retrieve

Convenience tool for retrieving original tool outputs after headroom
compression. When a tool output was replaced with a placeholder indicating
it was cleared to save context space, this tool provides the retrieval
instructions — the actual retrieval is done by the session_search tool.

Ride-along tool — the real retrieval fix is the _PRUNED_TOOL_PLACEHOLDER
change in context_compressor.py telling the model to call session_search.

Capped at 2 retrievals per key to prevent compress→retrieve→re-compress loops (G3).

KENSEI CUSTOM — fork patch.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

NAME: str = "headroom_retrieve"

TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "Look up how to retrieve original tool outputs that were compressed. "
            "Returns retrieval instructions that tell the model exactly how to use "
            "session_search to recover compressed content. "
            "Capped at 2 calls per unique query to prevent retrieval loops."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": (
                        "A keyword, tool name, or content fragment that identifies "
                        "the compressed content you want to retrieve. "
                        "Example: 'web_extract', 'pytest output', 'terminal'"
                    ),
                },
                "tool_name": {
                    "type": "string",
                    "description": (
                        "Optional: the specific tool whose output was compressed. "
                        "Helps narrow the search. Example: 'terminal', 'web_extract'"
                    ),
                },
            },
            "required": ["key"],
        },
    },
}

# Per-session retrieval cap: max 2 per distinct key
_retrieval_caps: Dict[str, int] = {}


def reset_caps() -> None:
    """Reset per-session retrieval caps (called at session start)."""
    global _retrieval_caps
    _retrieval_caps.clear()


def handler(
    key: str,
    tool_name: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Return retrieval instructions pointing the model to session_search.

    Does NOT perform the retrieval itself — session_search is a model-facing
    tool that handles the actual database query. This function provides the
    correct invocation pattern and enforces the retrieval cap.
    """
    cap_key = f"{tool_name or '*'}:{key}"
    count = _retrieval_caps.get(cap_key, 0)
    if count >= 2:
        return (
            f"[Retrieval capped for '{key}' — already retrieved {count} times. "
            f"Use the existing retrieved content rather than requesting again.]"
        )
    _retrieval_caps[cap_key] = count + 1

    # Build the session_search guidance
    search_query = key
    if tool_name:
        search_query = f"{tool_name} {key}"

    return (
        f"To retrieve the original content for '{key}', call the session_search tool "
        f"with these parameters:\n\n"
        f"  session_search(\n"
        f"    query=\"{search_query}\",\n"
        f"    role_filter=\"tool\"\n"
        f"  )\n\n"
        f"The session database preserves full tool outputs. "
        f"Look for messages matching your query in the results. "
        f"If the tool output was compressed by headroom, use "
        f"a content keyword or the tool name as the query.\n\n"
        f"[Retrieval {_retrieval_caps[cap_key]}/2 for '{key}'. "
        f"{2 - _retrieval_caps[cap_key]} remaining.]"
    )
