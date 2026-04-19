#!/usr/bin/env python3
"""
Compress Context Tool Module - Autonomous Context Management

Provides a native tool for agents to autonomously compress conversation context
during execution. Agents can trigger compression when needed to reduce token usage
or before large tasks. This wraps the existing ContextCompressor logic.
"""

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def compress_context_tool(
    protect_last_n: int = 20,
    target_ratio: float = 0.20,
    focus_topic: Optional[str] = None,
    agent: Optional[Any] = None,
) -> str:
    """
    Compress conversation context to reduce token usage.

    Summarizes older messages while protecting recent context. Use when:
    - Conversation grows too long (approaching context limit)
    - Before executing large tasks that need headroom
    - When agent decides context management is needed

    Args:
        protect_last_n: how many recent messages to keep uncompressed (default: 20)
        target_ratio: target compression ratio 0-1 (default: 0.20 = 20%)
        focus_topic: optional topic to emphasize in summary
        agent: the AIAgent instance (passed via kw from run_agent)

    Returns:
        JSON string with compression results: message counts before/after,
        token savings, and whether compression was performed.
    """
    if agent is None:
        return tool_error("Compress tool requires agent context")

    compressor = getattr(agent, "context_compressor", None)
    if not compressor:
        return tool_error("Context compressor not initialized on agent")

    try:
        # Signal to the agent that compression was requested
        # The agent loop will check for this flag and perform actual compression
        # in its main control flow where it can properly manage session state

        # Store compression request parameters on the agent for the loop to pick up
        if not hasattr(agent, '_pending_compression_request'):
            agent._pending_compression_request = {}

        agent._pending_compression_request = {
            'protect_last_n': protect_last_n,
            'target_ratio': target_ratio,
            'focus_topic': focus_topic,
        }

        # Return status indicating compression was requested
        return json.dumps({
            "status": "compression_queued",
            "message": (
                "Compression request queued. The agent's main loop will perform "
                "compression at the next opportunity (typically after the current "
                "tool finishes processing)."
            ),
            "parameters": {
                "protect_last_n": protect_last_n,
                "target_ratio": target_ratio,
                "focus_topic": focus_topic or "none",
            },
            "note": (
                "Actual compression happens asynchronously in the agent loop. "
                "You'll see token reduction in subsequent API calls."
            ),
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Compression request failed: {e}", exc_info=True)
        return tool_error(f"Compression request failed: {str(e)}")


def check_compress_context_requirements() -> bool:
    """Compress context tool is always available if agent is running."""
    return True


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

COMPRESS_CONTEXT_SCHEMA = {
    "name": "compress_context",
    "description": (
        "Compress conversation context to reduce token usage and free up context space. "
        "Summarizes older messages while protecting recent context. "
        "Use when conversation grows too long or before large tasks.\n\n"
        "Parameters (all optional, use defaults for standard compression):\n"
        "- protect_last_n: how many recent messages to keep uncompressed (default: 20)\n"
        "- target_ratio: compression target as ratio 0-1 (e.g., 0.20 = 20%, default: 0.20)\n"
        "- focus_topic: optional topic to emphasize in summary\n\n"
        "Compression happens asynchronously in the agent's main loop. You'll see "
        "token reduction in subsequent API calls after compression completes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "protect_last_n": {
                "type": "integer",
                "description": "Number of recent messages to preserve uncompressed",
                "default": 20,
                "minimum": 5,
                "maximum": 100,
            },
            "target_ratio": {
                "type": "number",
                "description": "Target compression ratio (e.g., 0.20 = compress to 20% of original size)",
                "default": 0.20,
                "minimum": 0.05,
                "maximum": 0.5,
            },
            "focus_topic": {
                "type": "string",
                "description": (
                    "Optional: guide the summarizer to emphasize this topic while compressing. "
                    "Useful for preserving context about specific ongoing work."
                ),
            },
        },
        "required": [],
    }
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="compress_context",
    toolset="memory",
    schema=COMPRESS_CONTEXT_SCHEMA,
    handler=lambda args, **kw: compress_context_tool(
        protect_last_n=args.get("protect_last_n", 20),
        target_ratio=args.get("target_ratio", 0.20),
        summary_model=args.get("summary_model"),
        summary_provider=args.get("summary_provider"),
        focus_topic=args.get("focus_topic"),
        agent=kw.get("agent")
    ),
    check_fn=check_compress_context_requirements,
    emoji="🗜️",
)
