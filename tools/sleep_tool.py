#!/usr/bin/env python3
"""
Sleep Memory Tool - Memory consolidation and cleaning for Hermes Agent.

Provides tools to clean and consolidate memories based on session usage patterns.
Inspired by human sleep-memory consolidation mechanisms.

Key features:
1. Session analysis: Analyzes historical sessions to identify important vs unimportant content
2. Adaptive vocabulary learning: Learns important/unimportant words from user's own session patterns
3. Memory scoring: Scores memories based on learned vocabulary and length
4. Intelligent cleaning: Removes low-scoring memories while preserving important information

The tool is self-adaptive — it learns from each user's unique interaction patterns,
avoiding hardcoded rules that might not generalize across different users.
"""

import json
import logging
from typing import Optional

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Core tool functions
# -----------------------------------------------------------------------------

def check_sleep_requirements() -> bool:
    """Check if sleep tool requirements are met."""
    # No external dependencies required
    return True

def sleep_memory(
    mode: str = "quick",
    apply_changes: bool = False,
    task_id: Optional[str] = None,
) -> str:
    """
    Perform memory consolidation (sleep mode).
    
    Args:
        mode: "quick" for fast filtering, "deep" for thorough analysis
        apply_changes: When True, persist deletions instead of previewing them
        task_id: Optional task ID for context
    
    Returns:
        JSON string with results and statistics
    """
    try:
        # Import here to avoid circular dependencies
        from agent.sleep_engine import SleepEngine
        from tools.memory_tool import MemoryStore
        
        # Load memory store
        memory_store = MemoryStore()
        memory_store.load_from_disk()
        
        # Create sleep engine
        engine = SleepEngine(memory_store)
        
        # Run sleep
        report = engine.sleep(mode, apply_changes=apply_changes)
        
        # Convert to JSON string
        return json.dumps(report, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Sleep memory failed: {e}")
        return tool_error(f"Sleep memory failed: {e}")

SLEEP_MEMORY_SCHEMA = {
    "name": "sleep_memory",
    "description": (
        "Perform memory consolidation (sleep mode) to clean redundant memories "
        "based on session usage patterns. Analyzes historical sessions to learn "
        "important vs unimportant vocabulary, then scores and filters memories. "
        "Mode: 'quick' for fast filtering, 'deep' for thorough analysis. "
        "By default this is a dry run; set apply_changes=true to write the filtered "
        "memory set back to MEMORY.md. "
        "Returns detailed statistics including sessions analyzed, memories deleted, "
        "vocabulary learned, and space saved."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "description": "Sleep mode: 'quick' (fast filtering) or 'deep' (thorough analysis)",
                "enum": ["quick", "deep"],
                "default": "quick"
            },
            "apply_changes": {
                "type": "boolean",
                "description": "When true, persist the filtered memory set to MEMORY.md. When false, preview only.",
                "default": False,
            }
        },
        "required": []
    }
}

# -----------------------------------------------------------------------------
# Tool registration
# -----------------------------------------------------------------------------

registry.register(
    name="sleep_memory",
    toolset="memory",
    schema=SLEEP_MEMORY_SCHEMA,
    handler=lambda args, **kw: sleep_memory(
        mode=args.get("mode", "quick"),
        apply_changes=args.get("apply_changes", False),
        task_id=kw.get("task_id")
    ),
    check_fn=check_sleep_requirements,
    emoji="🌙",
)
