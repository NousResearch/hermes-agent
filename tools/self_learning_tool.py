#!/usr/bin/env python3
"""
Self-Learning Tool - Failure Analysis & Skill Evolution

Provides tools for analyzing tool call failures, detecting patterns,
extracting lessons, and suggesting skill improvements.

Part of Vibe Coding enhancements for Hermes Agent.
"""

import json
from typing import Dict, Any, Optional

from tools.registry import registry
from agent.self_learning import (
    SelfLearner,
    analyze_failure,
    get_recurring_failures,
    get_pending_skill_suggestions,
)


def _check_requirements() -> bool:
    """Self-learning is always available (no external dependencies)."""
    return True


def _handle_analyze_failure(args: Dict[str, Any], **kwargs) -> str:
    """Handle analyze_failure tool call."""
    tool_name = args.get("tool_name", "")
    tool_args = args.get("args", {})
    error_message = args.get("error_message", "")
    context = args.get("context", {})
    
    if not tool_name or not error_message:
        return json.dumps({
            "error": "tool_name and error_message are required",
            "status": "failed",
        })
    
    result = analyze_failure(tool_name, tool_args, error_message, context)
    return json.dumps(result, indent=2)


def _handle_get_lessons(args: Dict[str, Any], **kwargs) -> str:
    """Handle get_learned_lessons tool call."""
    learner = SelfLearner()
    lessons = learner.get_learned_lessons()
    return json.dumps({
        "lessons": lessons,
        "count": len(lessons),
    }, indent=2)


def _handle_get_failures(args: Dict[str, Any], **kwargs) -> str:
    """Handle get_failure_history tool call."""
    tool_name = args.get("tool_name")  # Optional filter
    learner = SelfLearner()
    history = learner.get_failure_history(tool_name)
    return json.dumps({
        "failures": history,
        "count": len(history),
    }, indent=2)


def _handle_suggest_skill(args: Dict[str, Any], **kwargs) -> str:
    """Handle suggest_skill_update tool call."""
    skill_name = args.get("skill_name")  # Optional
    learner = SelfLearner()
    suggestion = learner.suggest_skill_update(skill_name)
    return json.dumps(suggestion, indent=2)


def _handle_clear_history(args: Dict[str, Any], **kwargs) -> str:
    """Handle clear_failure_history tool call."""
    learner = SelfLearner()
    learner.clear_history()
    return json.dumps({
        "status": "cleared",
        "message": "In-memory failure history cleared. Learned lessons are preserved.",
    })


def _handle_get_memory_entries(args: Dict[str, Any], **kwargs) -> str:
    """Handle get_memory_entries tool call - for memory tool integration."""
    learner = SelfLearner()
    entries = learner.get_memory_entries()
    return json.dumps({
        "memory_entries": entries,
        "count": len(entries),
    }, indent=2)


# Register main analysis tool
registry.register(
    name="analyze_failure",
    toolset="core",
    schema={
        "name": "analyze_failure",
        "description": (
            "Analyze a tool call failure to extract lessons and detect patterns. "
            "Use this when a tool call fails to understand what went wrong and get "
            "suggestions for avoiding similar failures. Tracks recurring failure patterns "
            "(3+ occurrences) and generates learned lessons automatically."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Name of the tool that failed (e.g., 'terminal', 'read_file')",
                },
                "args": {
                    "type": "object",
                    "description": "Arguments that were passed to the failing tool call",
                },
                "error_message": {
                    "type": "string",
                    "description": "The error message from the failed tool call",
                },
                "context": {
                    "type": "object",
                    "description": "Additional context (cwd, session_id, etc.)",
                },
            },
            "required": ["tool_name", "error_message"],
        },
    },
    handler=_handle_analyze_failure,
    check_fn=_check_requirements,
    requires_env=[],
    is_async=False,
    description="Analyze tool call failures and detect patterns",
    emoji="🧠",
)

# Register lesson retrieval tool
registry.register(
    name="get_learned_lessons",
    toolset="core",
    schema={
        "name": "get_learned_lessons",
        "description": (
            "Retrieve all learned lessons from failure analysis. "
            "Use this to review what the agent has learned from past failures."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    handler=_handle_get_lessons,
    check_fn=_check_requirements,
    requires_env=[],
    is_async=False,
    description="Get learned lessons from failure analysis",
    emoji="📚",
)

# Register failure history tool
registry.register(
    name="get_failure_history",
    toolset="core",
    schema={
        "name": "get_failure_history",
        "description": (
            "Get failure history across all tool calls. "
            "Optionally filter by tool name. Shows patterns and occurrence counts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Filter to specific tool (optional)",
                },
            },
        },
    },
    handler=_handle_get_failures,
    check_fn=_check_requirements,
    requires_env=[],
    is_async=False,
    description="Get failure history with pattern detection",
    emoji="📊",
)

# Register skill suggestion tool
registry.register(
    name="suggest_skill_update",
    toolset="core",
    schema={
        "name": "suggest_skill_update",
        "description": (
            "Generate a skill update suggestion based on failure patterns. "
            "Use this after analyzing failures to get recommendations for creating "
            "or updating skills to avoid similar issues in the future."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Existing skill to update (optional, creates new if omitted)",
                },
            },
        },
    },
    handler=_handle_suggest_skill,
    check_fn=_check_requirements,
    requires_env=[],
    is_async=False,
    description="Suggest skill updates from failure patterns",
    emoji="💡",
)

# Register clear history tool
registry.register(
    name="clear_failure_history",
    toolset="core",
    schema={
        "name": "clear_failure_history",
        "description": (
            "Clear in-memory failure history. "
            "Learned lessons are preserved on disk. "
            "Use this to reset pattern detection for a new session."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    handler=_handle_clear_history,
    check_fn=_check_requirements,
    requires_env=[],
    is_async=False,
    description="Clear failure history",
    emoji="🧹",
)