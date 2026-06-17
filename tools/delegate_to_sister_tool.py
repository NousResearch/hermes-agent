#!/usr/bin/env python3
"""
delegate_to_sister_tool.py

Tool for Astra to delegate tasks to specialized sister agents.
This enables the Astra -> Sister -> Astra hierarchical delegation flow.

The tool uses the existing subagent infrastructure (delegate_task) but
provides a specialized interface that:
1. Resolves the sister by ID from the SisterRegistry
2. Builds the sister's byte-stable system prompt via SisterPromptLoader
3. Spawns a child agent with the sister's identity and appropriate toolsets
4. Returns the sister's response to Astra for synthesis
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from tools.registry import registry, tool_error


# Tool schema
DELEGATE_TO_SISTER_SCHEMA = {
    "name": "delegate_to_sister",
    "description": (
        "Delegate a specialized task to a sister agent. "
        "The sister will execute with her own identity, system prompt, and appropriate toolsets. "
        "Use this when a task requires deep domain expertise (legal, creative, research, code, etc.). "
        "The orchestrator (Astra) receives the result and synthesizes it for the user."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "sister_id": {
                "type": "string",
                "description": "ID of the sister to delegate to (e.g., 'novus', 'helena', 'vitoria', 'luna', 'maya', 'ada', 'bia', 'daine', 'clara', 'larissa', 'nova'). Legacy aliases also work: 'fofoqueiro'->bia, 'vini'->vitoria, 'larissinha'->larissa, 'daiane'->daine.",
            },
            "prompt": {
                "type": "string",
                "description": "The task description for the sister. Be specific and self-contained — the sister knows nothing about your conversation history.",
            },
            "toolsets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional toolset override. If omitted, the sister's default toolsets are used based on her domain.",
            },
            "risk_level": {
                "type": "string",
                "enum": ["low", "standard", "high", "production"],
                "description": "Risk level for this delegation. Determines model routing via HARP. Default: sister's configured risk_level.",
            },
        },
        "required": ["sister_id", "prompt"],
    },
}


def _resolve_sister_toolsets(sister_id: str, override: Optional[List[str]] = None) -> List[str]:
    """Determine appropriate toolsets for a sister based on her domain."""
    if override:
        return override

    # Domain-based toolset defaults
    domain_toolsets = {
        "Core Intelligence": ["coding", "web", "browser", "delegation"],
        "Business & Law": ["web", "file", "terminal", "delegation"],
        "Creative & Support": ["web", "file", "vision", "delegation", "image_gen"],
    }

    # Sister-specific overrides
    sister_overrides = {
        "novus": ["coding", "terminal", "file", "delegation"],  # Local code focus
        "nova": ["browser", "web", "vision", "delegation"],     # Browser/vision focus
        "luna": ["web", "browser", "file", "delegation"],       # Research focus
        "maya": ["coding", "terminal", "file", "delegation"],   # Implementation focus
        "helena": ["web", "file", "delegation"],                # Legal research
        "larissa": ["web", "file", "messaging", "delegation"],  # Customer service
        "clara": ["web", "file", "messaging", "delegation"],    # Sales
        "bia": ["web", "browser", "file", "delegation"],        # Signal monitoring
        "vitoria": ["web", "file", "vision", "image_gen", "delegation"],  # Creative
        "daine": ["web", "file", "coding", "delegation"],       # Data analysis
        "ada": ["coding", "terminal", "file", "delegation"],    # Code review
    }

    # Default to sister-specific if available, else domain-based
    if sister_id in sister_overrides:
        return sister_overrides[sister_id]

    # Fallback: try to get domain from registry
    try:
        from agent.sister_registry import get_registry
        registry = get_registry()
        sister = registry.get(sister_id)
        if sister and sister.domain in domain_toolsets:
            return domain_toolsets[sister.domain]
    except Exception:
        pass

    # Ultimate fallback
    return ["web", "file", "delegation"]


def _get_sister_model_hint(sister_id: str, risk_level: Optional[str] = None) -> Optional[str]:
    """Get model preference hint for HARP routing."""
    try:
        from agent.sister_registry import get_registry
        reg = get_registry()
        sister = reg.get(sister_id)
        if sister:
            return sister.model_preference
    except Exception:
        pass
    return None


async def _delegate_to_sister_impl(
    sister_id: str,
    prompt: str,
    toolsets: Optional[List[str]] = None,
    risk_level: Optional[str] = None,
    parent_agent=None,
) -> str:
    """
    Internal implementation that spawns a sister agent and returns her response.

    This uses the existing delegate_task infrastructure but with sister-specific
    configuration (system prompt, toolsets, model hint).
    """
    # Resolve sister
    try:
        from agent.sister_registry import get_registry
        from agent.sister_prompt_loader import get_prompt_loader
    except ImportError:
        return tool_error("Sister system not available. Run 'hermes sister reload' to initialize.")

    reg = get_registry()
    prompt_loader = get_prompt_loader()

    sister = reg.get(sister_id)
    if not sister:
        available = [s.id for s in reg.get_all()]
        return tool_error(
            f"Sister '{sister_id}' not found. Available: {', '.join(available)}"
        )

    # Build sister's system prompt
    sister_prompt = prompt_loader.build_sister_prompt(sister)
    system_prompt = sister_prompt.system_prompt

    # Resolve toolsets
    effective_toolsets = _resolve_sister_toolsets(sister_id, toolsets)

    # Resolve risk level
    effective_risk = risk_level or sister.risk_level

    # Prepare context for the sister
    context = (
        f"You are {sister.name} ({sister.id}), {sister.role}.\n"
        f"Domain: {sister.domain}\n"
        f"Archetype: {sister.archetype}\n"
        f"Style: {sister.personality_style}\n\n"
        f"CORE DIRECTIVE:\n{sister.core_directive}\n\n"
        f"DELEGATION SCOPE: {', '.join(sister.delegation_scope)}\n"
        f"RISK LEVEL: {effective_risk}\n\n"
        f"--- TASK FROM ORCHESTRATOR ---\n{prompt}"
    )

    # Use the existing delegate_task function from delegate_tool
    try:
        from tools.delegate_tool import delegate_task
    except ImportError:
        return tool_error("delegate_task not available")

    # Execute delegation
    # Note: We pass the sister's system prompt as part of the goal context
    # The child agent will receive this as its initial context
    result_json = delegate_task(
        goal=prompt,
        context=context,
        toolsets=effective_toolsets,
        role="leaf",  # Sisters are leaf agents - they don't delegate further
        sister_id=sister.id,
        parent_agent=parent_agent,
    )

    # Parse result
    try:
        result = json.loads(result_json)
        results = result.get("results", [])
        if results and len(results) > 0:
            entry = results[0]
            if entry.get("status") == "completed":
                return entry.get("summary", "Task completed (no summary returned)")
            else:
                return f"[Sister {sister.name} failed] {entry.get('error', 'Unknown error')}"
        return "Sister delegation completed but returned no result"
    except json.JSONDecodeError:
        return f"Sister delegation returned invalid result: {result_json}"


def delegate_to_sister(
    sister_id: str,
    prompt: str,
    toolsets: Optional[List[str]] = None,
    risk_level: Optional[str] = None,
    parent_agent=None,
) -> str:
    """
    Synchronous wrapper for delegate_to_sister.
    Uses the async bridge from model_tools.
    """
    try:
        import model_tools
        return model_tools._run_async(
            _delegate_to_sister_impl(
                sister_id=sister_id,
                prompt=prompt,
                toolsets=toolsets,
                risk_level=risk_level,
                parent_agent=parent_agent,
            )
        )
    except Exception as e:
        return tool_error(f"Delegation failed: {str(e)}")


# Register the tool
def _check_delegate_to_sister_availability() -> bool:
    """Check if sister system is available."""
    try:
        from agent.sister_registry import get_registry
        reg = get_registry()
        return len(reg.get_all()) > 0
    except Exception:
        return False


registry.register(
    name="delegate_to_sister",
    toolset="delegation",
    schema=DELEGATE_TO_SISTER_SCHEMA,
    handler=lambda args, **kw: delegate_to_sister(
        sister_id=args.get("sister_id"),
        prompt=args.get("prompt"),
        toolsets=args.get("toolsets"),
        risk_level=args.get("risk_level"),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=_check_delegate_to_sister_availability,
    emoji="👯",
)