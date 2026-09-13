"""Child toolset resolution for delegate_task: what a child may and may never use."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from toolsets import TOOLSETS
from tools.delegate_tool_config import _get_inherit_mcp_toolsets

logger = logging.getLogger("tools.delegate_tool")  # log-record parity with the origin module

# Tools that children must never have access to
DELEGATE_BLOCKED_TOOLS = frozenset(
    [
        "delegate_task",  # no recursive delegation
        "clarify",  # no user interaction
        "memory",  # no writes to shared MEMORY.md
        "send_message",  # no cross-platform side effects
        "cronjob_manage",  # no scheduling more work in the parent's name
    ]
)
DEFAULT_TOOLSETS = ["terminal", "file", "web"]

def _is_mcp_toolset_name(name: str) -> bool:
    """Return True for canonical MCP toolsets and their registered aliases."""
    if not name:
        return False
    if str(name).startswith("mcp-"):
        return True
    try:
        from tools.registry import registry
        target = registry.get_toolset_alias_target(str(name))
    except Exception:
        target = None
    return bool(target and str(target).startswith("mcp-"))

def _expand_parent_toolsets(parent_toolsets: set) -> set:
    """Add every toolset whose tools are a subset of the parent's tools: a parent on a composite like ``hermes-cli``
    must still let a child request ``web``/``terminal``; bare name intersection would reject them."""
    parent_tool_names = {t for ts_name in parent_toolsets for t in (TOOLSETS.get(ts_name) or {}).get("tools", [])}
    expanded = set(parent_toolsets)
    if parent_tool_names:
        expanded.update(
            ts_name for ts_name, ts_def in TOOLSETS.items()
            if ts_name not in expanded and ts_def.get("tools") and set(ts_def["tools"]).issubset(parent_tool_names)
        )
    return expanded

def _strip_blocked_tools(toolsets: List[str]) -> List[str]:
    """Remove toolsets whose tools are ALL blocked (derived from DELEGATE_BLOCKED_TOOLS so the two can't drift) plus
    composite toolsets children must never get (``delegation``, ``kanban``)."""
    blocked_toolset_names = {"delegation", "kanban"} | {
        name for name, defn in TOOLSETS.items() if all(t in DELEGATE_BLOCKED_TOOLS for t in defn.get("tools", []))
    }
    return [t for t in toolsets if t not in blocked_toolset_names]

def _blocked_toolsets_for_role(role: str) -> List[str]:
    """One-tool deny toolsets for the role; passed as ``disabled_toolsets`` so
    blocked names inside mixed bundles are subtracted AFTER composite expansion."""
    blocked_names = set(DELEGATE_BLOCKED_TOOLS)
    # Both roles keep the control surface; delegate_task itself rejects spawn for
    # leaves while permitting status/message/wait/cancel/resume.
    blocked_names.discard("delegate_task")
    return sorted(
        name for name, defn in TOOLSETS.items() if defn.get("tools") and set(defn.get("tools", ())).issubset(blocked_names)
    )

def _resolve_child_toolsets(
    parent_agent, toolsets: Optional[List[str]], effective_role: str
) -> tuple[List[str], List[str]]:
    """``(enabled_toolsets, disabled_toolsets)`` for a child. Children never gain tools the parent lacks: explicit
    ``toolsets`` are intersected with the parent's (composite-expanded) set, else the parent's enabled set is
    inherited. Blocked tools are stripped twice — whole blocked toolsets here, and exact one-tool deny toolsets via
    ``disabled_toolsets`` so blocked names inside mixed bundles (hermes-cli) are subtracted AFTER composite
    expansion and survive registry refreshes. Orchestrators get ``delegation`` re-added unconditionally
    (role-granted, not inherited)."""
    # enabled_toolsets=None means "all tools", so derive from loaded tool names.
    parent_enabled = getattr(parent_agent, "enabled_toolsets", None)
    if parent_enabled is not None:
        parent_toolsets = set(parent_enabled)
    elif parent_agent and hasattr(parent_agent, "valid_tool_names"):
        import model_tools
        parent_toolsets = {
            ts for name in parent_agent.valid_tool_names if (ts := model_tools.get_toolset_for_tool(name)) is not None
        }
    else:
        parent_toolsets = set(DEFAULT_TOOLSETS)

    if toolsets is not None:
        expanded_parent = _expand_parent_toolsets(parent_toolsets)
        child_toolsets = [t for t in toolsets if t in expanded_parent]
        if _get_inherit_mcp_toolsets():
            # Append any parent MCP toolsets missing from the narrowed child.
            child_toolsets += [
                name for name in sorted(parent_toolsets) if _is_mcp_toolset_name(name) and name not in child_toolsets
            ]
    elif parent_agent and parent_enabled is not None:
        child_toolsets = parent_enabled
    else:
        child_toolsets = sorted(parent_toolsets) or DEFAULT_TOOLSETS
    child_toolsets = _strip_blocked_tools(child_toolsets)
    parent_has_delegate = any(
        "delegate_task" in (TOOLSETS.get(name) or {}).get("tools", ()) for name in parent_toolsets
    )
    # Legacy inheritance retains the control surface.  An explicit request
    # allowlist is a ceiling and may remove it.
    if toolsets is None and parent_has_delegate and "delegation" not in child_toolsets:
        child_toolsets.append("delegation")

    raw_parent_disabled = getattr(parent_agent, "disabled_toolsets", None)
    inherited_disabled = (
        [str(name) for name in raw_parent_disabled] if isinstance(raw_parent_disabled, (list, tuple, set)) else []
    )
    if (
        effective_role == "orchestrator" and toolsets is None and parent_has_delegate
        and "delegation" not in inherited_disabled
    ):
        if "delegation" not in child_toolsets:
            child_toolsets.append("delegation")
    child_disabled_toolsets = list(
        dict.fromkeys(inherited_disabled + _blocked_toolsets_for_role(effective_role) + ["kanban"])
    )
    return child_toolsets, child_disabled_toolsets


def _tool_names_for_toolsets(toolsets: Optional[List[str]]) -> Optional[set[str]]:
    if toolsets is None:
        return None
    if not toolsets:
        return set()
    import model_tools
    definitions = model_tools.get_tool_definitions(
        enabled_toolsets=list(toolsets), disabled_toolsets=[], quiet_mode=True,
        skip_tool_search_assembly=True,
    ) or []
    return {
        item.get("function", {}).get("name")
        for item in definitions
        if isinstance(item, dict) and isinstance(item.get("function", {}).get("name"), str)
    }


def _apply_exact_tool_policy(
    child: Any,
    policy: Any,
    *,
    request_toolsets: Optional[List[str]] = None,
    request_blocked_tools: Optional[List[str]] = None,
    ancestor_allowed_tools: Any = None,
) -> None:
    """Apply every ancestor/request/profile ceiling to execution and presentation.

    The executable catalog remains complete even when tool-search collapses deferred
    schemas.  ``tools``/``valid_tool_names`` are only the model-visible projection;
    ``_worker_effective_tool_names`` is the exact dispatch authority.
    """
    visible = set(getattr(child, "valid_tool_names", set()) or set())
    current = set(getattr(child, "_executable_tool_names", visible) or set())
    ancestor = (
        set(ancestor_allowed_tools)
        if isinstance(ancestor_allowed_tools, (set, frozenset, list, tuple))
        else None
    )
    if ancestor is not None:
        current.intersection_update(ancestor)
    request_names = _tool_names_for_toolsets(request_toolsets)
    if request_names is not None:
        current.intersection_update(request_names)
    allowed_toolsets = getattr(policy, "allowed_toolsets", None) if policy is not None else None
    profile_names = _tool_names_for_toolsets(
        list(allowed_toolsets) if allowed_toolsets is not None else None)
    if profile_names is not None:
        current.intersection_update(profile_names)
    allowed = getattr(policy, "allowed_tools", None) if policy is not None else None
    allowed_mcp = getattr(policy, "allowed_mcp_tools", None) if policy is not None else None
    blocked = set(getattr(policy, "blocked_tools", ()) or ()) if policy is not None else set()
    blocked.update(request_blocked_tools or ())
    if allowed is not None:
        current.intersection_update(allowed)
    if allowed_mcp is not None:
        allowed_mcp_names = set(allowed_mcp)
        for name in tuple(current):
            try:
                import model_tools
                toolset = model_tools.get_toolset_for_tool(name)
            except Exception:
                toolset = None
            if _is_mcp_toolset_name(str(toolset or "")) and name not in allowed_mcp_names:
                current.discard(name)
    current.difference_update(blocked)
    bridge_names = set()
    try:
        from tools import tool_search as _ts
        defer_tools = _ts.load_config_readonly().effective_defer_tools
        if any(_ts.is_deferrable_tool_name(name, defer_tools) for name in current):
            bridge_names = set(_ts.BRIDGE_TOOL_NAMES)
    except Exception:
        pass
    visible_authorized = visible.intersection(current).union(visible.intersection(bridge_names))
    child.tools = [
        item for item in (getattr(child, "tools", None) or [])
        if item.get("function", {}).get("name") in visible_authorized
    ]
    child.valid_tool_names = visible_authorized
    child._worker_effective_tool_names = frozenset(current)
