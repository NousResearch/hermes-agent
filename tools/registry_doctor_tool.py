"""Registry doctor tool.

Inspects the live tool registry and toolset surface and returns a
metadata-only health envelope.  Intentionally side-effect free: ``check_fn``
always returns True so the tool remains available even when environment
checks are failing.

Uses stdlib only plus the existing ``tools.validation_helper``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tools.registry import registry
from tools.validation_helper import relay_success

logger = logging.getLogger(__name__)


def _resolve_toolset_names() -> List[str]:
    try:
        import toolsets as _toolsets_module

        return _toolsets_module.get_toolset_names()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("toolsets.get_toolset_names() unavailable: %s", exc)
        return sorted(registry.get_registered_toolset_names())


def _registered_toolsets() -> Set[str]:
    return {entry.toolset for entry in registry._snapshot_entries()}


def _missing_toolset_tools(toolset_names: List[str]) -> List[str]:
    """Registered tools whose toolset name is absent from both static TOOLSETS and live registry toolsets."""
    registered_toolsets = _registered_toolsets()
    known = registered_toolsets | set(toolset_names)
    return sorted(
        {entry.toolset for entry in registry._snapshot_entries() if entry.toolset not in known}
    )


def _empty_toolsets(toolset_names: List[str]) -> List[str]:
    """Static toolsets that resolve to no tools."""
    results = []
    for name in toolset_names:
        try:
            import toolsets as _toolsets_module
            resolved = _toolsets_module.resolve_toolset(name)
        except Exception:
            resolved = registry.get_tool_names_for_toolset(name)
        if not resolved:
            results.append(name)
    return sorted(results)


def _build_envelope() -> Dict[str, Any]:
    entries = list(registry._snapshot_entries())
    registered_tool_count = len(entries)
    toolset_names = _resolve_toolset_names()
    toolsets_count = len(toolset_names)
    known_toolsets = _registered_toolsets() | set(toolset_names)
    missing_toolset_tools = sorted(
        entry.toolset for entry in entries if entry.toolset not in known_toolsets
    )
    empty_toolsets = _empty_toolsets(toolset_names)
    orphan_count = len(missing_toolset_tools) + sum(
        1 for entry in entries if not entry.toolset
    )

    return {
        "registered_tool_count": registered_tool_count,
        "toolsets_count": toolsets_count,
        "missing_toolset_tools": missing_toolset_tools,
        "empty_toolsets": empty_toolsets,
        "orphan_count": orphan_count,
    }


def check_fn() -> bool:
    return True


def handle(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    envelope = _build_envelope()
    return relay_success(envelope)
