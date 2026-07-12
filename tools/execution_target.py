"""Tool execution target inference for split-runtime deployments.

This module is deliberately transport-free. It classifies registered tools by
metadata that already exists in ``tools.registry.ToolEntry`` so API/gateway
transports can decide where to execute a call without baking routing policy into
the registry itself.
"""

from __future__ import annotations

import enum
from typing import Any, Iterable


class ExecutionTarget(enum.Enum):
    """Where a tool call should execute."""

    SERVER = "server"
    LOCAL = "local"


# The first protocol version routes only read-only file tools. The toolset gate keeps the policy
# broad enough to configure by capability group; the tool-name gate prevents
# mutating file tools from becoming routable just because they share a toolset.
LOCAL_ROUTABLE_TOOLSETS: frozenset[str] = frozenset({"file"})
LOCAL_ROUTABLE_TOOLS: frozenset[str] = frozenset({"read_file", "search_files"})


def normalize_routed_toolsets(value: Any) -> frozenset[str]:
    """Normalize configured toolsets and clamp them to implemented targets."""

    if value is None:
        return LOCAL_ROUTABLE_TOOLSETS
    if isinstance(value, str):
        items: Iterable[Any] = value.split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = value
    else:
        items = [value]
    normalized = frozenset(str(item).strip() for item in items if str(item).strip())
    return normalized & LOCAL_ROUTABLE_TOOLSETS


def infer_execution_target(
    entry: Any,
    *,
    enabled: bool = False,
    routed_toolsets: frozenset[str] | None = None,
) -> ExecutionTarget:
    """Return the execution target for a registry ``ToolEntry``.

    Disabled split runtime, missing registry metadata, non-routed toolsets, and
    mutating file tools all resolve to ``SERVER``. This keeps the default path
    and every non-routed tool unchanged.
    """

    if not enabled or entry is None:
        return ExecutionTarget.SERVER
    routed = routed_toolsets if routed_toolsets is not None else LOCAL_ROUTABLE_TOOLSETS
    toolset = getattr(entry, "toolset", "") or ""
    name = getattr(entry, "name", "") or ""
    if toolset in routed and toolset in LOCAL_ROUTABLE_TOOLSETS and name in LOCAL_ROUTABLE_TOOLS:
        return ExecutionTarget.LOCAL
    return ExecutionTarget.SERVER
