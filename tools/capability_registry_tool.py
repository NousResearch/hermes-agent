"""Model-facing Executive Capability Registry tools."""

from __future__ import annotations

import json
from typing import Optional

from hermes_cli.capability_registry import find_capability, to_json
from hermes_cli.executive_bus_gate import executive_bus_enabled_for_current_context
from tools.registry import registry


def find_capability_tool(
    capability: str,
    risk: str = "READ",
    include_disabled: bool = False,
    test_credentials: bool = False,
    requester_profile: Optional[str] = None,
    max_concurrency: Optional[int] = None,
) -> str:
    result = find_capability(
        capability,
        requester_profile=requester_profile,
        risk=risk,  # retained for API shape / future ranking policy
        include_disabled=include_disabled,
        test_credentials=test_credentials,
        max_concurrency=max_concurrency,
    )
    return to_json(result)


registry.register(
    name="find_capability",
    toolset="executive_bus",
    schema={
        "name": "find_capability",
        "description": (
            "Find which Hermes executive profile owns a capability such as mcp:vercel, "
            "mcp:composio, tool:terminal, or toolset:browser. Returns redacted, "
            "workload-aware ranking metadata only; never returns credentials."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "capability": {"type": "string", "description": "Capability identifier, e.g. mcp:vercel"},
                "risk": {"type": "string", "enum": ["READ", "PREPARE", "CONSEQUENTIAL_WRITE"], "default": "READ"},
                "include_disabled": {"type": "boolean", "default": False},
                "test_credentials": {"type": "boolean", "default": False},
                "requester_profile": {"type": "string", "description": "Optional requesting profile; defaults to current profile."},
                "max_concurrency": {"type": "integer", "description": "Optional simple per-profile concurrency cap used for ranking."},
            },
            "required": ["capability"],
        },
    },
    handler=lambda args, **kw: find_capability_tool(
        capability=args.get("capability", ""),
        risk=args.get("risk", "READ"),
        include_disabled=bool(args.get("include_disabled", False)),
        test_credentials=bool(args.get("test_credentials", False)),
        requester_profile=args.get("requester_profile"),
        max_concurrency=args.get("max_concurrency"),
    ),
    check_fn=executive_bus_enabled_for_current_context,
)
