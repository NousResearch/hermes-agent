"""Central Beta orchestration tool available only in agent.mode=beta."""
from __future__ import annotations

import json
from typing import Any

from agent.beta.runtime import get_beta_runtime
from agent.beta_identity import BETA_MODE, resolve_agent_mode


def beta_orchestrate(request: str, parent_agent: Any) -> str:
    if not request or not request.strip():
        return json.dumps({"error": "request is required"}, ensure_ascii=False)
    if parent_agent is None:
        return json.dumps({"error": "Beta runtime requires parent_agent"}, ensure_ascii=False)
    run = get_beta_runtime(parent_agent).handle(request.strip())
    return run.model_dump_json()


def check_beta_mode() -> bool:
    return resolve_agent_mode() == BETA_MODE


BETA_ORCHESTRATE_SCHEMA = {
    "name": "beta_orchestrate",
    "description": (
        "Use this as the mandatory entry point for specialist work in Beta mode. "
        "It plans the request, selects registered specialists, applies risk and approval "
        "policy, delegates, validates evidence, and returns one auditable consolidated result. "
        "Do not call delegate_task directly for a Chief request when this tool is available."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": "The Chief's complete request, preserving relevant context and constraints.",
            }
        },
        "required": ["request"],
    },
}


from tools.registry import registry

registry.register(
    name="beta_orchestrate",
    toolset="delegation",
    schema=BETA_ORCHESTRATE_SCHEMA,
    handler=lambda args, **kw: beta_orchestrate(
        request=args.get("request", ""),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=check_beta_mode,
    emoji="🎩",
)
