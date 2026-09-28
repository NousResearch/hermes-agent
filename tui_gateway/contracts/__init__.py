"""Wire contracts (see ``base.py``). Importing this package fills the registry tables; every topic
module is listed here so the generator and the runtime see the same catalog."""

from . import (  # noqa: F401
    billing_delegation_pets,
    common,
    config_free_tier_control,
    connectors,
    connectors_operation,
    display,
    events,
    groups_bot_relay,
    liveness,
    profiles_vault_complete_foreign_subagents,
    projects_pets,
    prompt_voice,
    server_requests,
    sessions,
    tools_commands,
    tools_mcp_plugins,
)
from .base import JsonValue, Params, Payload, Result, WireEnum
from .registry import EVENTS, METHODS, SERVER_REQUESTS

__all__ = ["EVENTS", "METHODS", "SERVER_REQUESTS", "JsonValue", "Params", "Payload", "Result", "WireEnum"]
