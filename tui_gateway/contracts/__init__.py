"""Wire contracts (see ``base.py``). Importing this package fills the registry tables; every topic
module is listed here so the generator and the runtime see the same catalog."""

from . import (  # noqa: F401
    common,
    config_control,
    delegation_pets,
    connectors,
    connectors_operation,
    display,
    events,
    groups_bot_relay,
    i18n,
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
