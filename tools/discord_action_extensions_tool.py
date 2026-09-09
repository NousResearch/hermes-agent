"""Compose bounded Discord actions into the canonical model tools.

Feature modules under :mod:`tools.discord_api` register action metadata only.
This installer imports the existing :mod:`tools.discord_tool` owner, extends
its action/schema/policy maps, and re-registers the same public ``discord`` and
``discord_admin`` tools. Credentials, config allowlists, REST transport, and
error policy therefore remain owned by one canonical implementation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from tools import discord_tool as _discord
from tools.discord_api.action_registry import (
    DiscordAction,
    discover_discord_actions,
    get_discord_actions,
)
from tools.registry import registry, tool_error


discover_discord_actions()
_EXTENSION_ACTIONS = get_discord_actions()

# Snapshot the built-in contract exactly once. Re-importing this installer must
# never treat a prior extension generation as a built-in or duplicate metadata.
_BUILTIN_ACTIONS: Dict[str, Any] = getattr(
    _discord,
    "_DISCORD_BUILTIN_ACTIONS",
    dict(_discord._ACTIONS),
)
_BUILTIN_CORE_NAMES = getattr(
    _discord,
    "_DISCORD_BUILTIN_CORE_NAMES",
    frozenset(_discord._CORE_ACTIONS),
)
_BUILTIN_ADMIN_NAMES = getattr(
    _discord,
    "_DISCORD_BUILTIN_ADMIN_NAMES",
    frozenset(_discord._ADMIN_ACTIONS),
)
_BUILTIN_ACTION_MANIFEST = getattr(
    _discord,
    "_DISCORD_BUILTIN_ACTION_MANIFEST",
    tuple(_discord._ACTION_MANIFEST),
)
_BUILTIN_REQUIRED_PARAMS = getattr(
    _discord,
    "_DISCORD_BUILTIN_REQUIRED_PARAMS",
    {name: list(params) for name, params in _discord._REQUIRED_PARAMS.items()},
)
_BUILTIN_INTENT_GATED_MEMBERS = getattr(
    _discord,
    "_DISCORD_BUILTIN_INTENT_GATED_MEMBERS",
    frozenset(_discord._INTENT_GATED_MEMBERS),
)
_BUILTIN_403_HINTS = getattr(
    _discord,
    "_DISCORD_BUILTIN_403_HINTS",
    dict(_discord._ACTION_403_HINT),
)
_BASE_BUILD_SCHEMA = getattr(
    _discord,
    "_DISCORD_BASE_BUILD_SCHEMA",
    _discord._build_schema,
)

_discord._DISCORD_BUILTIN_ACTIONS = _BUILTIN_ACTIONS
_discord._DISCORD_BUILTIN_CORE_NAMES = _BUILTIN_CORE_NAMES
_discord._DISCORD_BUILTIN_ADMIN_NAMES = _BUILTIN_ADMIN_NAMES
_discord._DISCORD_BUILTIN_ACTION_MANIFEST = _BUILTIN_ACTION_MANIFEST
_discord._DISCORD_BUILTIN_REQUIRED_PARAMS = _BUILTIN_REQUIRED_PARAMS
_discord._DISCORD_BUILTIN_INTENT_GATED_MEMBERS = _BUILTIN_INTENT_GATED_MEMBERS
_discord._DISCORD_BUILTIN_403_HINTS = _BUILTIN_403_HINTS
_discord._DISCORD_BASE_BUILD_SCHEMA = _BASE_BUILD_SCHEMA

_overlap = set(_BUILTIN_ACTIONS) & set(_EXTENSION_ACTIONS)
if _overlap:
    raise RuntimeError(
        "Discord extension action(s) collide with canonical built-ins: "
        + ", ".join(sorted(_overlap))
    )


def _action_handler(action: DiscordAction):
    return action.handler


_combined_actions: Dict[str, Any] = dict(_BUILTIN_ACTIONS)
_combined_actions.update(
    {name: _action_handler(action) for name, action in _EXTENSION_ACTIONS.items()}
)
_core_names = frozenset(_BUILTIN_CORE_NAMES) | frozenset(
    name for name, action in _EXTENSION_ACTIONS.items() if action.surface == "core"
)
_admin_names = frozenset(_BUILTIN_ADMIN_NAMES) | frozenset(
    name for name, action in _EXTENSION_ACTIONS.items() if action.surface == "admin"
)

_discord._ACTIONS = _combined_actions
_discord._CORE_ACTION_NAMES = _core_names
_discord._ADMIN_ACTION_NAMES = _admin_names
_discord._CORE_ACTIONS = {
    name: handler for name, handler in _combined_actions.items() if name in _core_names
}
_discord._ADMIN_ACTIONS = {
    name: handler for name, handler in _combined_actions.items() if name in _admin_names
}
_discord._ACTION_MANIFEST = list(_BUILTIN_ACTION_MANIFEST) + [
    (action.name, action.signature, action.description)
    for action in _EXTENSION_ACTIONS.values()
]
_discord._REQUIRED_PARAMS = {
    **{name: list(params) for name, params in _BUILTIN_REQUIRED_PARAMS.items()},
    **{name: list(action.required) for name, action in _EXTENSION_ACTIONS.items()},
}
_discord._INTENT_GATED_MEMBERS = frozenset(_BUILTIN_INTENT_GATED_MEMBERS) | frozenset(
    name for name, action in _EXTENSION_ACTIONS.items() if action.members_intent_required
)
_discord._ACTION_403_HINT = {
    **_BUILTIN_403_HINTS,
    **{
        name: action.permission_hint
        for name, action in _EXTENSION_ACTIONS.items()
        if action.permission_hint
    },
}


def _build_schema(
    actions: List[str],
    caps: Optional[Dict[str, Any]] = None,
    tool_name: str = "discord",
) -> Optional[Dict[str, Any]]:
    """Build the canonical schema plus selected extension properties."""
    schema = _BASE_BUILD_SCHEMA(actions, caps, tool_name)
    if schema is None:
        return None

    properties = schema["parameters"]["properties"]
    for action_name in actions:
        extension = _EXTENSION_ACTIONS.get(action_name)
        if extension is None:
            continue
        for property_name, property_schema in extension.properties.items():
            normalized = dict(property_schema)
            existing = properties.get(property_name)
            if existing is not None and existing != normalized:
                raise RuntimeError(
                    "Discord action schema collision for "
                    f"{property_name!r}: {action_name!r} disagrees with the canonical schema"
                )
            properties[property_name] = normalized
    return schema


_discord._build_schema = _build_schema


def _missing_required(value: Any) -> bool:
    """Treat only absence/null/empty-string as missing; preserve false/zero/objects."""
    return value is None or value == ""


def _run_discord_action(
    action: str,
    valid_actions: Dict[str, Any],
    tool_label: str,
    guild_id: str = "",
    channel_id: str = "",
    user_id: str = "",
    role_id: str = "",
    message_id: str = "",
    query: str = "",
    name: str = "",
    limit: int = 50,
    before: str = "",
    after: str = "",
    auto_archive_duration: int = 1440,
    **extra_params: Any,
) -> str:
    """Canonical dispatcher with lossless extension-argument forwarding."""
    token = _discord._get_bot_token()
    if not token:
        return tool_error("DISCORD_BOT_TOKEN not configured.")

    action_fn = valid_actions.get(action)
    if not action_fn:
        return tool_error(
            f"Unknown action: {action}",
            available_actions=list(valid_actions.keys()),
        )

    allowlist = _discord._load_allowed_actions_config()
    if allowlist is not None and action not in allowlist:
        return tool_error(
            f"Action '{action}' is disabled by config (discord.server_actions). "
            f"Allowed: {', '.join(allowlist) if allowlist else '<none>'}"
        )

    action_kwargs: Dict[str, Any] = {
        "guild_id": guild_id,
        "channel_id": channel_id,
        "user_id": user_id,
        "role_id": role_id,
        "message_id": message_id,
        "query": query,
        "name": name,
        "limit": limit,
        "before": before,
        "after": after,
        "auto_archive_duration": auto_archive_duration,
    }
    action_kwargs.update(extra_params)

    missing = [
        param
        for param in _discord._REQUIRED_PARAMS.get(action, [])
        if param not in action_kwargs or _missing_required(action_kwargs[param])
    ]
    if missing:
        return tool_error(
            f"Missing required parameters for '{action}': {', '.join(missing)}"
        )

    # Canonical authority always wins over any direct-call attempt to inject
    # credentials or a transport callable outside the model schema.
    action_kwargs["token"] = token
    action_kwargs["_request"] = _discord._discord_request

    try:
        return action_fn(**action_kwargs)
    except _discord.DiscordAPIError as exc:
        _discord.logger.warning(
            "Discord API error in %s action '%s': %s", tool_label, action, exc
        )
        if exc.status == 403:
            return tool_error(_discord._enrich_403(action, exc.body))
        return tool_error(str(exc))
    except Exception as exc:
        _discord.logger.exception(
            "Unexpected error in %s action '%s'", tool_label, action
        )
        return tool_error(f"Unexpected error: {exc}")


_discord._run_discord_action = _run_discord_action


def _make_handler(handler_fn):
    """Preserve built-in defaults without discarding extension properties."""
    def _handler(args, **_kwargs):
        call_args = dict(_discord._HANDLER_DEFAULTS)
        call_args.update(args or {})
        return handler_fn(**call_args)

    return _handler


_discord._make_handler = _make_handler
_discord._STATIC_CORE_SCHEMA = _build_schema(
    list(_discord._CORE_ACTIONS), caps={"detected": False}, tool_name="discord"
)
_discord._STATIC_ADMIN_SCHEMA = _build_schema(
    list(_discord._ADMIN_ACTIONS), caps={"detected": False}, tool_name="discord_admin"
)

# These are intentional same-name/same-toolset replacements. No second public
# tool is introduced; the canonical owner receives the composed contract.
registry.register(
    name="discord",
    toolset="discord",
    schema=_discord._STATIC_CORE_SCHEMA,
    handler=_make_handler(_discord.discord_core),
    check_fn=_discord.check_discord_tool_requirements,
    requires_env=["DISCORD_BOT_TOKEN"],
)

registry.register(
    name="discord_admin",
    toolset="discord_admin",
    schema=_discord._STATIC_ADMIN_SCHEMA,
    handler=_make_handler(_discord.discord_admin_handler),
    check_fn=_discord.check_discord_tool_requirements,
    requires_env=["DISCORD_BOT_TOKEN"],
)
