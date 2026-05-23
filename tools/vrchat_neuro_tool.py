"""Hermes tools for VedalAI Neuro API-compatible VRChat bridge helpers."""

from __future__ import annotations

import json

from tools.openclaw.neuro_bridge import (
    DEFAULT_GAME_NAME,
    build_action_force_message,
    build_neuro_bridge_bootstrap,
    build_vrchat_neuro_actions,
    handle_neuro_action_message,
    neuro_sdk_vendor_status,
)
from tools.openclaw.vrchat_autonomy import load_autonomy_profile
from tools.registry import registry


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


registry.register(
    name="vrchat_neuro_status",
    toolset="vrchat",
    schema={
        "name": "vrchat_neuro_status",
        "description": (
            "Read-only status for the vendored VedalAI neuro-sdk bridge and local VRChat autonomy profile. "
            "Does not open a websocket, send OSC, play audio, or change VRChat state."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "profile_path": {
                    "type": "string",
                    "description": "Optional profile JSON path. Default: Hermes home config/vrchat-autonomy-profile.json.",
                }
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: _json(_status_payload(args)),
    emoji="VR",
)


registry.register(
    name="vrchat_neuro_build_messages",
    toolset="vrchat",
    schema={
        "name": "vrchat_neuro_build_messages",
        "description": (
            "Build Neuro API startup, context, action registration, and optional force-action messages "
            "for a websocket harness. Does not connect to Neuro or VRChat."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "game": {
                    "type": "string",
                    "description": f"Neuro API game name. Default: {DEFAULT_GAME_NAME}.",
                },
                "profile_path": {
                    "type": "string",
                    "description": "Optional profile JSON path. Default: Hermes home config/vrchat-autonomy-profile.json.",
                },
                "context": {
                    "type": "string",
                    "description": "Optional initial context sent after startup.",
                },
                "silent_context": {
                    "type": "boolean",
                    "description": "Mark the context as silent. Default: true.",
                },
                "force_action_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional action names for a Neuro actions/force message.",
                },
                "force_query": {
                    "type": "string",
                    "description": "Optional query for a Neuro actions/force message.",
                },
                "force_state": {
                    "type": "string",
                    "description": "Optional state for a Neuro actions/force message.",
                },
                "force_priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                    "description": "Priority for the optional actions/force message. Default: low.",
                },
                "ephemeral_context": {
                    "type": "boolean",
                    "description": "Whether forced-action context is ephemeral. Default: true.",
                },
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: _json(
        _build_messages_payload(args)
    ),
    emoji="VR",
)


registry.register(
    name="vrchat_neuro_handle_action",
    toolset="vrchat",
    schema={
        "name": "vrchat_neuro_handle_action",
        "description": (
            "Validate one incoming Neuro API action message and route it through the local VRChat safety gate. "
            "Live OSC or audio can occur only when the profile is enabled, valid, not dry-run, and explicitly acknowledged."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "object",
                    "description": "Incoming Neuro websocket message with command=action.",
                },
                "profile_path": {
                    "type": "string",
                    "description": "Optional profile JSON path. Default: Hermes home config/vrchat-autonomy-profile.json.",
                },
                "game": {
                    "type": "string",
                    "description": f"Neuro API game name. Default: {DEFAULT_GAME_NAME}.",
                },
                "retry_on_failure": {
                    "type": "boolean",
                    "description": (
                        "Return action/result success=false on rejection, allowing Neuro to retry. "
                        "Default false avoids retry loops for local policy blocks."
                    ),
                },
            },
            "required": ["message"],
        },
    },
    handler=lambda args, **kw: _json(
        handle_neuro_action_message(
            args.get("message") or {},
            profile_path=args.get("profile_path") or None,
            game=args.get("game") or DEFAULT_GAME_NAME,
            retry_on_failure=bool(args.get("retry_on_failure", False)),
        )
    ),
    emoji="VR",
)


def _build_messages_payload(args: dict) -> dict:
    game = args.get("game") or DEFAULT_GAME_NAME
    payload = build_neuro_bridge_bootstrap(
        game=game,
        profile_path=args.get("profile_path") or None,
        context=args.get("context") or "",
        silent_context=bool(args.get("silent_context", True)),
    )
    force_query = args.get("force_query") or ""
    force_action_names = list(args.get("force_action_names") or [])
    if force_query and force_action_names:
        payload["messages"].append(
            build_action_force_message(
                action_names=force_action_names,
                query=force_query,
                state=args.get("force_state") or "",
                game=game,
                priority=args.get("force_priority") or "low",
                ephemeral_context=bool(args.get("ephemeral_context", True)),
            )
        )
    return payload


def _status_payload(args: dict) -> dict:
    vendor = neuro_sdk_vendor_status()
    profile_path = args.get("profile_path") or None
    return {
        "success": vendor["success"],
        "vendor": vendor,
        "profile": load_autonomy_profile(profile_path),
        "actions": build_vrchat_neuro_actions(profile_path=profile_path),
    }
