"""Hermes tool for preparing VRChat autonomy operator profiles."""

from __future__ import annotations

import json

from tools.openclaw.vrchat_profile import prepare_autonomy_profile
from tools.registry import registry


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


registry.register(
    name="vrchat_autonomy_prepare_profile",
    toolset="vrchat",
    schema={
        "name": "vrchat_autonomy_prepare_profile",
        "description": (
            "Prepare a local VRChat autonomy operator profile. Defaults to enabled private-test dry-run, "
            "VOICEVOX/ChatBox allowed, movement blocked, and no live OSC or audio. "
            "Live mode requires arm_live=true and the exact live acknowledgement."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "profile_path": {
                    "type": "string",
                    "description": "Optional profile JSON path. Default: Hermes home config/vrchat-autonomy-profile.json.",
                },
                "enabled": {
                    "type": "boolean",
                    "description": "Enable profile-driven ticks. Default: true.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["observe", "private_test", "trusted_instance", "public"],
                    "description": "Safety mode. Default: private_test.",
                },
                "audio_output_device": {
                    "type": "string",
                    "description": "Virtual cable playback device to verify and use. Default: CABLE Input.",
                },
                "vrchat_microphone_device": {
                    "type": "string",
                    "description": "VRChat microphone-side virtual cable device to verify. Default: CABLE Output.",
                },
                "require_harness": {
                    "type": "boolean",
                    "description": "Require Hypura harness readiness. Default: false.",
                },
                "allow_voice": {
                    "type": "boolean",
                    "description": "Allow VOICEVOX speech through the profile gate. Default: true.",
                },
                "allow_chatbox": {
                    "type": "boolean",
                    "description": "Allow VRChat ChatBox through the profile gate. Default: true.",
                },
                "allow_movement": {
                    "type": "boolean",
                    "description": "Allow movement-like actions. Public mode still rejects movement. Default: false.",
                },
                "allow_interrupt": {
                    "type": "boolean",
                    "description": "Allow critical speech interruption. Default: false.",
                },
                "allowed_avatar_actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Approved avatar action IDs. Omit to preserve existing profile entries.",
                },
                "avatar_action_descriptions": {
                    "type": "object",
                    "description": "Optional descriptions keyed by approved avatar action ID.",
                },
                "avatar_action_profiles": {
                    "type": "object",
                    "description": "Map action ID to safe avatar parameter writes. Omit to preserve existing entries.",
                },
                "persona": {
                    "type": "string",
                    "description": "Persona text for the side-model turn prompt.",
                },
                "task": {
                    "type": "string",
                    "description": "Task text for the side-model turn prompt.",
                },
                "voicevox_speaker": {
                    "type": "integer",
                    "description": "VOICEVOX speaker/style ID. Default: 8.",
                },
                "provider": {
                    "type": "string",
                    "description": "Optional auxiliary provider override.",
                },
                "model": {
                    "type": "string",
                    "description": "Optional auxiliary model override.",
                },
                "base_url": {
                    "type": "string",
                    "description": "Optional auxiliary model base URL override.",
                },
                "arm_live": {
                    "type": "boolean",
                    "description": "Write a non-dry-run profile only when live_ack exactly matches. Default: false.",
                },
                "live_ack": {
                    "type": "string",
                    "description": "Exact acknowledgement required for arm_live=true.",
                },
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: _json(
        prepare_autonomy_profile(
            profile_path=args.get("profile_path") or None,
            enabled=bool(args.get("enabled", True)),
            mode=args.get("mode", "private_test"),
            audio_output_device=args.get("audio_output_device") or "CABLE Input",
            vrchat_microphone_device=args.get("vrchat_microphone_device") or "CABLE Output",
            require_harness=bool(args.get("require_harness", False)),
            allow_voice=bool(args.get("allow_voice", True)),
            allow_chatbox=bool(args.get("allow_chatbox", True)),
            allow_movement=bool(args.get("allow_movement", False)),
            allow_interrupt=bool(args.get("allow_interrupt", False)),
            allowed_avatar_actions=(
                list(args.get("allowed_avatar_actions"))
                if args.get("allowed_avatar_actions") is not None
                else None
            ),
            avatar_action_descriptions=(
                dict(args.get("avatar_action_descriptions"))
                if args.get("avatar_action_descriptions") is not None
                else None
            ),
            avatar_action_profiles=(
                dict(args.get("avatar_action_profiles"))
                if args.get("avatar_action_profiles") is not None
                else None
            ),
            persona=args.get("persona") if args.get("persona") is not None else None,
            task=args.get("task") if args.get("task") is not None else None,
            voicevox_speaker=int(args.get("voicevox_speaker", 8)),
            provider=args.get("provider") if args.get("provider") is not None else None,
            model=args.get("model") if args.get("model") is not None else None,
            base_url=args.get("base_url") if args.get("base_url") is not None else None,
            arm_live=bool(args.get("arm_live", False)),
            live_ack=args.get("live_ack", ""),
        )
    ),
    emoji="VR",
)
