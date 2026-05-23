"""Hermes tool for auditing VRChat autonomy goal completion."""

from __future__ import annotations

import json

from tools.openclaw.vrchat_completion_audit import build_completion_audit
from tools.registry import registry


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


registry.register(
    name="vrchat_autonomy_completion_audit",
    toolset="vrchat",
    schema={
        "name": "vrchat_autonomy_completion_audit",
        "description": (
            "Audit the Neuro-style VRChat autonomy goal against current local evidence. "
            "Read-only: no OSC, audio playback, microphone capture, websocket connection, or profile arming."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "profile_path": {
                    "type": "string",
                    "description": "Optional profile JSON path. Default: Hermes home config/vrchat-autonomy-profile.json.",
                },
                "voicevox_url": {
                    "type": "string",
                    "description": "VOICEVOX Engine base URL. Default: http://127.0.0.1:50021",
                },
                "harness_url": {
                    "type": "string",
                    "description": "Hypura harness base URL. Default: http://127.0.0.1:18794",
                },
                "audio_output_device": {
                    "type": "string",
                    "description": "Optional virtual cable output device name to verify.",
                },
                "require_harness": {
                    "type": "boolean",
                    "description": "Require Hypura harness readiness for runtime-ready evidence. Default: false.",
                },
                "queue_path": {
                    "type": "string",
                    "description": "Optional observation queue JSONL path.",
                },
                "include_audio_devices": {
                    "type": "boolean",
                    "description": "List output-capable audio devices without opening streams. Default: false.",
                },
                "include_voicevox_synthesis": {
                    "type": "boolean",
                    "description": "Run VOICEVOX audio_query/synthesis without playback. Default: true.",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional JSON output path for the audit bundle.",
                },
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: _json(
        build_completion_audit(
            profile_path=args.get("profile_path") or None,
            voicevox_url=args.get("voicevox_url") or "http://127.0.0.1:50021",
            harness_url=args.get("harness_url") or "http://127.0.0.1:18794",
            audio_output_device=args.get("audio_output_device") or None,
            require_harness=bool(args.get("require_harness", False)),
            queue_path=args.get("queue_path") or None,
            include_audio_devices=bool(args.get("include_audio_devices", False)),
            include_voicevox_synthesis=bool(args.get("include_voicevox_synthesis", True)),
            output_path=args.get("output_path") or None,
        )
    ),
    emoji="VR",
)
