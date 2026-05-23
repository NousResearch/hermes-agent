"""Hermes tool for read-only VRChat runtime mismatch diagnosis."""

from __future__ import annotations

import json

from tools.openclaw.vrchat_preflight import build_runtime_doctor
from tools.registry import registry


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


registry.register(
    name="vrchat_autonomy_runtime_doctor",
    toolset="vrchat",
    schema={
        "name": "vrchat_autonomy_runtime_doctor",
        "description": (
            "Diagnose local VRChat and VOICEVOX readiness mismatches with read-only probes. "
            "Does not send OSC, play audio, record microphone input, open a Neuro websocket, or arm a live profile."
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
                    "description": "Require Hypura harness readiness for ready=true. Default: false.",
                },
                "queue_path": {
                    "type": "string",
                    "description": "Optional observation queue JSONL path.",
                },
                "include_audio_devices": {
                    "type": "boolean",
                    "description": "List output-capable audio devices without opening streams. Default: true.",
                },
                "max_audio_devices": {
                    "type": "integer",
                    "description": "Maximum audio output devices to list. Default: 20.",
                },
                "operator_reported_vrchat": {
                    "type": "boolean",
                    "description": "Set true when the operator says VRChat is already running.",
                },
                "operator_reported_voicevox": {
                    "type": "boolean",
                    "description": "Set true when the operator says VOICEVOX is already running.",
                },
                "voicevox_probe_timeout": {
                    "type": "number",
                    "description": "Per-candidate VOICEVOX URL probe timeout in seconds. Default: 1.0.",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional JSON output path for the doctor result.",
                },
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: _json(
        build_runtime_doctor(
            profile_path=args.get("profile_path") or None,
            voicevox_url=args.get("voicevox_url") or "http://127.0.0.1:50021",
            harness_url=args.get("harness_url") or "http://127.0.0.1:18794",
            audio_output_device=args.get("audio_output_device") or None,
            require_harness=bool(args.get("require_harness", False)),
            queue_path=args.get("queue_path") or None,
            include_audio_devices=bool(args.get("include_audio_devices", True)),
            max_audio_devices=int(args.get("max_audio_devices", 20)),
            operator_reported_vrchat=bool(args.get("operator_reported_vrchat", False)),
            operator_reported_voicevox=bool(args.get("operator_reported_voicevox", False)),
            voicevox_probe_timeout=float(args.get("voicevox_probe_timeout", 1.0)),
            output_path=args.get("output_path") or None,
        )
    ),
    emoji="VR",
)
