"""Hermes tool for read-only VRChat autonomy preflight evidence."""

from __future__ import annotations

import json

from tools.openclaw.vrchat_preflight import build_preflight_bundle
from tools.registry import registry


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


registry.register(
    name="vrchat_autonomy_preflight_bundle",
    toolset="vrchat",
    schema={
        "name": "vrchat_autonomy_preflight_bundle",
        "description": (
            "Collect a read-only evidence bundle before VRChat autonomy live smoke tests. "
            "Checks profile, readiness, Neuro SDK vendor status, observation queue, optional audio devices, "
            "and optional no-playback VOICEVOX synthesis. "
            "Does not send OSC, play audio, record microphone input, or open a Neuro websocket."
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
                "include_voicevox_synthesis": {
                    "type": "boolean",
                    "description": "Run VOICEVOX audio_query/synthesis without playback. Default: false.",
                },
                "voicevox_synthesis_text": {
                    "type": "string",
                    "description": "Short text for the no-playback VOICEVOX synthesis probe.",
                },
                "voicevox_synthesis_speaker": {
                    "type": "integer",
                    "description": "VOICEVOX speaker/style ID for the no-playback synthesis probe.",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional JSON output path for the evidence bundle.",
                },
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: _json(
        build_preflight_bundle(
            profile_path=args.get("profile_path") or None,
            voicevox_url=args.get("voicevox_url") or "http://127.0.0.1:50021",
            harness_url=args.get("harness_url") or "http://127.0.0.1:18794",
            audio_output_device=args.get("audio_output_device") or None,
            require_harness=bool(args.get("require_harness", False)),
            queue_path=args.get("queue_path") or None,
            include_audio_devices=bool(args.get("include_audio_devices", True)),
            max_audio_devices=int(args.get("max_audio_devices", 20)),
            include_voicevox_synthesis=bool(args.get("include_voicevox_synthesis", False)),
            voicevox_synthesis_text=args.get("voicevox_synthesis_text") or "\u30c6\u30b9\u30c8",
            voicevox_synthesis_speaker=args.get("voicevox_synthesis_speaker"),
            output_path=args.get("output_path") or None,
        )
    ),
    emoji="VR",
)
