"""Hermes tool for read-only VRChat autonomy readiness waiting."""

from __future__ import annotations

import json

from tools.openclaw.vrchat_preflight import wait_for_readiness
from tools.registry import registry


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


registry.register(
    name="vrchat_autonomy_wait_ready",
    toolset="vrchat",
    schema={
        "name": "vrchat_autonomy_wait_ready",
        "description": (
            "Poll read-only VRChat autonomy readiness until ready or timeout. "
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
                "timeout_sec": {
                    "type": "number",
                    "description": "Maximum seconds to wait. Default: 120.0.",
                },
                "interval_sec": {
                    "type": "number",
                    "description": "Seconds between polls. Default: 5.0.",
                },
                "max_snapshots": {
                    "type": "integer",
                    "description": "Maximum poll summaries to retain. Default: 25.",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional JSON output path for the wait result.",
                },
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: _json(
        wait_for_readiness(
            profile_path=args.get("profile_path") or None,
            voicevox_url=args.get("voicevox_url") or "http://127.0.0.1:50021",
            harness_url=args.get("harness_url") or "http://127.0.0.1:18794",
            audio_output_device=args.get("audio_output_device") or None,
            require_harness=bool(args.get("require_harness", False)),
            queue_path=args.get("queue_path") or None,
            timeout_sec=float(args.get("timeout_sec", 120.0)),
            interval_sec=float(args.get("interval_sec", 5.0)),
            max_snapshots=int(args.get("max_snapshots", 25)),
            output_path=args.get("output_path") or None,
        )
    ),
    emoji="VR",
)
