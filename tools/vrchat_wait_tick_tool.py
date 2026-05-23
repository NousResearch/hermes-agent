"""Hermes tool for readiness wait followed by a gated VRChat autonomy tick."""

from __future__ import annotations

import json

from tools.openclaw.vrchat_preflight import wait_for_readiness_then_tick
from tools.registry import registry


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


registry.register(
    name="vrchat_autonomy_wait_then_tick",
    toolset="vrchat",
    schema={
        "name": "vrchat_autonomy_wait_then_tick",
        "description": (
            "Wait for read-only VRChat autonomy readiness, then run one gated profile heartbeat tick. "
            "Live profiles require allow_live_profile=true and the exact live acknowledgement."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "profile_path": {
                    "type": "string",
                    "description": "Optional profile JSON path. Default: Hermes home config/vrchat-autonomy-profile.json.",
                },
                "observations": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Optional observations to process if readiness succeeds and the tick runs.",
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
                "persist_heartbeat": {
                    "type": "boolean",
                    "description": "Persist heartbeat state under Hermes home. Default: true.",
                },
                "allow_live_profile": {
                    "type": "boolean",
                    "description": "Permit a non-dry-run profile. Requires live_ack. Default: false.",
                },
                "live_ack": {
                    "type": "string",
                    "description": "Exact live acknowledgement required for non-dry-run profiles.",
                },
                "emergency_stop": {
                    "type": "boolean",
                    "description": "Disable loop state through the profile tick emergency stop path. Default: false.",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional JSON output path for the wait/tick result.",
                },
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: _json(
        wait_for_readiness_then_tick(
            profile_path=args.get("profile_path") or None,
            observations=list(args.get("observations") or []),
            voicevox_url=args.get("voicevox_url") or "http://127.0.0.1:50021",
            harness_url=args.get("harness_url") or "http://127.0.0.1:18794",
            audio_output_device=args.get("audio_output_device") or None,
            require_harness=bool(args.get("require_harness", False)),
            queue_path=args.get("queue_path") or None,
            timeout_sec=float(args.get("timeout_sec", 120.0)),
            interval_sec=float(args.get("interval_sec", 5.0)),
            max_snapshots=int(args.get("max_snapshots", 25)),
            persist_heartbeat=bool(args.get("persist_heartbeat", True)),
            allow_live_profile=bool(args.get("allow_live_profile", False)),
            live_ack=args.get("live_ack", ""),
            emergency_stop=bool(args.get("emergency_stop", False)),
            output_path=args.get("output_path") or None,
        )
    ),
    emoji="VR",
)
