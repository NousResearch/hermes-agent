"""Hermes tool for dry-run multimodal VRChat conversation proof."""

from __future__ import annotations

import json

from tools.openclaw.vrchat_conversation import run_multimodal_conversation_dry_run
from tools.registry import registry


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


registry.register(
    name="vrchat_autonomy_conversation_dry_run",
    toolset="vrchat",
    schema={
        "name": "vrchat_autonomy_conversation_dry_run",
        "description": (
            "Run a dry-run multimodal VRChat conversation proof through observation normalization, "
            "ChatBox/VOICEVOX planning, and Neuro action routing without live output."
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
                    "description": "Optional observations. Defaults to representative vision/STT/ChatBox/operator events.",
                },
                "decision": {
                    "type": "object",
                    "description": "Optional structured decision override. Defaults to a safe short dry-run response.",
                },
                "persist_observations": {
                    "type": "boolean",
                    "description": "Persist normalized observations to the queue. Default: false.",
                },
                "queue_path": {
                    "type": "string",
                    "description": "Optional observation queue JSONL path.",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional JSON output path for the dry-run proof.",
                },
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: _json(
        run_multimodal_conversation_dry_run(
            profile_path=args.get("profile_path") or None,
            observations=list(args.get("observations") or []) or None,
            decision=dict(args.get("decision") or {}),
            persist_observations=bool(args.get("persist_observations", False)),
            queue_path=args.get("queue_path") or None,
            output_path=args.get("output_path") or None,
        )
    ),
    emoji="VR",
)
