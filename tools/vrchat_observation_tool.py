"""Hermes tools for VRChat multimodal observation ingestion."""

from __future__ import annotations

import json

from tools.openclaw.vrchat_observations import (
    build_observation_from_osc,
    ingest_observations,
    observation_queue_status,
)
from tools.registry import registry


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


registry.register(
    name="vrchat_observation_ingest",
    toolset="vrchat",
    schema={
        "name": "vrchat_observation_ingest",
        "description": (
            "Validate and queue bounded multimodal observations for the VRChat autonomy loop. "
            "Accepted sources are textBox, speechToText, visionObservation, streamComment, operator, and system. "
            "Does not call a model, send OSC, or play audio."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Observation events to validate and queue.",
                },
                "queue_path": {
                    "type": "string",
                    "description": "Optional queue JSONL path. Default: Hermes home state queue.",
                },
                "persist": {
                    "type": "boolean",
                    "description": "Persist accepted observations. Default: true.",
                },
            },
            "required": ["observations"],
        },
    },
    handler=lambda args, **kw: _json(
        ingest_observations(
            list(args.get("observations") or []),
            queue_path=args.get("queue_path") or None,
            persist=bool(args.get("persist", True)),
        )
    ),
    emoji="VR",
)


registry.register(
    name="vrchat_observation_from_osc",
    toolset="vrchat",
    schema={
        "name": "vrchat_observation_from_osc",
        "description": (
            "Convert one incoming VRChat OSC event into a queued observation. "
            "ChatBox input becomes textBox context; avatar parameter observation is disabled unless explicitly allowed. "
            "Does not send OSC or play audio."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "address": {"type": "string", "description": "Incoming OSC address."},
                "args": {"type": "array", "description": "Incoming OSC argument values."},
                "queue_path": {
                    "type": "string",
                    "description": "Optional queue JSONL path. Default: Hermes home state queue.",
                },
                "allow_avatar_parameters": {
                    "type": "boolean",
                    "description": "Allow avatar parameter changes to be queued as system observations. Default: false.",
                },
                "persist": {
                    "type": "boolean",
                    "description": "Persist accepted observation. Default: true.",
                },
            },
            "required": ["address"],
        },
    },
    handler=lambda args, **kw: _json(_ingest_osc_payload(args)),
    emoji="VR",
)


registry.register(
    name="vrchat_observation_queue_status",
    toolset="vrchat",
    schema={
        "name": "vrchat_observation_queue_status",
        "description": (
            "Read-only status and preview for the VRChat autonomy observation queue. "
            "Does not consume observations, call a model, send OSC, or play audio."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "queue_path": {
                    "type": "string",
                    "description": "Optional queue JSONL path. Default: Hermes home state queue.",
                },
                "max_preview": {
                    "type": "integer",
                    "description": "Maximum observations to include in preview. Default: 5.",
                },
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: _json(
        observation_queue_status(
            queue_path=args.get("queue_path") or None,
            max_preview=int(args.get("max_preview", 5)),
        )
    ),
    emoji="VR",
)


def _ingest_osc_payload(args: dict) -> dict:
    converted = build_observation_from_osc(
        args.get("address") or "",
        list(args.get("args") or []),
        allow_avatar_parameters=bool(args.get("allow_avatar_parameters", False)),
    )
    if not converted["success"]:
        return {**converted, "queued": False}
    queued = ingest_observations(
        [converted["observation"]],
        queue_path=args.get("queue_path") or None,
        persist=bool(args.get("persist", True)),
    )
    return {**converted, "queue": queued, "queued": bool(queued["queued"])}
