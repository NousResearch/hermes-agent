"""Hermes tools for staged VRChat private smoke checks."""

from __future__ import annotations

import json

from tools.openclaw.vrchat_smoke import prepare_private_smoke, run_private_smoke, wait_then_private_smoke
from tools.registry import registry


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


registry.register(
    name="vrchat_autonomy_private_smoke",
    toolset="vrchat",
    schema={
        "name": "vrchat_autonomy_private_smoke",
        "description": (
            "Run a staged private-instance VRChat autonomy smoke check. "
            "Defaults to dry-run. Live ChatBox, VOICEVOX, or avatar action execution requires readiness, "
            "a valid enabled non-dry-run profile, and the exact live acknowledgement."
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
                "require_harness": {
                    "type": "boolean",
                    "description": "Require harness readiness. Default: false.",
                },
                "audio_output_device": {
                    "type": "string",
                    "description": "Optional virtual cable output device name to verify.",
                },
                "chatbox_text": {
                    "type": "string",
                    "description": "Short ChatBox smoke text. Default: Hermes VRChat private smoke test.",
                },
                "speak_text": {
                    "type": "string",
                    "description": "Short VOICEVOX smoke text. Default: Hermes smoke test.",
                },
                "avatar_action": {
                    "type": "string",
                    "description": "Optional approved avatar action ID from the profile.",
                },
                "live": {
                    "type": "boolean",
                    "description": "Attempt live actuation only if all gates pass. Default: false.",
                },
                "live_ack": {
                    "type": "string",
                    "description": "Exact live actuation acknowledgement required for live=true.",
                },
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: _json(
        run_private_smoke(
            profile_path=args.get("profile_path") or None,
            voicevox_url=args.get("voicevox_url") or "http://127.0.0.1:50021",
            harness_url=args.get("harness_url") or "http://127.0.0.1:18794",
            require_harness=bool(args.get("require_harness", False)),
            audio_output_device=args.get("audio_output_device") or None,
            chatbox_text=args.get("chatbox_text") or "Hermes VRChat private smoke test.",
            speak_text=args.get("speak_text") or "Hermes smoke test.",
            avatar_action=args.get("avatar_action") or "",
            live=bool(args.get("live", False)),
            live_ack=args.get("live_ack") or "",
        )
    ),
    emoji="VR",
)


registry.register(
    name="vrchat_autonomy_prepare_private_smoke",
    toolset="vrchat",
    schema={
        "name": "vrchat_autonomy_prepare_private_smoke",
        "description": (
            "Build a read-only private live-smoke readiness report and dry-run action plan. "
            "It evaluates the live gates but never sends ChatBox, plays VOICEVOX audio, "
            "or writes avatar parameters."
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
                "require_harness": {
                    "type": "boolean",
                    "description": "Require harness readiness. Default: false.",
                },
                "audio_output_device": {
                    "type": "string",
                    "description": "Optional virtual cable output device name to verify.",
                },
                "chatbox_text": {
                    "type": "string",
                    "description": "Short ChatBox smoke text. Default: Hermes VRChat private smoke test.",
                },
                "speak_text": {
                    "type": "string",
                    "description": "Short VOICEVOX smoke text. Default: Hermes smoke test.",
                },
                "avatar_action": {
                    "type": "string",
                    "description": "Optional approved avatar action ID from the profile.",
                },
                "live_ack": {
                    "type": "string",
                    "description": "Exact live actuation acknowledgement to validate readiness for live smoke.",
                },
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: _json(
        prepare_private_smoke(
            profile_path=args.get("profile_path") or None,
            voicevox_url=args.get("voicevox_url") or "http://127.0.0.1:50021",
            harness_url=args.get("harness_url") or "http://127.0.0.1:18794",
            require_harness=bool(args.get("require_harness", False)),
            audio_output_device=args.get("audio_output_device") or None,
            chatbox_text=args.get("chatbox_text") or "Hermes VRChat private smoke test.",
            speak_text=args.get("speak_text") or "Hermes smoke test.",
            avatar_action=args.get("avatar_action") or "",
            live_ack=args.get("live_ack") or "",
        )
    ),
    emoji="VR",
)


registry.register(
    name="vrchat_autonomy_wait_then_private_smoke",
    toolset="vrchat",
    schema={
        "name": "vrchat_autonomy_wait_then_private_smoke",
        "description": (
            "Wait for read-only VRChat readiness, then prepare a private smoke plan. "
            "Live ChatBox, VOICEVOX, or avatar action execution requires allow_live_smoke=true, "
            "readiness, a valid non-dry-run profile, and the exact live acknowledgement."
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
                "require_harness": {
                    "type": "boolean",
                    "description": "Require harness readiness. Default: false.",
                },
                "audio_output_device": {
                    "type": "string",
                    "description": "Optional virtual cable output device name to verify.",
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
                "chatbox_text": {
                    "type": "string",
                    "description": "Short ChatBox smoke text. Default: Hermes VRChat private smoke test.",
                },
                "speak_text": {
                    "type": "string",
                    "description": "Short VOICEVOX smoke text. Default: Hermes smoke test.",
                },
                "avatar_action": {
                    "type": "string",
                    "description": "Optional approved avatar action ID from the profile.",
                },
                "allow_live_smoke": {
                    "type": "boolean",
                    "description": "Attempt live private smoke only if all gates pass. Default: false.",
                },
                "live_ack": {
                    "type": "string",
                    "description": "Exact live actuation acknowledgement required when allow_live_smoke=true.",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional JSON output path for the wait/private-smoke result.",
                },
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: _json(
        wait_then_private_smoke(
            profile_path=args.get("profile_path") or None,
            voicevox_url=args.get("voicevox_url") or "http://127.0.0.1:50021",
            harness_url=args.get("harness_url") or "http://127.0.0.1:18794",
            require_harness=bool(args.get("require_harness", False)),
            audio_output_device=args.get("audio_output_device") or None,
            queue_path=args.get("queue_path") or None,
            timeout_sec=float(args.get("timeout_sec", 120.0)),
            interval_sec=float(args.get("interval_sec", 5.0)),
            max_snapshots=int(args.get("max_snapshots", 25)),
            chatbox_text=args.get("chatbox_text") or "Hermes VRChat private smoke test.",
            speak_text=args.get("speak_text") or "Hermes smoke test.",
            avatar_action=args.get("avatar_action") or "",
            allow_live_smoke=bool(args.get("allow_live_smoke", False)),
            live_ack=args.get("live_ack") or "",
            output_path=args.get("output_path") or None,
        )
    ),
    emoji="VR",
)
