"""Agent tools for safe VRChat autonomous-avatar readiness and validation."""

from __future__ import annotations

import json

from tools.openclaw.vrchat_autonomy import (
    build_decision_request,
    enqueue_observation,
    load_autonomy_profile,
    plan_autonomy_turn,
    run_autonomy_decision_turn,
    validate_agent_decision,
    vrchat_autonomy_heartbeat_tick,
    vrchat_autonomy_loop_tick,
    vrchat_autonomy_profile_tick,
    vrchat_autonomy_heartbeat,
    vrchat_autonomy_readiness,
)
from tools.registry import registry


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


registry.register(
    name="vrchat_autonomy_status",
    toolset="vrchat",
    schema={
        "name": "vrchat_autonomy_status",
        "description": (
            "Read-only readiness check for a Hermes VRChat autonomous avatar loop. "
            "Checks VRChat process, python-osc, VOICEVOX, optional harness, and optional output device. "
            "Does not send OSC, play audio, record microphone input, or change VRChat state."
        ),
        "parameters": {
            "type": "object",
            "properties": {
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
                    "description": "Require harness readiness for ready=true. Default: false.",
                },
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: _json(
        vrchat_autonomy_readiness(
            voicevox_url=args.get("voicevox_url") or "http://127.0.0.1:50021",
            harness_url=args.get("harness_url") or "http://127.0.0.1:18794",
            audio_output_device=args.get("audio_output_device") or None,
            require_harness=bool(args.get("require_harness", False)),
        )
    ),
    emoji="VR",
)


registry.register(
    name="vrchat_autonomy_heartbeat",
    toolset="vrchat",
    schema={
        "name": "vrchat_autonomy_heartbeat",
        "description": (
            "Read-only heartbeat for VRChat launch/readiness changes. "
            "Persists a small local state file and returns notify=false for healthy no-op states. "
            "Does not send OSC, play audio, record microphone input, or change VRChat state."
        ),
        "parameters": {
            "type": "object",
            "properties": {
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
                    "description": "Require harness readiness for ready=true. Default: false.",
                },
                "persist": {
                    "type": "boolean",
                    "description": "Persist heartbeat state under Hermes home. Default: true.",
                },
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: _json(
        vrchat_autonomy_heartbeat(
            voicevox_url=args.get("voicevox_url") or "http://127.0.0.1:50021",
            harness_url=args.get("harness_url") or "http://127.0.0.1:18794",
            audio_output_device=args.get("audio_output_device") or None,
            require_harness=bool(args.get("require_harness", False)),
            persist=bool(args.get("persist", True)),
        )
    ),
    emoji="VR",
)


registry.register(
    name="vrchat_autonomy_build_decision_request",
    toolset="vrchat",
    schema={
        "name": "vrchat_autonomy_build_decision_request",
        "description": (
            "Build a structured LLM request for one safe VRChat autonomy decision from text, STT, "
            "vision observation, stream comment, operator, or system observations. Does not call an LLM."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Bounded observation events to include as context.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["observe", "private_test", "trusted_instance", "public"],
                    "description": "Safety mode. Default: observe.",
                },
                "allowed_avatar_actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Approved avatar action IDs.",
                },
                "avatar_action_descriptions": {
                    "type": "object",
                    "description": "Optional map of action IDs to plain descriptions.",
                },
                "allow_voice": {"type": "boolean", "description": "Whether voice may be proposed."},
                "allow_chatbox": {"type": "boolean", "description": "Whether ChatBox may be proposed."},
                "allow_movement": {"type": "boolean", "description": "Whether movement-like action may be proposed."},
                "allow_interrupt": {"type": "boolean", "description": "Whether critical interruption may be proposed."},
                "persona": {"type": "string", "description": "Optional short persona guidance."},
                "task": {"type": "string", "description": "Optional specific decision task."},
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: _json(
        build_decision_request(
            observations=list(args.get("observations") or []),
            mode=args.get("mode", "observe"),
            allowed_avatar_actions=list(args.get("allowed_avatar_actions") or []),
            avatar_action_descriptions=dict(args.get("avatar_action_descriptions") or {}),
            allow_voice=bool(args.get("allow_voice", False)),
            allow_chatbox=bool(args.get("allow_chatbox", False)),
            allow_movement=bool(args.get("allow_movement", False)),
            allow_interrupt=bool(args.get("allow_interrupt", False)),
            persona=args.get("persona", ""),
            task=args.get("task", ""),
        )
    ),
    emoji="VR",
)


registry.register(
    name="vrchat_autonomy_validate_decision",
    toolset="vrchat",
    schema={
        "name": "vrchat_autonomy_validate_decision",
        "description": (
            "Validate an untrusted model decision before VRChat/VOICEVOX actuation. "
            "Rejects raw OSC, overlong ChatBox/speech text, unknown avatar actions, and disabled modes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "object",
                    "description": "Model decision object to validate.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["observe", "private_test", "trusted_instance", "public"],
                    "description": "Safety mode. Default: observe.",
                },
                "allowed_avatar_actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Approved avatar action IDs for this avatar profile.",
                },
                "allow_voice": {
                    "type": "boolean",
                    "description": "Allow VOICEVOX speech in this validation pass. Default: false.",
                },
                "allow_chatbox": {
                    "type": "boolean",
                    "description": "Allow ChatBox output in this validation pass. Default: false.",
                },
                "allow_movement": {
                    "type": "boolean",
                    "description": "Allow movement-like actions. Public mode still blocks them. Default: false.",
                },
                "allow_interrupt": {
                    "type": "boolean",
                    "description": "Allow critical speech interruption. Default: false.",
                },
            },
            "required": ["decision"],
        },
    },
    handler=lambda args, **kw: _json(
        validate_agent_decision(
            args.get("decision") or {},
            mode=args.get("mode", "observe"),
            allowed_avatar_actions=list(args.get("allowed_avatar_actions") or []),
            allow_voice=bool(args.get("allow_voice", False)),
            allow_chatbox=bool(args.get("allow_chatbox", False)),
            allow_movement=bool(args.get("allow_movement", False)),
            allow_interrupt=bool(args.get("allow_interrupt", False)),
        )
    ),
    emoji="VR",
)


registry.register(
    name="vrchat_autonomy_enqueue_observation",
    toolset="vrchat",
    schema={
        "name": "vrchat_autonomy_enqueue_observation",
        "description": (
            "Queue one bounded multimodal observation for a later VRChat autonomy loop tick. "
            "Accepted sources include textBox, speechToText, visionObservation, streamComment, operator, and system."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "observation": {
                    "type": "object",
                    "description": "Observation object with source/type plus text, summary, or content.",
                },
                "persist": {
                    "type": "boolean",
                    "description": "Persist to the Hermes-home observation queue. Default: true.",
                },
            },
            "required": ["observation"],
        },
    },
    handler=lambda args, **kw: _json(
        enqueue_observation(
            args.get("observation") or {},
            persist=bool(args.get("persist", True)),
        )
    ),
    emoji="VR",
)


registry.register(
    name="vrchat_autonomy_profile_status",
    toolset="vrchat",
    schema={
        "name": "vrchat_autonomy_profile_status",
        "description": (
            "Load and validate the local VRChat autonomy operator profile. "
            "The profile defaults to disabled, observe mode, and dry-run."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "profile_path": {
                    "type": "string",
                    "description": "Optional profile JSON path. Default: Hermes home config/vrchat-autonomy-profile.json.",
                },
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: _json(
        load_autonomy_profile(args.get("profile_path") or None)
    ),
    emoji="VR",
)


registry.register(
    name="vrchat_autonomy_profile_tick",
    toolset="vrchat",
    schema={
        "name": "vrchat_autonomy_profile_tick",
        "description": (
            "Run one VRChat autonomy loop tick using the local operator profile. "
            "Missing, invalid, or disabled profiles perform no actuation."
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
                    "description": "Optional observations to process with the persisted queue.",
                },
                "emergency_stop": {
                    "type": "boolean",
                    "description": "Disable the loop state and perform no actuation. Default: false.",
                },
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: _json(
        vrchat_autonomy_profile_tick(
            profile_path=args.get("profile_path") or None,
            observations=list(args.get("observations") or []),
            emergency_stop=bool(args.get("emergency_stop", False)),
        )
    ),
    emoji="VR",
)


registry.register(
    name="vrchat_autonomy_heartbeat_tick",
    toolset="vrchat",
    schema={
        "name": "vrchat_autonomy_heartbeat_tick",
        "description": (
            "Run read-only VRChat launch/readiness heartbeat and optionally one profile-driven autonomy tick. "
            "Ticks only on ready launch/completion events unless force flags are provided; live profiles require "
            "allow_live_profile=true and the exact live acknowledgement."
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
                    "description": "Optional observations to process when the heartbeat triggers a profile tick.",
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
                    "description": "Require harness readiness for heartbeat ready=true. Default: false.",
                },
                "persist_heartbeat": {
                    "type": "boolean",
                    "description": "Persist heartbeat state under Hermes home. Default: true.",
                },
                "tick_on_ready_event": {
                    "type": "boolean",
                    "description": "Run a tick on VRCHAT_LAUNCHED_READY or READINESS_COMPLETE. Default: true.",
                },
                "tick_when_already_ready": {
                    "type": "boolean",
                    "description": "Run a tick even when readiness is already stable. Default: false.",
                },
                "force_tick": {
                    "type": "boolean",
                    "description": "Run a profile tick regardless of heartbeat event, while still honoring readiness/profile gates.",
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
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: _json(
        vrchat_autonomy_heartbeat_tick(
            profile_path=args.get("profile_path") or None,
            observations=list(args.get("observations") or []),
            voicevox_url=args.get("voicevox_url") or "http://127.0.0.1:50021",
            harness_url=args.get("harness_url") or "http://127.0.0.1:18794",
            audio_output_device=args.get("audio_output_device") or None,
            require_harness=bool(args.get("require_harness", False)),
            persist_heartbeat=bool(args.get("persist_heartbeat", True)),
            tick_on_ready_event=bool(args.get("tick_on_ready_event", True)),
            tick_when_already_ready=bool(args.get("tick_when_already_ready", False)),
            force_tick=bool(args.get("force_tick", False)),
            allow_live_profile=bool(args.get("allow_live_profile", False)),
            live_ack=args.get("live_ack", ""),
            emergency_stop=bool(args.get("emergency_stop", False)),
        )
    ),
    emoji="VR",
)


registry.register(
    name="vrchat_autonomy_run_turn",
    toolset="vrchat",
    schema={
        "name": "vrchat_autonomy_run_turn",
        "description": (
            "Call the configured Hermes auxiliary model for one structured VRChat autonomy decision, "
            "then validate and plan or execute it. Dry-run is true by default."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "Bounded context events such as textBox, speechToText, "
                        "visionObservation, streamComment, operator, or system."
                    ),
                },
                "mode": {
                    "type": "string",
                    "enum": ["observe", "private_test", "trusted_instance", "public"],
                    "description": "Safety mode. Default: observe.",
                },
                "allowed_avatar_actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Approved avatar action IDs for this turn.",
                },
                "avatar_action_descriptions": {
                    "type": "object",
                    "description": "Optional map of approved action IDs to plain descriptions.",
                },
                "avatar_action_profiles": {
                    "type": "object",
                    "description": "Map action ID to safe avatar parameter writes.",
                },
                "allow_voice": {"type": "boolean", "description": "Allow VOICEVOX speech."},
                "allow_chatbox": {"type": "boolean", "description": "Allow ChatBox output."},
                "allow_movement": {"type": "boolean", "description": "Allow movement-like actions."},
                "allow_interrupt": {"type": "boolean", "description": "Allow critical interruption."},
                "persona": {"type": "string", "description": "Optional short persona guidance."},
                "task": {"type": "string", "description": "Optional specific decision task."},
                "dry_run": {
                    "type": "boolean",
                    "description": "Return the plan without sending OSC or audio. Default: true.",
                },
                "output_device": {
                    "description": "Optional VOICEVOX output device index/name for virtual cable routing.",
                },
                "voicevox_speaker": {
                    "type": "integer",
                    "description": "VOICEVOX speaker/style ID. Default: 8.",
                },
                "chatbox_immediate": {
                    "type": "boolean",
                    "description": "Send ChatBox immediately when dry_run is false. Default: true.",
                },
                "provider": {
                    "type": "string",
                    "description": "Optional auxiliary provider override; otherwise config.yaml is used.",
                },
                "model": {
                    "type": "string",
                    "description": "Optional auxiliary model override; otherwise config.yaml is used.",
                },
                "base_url": {
                    "type": "string",
                    "description": "Optional OpenAI-compatible endpoint override.",
                },
                "timeout": {
                    "type": "number",
                    "description": "Optional LLM call timeout in seconds.",
                },
                "temperature": {
                    "type": "number",
                    "description": "Optional LLM temperature. Default: 0.",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Optional max output tokens. Default: 320.",
                },
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: _json(
        run_autonomy_decision_turn(
            observations=list(args.get("observations") or []),
            mode=args.get("mode", "observe"),
            allowed_avatar_actions=list(args.get("allowed_avatar_actions") or []),
            avatar_action_descriptions=dict(args.get("avatar_action_descriptions") or {}),
            avatar_action_profiles=dict(args.get("avatar_action_profiles") or {}),
            allow_voice=bool(args.get("allow_voice", False)),
            allow_chatbox=bool(args.get("allow_chatbox", False)),
            allow_movement=bool(args.get("allow_movement", False)),
            allow_interrupt=bool(args.get("allow_interrupt", False)),
            persona=args.get("persona", ""),
            task=args.get("task", ""),
            dry_run=bool(args.get("dry_run", True)),
            output_device=args.get("output_device"),
            voicevox_speaker=int(args.get("voicevox_speaker", 8)),
            chatbox_immediate=bool(args.get("chatbox_immediate", True)),
            provider=args.get("provider") or None,
            model=args.get("model") or None,
            base_url=args.get("base_url") or None,
            timeout=args.get("timeout"),
            temperature=args.get("temperature", 0.0),
            max_tokens=int(args.get("max_tokens", 320)),
        )
    ),
    emoji="VR",
)


registry.register(
    name="vrchat_autonomy_loop_tick",
    toolset="vrchat",
    schema={
        "name": "vrchat_autonomy_loop_tick",
        "description": (
            "Run one safe periodic VRChat autonomy loop tick from explicit or queued observations. "
            "The loop is disabled unless enabled=true and still defaults to dry-run."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": "Enable this tick. Default: false.",
                },
                "emergency_stop": {
                    "type": "boolean",
                    "description": "Disable the loop state and perform no actuation. Default: false.",
                },
                "observations": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Optional observations to process along with the persisted queue.",
                },
                "consume_queue": {
                    "type": "boolean",
                    "description": "Remove queued observations after reading them. Default: true.",
                },
                "max_observations": {
                    "type": "integer",
                    "description": "Maximum queued observations to process. Default: 12.",
                },
                "min_turn_interval_sec": {
                    "type": "number",
                    "description": "Minimum seconds between model turns. Default: 10.",
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
                    "description": "Require harness readiness for loop execution. Default: false.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["observe", "private_test", "trusted_instance", "public"],
                    "description": "Safety mode. Default: observe.",
                },
                "allowed_avatar_actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Approved avatar action IDs for this turn.",
                },
                "avatar_action_descriptions": {
                    "type": "object",
                    "description": "Optional map of approved action IDs to plain descriptions.",
                },
                "avatar_action_profiles": {
                    "type": "object",
                    "description": "Map action ID to safe avatar parameter writes.",
                },
                "allow_voice": {"type": "boolean", "description": "Allow VOICEVOX speech."},
                "allow_chatbox": {"type": "boolean", "description": "Allow ChatBox output."},
                "allow_movement": {"type": "boolean", "description": "Allow movement-like actions."},
                "allow_interrupt": {"type": "boolean", "description": "Allow critical interruption."},
                "persona": {"type": "string", "description": "Optional short persona guidance."},
                "task": {"type": "string", "description": "Optional specific decision task."},
                "dry_run": {
                    "type": "boolean",
                    "description": "Return the plan without sending OSC or audio. Default: true.",
                },
                "output_device": {
                    "description": "Optional VOICEVOX output device index/name for virtual cable routing.",
                },
                "voicevox_speaker": {
                    "type": "integer",
                    "description": "VOICEVOX speaker/style ID. Default: 8.",
                },
                "chatbox_immediate": {
                    "type": "boolean",
                    "description": "Send ChatBox immediately when dry_run is false. Default: true.",
                },
                "provider": {"type": "string", "description": "Optional auxiliary provider override."},
                "model": {"type": "string", "description": "Optional auxiliary model override."},
                "base_url": {"type": "string", "description": "Optional OpenAI-compatible endpoint override."},
                "timeout": {"type": "number", "description": "Optional LLM timeout in seconds."},
                "temperature": {"type": "number", "description": "Optional LLM temperature. Default: 0."},
                "max_tokens": {"type": "integer", "description": "Optional max output tokens. Default: 320."},
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: _json(
        vrchat_autonomy_loop_tick(
            enabled=bool(args.get("enabled", False)),
            emergency_stop=bool(args.get("emergency_stop", False)),
            observations=list(args.get("observations") or []),
            consume_queue=bool(args.get("consume_queue", True)),
            max_observations=int(args.get("max_observations", 12)),
            min_turn_interval_sec=float(args.get("min_turn_interval_sec", 10.0)),
            voicevox_url=args.get("voicevox_url") or "http://127.0.0.1:50021",
            harness_url=args.get("harness_url") or "http://127.0.0.1:18794",
            audio_output_device=args.get("audio_output_device") or None,
            require_harness=bool(args.get("require_harness", False)),
            mode=args.get("mode", "observe"),
            allowed_avatar_actions=list(args.get("allowed_avatar_actions") or []),
            avatar_action_descriptions=dict(args.get("avatar_action_descriptions") or {}),
            avatar_action_profiles=dict(args.get("avatar_action_profiles") or {}),
            allow_voice=bool(args.get("allow_voice", False)),
            allow_chatbox=bool(args.get("allow_chatbox", False)),
            allow_movement=bool(args.get("allow_movement", False)),
            allow_interrupt=bool(args.get("allow_interrupt", False)),
            persona=args.get("persona", ""),
            task=args.get("task", ""),
            dry_run=bool(args.get("dry_run", True)),
            output_device=args.get("output_device"),
            voicevox_speaker=int(args.get("voicevox_speaker", 8)),
            chatbox_immediate=bool(args.get("chatbox_immediate", True)),
            provider=args.get("provider") or None,
            model=args.get("model") or None,
            base_url=args.get("base_url") or None,
            timeout=args.get("timeout"),
            temperature=args.get("temperature", 0.0),
            max_tokens=int(args.get("max_tokens", 320)),
        )
    ),
    emoji="VR",
)


registry.register(
    name="vrchat_autonomy_plan_turn",
    toolset="vrchat",
    schema={
        "name": "vrchat_autonomy_plan_turn",
        "description": (
            "Plan one Neuro-sama-style VRChat autonomous turn from observations and a structured decision. "
            "Dry-run is true by default. When dry-run is false, only validated ChatBox, VOICEVOX, "
            "and approved avatar profile actions are executed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "Bounded context events such as textBox, speechToText, "
                        "visionObservation, streamComment, operator, or system."
                    ),
                },
                "decision": {
                    "type": "object",
                    "description": "Structured model decision to validate and plan.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["observe", "private_test", "trusted_instance", "public"],
                    "description": "Safety mode. Default: observe.",
                },
                "allowed_avatar_actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Approved avatar action IDs for this turn.",
                },
                "avatar_action_profiles": {
                    "type": "object",
                    "description": "Map action ID to safe avatar parameter writes, e.g. {'wave': [{'name':'Wave','value':true}]}",
                },
                "allow_voice": {
                    "type": "boolean",
                    "description": "Allow VOICEVOX speech. Default: false.",
                },
                "allow_chatbox": {
                    "type": "boolean",
                    "description": "Allow ChatBox output. Default: false.",
                },
                "allow_movement": {
                    "type": "boolean",
                    "description": "Allow movement-like actions. Public mode still blocks them. Default: false.",
                },
                "allow_interrupt": {
                    "type": "boolean",
                    "description": "Allow critical speech interruption. Default: false.",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Return the plan without sending OSC or audio. Default: true.",
                },
                "output_device": {
                    "description": "Optional VOICEVOX output device index/name for virtual cable routing.",
                },
                "voicevox_speaker": {
                    "type": "integer",
                    "description": "VOICEVOX speaker/style ID. Default: 8.",
                },
                "chatbox_immediate": {
                    "type": "boolean",
                    "description": "Send ChatBox immediately when dry_run is false. Default: true.",
                },
            },
            "required": ["decision"],
        },
    },
    handler=lambda args, **kw: _json(
        plan_autonomy_turn(
            observations=list(args.get("observations") or []),
            decision=args.get("decision") or {},
            mode=args.get("mode", "observe"),
            allowed_avatar_actions=list(args.get("allowed_avatar_actions") or []),
            avatar_action_profiles=dict(args.get("avatar_action_profiles") or {}),
            allow_voice=bool(args.get("allow_voice", False)),
            allow_chatbox=bool(args.get("allow_chatbox", False)),
            allow_movement=bool(args.get("allow_movement", False)),
            allow_interrupt=bool(args.get("allow_interrupt", False)),
            dry_run=bool(args.get("dry_run", True)),
            output_device=args.get("output_device"),
            voicevox_speaker=int(args.get("voicevox_speaker", 8)),
            chatbox_immediate=bool(args.get("chatbox_immediate", True)),
        )
    ),
    emoji="VR",
)
