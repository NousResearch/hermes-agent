"""Completion audit for the Hermes VRChat Neuro-style autonomy goal."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tools.openclaw.neuro_bridge import handle_neuro_action_message
from tools.openclaw.vrchat_autonomy import LIVE_ACTUATION_ACK, plan_autonomy_turn
from tools.openclaw.vrchat_conversation import run_multimodal_conversation_dry_run
from tools.openclaw.vrchat_preflight import build_preflight_bundle, build_runtime_doctor
from tools.openclaw.vrchat_smoke import prepare_private_smoke


def build_completion_audit(
    *,
    profile_path: str | Path | None = None,
    voicevox_url: str = "http://127.0.0.1:50021",
    harness_url: str = "http://127.0.0.1:18794",
    audio_output_device: str | None = None,
    require_harness: bool = False,
    queue_path: str | Path | None = None,
    include_audio_devices: bool = False,
    include_voicevox_synthesis: bool = True,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build an evidence-backed audit for the active VRChat autonomy objective."""
    preflight = build_preflight_bundle(
        profile_path=profile_path,
        voicevox_url=voicevox_url,
        harness_url=harness_url,
        audio_output_device=audio_output_device,
        require_harness=require_harness,
        queue_path=queue_path,
        include_audio_devices=include_audio_devices,
        include_voicevox_synthesis=include_voicevox_synthesis,
    )
    registration = _tool_registration_status()
    docs = _documentation_status()
    dry_run_turn = _dry_run_multimodal_turn(preflight.get("profile", {}))
    neuro_action_route = _neuro_action_dry_run_route(preflight.get("profile", {}))
    conversation_dry_run = _conversation_dry_run(preflight.get("profile", {}))
    private_smoke_prepare = _private_smoke_prepare(
        preflight.get("profile", {}),
        voicevox_url=voicevox_url,
        harness_url=harness_url,
        audio_output_device=audio_output_device,
        require_harness=require_harness,
    )
    runtime_doctor = build_runtime_doctor(
        voicevox_url=voicevox_url,
        harness_url=harness_url,
        audio_output_device=audio_output_device,
        require_harness=require_harness,
        queue_path=queue_path,
        include_audio_devices=include_audio_devices,
        voicevox_probe_timeout=0.25,
        preflight_bundle=preflight,
    )
    requirements = _requirements(
        preflight,
        registration,
        docs,
        dry_run_turn,
        neuro_action_route,
        conversation_dry_run,
        private_smoke_prepare,
        runtime_doctor,
    )
    incomplete = [item["id"] for item in requirements if item["status"] != "achieved"]
    readiness_checks = preflight.get("readiness", {}).get("checks", {})
    vrchat_process = readiness_checks.get("vrchat_process", {})
    voicevox = readiness_checks.get("voicevox", {})
    voicevox_process = voicevox.get("process", {})
    virtual_cable_route = preflight.get("audio", {}).get("virtual_cable_route", {})
    voicevox_synthesis = preflight.get("voicevox_synthesis", {})
    audit = {
        "success": True,
        "created_at": int(time.time()),
        "objective": (
            "Deep-research Neuro-sama-style VRChat autonomous multimodal agent design for "
            "hermes-agent: heartbeat VRChat launch detection, Python OSC control/textbox, "
            "VOICEVOX speech through virtual cable microphone, safe autonomous OSC actions, "
            "and multimodal conversation integration."
        ),
        "complete": not incomplete,
        "incomplete_requirements": incomplete,
        "requirements": requirements,
        "preflight_summary": {
            "ready": bool(preflight.get("readiness", {}).get("ready")),
            "missing": list(preflight.get("readiness", {}).get("missing") or []),
            "vrchat_process_phase": vrchat_process.get("phase"),
            "vrchat_process_diagnostic": vrchat_process.get("diagnostic"),
            "voicevox_ok": voicevox.get("ok"),
            "voicevox_process_phase": voicevox_process.get("phase"),
            "voicevox_process_diagnostic": voicevox_process.get("diagnostic"),
            "voicevox_synthesis_ok": voicevox_synthesis.get("ok"),
            "voicevox_synthesis_size_bytes": voicevox_synthesis.get("size_bytes"),
            "virtual_cable_route_ok": virtual_cable_route.get("ok"),
            "virtual_cable_microphone_device": virtual_cable_route.get("microphone_device", ""),
            "runtime_doctor_status": runtime_doctor.get("summary", {}).get("status"),
            "runtime_doctor_mismatches": runtime_doctor.get("summary", {}).get("operator_mismatches", []),
            "live_smoke_gate": preflight.get("live_smoke_gate", {}),
            "profile_path": preflight.get("profile", {}).get("path"),
        },
        "tool_registration": registration,
        "documentation": docs,
        "dry_run_multimodal_turn": dry_run_turn,
        "neuro_action_dry_run_route": neuro_action_route,
        "conversation_dry_run": conversation_dry_run,
        "private_smoke_prepare": private_smoke_prepare,
        "runtime_doctor": runtime_doctor,
        "commands": {
            **dict(preflight.get("commands") or {}),
            "runtime_doctor": runtime_doctor.get("commands", {}).get("runtime_doctor", []),
            "completion_audit": [
                "py",
                "-3.12",
                "scripts\\vrchat_completion_audit.py",
                "--profile",
                preflight.get("profile", {}).get("path")
                or "<Hermes home>\\config\\vrchat-autonomy-profile.json",
            ],
        },
        "safety": {
            "actuation_performed": False,
            "chatbox_sent": False,
            "speech_played": False,
            "avatar_parameters_written": False,
            "microphone_recorded": False,
            "websocket_opened": False,
        },
    }
    if output_path:
        path = Path(output_path).expanduser()
        audit["output_path"] = str(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def _requirements(
    preflight: dict[str, Any],
    registration: dict[str, bool],
    docs: dict[str, Any],
    dry_run_turn: dict[str, Any],
    neuro_action_route: dict[str, Any],
    conversation_dry_run: dict[str, Any],
    private_smoke_prepare: dict[str, Any],
    runtime_doctor: dict[str, Any],
) -> list[dict[str, Any]]:
    readiness = preflight.get("readiness", {})
    checks = readiness.get("checks", {})
    profile_state = preflight.get("profile", {})
    profile = profile_state.get("profile", {})
    live_gate = preflight.get("live_smoke_gate", {})
    vendor = preflight.get("vendor", {}).get("neuro_sdk", {})
    observations = preflight.get("observations", {})
    virtual_cable_route = preflight.get("audio", {}).get("virtual_cable_route", {})
    voicevox_synthesis = preflight.get("voicevox_synthesis", {})

    return [
        _item(
            "primary_source_research",
            "Primary-source grounding exists for VRChat OSC, VOICEVOX Engine, and Neuro SDK protocol surfaces.",
            docs["guide_exists"] and docs["has_primary_sources"] and bool(vendor.get("success")),
            [
                docs["guide_path"],
                "VRChat OSC docs",
                "VOICEVOX Engine repository",
                "VedalAI neuro-sdk API specification",
            ],
            [],
        ),
        _item(
            "neuro_sdk_vendor",
            "VedalAI neuro-sdk is vendored locally and exposes API specification files.",
            bool(vendor.get("success")) and bool(vendor.get("specification_exists")),
            [vendor.get("path", ""), vendor.get("commit", "")],
            [] if vendor.get("success") else ["vendor_neuro_sdk_missing_or_incomplete"],
        ),
        _item(
            "hermes_tool_surface",
            "Hermes exposes profile, heartbeat, preflight, smoke, Neuro bridge, observation, and VOICEVOX tools.",
            all(registration.values()),
            [name for name, ok in registration.items() if ok],
            [name for name, ok in registration.items() if not ok],
        ),
        _item(
            "heartbeat_launch_detection",
            "VRChat launch/readiness heartbeat is implemented and available.",
            registration.get("vrchat_autonomy_heartbeat_tick", False)
            and registration.get("vrchat_autonomy_wait_ready", False)
            and registration.get("vrchat_autonomy_wait_then_tick", False),
            [
                "vrchat_autonomy_heartbeat",
                "vrchat_autonomy_heartbeat_tick",
                "vrchat_autonomy_wait_ready",
                "vrchat_autonomy_wait_then_tick",
            ],
            [],
        ),
        _item(
            "vrchat_runtime_ready",
            "Current local runtime has VRChat process, python-osc, VOICEVOX, and configured audio output ready.",
            bool(readiness.get("ready")),
            [json.dumps(readiness.get("checks", {}), ensure_ascii=False)],
            list(readiness.get("missing") or []),
        ),
        _item(
            "python_osc_chatbox_surface",
            "Python OSC ChatBox/control surface is available through Hermes tools and readiness.",
            registration.get("vrchat_autonomy_plan_turn", False)
            and registration.get("vrchat_autonomy_private_smoke", False)
            and bool(checks.get("python_osc", {}).get("ok")),
            ["vrchat_autonomy_plan_turn", "vrchat_autonomy_private_smoke", "python-osc"],
            [] if checks.get("python_osc", {}).get("ok") else ["python_osc_unavailable"],
        ),
        _item(
            "voicevox_virtual_cable",
            "VOICEVOX is reachable, can synthesize a no-playback WAV, and the configured virtual cable playback plus VRChat microphone side are detected.",
            bool(checks.get("voicevox", {}).get("ok"))
            and bool(voicevox_synthesis.get("ok"))
            and bool(checks.get("audio_output_device", {}).get("ok"))
            and bool(virtual_cable_route.get("ok")),
            [
                json.dumps(checks.get("voicevox", {}), ensure_ascii=False),
                json.dumps(voicevox_synthesis, ensure_ascii=False),
                json.dumps(checks.get("audio_output_device", {}), ensure_ascii=False),
                json.dumps(virtual_cable_route, ensure_ascii=False),
            ],
            _voice_audio_blockers(checks, virtual_cable_route, voicevox_synthesis),
        ),
        _item(
            "safe_autonomous_actions",
            "Autonomous decisions are profile-gated with dry-run/live ACK safety and movement defaults.",
            bool(profile_state.get("success"))
            and bool(profile.get("enabled"))
            and bool(profile.get("allow_voice"))
            and bool(profile.get("allow_chatbox"))
            and not bool(profile.get("allow_movement"))
            and profile.get("live_actuation_ack", "") in {"", LIVE_ACTUATION_ACK},
            [
                f"mode={profile.get('mode')}",
                f"dry_run={profile.get('dry_run')}",
                f"allow_voice={profile.get('allow_voice')}",
                f"allow_chatbox={profile.get('allow_chatbox')}",
                f"allow_movement={profile.get('allow_movement')}",
            ],
            _profile_blockers(profile_state, profile),
        ),
        _item(
            "dry_run_multimodal_turn",
            "A bounded multimodal observation can produce a dry-run ChatBox and VOICEVOX action plan without actuation.",
            bool(dry_run_turn.get("success"))
            and bool(dry_run_turn.get("has_chatbox"))
            and bool(dry_run_turn.get("has_voice"))
            and not bool(dry_run_turn.get("safety", {}).get("actuation_performed")),
            [
                json.dumps(dry_run_turn.get("planned_kinds", []), ensure_ascii=False),
                json.dumps(dry_run_turn.get("safety", {}), ensure_ascii=False),
            ],
            list(dry_run_turn.get("blockers") or []),
        ),
        _item(
            "multimodal_observation_loop",
            "Multimodal observation queue and Neuro action bridge are available for conversation context.",
            registration.get("vrchat_observation_ingest", False)
            and registration.get("vrchat_observation_from_osc", False)
            and registration.get("vrchat_neuro_handle_action", False)
            and bool(observations.get("success")),
            [
                "vrchat_observation_ingest",
                "vrchat_observation_from_osc",
                "vrchat_neuro_handle_action",
                observations.get("path", ""),
            ],
            [],
        ),
        _item(
            "conversation_dry_run_harness",
            "A CLI/tool harness can dry-run multimodal conversation observations through ChatBox, VOICEVOX, and Neuro routing.",
            bool(conversation_dry_run.get("success")),
            [
                json.dumps(conversation_dry_run.get("planned_kinds", []), ensure_ascii=False),
                json.dumps(conversation_dry_run.get("safety", {}), ensure_ascii=False),
            ],
            list(conversation_dry_run.get("blockers") or []),
        ),
        _item(
            "runtime_doctor_harness",
            "A read-only runtime doctor can explain VRChat and VOICEVOX readiness mismatches before live smoke.",
            bool(runtime_doctor.get("success")) and registration.get("vrchat_autonomy_runtime_doctor", False),
            [
                json.dumps(runtime_doctor.get("summary", {}), ensure_ascii=False),
                json.dumps(runtime_doctor.get("safety", {}), ensure_ascii=False),
            ],
            [] if runtime_doctor.get("success") else ["runtime_doctor_failed"],
        ),
        _item(
            "private_smoke_prepare_harness",
            "A read-only private live-smoke preparation harness evaluates gates and dry-run actions.",
            bool(private_smoke_prepare.get("success"))
            and registration.get("vrchat_autonomy_prepare_private_smoke", False)
            and "chatbox" in private_smoke_prepare.get("planned_kinds", [])
            and "voice" in private_smoke_prepare.get("planned_kinds", [])
            and not bool(private_smoke_prepare.get("safety", {}).get("actuation_performed")),
            [
                private_smoke_prepare.get("code", ""),
                json.dumps(private_smoke_prepare.get("planned_kinds", []), ensure_ascii=False),
                json.dumps(private_smoke_prepare.get("safety", {}), ensure_ascii=False),
            ],
            list(private_smoke_prepare.get("blockers") or []),
        ),
        _item(
            "wait_then_private_smoke_harness",
            "A wait-then-private-smoke harness waits for readiness and defaults to preparation without live output.",
            registration.get("vrchat_autonomy_wait_then_private_smoke", False)
            and "wait_readiness_then_private_smoke_prepare" in dict(preflight.get("commands") or {}),
            [
                "vrchat_autonomy_wait_then_private_smoke",
                json.dumps(
                    dict(preflight.get("commands") or {}).get("wait_readiness_then_private_smoke_prepare", []),
                    ensure_ascii=False,
                ),
            ],
            [] if registration.get("vrchat_autonomy_wait_then_private_smoke", False) else ["tool_not_registered"],
        ),
        _item(
            "neuro_action_dry_run_route",
            "A Neuro API action can route through Hermes safety gates into a dry-run ChatBox and VOICEVOX action plan.",
            bool(neuro_action_route.get("success"))
            and bool(neuro_action_route.get("has_chatbox"))
            and bool(neuro_action_route.get("has_voice"))
            and not bool(neuro_action_route.get("safety", {}).get("actuation_performed")),
            [
                json.dumps(neuro_action_route.get("planned_kinds", []), ensure_ascii=False),
                json.dumps(neuro_action_route.get("action_result", {}), ensure_ascii=False),
            ],
            list(neuro_action_route.get("blockers") or []),
        ),
        _item(
            "private_live_smoke_verified",
            "A private-instance live smoke can run and has explicit readiness, non-dry-run profile, and live ACK.",
            bool(live_gate.get("ready_for_live_private_smoke")),
            [json.dumps(live_gate, ensure_ascii=False)],
            list(live_gate.get("blocked_reasons") or ["live_smoke_evidence_missing"]),
        ),
    ]


def _item(
    item_id: str,
    requirement: str,
    achieved: bool,
    evidence: list[str],
    blockers: list[str],
) -> dict[str, Any]:
    return {
        "id": item_id,
        "requirement": requirement,
        "status": "achieved" if achieved else "incomplete",
        "evidence": [str(item) for item in evidence if item],
        "blockers": [str(item) for item in blockers if item],
    }


def _tool_registration_status() -> dict[str, bool]:
    names = [
        "vrchat_autonomy_status",
        "vrchat_autonomy_heartbeat",
        "vrchat_autonomy_heartbeat_tick",
        "vrchat_autonomy_wait_ready",
        "vrchat_autonomy_wait_then_tick",
        "vrchat_autonomy_runtime_doctor",
        "vrchat_autonomy_prepare_profile",
        "vrchat_autonomy_preflight_bundle",
        "vrchat_autonomy_prepare_private_smoke",
        "vrchat_autonomy_wait_then_private_smoke",
        "vrchat_autonomy_private_smoke",
        "vrchat_autonomy_plan_turn",
        "vrchat_autonomy_conversation_dry_run",
        "vrchat_neuro_status",
        "vrchat_neuro_build_messages",
        "vrchat_neuro_handle_action",
        "vrchat_observation_ingest",
        "vrchat_observation_from_osc",
        "vrchat_observation_queue_status",
        "voicevox_speak",
    ]
    try:
        from tools.registry import discover_builtin_tools, registry

        discover_builtin_tools()
        return {name: registry.get_entry(name) is not None for name in names}
    except Exception:
        return {name: False for name in names}


def _documentation_status() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    guide = root / "docs" / "migration" / "vrchat_neurosama_autonomy.md"
    text = ""
    try:
        text = guide.read_text(encoding="utf-8")
    except OSError:
        pass
    required_markers = [
        "VRChat OSC Overview",
        "VRChat OSC as Input Controller",
        "VedalAI neuro-sdk API specification",
        "VOICEVOX Engine repository",
    ]
    return {
        "guide_path": str(guide),
        "guide_exists": guide.is_file(),
        "has_primary_sources": all(marker in text for marker in required_markers),
        "required_markers": required_markers,
    }


def _voice_audio_blockers(
    checks: dict[str, Any],
    virtual_cable_route: dict[str, Any],
    voicevox_synthesis: dict[str, Any],
) -> list[str]:
    blockers = []
    if not checks.get("voicevox", {}).get("ok"):
        blockers.append("voicevox_not_ready")
    if voicevox_synthesis.get("included") and not voicevox_synthesis.get("ok"):
        blockers.append("voicevox_synthesis_failed")
        if voicevox_synthesis.get("wav_header_ok") is False:
            blockers.append("voicevox_wav_header_invalid")
    elif not voicevox_synthesis.get("included"):
        blockers.append("voicevox_synthesis_not_checked")
    audio = checks.get("audio_output_device", {})
    if audio.get("configured") and not audio.get("ok"):
        blockers.append("audio_output_device_not_found")
    elif not audio.get("configured"):
        blockers.append("audio_output_device_not_configured")
    if virtual_cable_route.get("configured") and not virtual_cable_route.get("ok"):
        if virtual_cable_route.get("playback", {}).get("ok") is False:
            blockers.append("virtual_cable_playback_device_not_found")
        if virtual_cable_route.get("microphone", {}).get("ok") is False:
            blockers.append("virtual_cable_microphone_device_not_found")
    elif not virtual_cable_route.get("configured"):
        blockers.append("virtual_cable_route_not_configured")
    return blockers


def _profile_blockers(profile_state: dict[str, Any], profile: dict[str, Any]) -> list[str]:
    blockers = []
    if not profile_state.get("success"):
        blockers.extend(profile_state.get("errors") or ["profile_invalid"])
    if not bool(profile.get("enabled")):
        blockers.append("profile_disabled")
    if not bool(profile.get("allow_voice")):
        blockers.append("voice_not_allowed")
    if not bool(profile.get("allow_chatbox")):
        blockers.append("chatbox_not_allowed")
    if bool(profile.get("allow_movement")):
        blockers.append("movement_allowed")
    ack = profile.get("live_actuation_ack", "")
    if ack not in {"", LIVE_ACTUATION_ACK}:
        blockers.append("invalid_live_ack")
    return blockers


def _dry_run_multimodal_turn(profile_state: dict[str, Any]) -> dict[str, Any]:
    profile = profile_state.get("profile", {})
    if not profile_state.get("success") or not bool(profile.get("enabled")):
        return {
            "success": False,
            "planned_kinds": [],
            "has_chatbox": False,
            "has_voice": False,
            "safety": _empty_safety(),
            "blockers": _profile_blockers(profile_state, profile),
        }
    turn = plan_autonomy_turn(
        observations=[
            {
                "source": "visionObservation",
                "summary": "The operator is present in a private VRChat readiness check.",
            },
            {
                "source": "operator",
                "text": "Dry-run multimodal audit. Do not send OSC or play audio.",
            },
        ],
        decision={
            "speak_text": "Hermes dry-run voice plan.",
            "chatbox_text": "Hermes dry-run chatbox plan.",
            "emotion": "neutral",
            "avatar_action": "",
            "urgency": "low",
        },
        mode=profile.get("mode", "observe"),
        allowed_avatar_actions=list(profile.get("allowed_avatar_actions") or []),
        avatar_action_profiles=dict(profile.get("avatar_action_profiles") or {}),
        allow_voice=bool(profile.get("allow_voice", False)),
        allow_chatbox=bool(profile.get("allow_chatbox", False)),
        allow_movement=bool(profile.get("allow_movement", False)),
        allow_interrupt=bool(profile.get("allow_interrupt", False)),
        dry_run=True,
        output_device=profile.get("output_device") or profile.get("audio_output_device") or None,
        voicevox_speaker=int(profile.get("voicevox_speaker", 8)),
        chatbox_immediate=bool(profile.get("chatbox_immediate", True)),
    )
    planned_kinds = [action.get("kind", "") for action in turn.get("planned_actions", [])]
    blockers = []
    if not turn.get("success"):
        blockers.extend(turn.get("decision", {}).get("blocked_reasons") or ["dry_run_turn_failed"])
    if "chatbox" not in planned_kinds:
        blockers.append("chatbox_plan_missing")
    if "voice" not in planned_kinds:
        blockers.append("voice_plan_missing")
    if turn.get("safety", {}).get("actuation_performed"):
        blockers.append("dry_run_actuation_detected")
    return {
        "success": bool(turn.get("success")) and not blockers,
        "planned_kinds": planned_kinds,
        "has_chatbox": "chatbox" in planned_kinds,
        "has_voice": "voice" in planned_kinds,
        "safety": turn.get("safety", _empty_safety()),
        "blockers": blockers,
        "turn": turn,
    }


def _neuro_action_dry_run_route(profile_state: dict[str, Any]) -> dict[str, Any]:
    profile_path = profile_state.get("path")
    if not profile_state.get("success") or not bool(profile_state.get("profile", {}).get("enabled")):
        return {
            "success": False,
            "planned_kinds": [],
            "has_chatbox": False,
            "has_voice": False,
            "safety": _empty_safety(),
            "action_result": {},
            "blockers": _profile_blockers(profile_state, profile_state.get("profile", {})),
        }

    message = {
        "command": "action",
        "data": {
            "id": "completion-audit-neuro-action",
            "name": "vrchat_autonomy_turn",
            "data": json.dumps(
                {
                    "speak_text": "Hermes Neuro dry-run voice plan.",
                    "chatbox_text": "Hermes Neuro dry-run chatbox plan.",
                    "emotion": "neutral",
                    "avatar_action": "",
                    "urgency": "low",
                },
                ensure_ascii=False,
            ),
        },
    }
    result = handle_neuro_action_message(
        message,
        profile_path=profile_path,
        force_dry_run=True,
    )
    turn = result.get("turn") or {}
    planned_kinds = [action.get("kind", "") for action in turn.get("planned_actions", [])]
    blockers = []
    if not result.get("success"):
        blockers.extend((result.get("decision") or {}).get("blocked_reasons") or ["neuro_action_route_failed"])
    if "chatbox" not in planned_kinds:
        blockers.append("chatbox_plan_missing")
    if "voice" not in planned_kinds:
        blockers.append("voice_plan_missing")
    if turn.get("safety", {}).get("actuation_performed"):
        blockers.append("force_dry_run_actuation_detected")
    return {
        "success": bool(result.get("success")) and not blockers,
        "planned_kinds": planned_kinds,
        "has_chatbox": "chatbox" in planned_kinds,
        "has_voice": "voice" in planned_kinds,
        "safety": turn.get("safety", _empty_safety()),
        "action_result": result.get("action_result", {}),
        "blockers": blockers,
        "result": result,
    }


def _conversation_dry_run(profile_state: dict[str, Any]) -> dict[str, Any]:
    profile_path = profile_state.get("path")
    result = run_multimodal_conversation_dry_run(profile_path=profile_path)
    return {
        "success": bool(result.get("success")),
        "planned_kinds": list(result.get("planned_kinds") or []),
        "has_chatbox": bool(result.get("has_chatbox")),
        "has_voice": bool(result.get("has_voice")),
        "safety": result.get("safety", _empty_safety()),
        "blockers": list(result.get("blockers") or []),
        "result": result,
    }


def _private_smoke_prepare(
    profile_state: dict[str, Any],
    *,
    voicevox_url: str,
    harness_url: str,
    audio_output_device: str | None,
    require_harness: bool,
) -> dict[str, Any]:
    profile_path = profile_state.get("path")
    result = prepare_private_smoke(
        profile_path=profile_path,
        voicevox_url=voicevox_url,
        harness_url=harness_url,
        audio_output_device=audio_output_device,
        require_harness=require_harness,
        live_ack=LIVE_ACTUATION_ACK,
    )
    blockers = list(result.get("live_gate", {}).get("blocked_reasons") or [])
    planned_kinds = list(result.get("planned_kinds") or [])
    if "chatbox" not in planned_kinds:
        blockers.append("chatbox_plan_missing")
    if "voice" not in planned_kinds:
        blockers.append("voice_plan_missing")
    if not result.get("success"):
        blockers.append(result.get("code") or "private_smoke_prepare_failed")
    return {
        "success": bool(result.get("success")),
        "code": str(result.get("code") or ""),
        "would_execute_live": bool(result.get("would_execute_live")),
        "planned_kinds": planned_kinds,
        "safety": result.get("safety", _empty_safety()),
        "blockers": blockers,
        "result": result,
    }


def _empty_safety() -> dict[str, bool]:
    return {
        "actuation_performed": False,
        "chatbox_sent": False,
        "speech_played": False,
        "avatar_parameters_written": False,
    }
