"""Private-instance smoke checks for Hermes VRChat autonomy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.openclaw.vrchat_autonomy import (
    LIVE_ACTUATION_ACK,
    load_autonomy_profile,
    plan_autonomy_turn,
    vrchat_autonomy_readiness,
)
from tools.openclaw.vrchat_preflight import wait_for_readiness


def run_private_smoke(
    *,
    profile_path: str | Path | None = None,
    voicevox_url: str = "http://127.0.0.1:50021",
    harness_url: str = "http://127.0.0.1:18794",
    require_harness: bool = False,
    audio_output_device: str | None = None,
    chatbox_text: str = "Hermes VRChat private smoke test.",
    speak_text: str = "Hermes smoke test.",
    avatar_action: str = "",
    live: bool = False,
    live_ack: str = "",
) -> dict[str, Any]:
    """Run a staged private smoke plan, executing only under explicit live gates."""
    profile_state = load_autonomy_profile(profile_path)
    profile = profile_state.get("profile", {})
    output_device = profile.get("output_device") or audio_output_device
    readiness = vrchat_autonomy_readiness(
        voicevox_url=voicevox_url or profile.get("voicevox_url") or "http://127.0.0.1:50021",
        harness_url=harness_url or profile.get("harness_url") or "http://127.0.0.1:18794",
        audio_output_device=audio_output_device or output_device or profile.get("audio_output_device") or None,
        require_harness=bool(require_harness or profile.get("require_harness", False)),
    )
    live_gate = _live_gate(profile_state, readiness, live=live, live_ack=live_ack)
    dry_run = not bool(live and live_gate["allowed"])

    if not profile_state["success"] or not bool(profile.get("enabled", False)):
        return {
            "success": False,
            "code": "PROFILE_BLOCKED",
            "profile": profile_state,
            "readiness": readiness,
            "live_gate": live_gate,
            "turn": None,
            "safety": readiness["safety"],
        }

    decision = {
        "speak_text": speak_text if bool(profile.get("allow_voice", False)) else "",
        "chatbox_text": chatbox_text if bool(profile.get("allow_chatbox", False)) else "",
        "emotion": "neutral",
        "avatar_action": avatar_action,
        "urgency": "low",
    }
    turn = plan_autonomy_turn(
        observations=[{"source": "operator", "text": "Private VRChat smoke test."}],
        decision=decision,
        mode=profile.get("mode", "observe"),
        allowed_avatar_actions=list(profile.get("allowed_avatar_actions") or []),
        avatar_action_profiles=dict(profile.get("avatar_action_profiles") or {}),
        allow_voice=bool(profile.get("allow_voice", False)),
        allow_chatbox=bool(profile.get("allow_chatbox", False)),
        allow_movement=bool(profile.get("allow_movement", False)),
        allow_interrupt=bool(profile.get("allow_interrupt", False)),
        dry_run=dry_run,
        output_device=output_device or None,
        voicevox_speaker=int(profile.get("voicevox_speaker", 8)),
        chatbox_immediate=bool(profile.get("chatbox_immediate", True)),
    )
    code = "LIVE_SMOKE_DONE" if live and live_gate["allowed"] and turn["success"] else "DRY_RUN_SMOKE_DONE"
    if not turn["success"]:
        code = "SMOKE_BLOCKED"
    return {
        "success": bool(turn["success"]),
        "code": code,
        "profile": profile_state,
        "readiness": readiness,
        "live_gate": live_gate,
        "turn": turn,
        "safety": turn.get("safety", readiness["safety"]),
    }


def prepare_private_smoke(
    *,
    profile_path: str | Path | None = None,
    voicevox_url: str = "http://127.0.0.1:50021",
    harness_url: str = "http://127.0.0.1:18794",
    require_harness: bool = False,
    audio_output_device: str | None = None,
    chatbox_text: str = "Hermes VRChat private smoke test.",
    speak_text: str = "Hermes smoke test.",
    avatar_action: str = "",
    live_ack: str = "",
) -> dict[str, Any]:
    """Build a read-only live-smoke readiness report and dry-run plan."""
    profile_state = load_autonomy_profile(profile_path)
    profile = profile_state.get("profile", {})
    output_device = profile.get("output_device") or audio_output_device
    readiness = vrchat_autonomy_readiness(
        voicevox_url=voicevox_url or profile.get("voicevox_url") or "http://127.0.0.1:50021",
        harness_url=harness_url or profile.get("harness_url") or "http://127.0.0.1:18794",
        audio_output_device=audio_output_device or output_device or profile.get("audio_output_device") or None,
        require_harness=bool(require_harness or profile.get("require_harness", False)),
    )
    live_gate = _live_gate(profile_state, readiness, live=True, live_ack=live_ack)
    decision = {
        "speak_text": speak_text if bool(profile.get("allow_voice", False)) else "",
        "chatbox_text": chatbox_text if bool(profile.get("allow_chatbox", False)) else "",
        "emotion": "neutral",
        "avatar_action": avatar_action,
        "urgency": "low",
    }
    if not profile_state["success"] or not bool(profile.get("enabled", False)):
        turn = None
        code = "PROFILE_BLOCKED"
        success = False
    else:
        turn = plan_autonomy_turn(
            observations=[{"source": "operator", "text": "Private VRChat smoke readiness rehearsal."}],
            decision=decision,
            mode=profile.get("mode", "observe"),
            allowed_avatar_actions=list(profile.get("allowed_avatar_actions") or []),
            avatar_action_profiles=dict(profile.get("avatar_action_profiles") or {}),
            allow_voice=bool(profile.get("allow_voice", False)),
            allow_chatbox=bool(profile.get("allow_chatbox", False)),
            allow_movement=bool(profile.get("allow_movement", False)),
            allow_interrupt=bool(profile.get("allow_interrupt", False)),
            dry_run=True,
            output_device=output_device or None,
            voicevox_speaker=int(profile.get("voicevox_speaker", 8)),
            chatbox_immediate=bool(profile.get("chatbox_immediate", True)),
        )
        success = bool(turn.get("success"))
        code = "PRIVATE_SMOKE_READY" if live_gate["allowed"] and success else "PRIVATE_SMOKE_BLOCKED"
    planned_kinds = [str(item.get("kind")) for item in (turn or {}).get("planned_actions", [])]
    return {
        "success": success,
        "code": code,
        "profile": profile_state,
        "readiness": readiness,
        "live_gate": live_gate,
        "would_execute_live": bool(live_gate["allowed"] and success),
        "planned_kinds": planned_kinds,
        "turn": turn,
        "safety": (turn or {}).get("safety", readiness["safety"]),
    }


def wait_then_private_smoke(
    *,
    profile_path: str | Path | None = None,
    voicevox_url: str = "http://127.0.0.1:50021",
    harness_url: str = "http://127.0.0.1:18794",
    require_harness: bool = False,
    audio_output_device: str | None = None,
    queue_path: str | Path | None = None,
    include_audio_devices: bool = False,
    max_audio_devices: int = 20,
    timeout_sec: float = 120.0,
    interval_sec: float = 5.0,
    max_snapshots: int = 25,
    chatbox_text: str = "Hermes VRChat private smoke test.",
    speak_text: str = "Hermes smoke test.",
    avatar_action: str = "",
    allow_live_smoke: bool = False,
    live_ack: str = "",
    output_path: str | Path | None = None,
    _sleep=None,
    _clock=None,
) -> dict[str, Any]:
    """Wait for readiness, then prepare private smoke before any optional live run."""
    wait_kwargs: dict[str, Any] = {}
    if _sleep is not None:
        wait_kwargs["_sleep"] = _sleep
    if _clock is not None:
        wait_kwargs["_clock"] = _clock
    wait_result = wait_for_readiness(
        profile_path=profile_path,
        voicevox_url=voicevox_url,
        harness_url=harness_url,
        audio_output_device=audio_output_device,
        require_harness=require_harness,
        queue_path=queue_path,
        include_audio_devices=include_audio_devices,
        max_audio_devices=max_audio_devices,
        timeout_sec=timeout_sec,
        interval_sec=interval_sec,
        max_snapshots=max_snapshots,
        **wait_kwargs,
    )

    prepare: dict[str, Any] | None = None
    smoke: dict[str, Any] | None = None
    code = "WAIT_READY_TIMEOUT"
    success = True
    safety = _safety_flags(wait_result.get("safety", {}))

    if wait_result.get("ready"):
        prepare = prepare_private_smoke(
            profile_path=profile_path,
            voicevox_url=voicevox_url,
            harness_url=harness_url,
            require_harness=require_harness,
            audio_output_device=audio_output_device,
            chatbox_text=chatbox_text,
            speak_text=speak_text,
            avatar_action=avatar_action,
            live_ack=live_ack,
        )
        safety = _safety_flags(prepare.get("safety", {}))
        if not prepare.get("success"):
            code = "WAIT_READY_PREPARE_FAILED"
            success = False
        elif not prepare.get("would_execute_live"):
            code = "WAIT_READY_PREPARE_BLOCKED"
        elif not allow_live_smoke:
            code = "WAIT_READY_PREPARED"
        else:
            smoke = run_private_smoke(
                profile_path=profile_path,
                voicevox_url=voicevox_url,
                harness_url=harness_url,
                require_harness=require_harness,
                audio_output_device=audio_output_device,
                chatbox_text=chatbox_text,
                speak_text=speak_text,
                avatar_action=avatar_action,
                live=True,
                live_ack=live_ack,
            )
            safety = _safety_flags(smoke.get("safety", {}))
            success = bool(smoke.get("success"))
            code = "WAIT_READY_LIVE_SMOKE_DONE" if success else "WAIT_READY_LIVE_SMOKE_FAILED"

    result = {
        "success": success,
        "code": code,
        "ready": bool(wait_result.get("ready")),
        "allow_live_smoke": bool(allow_live_smoke),
        "wait": wait_result,
        "prepare": prepare,
        "smoke": smoke,
        "dry_run": _dry_run_state(prepare, smoke),
        "safety": safety,
    }
    if output_path:
        path = Path(output_path).expanduser()
        result["output_path"] = str(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _live_gate(
    profile_state: dict[str, Any],
    readiness: dict[str, Any],
    *,
    live: bool,
    live_ack: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    profile = profile_state.get("profile", {})
    if not live:
        reasons.append("live_not_requested")
    if live_ack != LIVE_ACTUATION_ACK:
        reasons.append("live_ack_required")
    if not readiness.get("ready"):
        reasons.append("readiness_not_ready")
    if not profile_state.get("success"):
        reasons.append("profile_invalid")
    if not bool(profile.get("enabled", False)):
        reasons.append("profile_disabled")
    if bool(profile.get("dry_run", True)):
        reasons.append("profile_dry_run_true")
    return {
        "requested": bool(live),
        "allowed": bool(live and not reasons),
        "blocked_reasons": reasons,
    }


def _safety_flags(source: dict[str, Any] | None = None) -> dict[str, bool]:
    safety = {
        "actuation_performed": False,
        "chatbox_sent": False,
        "speech_played": False,
        "avatar_parameters_written": False,
        "microphone_recorded": False,
        "websocket_opened": False,
    }
    for key, value in (source or {}).items():
        if key in safety:
            safety[key] = bool(value)
    if safety["chatbox_sent"] or safety["speech_played"] or safety["avatar_parameters_written"]:
        safety["actuation_performed"] = True
    return safety


def _dry_run_state(prepare: dict[str, Any] | None, smoke: dict[str, Any] | None) -> bool | None:
    turn = (smoke or prepare or {}).get("turn")
    if isinstance(turn, dict):
        return bool(turn.get("dry_run"))
    return None
