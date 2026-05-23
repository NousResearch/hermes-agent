"""Dry-run multimodal conversation harness for VRChat autonomy."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tools.openclaw.neuro_bridge import handle_neuro_action_message
from tools.openclaw.vrchat_autonomy import load_autonomy_profile, plan_autonomy_turn
from tools.openclaw.vrchat_observations import build_observation, ingest_observations


DEFAULT_DRY_RUN_DECISION = {
    "speak_text": "Hermes multimodal dry-run voice response.",
    "chatbox_text": "Hermes multimodal dry-run chatbox response.",
    "emotion": "neutral",
    "avatar_action": "",
    "urgency": "low",
}


def build_sample_multimodal_observations() -> list[dict[str, Any]]:
    """Return representative non-persistent observations for local dry-run proof."""
    return [
        build_observation(
            source="visionObservation",
            summary="The operator is present in a private VRChat readiness check.",
        ),
        build_observation(
            source="speechToText",
            text="Can you say hello in VRChat when everything is ready?",
        ),
        build_observation(
            source="textBox",
            text="Private dry-run text box observation.",
            trust="vrchat_osc",
        ),
        build_observation(
            source="operator",
            text="Dry-run only. Do not send OSC or play audio.",
        ),
    ]


def run_multimodal_conversation_dry_run(
    *,
    profile_path: str | Path | None = None,
    observations: list[dict[str, Any]] | None = None,
    decision: dict[str, Any] | None = None,
    persist_observations: bool = False,
    queue_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Plan one multimodal VRChat conversation turn and Neuro route without actuation."""
    profile_state = load_autonomy_profile(profile_path)
    profile = profile_state.get("profile", {})
    source_observations = observations if observations is not None else build_sample_multimodal_observations()
    ingestion = ingest_observations(
        source_observations,
        queue_path=queue_path,
        persist=persist_observations,
    )
    normalized_decision = {
        **DEFAULT_DRY_RUN_DECISION,
        **(decision or {}),
    }
    if not normalized_decision.get("speak_text"):
        normalized_decision["speak_text"] = DEFAULT_DRY_RUN_DECISION["speak_text"]
    if not normalized_decision.get("chatbox_text"):
        normalized_decision["chatbox_text"] = DEFAULT_DRY_RUN_DECISION["chatbox_text"]

    if profile_state.get("success") and bool(profile.get("enabled")):
        turn = plan_autonomy_turn(
            observations=ingestion.get("accepted", []),
            decision=normalized_decision,
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
        neuro_route = handle_neuro_action_message(
            {
                "command": "action",
                "data": {
                    "id": "conversation-dry-run-neuro-action",
                    "name": "vrchat_autonomy_turn",
                    "data": json.dumps(normalized_decision, ensure_ascii=False),
                },
            },
            profile_path=profile_state.get("path"),
            force_dry_run=True,
        )
    else:
        turn = {
            "success": False,
            "dry_run": True,
            "planned_actions": [],
            "execution_results": [],
            "safety": _empty_safety(),
            "blocked_reasons": profile_state.get("errors") or ["profile_not_enabled_or_invalid"],
        }
        neuro_route = {
            "success": False,
            "turn": None,
            "action_result": {},
            "blocked_reasons": profile_state.get("errors") or ["profile_not_enabled_or_invalid"],
        }

    planned_kinds = [action.get("kind", "") for action in turn.get("planned_actions", [])]
    safety = _merge_safety(turn.get("safety"), (neuro_route.get("turn") or {}).get("safety"))
    result = {
        "success": bool(turn.get("success")) and bool(neuro_route.get("success")) and _has_expected_plan(planned_kinds),
        "created_at": int(time.time()),
        "profile": profile_state,
        "persisted_observations": bool(persist_observations),
        "ingestion": ingestion,
        "decision": normalized_decision,
        "turn": turn,
        "neuro_route": neuro_route,
        "planned_kinds": planned_kinds,
        "has_chatbox": "chatbox" in planned_kinds,
        "has_voice": "voice" in planned_kinds,
        "safety": safety,
        "blockers": _conversation_blockers(turn, neuro_route, planned_kinds, safety),
    }
    if output_path:
        path = Path(output_path).expanduser()
        result["output_path"] = str(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _has_expected_plan(planned_kinds: list[str]) -> bool:
    return "chatbox" in planned_kinds and "voice" in planned_kinds


def _conversation_blockers(
    turn: dict[str, Any],
    neuro_route: dict[str, Any],
    planned_kinds: list[str],
    safety: dict[str, bool],
) -> list[str]:
    blockers: list[str] = []
    if not turn.get("success"):
        blockers.append("turn_failed")
    if not neuro_route.get("success"):
        blockers.append("neuro_route_failed")
    if "chatbox" not in planned_kinds:
        blockers.append("chatbox_plan_missing")
    if "voice" not in planned_kinds:
        blockers.append("voice_plan_missing")
    if any(safety.values()):
        blockers.append("dry_run_actuation_detected")
    return blockers


def _merge_safety(*items: dict[str, Any] | None) -> dict[str, bool]:
    merged = _empty_safety()
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in merged:
            merged[key] = bool(merged[key] or item.get(key))
    return merged


def _empty_safety() -> dict[str, bool]:
    return {
        "actuation_performed": False,
        "chatbox_sent": False,
        "speech_played": False,
        "avatar_parameters_written": False,
    }
