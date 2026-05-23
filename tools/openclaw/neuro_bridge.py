"""Neuro API bridge helpers for safe VRChat autonomy integration."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from tools.openclaw.vrchat_autonomy import (
    VALID_EMOTIONS,
    VALID_URGENCY,
    load_autonomy_profile,
    plan_autonomy_turn,
    validate_agent_decision,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEURO_SDK_PATH = PROJECT_ROOT / "vendor" / "neuro-sdk"
DEFAULT_GAME_NAME = "Hermes VRChat"


def neuro_sdk_vendor_status() -> dict[str, Any]:
    """Return local clone status for the vendored VedalAI neuro-sdk."""
    spec = NEURO_SDK_PATH / "API" / "SPECIFICATION.md"
    readme = NEURO_SDK_PATH / "API" / "README.md"
    license_file = NEURO_SDK_PATH / "LICENSE.md"
    commit = ""
    if (NEURO_SDK_PATH / ".git").exists():
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(NEURO_SDK_PATH), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
        except Exception:
            commit = ""
    return {
        "success": spec.is_file() and license_file.is_file(),
        "path": str(NEURO_SDK_PATH),
        "commit": commit,
        "license": "MIT" if _file_contains(license_file, "MIT License") else "unknown",
        "api_readme": str(readme),
        "api_readme_exists": readme.is_file(),
        "specification": str(spec),
        "specification_exists": spec.is_file(),
    }


def build_startup_message(game: str = DEFAULT_GAME_NAME) -> dict[str, Any]:
    return {"command": "startup", "game": game}


def build_context_message(
    message: str,
    *,
    game: str = DEFAULT_GAME_NAME,
    silent: bool = True,
) -> dict[str, Any]:
    return {
        "command": "context",
        "game": game,
        "data": {"message": str(message)[:2000], "silent": bool(silent)},
    }


def build_action_register_message(
    *,
    game: str = DEFAULT_GAME_NAME,
    profile_path: str | Path | None = None,
) -> dict[str, Any]:
    return {
        "command": "actions/register",
        "game": game,
        "data": {"actions": build_vrchat_neuro_actions(profile_path=profile_path)},
    }


def build_action_force_message(
    *,
    action_names: list[str],
    query: str,
    state: str = "",
    game: str = DEFAULT_GAME_NAME,
    priority: str = "low",
    ephemeral_context: bool = True,
) -> dict[str, Any]:
    if priority not in VALID_URGENCY:
        priority = "low"
    return {
        "command": "actions/force",
        "game": game,
        "data": {
            "state": state[:4000],
            "query": query[:1000],
            "ephemeral_context": bool(ephemeral_context),
            "priority": priority,
            "action_names": list(action_names),
        },
    }


def build_action_result_message(
    action_id: str,
    *,
    success: bool,
    message: str = "",
    game: str = DEFAULT_GAME_NAME,
) -> dict[str, Any]:
    return {
        "command": "action/result",
        "game": game,
        "data": {
            "id": str(action_id),
            "success": bool(success),
            "message": str(message)[:500],
        },
    }


def build_vrchat_neuro_actions(profile_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Build Neuro API action catalog from the local VRChat autonomy profile."""
    profile_state = load_autonomy_profile(profile_path)
    profile = profile_state.get("profile", {})
    allowed_actions = list(profile.get("allowed_avatar_actions") or [])
    avatar_enum = ["", *allowed_actions]
    actions = [
        {
            "name": "vrchat_autonomy_turn",
            "description": (
                "Choose one safe VRChat response. Empty fields are allowed when no action is useful."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "speak_text": {"type": "string"},
                    "chatbox_text": {"type": "string"},
                    "emotion": {"type": "string", "enum": sorted(VALID_EMOTIONS)},
                    "avatar_action": {"type": "string", "enum": avatar_enum},
                    "urgency": {"type": "string", "enum": sorted(VALID_URGENCY)},
                },
                "required": ["speak_text", "chatbox_text", "emotion", "avatar_action", "urgency"],
            },
        },
        {
            "name": "vrchat_speak",
            "description": "Speak one short line through the approved VOICEVOX route.",
            "schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "emotion": {"type": "string", "enum": sorted(VALID_EMOTIONS)},
                },
                "required": ["text"],
            },
        },
        {
            "name": "vrchat_chatbox",
            "description": "Send one short line to the VRChat ChatBox.",
            "schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    ]
    if allowed_actions:
        actions.append(
            {
                "name": "vrchat_avatar_action",
                "description": "Run one approved avatar action profile.",
                "schema": {
                    "type": "object",
                    "properties": {"avatar_action": {"type": "string", "enum": allowed_actions}},
                    "required": ["avatar_action"],
                },
            }
        )
    return actions


def build_neuro_bridge_bootstrap(
    *,
    game: str = DEFAULT_GAME_NAME,
    profile_path: str | Path | None = None,
    context: str = "",
    silent_context: bool = True,
) -> dict[str, Any]:
    """Build startup/context/action messages for a Neuro websocket client."""
    messages = [build_startup_message(game), build_action_register_message(game=game, profile_path=profile_path)]
    if context:
        messages.insert(1, build_context_message(context, game=game, silent=silent_context))
    return {
        "success": True,
        "vendor": neuro_sdk_vendor_status(),
        "profile": load_autonomy_profile(profile_path),
        "messages": messages,
    }


def handle_neuro_action_message(
    message: dict[str, Any],
    *,
    profile_path: str | Path | None = None,
    game: str = DEFAULT_GAME_NAME,
    retry_on_failure: bool = False,
    force_dry_run: bool = False,
) -> dict[str, Any]:
    """Validate one Neuro incoming action and map it to the VRChat safety gate."""
    if not isinstance(message, dict) or message.get("command") != "action":
        return _action_rejection(
            "unknown",
            "unsupported_message",
            game=game,
            retry_on_failure=retry_on_failure,
        )
    data = message.get("data") or {}
    if not isinstance(data, dict):
        return _action_rejection(
            "unknown",
            "action_data_must_be_object",
            game=game,
            retry_on_failure=retry_on_failure,
        )
    action_id = str(data.get("id") or "unknown")
    action_name = str(data.get("name") or "")
    action_payload = _parse_action_payload(data.get("data"))
    if not action_payload["success"]:
        return _action_rejection(
            action_id,
            action_payload["error"],
            game=game,
            retry_on_failure=retry_on_failure,
        )

    profile_state = load_autonomy_profile(profile_path)
    profile = profile_state.get("profile", {})
    if not profile_state["success"] or not bool(profile.get("enabled", False)):
        return _action_rejection(
            action_id,
            "profile_not_enabled_or_invalid",
            game=game,
            retry_on_failure=retry_on_failure,
            profile=profile_state,
        )

    decision = _decision_from_neuro_action(action_name, action_payload["data"])
    if decision is None:
        return _action_rejection(
            action_id,
            f"unsupported_action:{action_name}",
            game=game,
            retry_on_failure=retry_on_failure,
            profile=profile_state,
        )

    validation = validate_agent_decision(
        decision,
        mode=profile.get("mode", "observe"),
        allowed_avatar_actions=list(profile.get("allowed_avatar_actions") or []),
        allow_voice=bool(profile.get("allow_voice", False)),
        allow_chatbox=bool(profile.get("allow_chatbox", False)),
        allow_movement=bool(profile.get("allow_movement", False)),
        allow_interrupt=bool(profile.get("allow_interrupt", False)),
    )
    if not validation["success"]:
        result = _action_result_for_policy_block(action_id, validation["blocked_reasons"], game)
        return {
            "success": False,
            "action_id": action_id,
            "action_name": action_name,
            "profile": profile_state,
            "decision": validation,
            "turn": None,
            "action_result": result,
        }

    turn = plan_autonomy_turn(
        observations=[{"source": "system", "text": f"Neuro action requested: {action_name}"}],
        decision=decision,
        mode=profile.get("mode", "observe"),
        allowed_avatar_actions=list(profile.get("allowed_avatar_actions") or []),
        avatar_action_profiles=dict(profile.get("avatar_action_profiles") or {}),
        allow_voice=bool(profile.get("allow_voice", False)),
        allow_chatbox=bool(profile.get("allow_chatbox", False)),
        allow_movement=bool(profile.get("allow_movement", False)),
        allow_interrupt=bool(profile.get("allow_interrupt", False)),
        dry_run=True if force_dry_run else bool(profile.get("dry_run", True)),
        output_device=profile.get("output_device"),
        voicevox_speaker=int(profile.get("voicevox_speaker", 8)),
        chatbox_immediate=bool(profile.get("chatbox_immediate", True)),
    )
    result_success = bool(turn["success"]) or not retry_on_failure
    result_message = _turn_result_message(turn)
    return {
        "success": bool(turn["success"]),
        "action_id": action_id,
        "action_name": action_name,
        "profile": profile_state,
        "decision": turn.get("decision"),
        "turn": turn,
        "action_result": build_action_result_message(
            action_id,
            success=result_success,
            message=result_message,
            game=game,
        ),
    }


def _decision_from_neuro_action(action_name: str, data: dict[str, Any]) -> dict[str, Any] | None:
    if action_name == "vrchat_autonomy_turn":
        return {
            "speak_text": str(data.get("speak_text") or ""),
            "chatbox_text": str(data.get("chatbox_text") or ""),
            "emotion": str(data.get("emotion") or "neutral"),
            "avatar_action": str(data.get("avatar_action") or ""),
            "urgency": str(data.get("urgency") or "low"),
        }
    if action_name == "vrchat_speak":
        return {
            "speak_text": str(data.get("text") or ""),
            "chatbox_text": "",
            "emotion": str(data.get("emotion") or "neutral"),
            "avatar_action": "",
            "urgency": "low",
        }
    if action_name == "vrchat_chatbox":
        return {
            "speak_text": "",
            "chatbox_text": str(data.get("text") or ""),
            "emotion": "neutral",
            "avatar_action": "",
            "urgency": "low",
        }
    if action_name == "vrchat_avatar_action":
        return {
            "speak_text": "",
            "chatbox_text": "",
            "emotion": "neutral",
            "avatar_action": str(data.get("avatar_action") or ""),
            "urgency": "low",
        }
    return None


def _parse_action_payload(raw: Any) -> dict[str, Any]:
    if raw in (None, ""):
        return {"success": True, "data": {}}
    if isinstance(raw, dict):
        return {"success": True, "data": raw}
    if not isinstance(raw, str):
        return {"success": False, "error": "action_payload_must_be_json_object"}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"success": False, "error": "action_payload_invalid_json"}
    if not isinstance(data, dict):
        return {"success": False, "error": "action_payload_must_be_object"}
    return {"success": True, "data": data}


def _action_rejection(
    action_id: str,
    reason: str,
    *,
    game: str,
    retry_on_failure: bool,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "success": False,
        "action_id": action_id,
        "profile": profile,
        "turn": None,
        "action_result": build_action_result_message(
            action_id,
            success=bool(not retry_on_failure),
            message=f"Action rejected: {reason}",
            game=game,
        ),
    }


def _action_result_for_policy_block(
    action_id: str,
    reasons: list[str],
    game: str,
) -> dict[str, Any]:
    return build_action_result_message(
        action_id,
        success=True,
        message="Action blocked by local VRChat safety policy: " + ", ".join(reasons)[:400],
        game=game,
    )


def _turn_result_message(turn: dict[str, Any]) -> str:
    if not turn.get("success"):
        reasons = turn.get("decision", {}).get("blocked_reasons", [])
        return "Action blocked by local VRChat safety policy: " + ", ".join(reasons)[:400]
    if turn.get("dry_run"):
        return "Dry-run planned safely; no OSC or audio was sent."
    safety = turn.get("safety", {})
    if safety.get("actuation_performed"):
        return "Action validated and executed through approved VRChat/VOICEVOX surfaces."
    return "Action validated; no actuation was needed."


def _file_contains(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
