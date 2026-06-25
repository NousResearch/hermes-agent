"""Prepare local VRChat autonomy operator profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.openclaw.vrchat_autonomy import (
    DEFAULT_HARNESS_URL,
    DEFAULT_MIN_TURN_INTERVAL_SEC,
    DEFAULT_VOICEVOX_URL,
    LIVE_ACTUATION_ACK,
    MAX_LOOP_OBSERVATIONS,
    VALID_MODES,
    load_autonomy_profile,
    validate_autonomy_profile,
)

DEFAULT_AUDIO_OUTPUT_DEVICE = "CABLE Input"
DEFAULT_VRCHAT_MICROPHONE_DEVICE = "CABLE Output"
DEFAULT_PERSONA = "Calm, concise VRChat companion. Do not mention hidden reasoning."
DEFAULT_TASK = "Choose one safe response to the latest VRChat observations."


def prepare_autonomy_profile(
    *,
    profile_path: str | Path | None = None,
    enabled: bool = True,
    mode: str = "private_test",
    audio_output_device: str = DEFAULT_AUDIO_OUTPUT_DEVICE,
    vrchat_microphone_device: str = DEFAULT_VRCHAT_MICROPHONE_DEVICE,
    output_device: str | None = None,
    voicevox_url: str = DEFAULT_VOICEVOX_URL,
    harness_url: str = DEFAULT_HARNESS_URL,
    require_harness: bool = False,
    allow_voice: bool = True,
    allow_chatbox: bool = True,
    allow_movement: bool = False,
    allow_interrupt: bool = False,
    consume_queue: bool = True,
    max_observations: int = MAX_LOOP_OBSERVATIONS,
    min_turn_interval_sec: float = DEFAULT_MIN_TURN_INTERVAL_SEC,
    voicevox_speaker: int = 8,
    chatbox_immediate: bool = True,
    tts_backend: str = "irodori",
    irodori_voice: str = "hakua",
    irodori_speed: float = 1.0,
    irodori_base_url: str = "",
    persona: str | None = None,
    task: str | None = None,
    allowed_avatar_actions: list[str] | None = None,
    avatar_action_descriptions: dict[str, str] | None = None,
    avatar_action_profiles: dict[str, list[dict[str, Any]]] | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    temperature: float = 0.0,
    max_tokens: int = 320,
    arm_live: bool = False,
    live_ack: str = "",
) -> dict[str, Any]:
    """Write a validated local operator profile, defaulting to dry-run."""
    loaded = load_autonomy_profile(profile_path)
    path = Path(loaded["path"])
    profile = dict(loaded.get("profile") or {})

    if arm_live and live_ack != LIVE_ACTUATION_ACK:
        return {
            "success": False,
            "written": False,
            "path": str(path),
            "code": "LIVE_ACK_REQUIRED",
            "message": "Exact live acknowledgement is required before writing a non-dry-run profile.",
            "required_live_ack": LIVE_ACTUATION_ACK,
            "profile": profile,
            "validation": validate_autonomy_profile(profile),
            "safety": _safety_flags(),
        }

    profile.update(
        {
            "enabled": bool(enabled),
            "mode": str(mode or "private_test"),
            "dry_run": not bool(arm_live),
            "consume_queue": bool(consume_queue),
            "max_observations": max(0, int(max_observations)),
            "min_turn_interval_sec": max(0.0, float(min_turn_interval_sec)),
            "voicevox_url": str(voicevox_url or DEFAULT_VOICEVOX_URL),
            "harness_url": str(harness_url or DEFAULT_HARNESS_URL),
            "audio_output_device": str(audio_output_device or ""),
            "vrchat_microphone_device": str(vrchat_microphone_device or ""),
            "require_harness": bool(require_harness),
            "allow_voice": bool(allow_voice),
            "allow_chatbox": bool(allow_chatbox),
            "allow_movement": bool(allow_movement),
            "allow_interrupt": bool(allow_interrupt),
            "output_device": str(output_device if output_device is not None else audio_output_device or ""),
            "voicevox_speaker": int(voicevox_speaker),
            "chatbox_immediate": bool(chatbox_immediate),
            "tts_backend": str(tts_backend or "irodori"),
            "irodori_voice": str(irodori_voice or "hakua"),
            "irodori_speed": float(irodori_speed),
            "irodori_base_url": str(irodori_base_url or ""),
            "temperature": float(temperature),
            "max_tokens": max(1, int(max_tokens)),
            "live_actuation_ack": LIVE_ACTUATION_ACK if arm_live else "",
        }
    )

    if persona is None:
        if not profile.get("persona"):
            profile["persona"] = DEFAULT_PERSONA
    else:
        profile["persona"] = str(persona)
    if task is None:
        if not profile.get("task"):
            profile["task"] = DEFAULT_TASK
    else:
        profile["task"] = str(task)
    if provider is not None:
        profile["provider"] = str(provider)
    if model is not None:
        profile["model"] = str(model)
    if base_url is not None:
        profile["base_url"] = str(base_url)
    if timeout is not None:
        profile["timeout"] = timeout

    if allowed_avatar_actions is not None:
        profile["allowed_avatar_actions"] = _coerce_string_list(allowed_avatar_actions)
    if avatar_action_descriptions is not None:
        profile["avatar_action_descriptions"] = dict(avatar_action_descriptions)
    if avatar_action_profiles is not None:
        profile["avatar_action_profiles"] = dict(avatar_action_profiles)

    if profile["mode"] not in VALID_MODES:
        validation = {
            "success": False,
            "errors": [f"invalid_mode:{profile['mode']}"],
            "warnings": [],
        }
    else:
        validation = validate_autonomy_profile(profile)
    if not validation["success"]:
        return {
            "success": False,
            "written": False,
            "path": str(path),
            "code": "PROFILE_VALIDATION_FAILED",
            "message": "Profile was not written because validation failed.",
            "required_live_ack": LIVE_ACTUATION_ACK,
            "profile": profile,
            "validation": validation,
            "safety": _safety_flags(),
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "success": True,
        "written": True,
        "path": str(path),
        "code": "LIVE_PROFILE_ARMED" if arm_live else "DRY_RUN_PROFILE_READY",
        "message": "VRChat autonomy profile prepared.",
        "required_live_ack": LIVE_ACTUATION_ACK,
        "profile": profile,
        "validation": validation,
        "safety": _safety_flags(),
    }


def _coerce_string_list(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            result.append(text)
    return result


def _safety_flags() -> dict[str, bool]:
    return {
        "actuation_performed": False,
        "chatbox_sent": False,
        "speech_played": False,
        "avatar_parameters_written": False,
    }
