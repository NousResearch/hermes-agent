"""Validated ElevenLabs request options shared by sync and streaming TTS."""

from __future__ import annotations

import inspect
import math
from typing import Any, Dict, Optional


_VOICE_SETTING_KEYS = frozenset({
    "stability", "similarity_boost", "style", "use_speaker_boost", "speed",
})
_CONTROLLED_CONVERT_KEYS = frozenset({
    "text", "voice_id", "model_id", "output_format", "language_code", "voice_settings",
})
_KNOWN_CONVERT_OPTION_KEYS = frozenset({
    "enable_logging", "optimize_streaming_latency", "pronunciation_dictionary_locators",
    "seed", "previous_text", "next_text", "previous_request_ids", "next_request_ids",
    "use_pvc_as_ivc", "apply_text_normalization", "apply_language_text_normalization",
    "request_options",
})
_V4_MODELS = frozenset({"eleven_v4", "eleven_v4_turbo"})
# ElevenLabs documents ``speed`` as 0.7..1.2 (1.0 = neutral). Like the xAI path, out-of-band
# values are clamped so a global ``tts.speed`` tuned for another provider never 400s the request.
ELEVENLABS_SPEED_MIN = 0.7
ELEVENLABS_SPEED_MAX = 1.2


def _allowed_convert_options(convert_method: Any) -> frozenset[str]:
    allowed = set(_KNOWN_CONVERT_OPTION_KEYS)
    try:
        for name, parameter in inspect.signature(convert_method).parameters.items():
            if parameter.kind in {
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }:
                allowed.add(name)
    except (TypeError, ValueError):
        pass
    return frozenset(allowed - _CONTROLLED_CONVERT_KEYS)


def _convert_options(el_config: Dict[str, Any], convert_method: Any) -> Dict[str, Any]:
    raw = el_config.get("convert_options", el_config.get("options", {}))
    if raw in (None, {}):
        return {}
    if not isinstance(raw, dict):
        raise ValueError("tts.elevenlabs.convert_options must be a mapping")
    controlled = sorted(set(raw) & _CONTROLLED_CONVERT_KEYS)
    if controlled:
        raise ValueError(
            "ElevenLabs convert_options cannot override Hermes-managed option(s): "
            + ", ".join(controlled)
        )
    unknown = sorted(set(raw) - _allowed_convert_options(convert_method))
    if unknown:
        raise ValueError("Unknown ElevenLabs convert_options option(s): " + ", ".join(unknown))
    return dict(raw)


def _normalized_speed(value: Any) -> Optional[float]:
    """``None``/empty is unset; malformed or non-finite values are operator errors, not SDK 400s."""
    if value in (None, ""):
        return None
    try:
        speed = math.nan if isinstance(value, bool) else float(value)
    except (TypeError, ValueError):
        speed = math.nan
    if not math.isfinite(speed):
        raise ValueError(f"ElevenLabs speed must be a finite number, got {value!r}")
    return max(ELEVENLABS_SPEED_MIN, min(ELEVENLABS_SPEED_MAX, speed))


def _voice_settings(
    el_config: Dict[str, Any], tts_config: Optional[Dict[str, Any]], model_id: str,
) -> Any:
    nested = el_config.get("voice_settings")
    if nested is not None and not isinstance(nested, dict):
        raise ValueError("tts.elevenlabs.voice_settings must be a mapping")
    raw = {key: el_config[key] for key in _VOICE_SETTING_KEYS if key in el_config}
    raw.update(nested or {})
    if "speed" not in raw and isinstance(tts_config, dict):
        raw["speed"] = tts_config.get("speed")
    speed = _normalized_speed(raw.pop("speed", None))
    if speed is not None:
        raw["speed"] = speed

    unknown = sorted(set(raw) - _VOICE_SETTING_KEYS)
    if unknown:
        raise ValueError("Unknown ElevenLabs voice_settings option(s): " + ", ".join(unknown))

    if model_id in _V4_MODELS:
        incompatible = []
        if raw.get("style") not in (None, 0, 0.0):
            incompatible.append("style")
        if raw.get("speed") not in (None, 1, 1.0):
            incompatible.append("speed")
        if raw.get("use_speaker_boost") not in (None, False):
            incompatible.append("use_speaker_boost")
        if incompatible:
            raise ValueError(
                f"ElevenLabs model {model_id} does not support voice setting(s): "
                + ", ".join(incompatible)
            )
        raw.pop("style", None)
        raw.pop("speed", None)
        raw.pop("use_speaker_boost", None)

    if not raw:
        return None
    try:
        from elevenlabs import VoiceSettings
    except ImportError:
        return raw
    return VoiceSettings(**raw)


def build_elevenlabs_convert_kwargs(
    *, text: str, voice_id: str, model_id: str, output_format: str,
    el_config: Dict[str, Any], tts_config: Optional[Dict[str, Any]] = None,
    convert_method: Any = None,
) -> Dict[str, Any]:
    """Build and validate kwargs before calling the ElevenLabs SDK."""
    kwargs = _convert_options(el_config, convert_method)
    kwargs.update(text=text, voice_id=voice_id, model_id=model_id, output_format=output_format)
    language_code = el_config.get("language_code")
    if language_code not in (None, ""):
        if not isinstance(language_code, str):
            raise ValueError("tts.elevenlabs.language_code must be a string")
        if language_code.strip():
            kwargs["language_code"] = language_code.strip()
    settings = _voice_settings(el_config, tts_config, model_id)
    if settings is not None:
        kwargs["voice_settings"] = settings
    return kwargs
