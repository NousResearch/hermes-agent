"""Optional language detection and per-language voice routing for Edge TTS."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from utils import is_truthy_value

logger = logging.getLogger("tools.tts_tool")

MIN_LANGUAGE_LETTERS = 8
MIN_LANGUAGE_CONFIDENCE = 0.80

_DETECTION_LOCK = threading.Lock()


def _normalize_language_code(value: Any) -> str:
    """Normalize detector/config language codes to lower-case BCP-47 form."""
    return str(value or "").strip().replace("_", "-").lower()


def _detect_dominant_language(text: str) -> Optional[tuple[str, float]]:
    """Return ``(language, confidence)`` or ``None`` when detection is unsafe/unavailable."""
    if sum(character.isalpha() for character in text) < MIN_LANGUAGE_LETTERS:
        logger.info("TTS auto-language kept the fallback voice: text is too short to classify")
        return None

    try:
        from tools.lazy_deps import ensure

        ensure("tts.langdetect", prompt=False)
        from langdetect import DetectorFactory, detect_langs

        # langdetect documents ambiguous input as nondeterministic unless this
        # process-wide seed is fixed. Serialize the seed + classify operation.
        with _DETECTION_LOCK:
            DetectorFactory.seed = 0
            candidates = detect_langs(text)
    except Exception as exc:
        logger.warning("TTS auto-language detection unavailable; using fallback voice: %s", exc)
        return None

    if not candidates:
        logger.info("TTS auto-language kept the fallback voice: detector returned no candidates")
        return None
    language = _normalize_language_code(getattr(candidates[0], "lang", ""))
    try:
        confidence = float(getattr(candidates[0], "prob", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if not language or confidence < MIN_LANGUAGE_CONFIDENCE:
        logger.info(
            "TTS auto-language kept the fallback voice: language=%s confidence=%.3f",
            language or "unknown",
            confidence,
        )
        return None
    return language, confidence


def _normalized_voice_map(value: Any) -> Dict[str, str]:
    """Return valid normalized language-to-voice entries from a config value."""
    if not isinstance(value, dict):
        return {}
    voices: Dict[str, str] = {}
    for raw_language, raw_voice in value.items():
        language = _normalize_language_code(raw_language)
        voice = raw_voice.strip() if isinstance(raw_voice, str) else ""
        if language and voice:
            voices[language] = voice
    return voices


def with_edge_auto_language_voice(tts_config: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Copy *tts_config* with one detected Edge voice, preserving it unchanged on fallback."""
    edge_config = tts_config.get("edge")
    if not isinstance(edge_config, dict) or not is_truthy_value(edge_config.get("auto_language")):
        return tts_config
    voices = _normalized_voice_map(edge_config.get("voice_by_language"))
    if not voices:
        return tts_config

    detected = _detect_dominant_language(text)
    if detected is None:
        return tts_config
    language, confidence = detected
    voice = voices.get(language) or voices.get(language.partition("-")[0])
    if not voice:
        logger.info(
            "TTS auto-language kept the fallback voice: language=%s confidence=%.3f has no mapping",
            language,
            confidence,
        )
        return tts_config

    updated_edge = dict(edge_config)
    updated_edge["voice"] = voice
    updated = dict(tts_config)
    updated["edge"] = updated_edge
    logger.info(
        "TTS auto-language selected language=%s confidence=%.3f voice=%s",
        language,
        confidence,
        voice,
    )
    return updated
