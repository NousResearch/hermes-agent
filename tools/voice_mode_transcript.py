"""Pure-text helpers for voice mode: Whisper hallucination filter, voice-chat
stop phrases, and the TTS self-echo guard. No audio dependencies."""

import difflib
import re
from contextlib import suppress
from typing import Optional


def _voice_config() -> dict:
    """``voice`` section of config.yaml, or ``{}`` when missing, malformed, or the
    config system can't be imported (broken config mid-install)."""
    with suppress(Exception):
        from hermes_cli.config import load_config
        voice_cfg = load_config().get("voice", {})
        return voice_cfg if isinstance(voice_cfg, dict) else {}
    return {}


# Whisper commonly hallucinates these phrases on silent/near-silent audio
# (matched with trailing '.'/'!' stripped, so the bare form suffices).
WHISPER_HALLUCINATIONS = {
    "thank you", "thanks for watching", "subscribe to my channel", "like and subscribe",
    "please subscribe", "thank you for watching", "bye", "you", "the end",
    # Non-English hallucinations (common on silence)
    "продолжение следует", "sous-titres", "sous-titres réalisés par la communauté d'amara.org",
    "sottotitoli creati dalla comunità amara.org", "untertitel von stephanie geiges",
    "amara.org", "www.mooji.org", "ご視聴ありがとうございました",
}

# Repetitive hallucinations (e.g. "Thank you. Thank you. Thank you.")
_HALLUCINATION_REPEAT_RE = re.compile(r'^(?:thank you|thanks|bye|you|ok|okay|the end|\.|\s|,|!)+$',
                                      flags=re.IGNORECASE)


def is_whisper_hallucination(transcript: str) -> bool:
    """Check if a transcript is a known Whisper hallucination on silence."""
    cleaned = transcript.strip().lower()
    return (not cleaned or cleaned.rstrip('.!') in WHISPER_HALLUCINATIONS
            or bool(_HALLUCINATION_REPEAT_RE.match(cleaned)))


DEFAULT_VOICE_STOP_PHRASES = ("stop",)


def _load_voice_stop_phrases() -> tuple:
    """Configured ``voice.stop_phrases`` (default ``("stop",)``); an empty tuple disables
    the feature. Malformed config (dict, list of non-strings) falls back to the default
    rather than crashing the voice loop."""
    with suppress(Exception):
        raw = _voice_config().get("stop_phrases", DEFAULT_VOICE_STOP_PHRASES)
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, (list, tuple)):
            return tuple(str(p).strip().lower() for p in raw if isinstance(p, (str, int, float)) and str(p).strip())
    return DEFAULT_VOICE_STOP_PHRASES


def is_voice_stop_phrase(transcript: str, stop_phrases: Optional[tuple] = None) -> bool:
    """True when *transcript* is EXACTLY a configured stop phrase. Deliberately strict: the whole
    utterance — lowercased, surrounding punctuation stripped — must equal a phrase, so "stop doing
    that and try again" still reaches the agent. ``voice.stop_phrases: []`` disables."""
    cleaned = transcript.strip().lower().strip(".,!?;: \t\n\"'") if transcript else ""
    return bool(cleaned) and cleaned in (_load_voice_stop_phrases() if stop_phrases is None else stop_phrases)


# Similarity ratio (difflib.SequenceMatcher) above which a playback-phase barge transcript
# is treated as a self-capture of Hermes' own TTS: the full-duplex listener has no echo
# cancellation, so speaker bleed can be transcribed near-verbatim (TTS -> STT -> TTS loop).
DEFAULT_TTS_ECHO_SIMILARITY_THRESHOLD = 0.6

# Minimum normalized-transcript length before the sliding-window fallback runs. Below
# this a genuine one-word barge-in ("yes") landing verbatim inside a longer reply would
# score a trivial 1.0; a real self-capture spans pre-roll plus time-to-silence, so it is longer.
# See #75792.
MIN_FRAGMENT_LENGTH_FOR_ECHO = 10


def _normalize_for_echo_compare(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def is_tts_echo(transcript: str, spoken_text: str,
                threshold: float = DEFAULT_TTS_ECHO_SIMILARITY_THRESHOLD) -> bool:
    """True when *transcript* looks like a self-capture of *spoken_text*. Character-level similarity
    (language-agnostic): a genuine interjection rarely matches Hermes' own words, so a high ratio signals
    speaker-bleed (fail-closed guard for the playback-phase listener). Playback capture spans only pre-roll
    plus time-to-silence, so for long replies the transcript is a FRAGMENT and the whole-string ratio dilutes
    toward 0; when it misses, a transcript-sized window slides across `spoken_text`. Transcripts shorter than
    `MIN_FRAGMENT_LENGTH_FOR_ECHO` skip the fallback (a short interjection trivially matches a short window)."""
    a, b = _normalize_for_echo_compare(transcript or ""), _normalize_for_echo_compare(spoken_text or "")
    if not a or not b:
        return False

    def _similar(x: str, y: str) -> bool:
        return difflib.SequenceMatcher(None, x, y).ratio() >= threshold

    if _similar(a, b):
        return True
    if len(a) < MIN_FRAGMENT_LENGTH_FOR_ECHO or len(a) >= len(b):
        return False
    return any(_similar(a, b[start : start + len(a)]) for start in range(0, len(b) - len(a) + 1))


def voice_stop_hint() -> str:
    """One-line 'Say "stop" to end the voice chat.' hint for voice-mode start, using the first
    ``voice.stop_phrases`` entry ("" when disabled). Every surface announcing voice-mode start
    (CLI, TUI, desktop) uses this one owner instead of hardcoding the wording."""
    phrases = _load_voice_stop_phrases()
    return f'Say "{phrases[0]}" to end the voice chat.' if phrases else ""
