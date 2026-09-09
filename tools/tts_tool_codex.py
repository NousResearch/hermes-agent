"""Codex OAuth subscription TTS backend."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from threading import Event
from typing import Any, Dict

from tools.codex_web_audio import CodexSpeech, _OperationBudget, synthesize_codex_speech


def _has_codex_tts_backend() -> bool:
    try:
        from hermes_cli.auth import has_codex_runtime_credentials

        return has_codex_runtime_credentials()
    except Exception:
        return False


def _codex_tts_credentials() -> tuple[Any, Dict[str, Any]]:
    from agent.credential_pool import load_pool
    from tools.transcription_codex import _codex_stt_credentials_from_pool_entry

    pool = load_pool("openai-codex")
    entry = pool.select()
    if entry is None:
        raise ValueError(
            "OpenAI Codex OAuth credentials are unavailable. Run: hermes auth add openai-codex"
        )
    return pool, _codex_stt_credentials_from_pool_entry(entry)


def synthesize_codex_speech_with_credentials(
    text: str,
    pool: Any,
    credentials: Dict[str, Any],
    *,
    voice: str = "juniper",
    timeout: float = 120,
    cancel_event: Event | None = None,
) -> CodexSpeech:
    """Synthesize using the captured profile pool, with one matching-token refresh."""
    budget = _OperationBudget(time.monotonic() + timeout, cancel_event)
    api_key = credentials["api_key"]
    try:
        return synthesize_codex_speech(
            text, api_key, voice=voice, timeout=budget.remaining(), cancel_event=cancel_event
        )
    except RuntimeError as exc:
        message = str(exc)
        if "HTTP 401" not in message:
            raise
        budget.remaining()
        refreshed = pool.try_refresh_matching(
            api_key_hint=api_key,
            credential_id=credentials.get("credential_id"),
        )
        if refreshed is None or refreshed.runtime_api_key == api_key:
            raise
        credentials["api_key"] = refreshed.runtime_api_key
        return synthesize_codex_speech(
            text, refreshed.runtime_api_key, voice=voice, timeout=budget.remaining(),
            cancel_event=cancel_event,
        )


def _generate_openai_codex_tts(
    text: str, output_path: str, tts_config: Dict[str, Any]
) -> None:
    """Generate verified ChatGPT subscription read-aloud audio at *output_path*."""
    target = Path(output_path)
    if target.suffix.lower() not in {"", ".mp3", ".ogg"}:
        raise ValueError("Codex subscription TTS output must be .mp3 or .ogg")
    section = tts_config.get("openai_codex") or {}
    voice = str(section.get("voice") or "juniper")
    timeout = max(10.0, min(float(section.get("timeout") or 120), 180.0))
    pool, credentials = _codex_tts_credentials()
    speech = synthesize_codex_speech_with_credentials(
        text, pool, credentials, voice=voice, timeout=timeout
    )

    if target.suffix.lower() in {"", ".mp3"}:
        target.write_bytes(speech.audio)
        return

    # ChatGPT synthesize emits MP3. Voice-bubble platforms require Ogg/Opus,
    # so convert through a sibling temporary and publish exactly the requested path.
    fd, source = tempfile.mkstemp(
        prefix=target.stem + ".", suffix=".mp3", dir=str(target.parent)
    )
    os.close(fd)
    converted = ""
    try:
        Path(source).write_bytes(speech.audio)
        from tools.tts_tool_delivery import _convert_to_opus

        converted = _convert_to_opus(source) or ""
        if not converted or not Path(converted).is_file():
            raise RuntimeError(
                "ffmpeg could not convert Codex subscription speech to Ogg/Opus"
            )
        os.replace(converted, target)
    finally:
        Path(source).unlink(missing_ok=True)
        if converted:
            Path(converted).unlink(missing_ok=True)
