#!/usr/bin/env python3
"""Audio generation tool — Venice-backed (music + SFX).

Wraps Venice's ``/audio/queue`` + ``/audio/{id}`` (or ``/audio/complete``
for one-shot) so the agent can generate music, sound effects, or
vocalized audio in response to user prompts.

Distinct from :mod:`tools.tts_tool` (text-to-speech is for spoken words
on top of provided text). This tool generates *new* audio content from
a creative prompt.

Authentication: ``VENICE_API_KEY``. Base URL is overridable via
``VENICE_BASE_URL`` for the managed-Venice proxy.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from tools.registry import registry

logger = logging.getLogger(__name__)


DEFAULT_VENICE_BASE_URL = "https://api.venice.ai/api/v1"
DEFAULT_AUDIO_MODEL = "elevenlabs-music"
DEFAULT_DURATION_SECONDS = 15
DEFAULT_POLL_INTERVAL_SECONDS = 3
DEFAULT_POLL_TIMEOUT_SECONDS = 180


def _resolve_credentials() -> Tuple[str, str]:
    api_key = os.environ.get("VENICE_API_KEY", "").strip()
    base_url = (
        os.environ.get("VENICE_BASE_URL", "").strip()
        or DEFAULT_VENICE_BASE_URL
    ).rstrip("/")
    return api_key, base_url


def _audio_cache_dir() -> Path:
    from hermes_constants import get_hermes_home

    path = get_hermes_home() / "cache" / "audio"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_binary_audio(data: bytes, *, prefix: str, ext: str = "mp3") -> Path:
    import datetime

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    path = _audio_cache_dir() / f"{prefix}_{ts}_{short}.{ext}"
    path.write_bytes(data)
    return path


def check_audio_generate_requirements() -> bool:
    api_key, _ = _resolve_credentials()
    return bool(api_key)


async def _queue_and_poll(
    *,
    api_key: str,
    base_url: str,
    payload: Dict[str, Any],
    poll_timeout: int,
    poll_interval: int,
) -> Tuple[str, Dict[str, Any]]:
    """Submit to /audio/queue, then poll /audio/{id} until done.

    Returns ``(queue_id, body_when_done)``. Raises on terminal errors.
    """
    import httpx

    async with httpx.AsyncClient() as client:
        submit = await client.post(
            f"{base_url}/audio/queue",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "hermes-agent/audio-gen-venice",
            },
            json=payload,
            timeout=60,
        )
        submit.raise_for_status()
        submit_body = submit.json()
        queue_id = submit_body.get("queue_id") or submit_body.get("id")
        if not queue_id:
            raise RuntimeError("Venice audio queue response did not include queue_id")

        elapsed = 0.0
        last_status = "QUEUED"
        while elapsed < poll_timeout:
            poll = await client.get(
                f"{base_url}/audio/{queue_id}",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": "hermes-agent/audio-gen-venice",
                },
                timeout=30,
            )
            poll.raise_for_status()
            body = poll.json()
            last_status = str(body.get("status") or "").upper()
            if last_status in {"DONE", "COMPLETED", "SUCCEEDED"}:
                return queue_id, body
            if last_status in {"FAILED", "ERROR", "EXPIRED", "CANCELLED", "CANCELED"}:
                raise RuntimeError(
                    f"Venice audio generation ended with status {last_status}: {body.get('error') or body.get('message')}"
                )
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"Timed out waiting for Venice audio after {poll_timeout}s (last status: {last_status})")


def audio_generate_tool(
    prompt: str,
    model: Optional[str] = None,
    duration_seconds: Optional[int] = None,
    lyrics_prompt: Optional[str] = None,
    voice: Optional[str] = None,
    force_instrumental: Optional[bool] = None,
    language_code: Optional[str] = None,
) -> str:
    """Generate music or sound effects from a text prompt via Venice."""
    api_key, base_url = _resolve_credentials()
    if not api_key:
        return json.dumps(
            {
                "success": False,
                "error": "VENICE_API_KEY not set. Configure Venice in `hermes model` first.",
                "error_type": "missing_api_key",
            }
        )

    if not isinstance(prompt, str) or not prompt.strip():
        return json.dumps(
            {"success": False, "error": "prompt is required", "error_type": "missing_prompt"}
        )

    chosen_model = (model or DEFAULT_AUDIO_MODEL).strip() or DEFAULT_AUDIO_MODEL
    duration = duration_seconds if isinstance(duration_seconds, int) else DEFAULT_DURATION_SECONDS
    duration = max(1, min(300, duration))

    payload: Dict[str, Any] = {
        "model": chosen_model,
        "prompt": prompt.strip(),
        "duration_seconds": duration,
    }
    if isinstance(lyrics_prompt, str) and lyrics_prompt.strip():
        payload["lyrics_prompt"] = lyrics_prompt.strip()
    if isinstance(voice, str) and voice.strip():
        payload["voice"] = voice.strip()
    if isinstance(force_instrumental, bool):
        payload["force_instrumental"] = force_instrumental
    if isinstance(language_code, str) and language_code.strip():
        payload["language_code"] = language_code.strip()

    try:
        loop = asyncio.new_event_loop()
        try:
            queue_id, body = loop.run_until_complete(
                _queue_and_poll(
                    api_key=api_key,
                    base_url=base_url,
                    payload=payload,
                    poll_timeout=DEFAULT_POLL_TIMEOUT_SECONDS,
                    poll_interval=DEFAULT_POLL_INTERVAL_SECONDS,
                )
            )
        finally:
            loop.close()
    except TimeoutError as exc:
        return json.dumps({"success": False, "error": str(exc), "error_type": "timeout"})
    except RuntimeError as exc:
        return json.dumps({"success": False, "error": str(exc), "error_type": "api_error"})
    except Exception as exc:
        return json.dumps({"success": False, "error": f"Venice audio generation failed: {exc}", "error_type": "api_error"})

    # Venice's done-state payload typically includes a download_url. If
    # the response is base64 inline instead, decode it.
    url = body.get("download_url") or body.get("url") or (body.get("audio") or {}).get("url")
    b64 = (body.get("audio") or {}).get("b64") if isinstance(body.get("audio"), dict) else None

    audio_ref: Optional[str] = None
    if isinstance(url, str) and url.strip():
        audio_ref = url
    elif isinstance(b64, str) and b64:
        try:
            import base64

            data = base64.b64decode(b64)
            saved = _save_binary_audio(data, prefix=f"venice_audio_{chosen_model}", ext="mp3")
            audio_ref = str(saved)
        except Exception as exc:
            return json.dumps(
                {"success": False, "error": f"Could not save audio: {exc}", "error_type": "io_error"}
            )

    if not audio_ref:
        return json.dumps(
            {
                "success": False,
                "error": "Venice audio generation completed without an audio URL or payload",
                "error_type": "empty_response",
            }
        )

    return json.dumps(
        {
            "success": True,
            "audio": audio_ref,
            "model": body.get("model") or chosen_model,
            "provider": "venice",
            "queue_id": queue_id,
            "duration_seconds": duration,
        }
    )


AUDIO_GENERATE_SCHEMA = {
    "name": "audio_generate",
    "description": (
        "Generate music, sound effects, or vocalized audio from a text "
        "prompt via Venice. Different from text_to_speech (which reads "
        "provided text aloud) — this tool creates new audio content "
        "from a creative description. Returns a URL or saved file path."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Creative description of the audio (e.g. 'cyberpunk synth loop, 120bpm').",
            },
            "model": {
                "type": "string",
                "description": "Venice audio model (defaults to elevenlabs-music).",
            },
            "duration_seconds": {
                "type": "integer",
                "description": "Length of generated audio (1-300). Model may clamp shorter.",
                "default": 15,
            },
            "lyrics_prompt": {
                "type": "string",
                "description": "Optional lyrics for vocal/lyric-capable models.",
            },
            "voice": {
                "type": "string",
                "description": "Voice selection for voice-enabled models.",
            },
            "force_instrumental": {
                "type": "boolean",
                "description": "Force an instrumental track (no vocals).",
            },
            "language_code": {
                "type": "string",
                "description": "ISO 639-1 language code for lyric models (e.g. 'en', 'es').",
            },
        },
        "required": ["prompt"],
    },
}


registry.register(
    name="audio_generate",
    toolset="video_gen",
    schema=AUDIO_GENERATE_SCHEMA,
    handler=lambda **kw: audio_generate_tool(**kw),
    check_fn=check_audio_generate_requirements,
    requires_env=["VENICE_API_KEY"],
    is_async=False,
    emoji="🎵",
)
