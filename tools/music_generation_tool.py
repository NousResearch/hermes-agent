#!/usr/bin/env python3
"""
Music Generation Tool

Generates music using the SenseAudio API (senseaudio.cn).

Workflow:
  generate(prompt-only) → POST /v1/music/lyrics/create → structured lyrics
                       → POST /v1/music/song/create → task_id
  generate(with lyrics) → POST /v1/music/song/create → task_id
           → background task polls GET /v1/music/song/pending/:task_id
           → returns GenerateMusicRuntimeResult when done
  status   → GET /v1/music/song/pending/:task_id for the active task

Environment:
  SENSEAUDIO_API_KEY  — required
  SENSEAUDIO_BASE_URL — optional override (default: https://api.senseaudio.cn/v1)
"""

import asyncio
import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from agent.background_task import background_tasks, current_session_origin
from hermes_constants import get_hermes_dir
from tools.debug_helpers import DebugSession
from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PROVIDER = "senseaudio"
DEFAULT_SENSEAUDIO_BASE_URL = "https://api.senseaudio.cn/v1"
SENSEAUDIO_BASE_URL = os.getenv("SENSEAUDIO_BASE_URL", DEFAULT_SENSEAUDIO_BASE_URL)
SENSEAUDIO_MUSIC_MODEL = "senseaudio-music-1.0-260319"

POLL_INTERVAL = 5     # seconds between status checks
POLL_MAX_WAIT = 1800  # total timeout (30 minutes)

_debug = DebugSession("music_tools", env_var="MUSIC_TOOLS_DEBUG")

# session_key → {"provider": str, "task_id": str}
_active_music_tasks: Dict[str, Dict[str, str]] = {}
_active_music_tasks_lock = threading.Lock()


def _get_music_output_dir() -> Path:
    return get_hermes_dir("cache/music", "music_cache")


def _sanitize_track_stem(name: str) -> str:
    stem = Path(name).stem.strip()
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-.")
    return sanitized or "track"


def _download_track_to_local(audio_url: str, *, file_name: str) -> str:
    """Download a generated track to Hermes-managed local storage."""
    out_dir = _get_music_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file_name).suffix or ".mp3"
    output_path = out_dir / f"{_sanitize_track_stem(file_name)}-{uuid.uuid4().hex[:8]}{suffix}"

    with httpx.Client(timeout=60, follow_redirects=True) as client:
        resp = client.get(audio_url)
        _raise_for_status_with_details(resp)
        content = resp.content

    if not content:
        raise ValueError(f"Downloaded empty audio file from SenseAudio: {audio_url}")

    output_path.write_bytes(content)
    return str(output_path)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_music_config() -> Dict[str, Any]:
    """Load the ``music`` section from user config, falling back to defaults."""
    try:
        from hermes_cli.config import load_config
        config = load_config()
        return config.get("music", {})
    except ImportError:
        logger.debug("hermes_cli.config not available, using default music config")
        return {}
    except Exception as exc:
        logger.warning("Failed to load music config: %s", exc, exc_info=True)
        return {}


def _get_provider(music_config: Dict[str, Any]) -> str:
    """Return the configured music provider name."""
    return (music_config.get("provider") or DEFAULT_PROVIDER).lower().strip()


def _get_senseaudio_config(music_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the provider-specific SenseAudio config section."""
    config = music_config or {}
    provider_cfg = config.get("senseaudio", {})
    return provider_cfg if isinstance(provider_cfg, dict) else {}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _api_key(music_config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    senseaudio_cfg = _get_senseaudio_config(music_config)
    return (senseaudio_cfg.get("api_key") or os.getenv("SENSEAUDIO_API_KEY") or "").strip() or None


def _base_url(music_config: Optional[Dict[str, Any]] = None) -> str:
    senseaudio_cfg = _get_senseaudio_config(music_config)
    return (senseaudio_cfg.get("base_url") or os.getenv("SENSEAUDIO_BASE_URL") or DEFAULT_SENSEAUDIO_BASE_URL).rstrip("/")


def _model_name(music_config: Optional[Dict[str, Any]] = None) -> str:
    senseaudio_cfg = _get_senseaudio_config(music_config)
    return (senseaudio_cfg.get("model") or SENSEAUDIO_MUSIC_MODEL).strip()


def _auth_headers(music_config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {_api_key(music_config)}",
        "Content-Type": "application/json",
    }


def _raise_for_status_with_details(resp: httpx.Response) -> None:
    """Raise HTTP errors with provider response details when available."""
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = ""
        try:
            payload = resp.json()
        except Exception:
            payload = None

        if isinstance(payload, dict):
            message = payload.get("message") or payload.get("error")
            code = payload.get("code")
            ref_code = payload.get("ref_code")
            extras = []
            if code:
                extras.append(f"code={code}")
            if ref_code is not None:
                extras.append(f"ref_code={ref_code}")
            if message:
                detail = str(message)
                if extras:
                    detail = f"{detail} ({', '.join(extras)})"

        if not detail:
            text = resp.text.strip()
            if text:
                detail = text[:500]

        if detail:
            raise ValueError(f"SenseAudio API error {resp.status_code}: {detail}") from exc
        raise


def _create_lyrics_from_prompt(
    prompt: str,
    music_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[str]]:
    """Convert a free-form prompt into structured lyrics via SenseAudio."""
    body = {
        "prompt": prompt,
        "provider": _model_name(music_config),
    }

    url = f"{_base_url(music_config)}/music/lyrics/create"
    with httpx.Client(timeout=30) as client:
        resp = client.post(url, json=body, headers=_auth_headers(music_config))
        _raise_for_status_with_details(resp)
        data = resp.json()

    items = data.get("data") or []
    first = items[0] if items else {}
    lyrics_text = (first.get("text") or "").strip()
    if not lyrics_text:
        raise ValueError(f"SenseAudio did not return generated lyrics: {data}")

    title = (first.get("title") or "").strip() or None
    return {"lyrics": lyrics_text, "title": title}


def _create_song(
    prompt: str,
    *,
    lyrics: Optional[str] = None,
    instrumental: bool = False,
    style: Optional[str] = None,
    vocal_gender: Optional[str] = None,
    title: Optional[str] = None,
    music_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Submit a song creation request. Returns the SenseAudio task_id."""
    # custom_mode=True causes a 400 "无效的歌词格式" from the SenseAudio API.
    # Always use custom_mode=False with structured lyrics from lyrics/create.
    body: Dict[str, Any] = {
        "model": _model_name(music_config),
        "lyrics": lyrics,
        "custom_mode": False,
        "instrumental": instrumental,
    }
    if style:
        body["style"] = style
    if vocal_gender:
        body["vocal_gender"] = vocal_gender
    if title:
        body["title"] = title

    url = f"{_base_url(music_config)}/music/song/create"
    with httpx.Client(timeout=30) as client:
        resp = client.post(url, json=body, headers=_auth_headers(music_config))
        _raise_for_status_with_details(resp)
        data = resp.json()

    task_id = data.get("task_id")
    if not task_id:
        raise ValueError(f"SenseAudio did not return a task_id: {data}")
    return task_id


def _query_status(task_id: str, music_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Call the pending endpoint once. Returns the raw response dict."""
    url = f"{_base_url(music_config)}/music/song/pending/{task_id}"
    with httpx.Client(timeout=30) as client:
        resp = client.get(url, headers=_auth_headers(music_config))
        _raise_for_status_with_details(resp)
        return resp.json()


def _poll_sync(task_id: str, music_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Block until the task reaches SUCCESS/FAILED or POLL_MAX_WAIT elapses."""
    deadline = time.monotonic() + POLL_MAX_WAIT
    while time.monotonic() < deadline:
        data = _query_status(task_id, music_config=music_config)
        status = data.get("status", "")
        if status == "SUCCESS":
            return data
        if status == "FAILED":
            raise RuntimeError(f"SenseAudio task {task_id} failed: {data}")
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(
        f"SenseAudio task {task_id} did not complete within {POLL_MAX_WAIT}s"
    )


async def _poll_async(
    task_id: str,
    session_key: str,
    music_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Background coroutine: poll until done, return the agent wake message."""
    deadline = time.monotonic() + POLL_MAX_WAIT
    try:
        while time.monotonic() < deadline:
            data = _query_status(task_id, music_config=music_config)
            status = data.get("status", "")
            if status == "SUCCESS":
                result = _format_result(data, ignored_overrides=[])
                return _wake_message(result)
            if status == "FAILED":
                return f"Music generation failed (task_id={task_id})."
            await asyncio.sleep(POLL_INTERVAL)
        return f"Music generation timed out after {POLL_MAX_WAIT}s (task_id={task_id})."
    finally:
        with _active_music_tasks_lock:
            _active_music_tasks.pop(session_key, None)


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def _format_result(
    pending_data: Dict[str, Any],
    ignored_overrides: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Convert SenseAudio pending response → GenerateMusicRuntimeResult."""
    items = (pending_data.get("response") or {}).get("data") or []
    tracks = []
    lyrics_list = []

    for item in items:
        audio_url = item.get("audio_url", "")
        cover_url = item.get("cover_url", "")
        duration = item.get("duration")
        item_title = item.get("title", "")
        raw_lyrics = item.get("lyrics", "")
        file_name = f"{item_title or 'track'}.mp3"
        local_path = _download_track_to_local(audio_url, file_name=file_name) if audio_url else ""

        tracks.append({
            "url": audio_url,
            "mimeType": "audio/mpeg",
            "fileName": file_name,
            "localPath": local_path or None,
            "mediaTag": f"MEDIA:{local_path}" if local_path else None,
            "metadata": {
                "cover_url": cover_url,
                "duration": duration,
                "title": item_title,
                "source_url": audio_url,
            },
        })
        if raw_lyrics:
            lyrics_list.append(raw_lyrics)

    return {
        "tracks": tracks,
        "provider": "senseaudio",
        "model": SENSEAUDIO_MUSIC_MODEL,
        "lyrics": lyrics_list or None,
        "ignoredOverrides": ignored_overrides,
    }


def _wake_message(result: Dict[str, Any]) -> str:
    """Build the text injected into the session when background generation completes."""
    tracks = result.get("tracks") or []
    if not tracks:
        return "Music generation completed but no tracks were returned."

    lines = ["Music generation completed."]
    for track in tracks:
        meta = track.get("metadata") or {}
        label = meta.get("title") or track.get("fileName", "track")
        duration = meta.get("duration")
        suffix = f" ({duration}s)" if duration else ""
        lines.append(f"Track: {label}{suffix}")
        local_path = track.get("localPath")
        if local_path:
            lines.append(f"Saved: {local_path}")
            lines.append(track.get("mediaTag") or f"MEDIA:{local_path}")
        elif track.get("url"):
            lines.append(f"Source: {track['url']}")
        if meta.get("cover_url"):
            lines.append(f"Cover: {meta['cover_url']}")

    if result.get("lyrics"):
        lines.append(f"Lyrics:\n{result['lyrics'][0]}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Provider: SenseAudio
# ---------------------------------------------------------------------------

def _senseaudio_status(
    *,
    session_key: str,
    music_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Query status for the active SenseAudio task in this session."""
    with _active_music_tasks_lock:
        task_info = _active_music_tasks.get(session_key)

    task_id = (task_info or {}).get("task_id", "")
    if not task_id:
        handle = background_tasks.get_active(session_key)
        if handle:
            return tool_result(status=handle.status, task_id=handle.task_id)
        return tool_error("No active music generation task for this session.")

    try:
        data = _query_status(task_id, music_config=music_config)
        return tool_result(
            status=data.get("status"),
            task_id=task_id,
            details=data,
        )
    except Exception as exc:
        logger.error("Status query failed for task %s: %s", task_id, exc)
        return tool_error(f"Failed to query task status: {exc}")


def _senseaudio_generate(
    *,
    prompt: str,
    lyrics: Optional[str],
    instrumental: bool,
    style: Optional[str],
    vocal_gender: Optional[str],
    title: Optional[str],
    ignored_overrides: List[Dict[str, Any]],
    origin,
    session_key: str,
    music_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate music via SenseAudio with sync or detached completion."""
    if not prompt and not lyrics:
        return tool_error("prompt is required for music generation")

    if not _api_key(music_config):
        return tool_error("SENSEAUDIO_API_KEY environment variable not set")

    try:
        resolved_lyrics = lyrics
        resolved_title = title
        if not resolved_lyrics:
            generated = _create_lyrics_from_prompt(prompt, music_config=music_config)
            resolved_lyrics = generated["lyrics"]
            if not resolved_title:
                resolved_title = generated.get("title")

        task_id = _create_song(
            prompt,
            lyrics=resolved_lyrics,
            instrumental=instrumental,
            style=style,
            vocal_gender=vocal_gender,
            title=resolved_title,
            music_config=music_config,
        )
        logger.info("SenseAudio song task created: %s", task_id)
    except Exception as exc:
        logger.error("Failed to create SenseAudio song task: %s", exc, exc_info=True)
        return tool_error(f"Failed to start music generation: {exc}")

    if session_key:
        with _active_music_tasks_lock:
            _active_music_tasks[session_key] = {
                "provider": "senseaudio",
                "task_id": task_id,
            }

    if session_key:
        handle = background_tasks.create(
            coro=_poll_async(task_id, session_key, music_config=music_config),
            session_key=session_key,
            origin=origin,
            label="music generation",
        )
        if handle is not None:
            logger.info("Music generation running in background, task_id=%s", task_id)
            return tool_result(
                status="started",
                task_id=task_id,
                message="Music is generating in the background. You will be notified automatically when it's ready. Do not check status again unless the user explicitly asks for a progress update.",
            )

    try:
        pending_data = _poll_sync(task_id, music_config=music_config)
        result = _format_result(pending_data, ignored_overrides)
        _debug.log_call("music_generate_tool", {"task_id": task_id, "success": True})
        _debug.save()
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.error("Music generation polling failed: %s", exc, exc_info=True)
        _debug.log_call("music_generate_tool", {"task_id": task_id, "success": False, "error": str(exc)})
        _debug.save()
        return tool_error(f"Music generation failed: {exc}")
    finally:
        if session_key:
            with _active_music_tasks_lock:
                _active_music_tasks.pop(session_key, None)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def music_generate_tool(
    prompt: str = "",
    action: str = "generate",
    lyrics: Optional[str] = None,
    instrumental: bool = False,
    style: Optional[str] = None,
    vocal_gender: Optional[str] = None,
    title: Optional[str] = None,
    # Unsupported params — collected and surfaced in ignoredOverrides
    durationSeconds: Optional[float] = None,
    format: Optional[str] = None,       # noqa: A002
    image: Optional[str] = None,
    images: Optional[List[str]] = None,
) -> str:
    """
    Generate music using SenseAudio. Returns a JSON string.

    Args:
        prompt:          Music style/mood description (required for generate)
        action:          "generate" (default) or "status"
        lyrics:          Structured lyrics; if omitted the model improvises from prompt
        instrumental:    True for no-vocals generation
        style:           Style hint, e.g. "jazz", "pop"
        vocal_gender:    "f" or "m"
        title:           Optional song title
        durationSeconds: Ignored — SenseAudio does not support duration control
        format:          Ignored — output format is fixed by the provider
        image/images:    Ignored — SenseAudio music API has no image input

    Returns:
        JSON string with fields: tracks, provider, model, lyrics,
        ignoredOverrides (GenerateMusicRuntimeResult shape)
    """
    # Collect unsupported params so the caller knows what was dropped
    ignored_overrides: List[Dict[str, Any]] = []
    if durationSeconds is not None:
        ignored_overrides.append({"key": "durationSeconds", "value": durationSeconds})
    if format is not None:
        ignored_overrides.append({"key": "format", "value": format})
    if image is not None:
        ignored_overrides.append({"key": "image", "value": image})
    if images is not None:
        ignored_overrides.append({"key": "images", "value": images})

    music_config = _load_music_config()
    provider = _get_provider(music_config)
    origin = current_session_origin()
    session_key = origin.session_key

    if provider == "senseaudio":
        if action == "status":
            return _senseaudio_status(
                session_key=session_key,
                music_config=music_config,
            )
        return _senseaudio_generate(
            prompt=prompt,
            lyrics=lyrics,
            instrumental=instrumental,
            style=style,
            vocal_gender=vocal_gender,
            title=title,
            ignored_overrides=ignored_overrides,
            origin=origin,
            session_key=session_key,
            music_config=music_config,
        )

    return tool_error(f"Unknown music provider: {provider}")


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def check_music_generation_requirements() -> bool:
    try:
        import httpx  # noqa: F401
        music_config = _load_music_config()
        provider = _get_provider(music_config)
        if provider == "senseaudio":
            return bool(_api_key(music_config))
        return False
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MUSIC_GENERATE_SCHEMA = {
    "name": "music_generate",
    "description": (
        "Generate music from a text prompt. "
        "Generation runs in the background "
        "and you are notified automatically when the audio is ready. "
        "After starting a generation, do not call this tool again to poll for status unless the user explicitly asks for a progress update. "
        "Use action='status' only when the user explicitly wants to check an in-progress generation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "Text prompt describing the desired music — style, mood, theme, etc. "
                    "Required when action='generate'."
                ),
            },
            "action": {
                "type": "string",
                "enum": ["generate", "status"],
                "description": (
                    "'generate' starts a new generation task (default). "
                    "'status' queries the active task for this session, but only use it when the user explicitly asks to check progress."
                ),
                "default": "generate",
            },
            "lyrics": {
                "type": "string",
                "description": (
                    "Structured lyrics text. Supports section tags: "
                    "[verse], [chorus], [bridge], [intro-short], [outro-medium], etc. "
                    "If omitted the model generates music directly from the prompt."
                ),
            },
            "instrumental": {
                "type": "boolean",
                "description": "Set true to generate instrumental-only music (no vocals).",
                "default": False,
            },
            "style": {
                "type": "string",
                "description": "Musical style hint, e.g. 'jazz', 'lo-fi', 'classical'.",
            },
            "vocal_gender": {
                "type": "string",
                "enum": ["f", "m"],
                "description": "'f' for female vocals, 'm' for male vocals.",
            },
            "title": {
                "type": "string",
                "description": "Optional song title.",
            },
        },
        "required": [],
    },
}


def _handle_music_generate(args: Dict[str, Any], **kw) -> str:
    return music_generate_tool(
        prompt=args.get("prompt", ""),
        action=args.get("action", "generate"),
        lyrics=args.get("lyrics"),
        instrumental=bool(args.get("instrumental", False)),
        style=args.get("style"),
        vocal_gender=args.get("vocal_gender"),
        title=args.get("title"),
    )


registry.register(
    name="music_generate",
    toolset="music_gen",
    schema=MUSIC_GENERATE_SCHEMA,
    handler=_handle_music_generate,
    check_fn=check_music_generation_requirements,
    requires_env=["SENSEAUDIO_API_KEY"],
    is_async=False,
    emoji="🎵",
)
