"""Music generation tool — generates songs with lyrics and style prompt via the native provider."""

from __future__ import annotations

import json
import logging
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict

from tools.registry import registry

logger = logging.getLogger(__name__)

_MUSIC_VALID_FORMATS = {"mp3", "wav", "pcm"}


# ─── Helpers ──────────────────────────────────────────────────────────────


def _post_json(url: str, payload: Dict[str, Any], key: str,
               *, timeout: int = 180) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
    )
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _music_cache_dir() -> Path:
    try:
        from hermes_constants import get_hermes_dir
        return Path(get_hermes_dir("cache/music", "music_cache"))
    except Exception:
        return Path.home() / ".hermes" / "music_cache"


def _music_model_for_key(key: str) -> str:
    """``sk-cp-`` (Token Plan) → ``music-2.6``; others → ``music-2.6-free``."""
    return "music-2.6" if key.startswith("sk-cp-") else "music-2.6-free"


# ─── Core ─────────────────────────────────────────────────────────────────


def music_generate_tool(
    prompt: str,
    lyrics: str,
    output_format: str = "mp3",
    sample_rate: int = 44100,
    bitrate: int = 256000,
) -> str:
    try:
        from hermes_cli.provider_native_tools import native_api_url
    except Exception:
        return json.dumps({"success": False,
                           "error": "no music backend available"})

    from hermes_cli.provider_native_tools import native_credential
    url = native_api_url("/v1/music_generation")
    key = native_credential()
    if not url or not key:
        return json.dumps({"success": False,
                           "error": "no music backend configured"})

    fmt = (output_format or "mp3").lower()
    if fmt not in _MUSIC_VALID_FORMATS:
        fmt = "mp3"

    model = _music_model_for_key(key)
    payload = {
        "model": model,
        "prompt": (prompt or "").strip(),
        "lyrics": (lyrics or "").strip(),
        "audio_setting": {
            "sample_rate": sample_rate,
            "bitrate": bitrate,
            "format": fmt,
        },
        "output_format": "hex",
        "stream": False,
    }

    try:
        body = _post_json(url, payload, key)
    except urllib.error.HTTPError as exc:
        return json.dumps({"success": False,
                           "error": f"music API HTTP {exc.code}"})
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        return json.dumps({"success": False, "error": str(exc)})

    base = body.get("base_resp") or {}
    if isinstance(base, dict) and base.get("status_code") not in (0, None):
        return json.dumps({"success": False,
                           "error": f"music API error {base.get('status_code')}: "
                                    f"{base.get('status_msg', '')}"})

    data = body.get("data") or {}
    audio_hex = data.get("audio") if isinstance(data, dict) else None
    if not isinstance(audio_hex, str) or not audio_hex:
        return json.dumps({"success": False,
                           "error": "music API returned no audio"})

    try:
        audio_bytes = bytes.fromhex(audio_hex)
    except ValueError as exc:
        return json.dumps({"success": False,
                           "error": f"invalid hex audio: {exc}"})

    out_dir = _music_cache_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"music_{ts}.{fmt}"
    try:
        path.write_bytes(audio_bytes)
    except OSError as exc:
        return json.dumps({"success": False, "error": f"write failed: {exc}"})

    return json.dumps({
        "success": True,
        "model": model,
        "path": str(path),
        "format": fmt,
        "bytes": len(audio_bytes),
    }, ensure_ascii=False)


# ─── Check & registration ─────────────────────────────────────────────────


def check_music_generation_requirements() -> bool:
    try:
        from hermes_cli.provider_native_tools import provider_has_native_tool, _safe_load_config
        return provider_has_native_tool("music_gen", _safe_load_config())
    except Exception:
        return False


MUSIC_GENERATE_SCHEMA = {
    "name": "music_generate",
    "description": (
        "Generate a song from a style/mood prompt and lyrics. "
        "Supports structure tags like [Intro], [Verse], [Chorus], [Bridge], [Outro]."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Style and mood description (10-300 chars).",
            },
            "lyrics": {
                "type": "string",
                "description": "Song lyrics with optional structure tags (10-600 chars).",
            },
            "output_format": {
                "type": "string",
                "enum": ["mp3", "wav", "pcm"],
                "description": "Audio output format (default mp3).",
            },
        },
        "required": ["prompt", "lyrics"],
    },
}


def _handle_music_generate(args: Dict[str, Any], **kw: Any) -> str:
    prompt = args.get("prompt", "")
    lyrics = args.get("lyrics", "")
    if not prompt or not lyrics:
        return json.dumps({"success": False,
                           "error": "both prompt and lyrics are required"})
    return music_generate_tool(
        prompt=prompt,
        lyrics=lyrics,
        output_format=args.get("output_format", "mp3"),
    )


registry.register(
    name="music_generate",
    toolset="music_gen",
    schema=MUSIC_GENERATE_SCHEMA,
    handler=_handle_music_generate,
    check_fn=check_music_generation_requirements,
    emoji="\U0001f3b5",
)
