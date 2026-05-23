#!/usr/bin/env python3
"""Voicebox local sidecar connector.

Voicebox (jamiepine/voicebox) exposes a local REST API and Streamable HTTP MCP
server while the desktop app is running. This module keeps Hermes integration
thin and optional: Hermes never vendors Voicebox's ML runtime; it only talks to
``127.0.0.1`` when the user selected/started Voicebox.
"""

from __future__ import annotations

import json
import mimetypes
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error as urlerror
from urllib import parse, request

DEFAULT_VOICEBOX_BASE_URL = "http://127.0.0.1:17493"
DEFAULT_VOICEBOX_CLIENT_ID = "hermes-agent"
DEFAULT_VOICEBOX_TIMEOUT = 30.0
DEFAULT_VOICEBOX_POLL_TIMEOUT = 180.0
DEFAULT_VOICEBOX_POLL_INTERVAL = 1.0
DEFAULT_VOICEBOX_LANGUAGE = "en"


def _load_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def load_voicebox_config(tts_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return merged Voicebox settings from config.

    Canonical location is ``tts.voicebox`` because Voicebox can act as a TTS
    provider. A top-level ``voicebox`` block is also accepted for future
    non-TTS tools. Values under ``tts.voicebox`` win.
    """

    root = _load_config()
    raw_top = root.get("voicebox")
    top: Dict[str, Any] = raw_top if isinstance(raw_top, dict) else {}
    tts = tts_config if isinstance(tts_config, dict) else root.get("tts", {})
    raw_nested = tts.get("voicebox") if isinstance(tts, dict) else None
    nested: Dict[str, Any] = raw_nested if isinstance(raw_nested, dict) else {}
    merged: Dict[str, Any] = {**top, **nested}
    return merged


def _base_url(config: Optional[Dict[str, Any]] = None) -> str:
    cfg = config or load_voicebox_config()
    return str(cfg.get("base_url") or DEFAULT_VOICEBOX_BASE_URL).rstrip("/")


def _client_id(config: Optional[Dict[str, Any]] = None) -> str:
    cfg = config or load_voicebox_config()
    return str(cfg.get("client_id") or DEFAULT_VOICEBOX_CLIENT_ID).strip() or DEFAULT_VOICEBOX_CLIENT_ID


def _timeout(config: Optional[Dict[str, Any]] = None) -> float:
    cfg = config or load_voicebox_config()
    try:
        value = float(cfg.get("timeout", DEFAULT_VOICEBOX_TIMEOUT))
    except (TypeError, ValueError):
        return DEFAULT_VOICEBOX_TIMEOUT
    return value if value > 0 else DEFAULT_VOICEBOX_TIMEOUT


def _poll_timeout(config: Optional[Dict[str, Any]] = None) -> float:
    cfg = config or load_voicebox_config()
    try:
        value = float(cfg.get("poll_timeout", DEFAULT_VOICEBOX_POLL_TIMEOUT))
    except (TypeError, ValueError):
        return DEFAULT_VOICEBOX_POLL_TIMEOUT
    return value if value > 0 else DEFAULT_VOICEBOX_POLL_TIMEOUT


def _poll_interval(config: Optional[Dict[str, Any]] = None) -> float:
    cfg = config or load_voicebox_config()
    try:
        value = float(cfg.get("poll_interval", DEFAULT_VOICEBOX_POLL_INTERVAL))
    except (TypeError, ValueError):
        return DEFAULT_VOICEBOX_POLL_INTERVAL
    return max(0.1, value)


def _headers(config: Optional[Dict[str, Any]] = None, *, json_body: bool = False) -> Dict[str, str]:
    headers = {"X-Voicebox-Client-Id": _client_id(config)}
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers


def _url(base_url: str, path: str) -> str:
    return parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def _decode_error(exc: Exception) -> str:
    if isinstance(exc, urlerror.HTTPError):
        try:
            raw = exc.read().decode("utf-8", errors="replace")
            data = json.loads(raw)
            detail = data.get("detail") or data.get("error") or raw
        except Exception:
            detail = getattr(exc, "reason", str(exc))
        return f"HTTP {exc.code}: {detail}"
    if isinstance(exc, urlerror.URLError):
        return f"Connection failed: {exc.reason}"
    return str(exc)


def _http_json(
    method: str,
    path: str,
    *,
    payload: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    cfg = config or load_voicebox_config()
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        _url(_base_url(cfg), path),
        data=data,
        method=method.upper(),
        headers=_headers(cfg, json_body=payload is not None),
    )
    with request.urlopen(req, timeout=timeout or _timeout(cfg)) as resp:  # nosec B310 - localhost sidecar URL is user-configured
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body) if body else {}


def _http_bytes(path: str, *, config: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> bytes:
    cfg = config or load_voicebox_config()
    req = request.Request(_url(_base_url(cfg), path), method="GET", headers=_headers(cfg))
    with request.urlopen(req, timeout=timeout or _timeout(cfg)) as resp:  # nosec B310 - localhost sidecar URL is user-configured
        return resp.read()


def _post_multipart_file(
    path: str,
    file_path: str,
    *,
    fields: Optional[Dict[str, str]] = None,
    config: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    cfg = config or load_voicebox_config()
    boundary = f"----hermes-voicebox-{uuid.uuid4().hex}"
    p = Path(file_path).expanduser()
    content_type = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    parts: list[bytes] = []
    for key, value in (fields or {}).items():
        if value is None or value == "":
            continue
        parts.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode(),
            str(value).encode("utf-8"),
            b"\r\n",
        ])
    parts.extend([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{p.name}"\r\n'.encode(),
        f"Content-Type: {content_type}\r\n\r\n".encode(),
        p.read_bytes(),
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ])
    body = b"".join(parts)
    headers = _headers(cfg)
    headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    headers["Content-Length"] = str(len(body))
    req = request.Request(_url(_base_url(cfg), path), data=body, method="POST", headers=headers)
    with request.urlopen(req, timeout=timeout or _timeout(cfg)) as resp:  # nosec B310 - localhost sidecar URL is user-configured
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw else {}


def check_voicebox_available() -> bool:
    try:
        status = _http_json("GET", "/health", timeout=2.0)
    except Exception:
        return False
    return bool(status)


def _poll_generation(
    generation_id: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    cfg = config or load_voicebox_config()
    deadline = time.monotonic() + (timeout or _poll_timeout(cfg))
    last: Dict[str, Any] = {"id": generation_id, "status": "unknown"}
    while time.monotonic() < deadline:
        remaining = max(1.0, min(_timeout(cfg), deadline - time.monotonic()))
        req = request.Request(
            _url(_base_url(cfg), f"/generate/{generation_id}/status"),
            method="GET",
            headers=_headers(cfg),
        )
        try:
            with request.urlopen(req, timeout=remaining) as resp:  # nosec B310 - localhost sidecar URL is user-configured
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        last = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    if last.get("status") in {"completed", "failed", "not_found"}:
                        return last
        except Exception:
            # SSE can be interrupted by app restarts; retry until the deadline.
            pass
        time.sleep(_poll_interval(cfg))
    last["timeout"] = True
    return last


def _default_output_path(generation_id: str) -> Path:
    try:
        from tools.tts_tool import DEFAULT_OUTPUT_DIR

        out_dir = Path(DEFAULT_OUTPUT_DIR)
    except Exception:
        out_dir = Path.home() / ".hermes" / "cache" / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"voicebox_{generation_id}.wav"


def voicebox_speak_to_file(
    text: str,
    output_path: Optional[str] = None,
    *,
    profile: Optional[str] = None,
    engine: Optional[str] = None,
    personality: Optional[bool] = None,
    language: Optional[str] = None,
    wait: bool = True,
    tts_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Speak text through Voicebox REST and download the generated audio."""

    cfg = load_voicebox_config(tts_config)
    payload: Dict[str, Any] = {
        "text": text,
        "language": language or cfg.get("language") or DEFAULT_VOICEBOX_LANGUAGE,
    }
    for key, value in {
        "profile": profile if profile is not None else cfg.get("profile"),
        "engine": engine if engine is not None else cfg.get("engine"),
        "personality": personality if personality is not None else cfg.get("personality"),
    }.items():
        if value is not None and value != "":
            payload[key] = value

    generation = _http_json("POST", "/speak", payload=payload, config=cfg)
    generation_id = generation.get("id") or generation.get("generation_id")
    if not generation_id:
        raise ValueError(f"Voicebox did not return a generation id: {generation}")

    status = generation
    if wait:
        status = _poll_generation(str(generation_id), config=cfg)
        if status.get("status") == "failed":
            raise ValueError(f"Voicebox generation failed: {status.get('error') or status}")
        if status.get("status") == "not_found":
            raise ValueError(f"Voicebox generation not found: {generation_id}")
        if status.get("timeout"):
            raise TimeoutError(f"Voicebox generation timed out: {generation_id}")

    out = Path(output_path).expanduser() if output_path else _default_output_path(str(generation_id))
    out.parent.mkdir(parents=True, exist_ok=True)
    audio = _http_bytes(f"/audio/{generation_id}", config=cfg)
    if not audio:
        raise ValueError(f"Voicebox returned empty audio for generation {generation_id}")
    out.write_bytes(audio)
    return {
        "success": True,
        "file_path": str(out),
        "generation_id": str(generation_id),
        "generation": generation,
        "status": status,
        "provider": "voicebox",
    }


def voicebox_status_tool(include_profiles: bool = False) -> str:
    try:
        cfg = load_voicebox_config()
        status = _http_json("GET", "/health", config=cfg)
        result: Dict[str, Any] = {
            "success": True,
            "base_url": _base_url(cfg),
            "client_id": _client_id(cfg),
            "status": status,
        }
        if include_profiles:
            result["profiles"] = _http_json("GET", "/profiles", config=cfg)
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        cfg = load_voicebox_config()
        return json.dumps({
            "success": False,
            "base_url": _base_url(cfg),
            "client_id": _client_id(cfg),
            "error": _decode_error(exc),
            "setup_needed": "Start Voicebox and verify it is listening on the configured base_url.",
        }, ensure_ascii=False)


def voicebox_list_profiles_tool() -> str:
    try:
        cfg = load_voicebox_config()
        return json.dumps({
            "success": True,
            "profiles": _http_json("GET", "/profiles", config=cfg),
        }, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"success": False, "error": _decode_error(exc)}, ensure_ascii=False)


def voicebox_speak_tool(
    text: str,
    output_path: Optional[str] = None,
    profile: Optional[str] = None,
    engine: Optional[str] = None,
    personality: Optional[bool] = None,
    language: Optional[str] = None,
    wait: bool = True,
) -> str:
    if not text or not text.strip():
        return json.dumps({"success": False, "error": "Text is required"}, ensure_ascii=False)
    try:
        result = voicebox_speak_to_file(
            text,
            output_path,
            profile=profile,
            engine=engine,
            personality=personality,
            language=language,
            wait=wait,
        )
        file_path = result["file_path"]
        result["media_tag"] = f"MEDIA:{file_path}"
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"success": False, "error": _decode_error(exc)}, ensure_ascii=False)


def voicebox_transcribe_tool(
    audio_path: str,
    language: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    if not audio_path:
        return json.dumps({"success": False, "error": "audio_path is required"}, ensure_ascii=False)
    p = Path(audio_path).expanduser()
    if not p.is_file():
        return json.dumps({"success": False, "error": f"File not found: {p}"}, ensure_ascii=False)
    try:
        fields = {"language": language or ""}
        # Voicebox REST /transcribe currently accepts language. MCP also accepts
        # model; keep it optional here for forward compatibility without failing
        # older Voicebox builds that ignore unknown multipart fields.
        if model:
            fields["model"] = model
        data = _post_multipart_file("/transcribe", str(p), fields=fields)
        return json.dumps({"success": True, **data}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"success": False, "error": _decode_error(exc)}, ensure_ascii=False)


VOICEBOX_STATUS_SCHEMA = {
    "name": "voicebox_status",
    "description": "Check the local Voicebox sidecar health and configuration. Use when Voicebox voice cloning/TTS is requested.",
    "parameters": {
        "type": "object",
        "properties": {
            "include_profiles": {"type": "boolean", "description": "Also fetch available Voicebox profiles."},
        },
    },
}

VOICEBOX_LIST_PROFILES_SCHEMA = {
    "name": "voicebox_list_profiles",
    "description": "List available local Voicebox voice profiles.",
    "parameters": {"type": "object", "properties": {}},
}

VOICEBOX_SPEAK_SCHEMA = {
    "name": "voicebox_speak",
    "description": "Speak text through the local Voicebox app and download the generated audio file.",
    "parameters": {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string", "description": "Text to speak."},
            "output_path": {"type": "string", "description": "Optional local output audio path."},
            "profile": {"type": "string", "description": "Optional Voicebox profile name or id."},
            "engine": {"type": "string", "description": "Optional Voicebox engine, e.g. qwen, chatterbox, kokoro."},
            "personality": {"type": "boolean", "description": "Use the profile personality rewrite before speaking."},
            "language": {"type": "string", "description": "Language code, default en."},
            "wait": {"type": "boolean", "description": "Wait for completion before downloading audio. Defaults true."},
        },
    },
}

VOICEBOX_TRANSCRIBE_SCHEMA = {
    "name": "voicebox_transcribe",
    "description": "Transcribe a local audio file through Voicebox's local Whisper backend.",
    "parameters": {
        "type": "object",
        "required": ["audio_path"],
        "properties": {
            "audio_path": {"type": "string", "description": "Absolute or user-relative local audio path."},
            "language": {"type": "string", "description": "Optional language code."},
            "model": {"type": "string", "description": "Optional Whisper model size when supported by Voicebox."},
        },
    },
}


from tools.registry import registry

registry.register(
    name="voicebox_status",
    toolset="voicebox",
    schema=VOICEBOX_STATUS_SCHEMA,
    handler=lambda args, **kw: voicebox_status_tool(bool(args.get("include_profiles", False))),
    emoji="🎙️",
)
registry.register(
    name="voicebox_list_profiles",
    toolset="voicebox",
    schema=VOICEBOX_LIST_PROFILES_SCHEMA,
    handler=lambda args, **kw: voicebox_list_profiles_tool(),
    emoji="🎙️",
)
registry.register(
    name="voicebox_speak",
    toolset="voicebox",
    schema=VOICEBOX_SPEAK_SCHEMA,
    handler=lambda args, **kw: voicebox_speak_tool(
        text=args.get("text", ""),
        output_path=args.get("output_path"),
        profile=args.get("profile"),
        engine=args.get("engine"),
        personality=args.get("personality"),
        language=args.get("language"),
        wait=args.get("wait", True),
    ),
    emoji="🎙️",
)
registry.register(
    name="voicebox_transcribe",
    toolset="voicebox",
    schema=VOICEBOX_TRANSCRIBE_SCHEMA,
    handler=lambda args, **kw: voicebox_transcribe_tool(
        audio_path=args.get("audio_path", ""),
        language=args.get("language"),
        model=args.get("model"),
    ),
    emoji="🎙️",
)
