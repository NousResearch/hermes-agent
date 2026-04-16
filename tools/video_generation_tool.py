"""Video generation tool — generates short video clips via the native provider."""

from __future__ import annotations

import json
import logging
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

_VIDEO_MODEL = "MiniMax-Hailuo-2.3"
_VIDEO_VALID_RESOLUTIONS = {"768P", "1080P"}
_VIDEO_VALID_DURATIONS = {6, 10}
_VIDEO_POLL_SECONDS = 15
_VIDEO_POLL_MAX = 40  # 40 × 15s = 10 min ceiling


# ─── Helpers ──────────────────────────────────────────────────────────────


def _post_json(url: str, payload: Dict[str, Any], key: str,
               *, timeout: int = 120) -> Dict[str, Any]:
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


def _get_json(url: str, key: str, *, timeout: int = 60) -> Dict[str, Any]:
    req = urllib.request.Request(
        url, method="GET",
        headers={"Authorization": f"Bearer {key}"},
    )
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _download(url: str, output_path: Path, *, timeout: int = 300) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(url, timeout=timeout, context=ctx) as resp:
        with open(output_path, "wb") as fh:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                fh.write(chunk)


def _video_cache_dir() -> Path:
    try:
        from hermes_constants import get_hermes_dir
        return Path(get_hermes_dir("cache/videos", "video_cache"))
    except Exception:
        return Path.home() / ".hermes" / "video_cache"


def _coerce_image_source(src: str) -> Optional[str]:
    """Normalise a local path / URL / data-URI for first_frame_image."""
    import base64
    import mimetypes

    src = (src or "").strip()
    if not src:
        return None
    if src.startswith(("data:", "http://", "https://")):
        return src
    if src.startswith("file://"):
        src = src[len("file://"):]
    path = Path(src).expanduser()
    if not path.is_file():
        return None
    mime, _ = mimetypes.guess_type(str(path))
    if not mime or not mime.startswith("image/"):
        return None
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


# ─── Core ─────────────────────────────────────────────────────────────────


def video_generate_tool(
    prompt: str,
    duration: Optional[int] = None,
    resolution: Optional[str] = None,
    first_frame_image: Optional[str] = None,
) -> str:
    try:
        from hermes_cli.provider_native_tools import native_api_url
    except Exception:
        return json.dumps({"success": False,
                           "error": "no video backend available"})

    from hermes_cli.provider_native_tools import native_credential
    url = native_api_url("/v1/video_generation")
    key = native_credential()
    if not url or not key:
        return json.dumps({"success": False,
                           "error": "no video backend configured"})

    payload: Dict[str, Any] = {
        "model": _VIDEO_MODEL,
        "prompt": (prompt or "").strip(),
    }
    if duration in _VIDEO_VALID_DURATIONS:
        payload["duration"] = duration
    if resolution and resolution.upper() in _VIDEO_VALID_RESOLUTIONS:
        payload["resolution"] = resolution.upper()
    if first_frame_image:
        coerced = _coerce_image_source(first_frame_image)
        if coerced:
            payload["first_frame_image"] = coerced

    # 1. Submit task.
    try:
        body = _post_json(url, payload, key)
    except urllib.error.HTTPError as exc:
        return json.dumps({"success": False,
                           "error": f"video API HTTP {exc.code}"})
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        return json.dumps({"success": False, "error": str(exc)})

    base = body.get("base_resp") or {}
    if isinstance(base, dict) and base.get("status_code") not in (0, None):
        return json.dumps({"success": False,
                           "error": f"video API error {base.get('status_code')}: "
                                    f"{base.get('status_msg', '')}"})

    task_id = body.get("task_id")
    if not task_id:
        return json.dumps({"success": False,
                           "error": "video API returned no task_id"})

    # 2. Poll until terminal status.
    api_root = url.rsplit("/v1/", 1)[0]
    file_id: Optional[str] = None
    last_status = ""
    for _ in range(_VIDEO_POLL_MAX):
        time.sleep(_VIDEO_POLL_SECONDS)
        try:
            status_body = _get_json(
                f"{api_root}/v1/query/video_generation?task_id={task_id}", key)
        except Exception:
            continue
        last_status = str(status_body.get("status") or "")
        if last_status == "Success":
            file_id = status_body.get("file_id")
            break
        if last_status == "Fail":
            return json.dumps({"success": False,
                               "error": f"video generation failed (task {task_id})",
                               "task_id": task_id})

    if not file_id:
        return json.dumps({"success": False,
                           "error": f"video generation timed out after "
                                    f"{_VIDEO_POLL_SECONDS * _VIDEO_POLL_MAX}s "
                                    f"(last status: {last_status or 'unknown'})",
                           "task_id": task_id})

    # 3. Resolve download URL.
    try:
        file_body = _get_json(
            f"{api_root}/v1/files/retrieve?file_id={file_id}", key)
    except Exception as exc:
        return json.dumps({"success": False,
                           "error": f"file retrieve failed: {exc}"})
    download_url = (file_body.get("file") or {}).get("download_url")
    if not download_url:
        return json.dumps({"success": False,
                           "error": f"no download_url for file_id {file_id}"})

    # 4. Save locally.
    out_dir = _video_cache_dir()
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"video_{ts}.mp4"
    try:
        _download(download_url, path)
    except Exception as exc:
        return json.dumps({"success": False,
                           "error": f"video download failed: {exc}",
                           "url": download_url, "task_id": task_id})

    return json.dumps({
        "success": True,
        "model": _VIDEO_MODEL,
        "task_id": task_id,
        "path": str(path),
        "url": download_url,
    }, ensure_ascii=False)


# ─── Check & registration ─────────────────────────────────────────────────


def check_video_generation_requirements() -> bool:
    try:
        from hermes_cli.provider_native_tools import provider_has_native_tool, _safe_load_config
        return provider_has_native_tool("video_gen", _safe_load_config())
    except Exception:
        return False


VIDEO_GENERATE_SCHEMA = {
    "name": "video_generate",
    "description": (
        "Generate a short video clip from a text description. "
        "Optionally provide a first-frame image for image-to-video."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Scene description for the video.",
            },
            "duration": {
                "type": "integer",
                "enum": [6, 10],
                "description": "Video length in seconds (default 6).",
            },
            "resolution": {
                "type": "string",
                "enum": ["768P", "1080P"],
                "description": "Output resolution (default 768P).",
            },
            "first_frame_image": {
                "type": "string",
                "description": "Local path, URL, or data URI of the first frame image.",
            },
        },
        "required": ["prompt"],
    },
}


def _handle_video_generate(args: Dict[str, Any], **kw: Any) -> str:
    prompt = args.get("prompt", "")
    if not prompt or not isinstance(prompt, str):
        return json.dumps({"success": False, "error": "prompt is required"})
    return video_generate_tool(
        prompt=prompt,
        duration=args.get("duration"),
        resolution=args.get("resolution"),
        first_frame_image=args.get("first_frame_image"),
    )


registry.register(
    name="video_generate",
    toolset="video_gen",
    schema=VIDEO_GENERATE_SCHEMA,
    handler=_handle_video_generate,
    check_fn=check_video_generation_requirements,
    emoji="\U0001f3ac",
)
