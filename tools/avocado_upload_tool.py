"""Avocado image-upload bridge tool.

AVOCADO FORK: chat platforms (Telegram, etc.) cache inbound photos as local
files (e.g. ``<HERMES_HOME>/cache/images/img_ab12.jpg``) and the agent only
ever sees that local path. The Avocado MCP's ``edit_image`` and
``generate_video`` (image-to-video) tools need the image inside the user's
Avocado account: ``prepare_image_upload`` returns a signed PUT URL plus a
``file_id`` and expects *someone* to upload the bytes. On Claude Desktop
that someone is the human (or a curl one-liner); fleet "Super Agent"
profiles have no terminal toolset, so nothing could ever perform the PUT —
user-attached photos were a dead end for editing / image-to-video.

``avocado_upload_image`` closes the gap in one tool call:

    1. validate the local cached image (cache-dir allowlist + magic bytes)
    2. call the registered MCP ``prepare_image_upload`` tool — routed
       through the normal MCP handler, so per-user header overrides
       (x-avocado-mcp-key multiplexing) and circuit breakers apply
    3. HTTP PUT the bytes to the signed URL
    4. return the ``file_id`` for the downstream edit/video call
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from hermes_constants import get_hermes_dir, get_hermes_home
from tools.registry import registry

logger = logging.getLogger(__name__)

# prepare_image_upload accepts png/jpeg/webp only. Detect by magic bytes —
# the cached file's extension can lie (Telegram photos default to .jpg).
_MAGIC_MIME: Tuple[Tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
)

# Pre-flight cap before we even ask for a signed URL; the real limit comes
# back from prepare_image_upload (maxSizeMb, currently 12).
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024

_PREPARE_SUFFIX = "_prepare_image_upload"


def _detect_image_mime(data: bytes) -> Optional[str]:
    for magic, mime in _MAGIC_MIME:
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _allowed_media_roots() -> list:
    """Media cache directories the agent may upload from.

    Deliberately NOT the whole HERMES_HOME — that holds config/.env with
    credentials, and this tool pushes bytes to external storage. The magic
    -byte check is the second line of defense.
    """
    roots = {
        get_hermes_home() / "cache",
        get_hermes_dir("cache/images", "image_cache"),
        get_hermes_dir("cache/vision", "temp_vision_images"),
        get_hermes_dir("cache/videos", "video_cache"),
        get_hermes_dir("cache/documents", "document_cache"),
    }
    resolved = []
    for root in roots:
        try:
            resolved.append(root.resolve())
        except OSError:
            continue
    return resolved


def _path_is_allowed(path: Path) -> bool:
    for root in _allowed_media_roots():
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _find_prepare_entry():
    """Locate the registered MCP prepare_image_upload tool.

    Prefers the canonical ``avocado`` server name but falls back to any
    MCP server exposing the tool (profiles can name the server anything).
    """
    entry = registry.get_entry(f"mcp_avocado{_PREPARE_SUFFIX}")
    if entry is not None:
        return entry
    for name, toolset in registry.get_tool_to_toolset_map().items():
        if toolset.startswith("mcp-") and name.endswith(_PREPARE_SUFFIX):
            return registry.get_entry(name)
    return None


def _extract_upload_grant(raw: str) -> Tuple[Optional[dict], Optional[str]]:
    """Pull {uploadUrl, fileId, maxSizeMb} out of the MCP handler result.

    The MCP handler wraps tool output as ``{"result": <text-or-obj>}`` and
    may add ``structuredContent``; the avocado server returns its payload
    as a JSON text block. Try structured paths first, regex as last resort.
    """
    candidates = []
    try:
        outer = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        outer = None
    if isinstance(outer, dict):
        if outer.get("error"):
            return None, str(outer["error"])
        for key in ("structuredContent", "result"):
            val = outer.get(key)
            if isinstance(val, dict):
                candidates.append(val)
            elif isinstance(val, str):
                try:
                    parsed = json.loads(val)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    candidates.append(parsed)
    for cand in candidates:
        if cand.get("uploadUrl") and cand.get("fileId"):
            return cand, None

    url_m = re.search(r'"uploadUrl"\s*:\s*"([^"]+)"', raw or "")
    id_m = re.search(r'"fileId"\s*:\s*"([^"]+)"', raw or "")
    if url_m and id_m:
        size_m = re.search(r'"maxSizeMb"\s*:\s*([0-9.]+)', raw)
        return {
            "uploadUrl": url_m.group(1),
            "fileId": id_m.group(1),
            "maxSizeMb": float(size_m.group(1)) if size_m else None,
        }, None
    return None, (
        "prepare_image_upload did not return an uploadUrl/fileId "
        f"(raw response: {(raw or '')[:300]})"
    )


def _tool_error(message: str) -> str:
    return json.dumps({"success": False, "error": message}, ensure_ascii=False)


def _handle_avocado_upload_image(args: Dict[str, Any], **kwargs: Any) -> str:
    file_path = str(args.get("file_path") or "").strip()
    purpose = str(args.get("purpose") or "edit").strip().lower()
    if purpose not in ("edit", "video"):
        purpose = "edit"
    if not file_path:
        return _tool_error("file_path is required")

    prepare_entry = _find_prepare_entry()
    if prepare_entry is None:
        return _tool_error(
            "No Avocado MCP server with a prepare_image_upload tool is "
            "connected — image upload to the user's Avocado account is "
            "unavailable."
        )

    try:
        path = Path(file_path).expanduser().resolve()
    except OSError as exc:
        return _tool_error(f"Invalid file_path: {exc}")
    if not path.is_file():
        return _tool_error(f"File not found: {file_path}")
    if not _path_is_allowed(path):
        return _tool_error(
            "file_path must point inside the media cache (the path given "
            "in the '[User sent an image: …]' note or returned by "
            "image_generate). Other locations are not uploadable."
        )
    size = path.stat().st_size
    if size > _MAX_UPLOAD_BYTES:
        return _tool_error(
            f"Image is {size / (1024 * 1024):.1f} MB — too large to upload."
        )

    data = path.read_bytes()
    mime = _detect_image_mime(data)
    if mime is None:
        return _tool_error(
            "Unsupported image format — only PNG, JPEG and WEBP can be "
            "uploaded for editing/video."
        )

    raw = prepare_entry.handler({"mime_type": mime, "purpose": purpose})
    grant, err = _extract_upload_grant(raw)
    if err or not grant:
        return _tool_error(f"prepare_image_upload failed: {err}")

    max_mb = grant.get("maxSizeMb")
    if isinstance(max_mb, (int, float)) and max_mb > 0 and size > max_mb * 1024 * 1024:
        return _tool_error(
            f"Image is {size / (1024 * 1024):.1f} MB but the upload limit "
            f"is {max_mb} MB."
        )

    import httpx

    try:
        resp = httpx.put(
            grant["uploadUrl"],
            content=data,
            headers={"Content-Type": mime},
            timeout=120.0,
            follow_redirects=False,
        )
    except httpx.HTTPError as exc:
        return _tool_error(f"Upload PUT failed: {exc}")
    if resp.status_code not in (200, 201):
        return _tool_error(
            f"Upload rejected (HTTP {resp.status_code}): {resp.text[:300]}"
        )

    file_id = grant["fileId"]
    prefix = prepare_entry.name[: -len(_PREPARE_SUFFIX)]
    logger.info(
        "avocado_upload_image: uploaded %s (%d bytes, %s) as %s",
        path.name, size, mime, file_id,
    )
    return json.dumps({
        "success": True,
        "file_id": file_id,
        "mime_type": mime,
        "size_bytes": size,
        "purpose": purpose,
        "next_step": (
            f"Image uploaded. Now call {prefix}_edit_image (to edit) or "
            f"{prefix}_generate_video (image-to-video) with "
            f"file_id='{file_id}'. Do NOT pass the local file path or "
            "base64 to those tools."
        ),
    }, ensure_ascii=False)


AVOCADO_UPLOAD_IMAGE_SCHEMA = {
    "name": "avocado_upload_image",
    "description": (
        "Upload a locally cached image to the user's Avocado AI account and "
        "get back a file_id. REQUIRED first step whenever the user attached "
        "a photo in chat (you saw '[User sent an image: /…/img_xxx.jpg]') "
        "and wants it edited or turned into a video: call this with that "
        "local path, then pass the returned file_id to the avocado "
        "edit_image or generate_video tool. Those tools cannot read local "
        "paths or base64. Also works for images you generated locally. "
        "Requires the Avocado MCP server to be connected."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": (
                    "Local path of the cached image — use the exact path "
                    "from the '[User sent an image: …]' note or from an "
                    "image_generate result."
                ),
            },
            "purpose": {
                "type": "string",
                "enum": ["edit", "video"],
                "description": (
                    "What the image will be used for next: 'edit' (default) "
                    "for edit_image, 'video' for image-to-video generation."
                ),
            },
        },
        "required": ["file_path"],
    },
}

# No check_fn on purpose: the first check_fn registered for a toolset
# becomes the toolset-level availability probe, and this module imports
# before image_generation_tool (alphabetical discovery) — gating all of
# image_gen on the Avocado MCP would be wrong. The handler fails with a
# clear message when no avocado server is connected instead.
registry.register(
    name="avocado_upload_image",
    toolset="image_gen",
    schema=AVOCADO_UPLOAD_IMAGE_SCHEMA,
    handler=_handle_avocado_upload_image,
    emoji="📤",
)
