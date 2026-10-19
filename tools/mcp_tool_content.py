"""Rendering of MCP tool-result content blocks into model-facing text: size
capping, _meta filtering, image/audio caching to MEDIA tags, resource links and
embedded resources."""

import base64
import json
import logging
import mimetypes
from typing import Any, Dict, List, Optional, Tuple
from tools.ansi_strip import strip_unicode_tags
from tools.mcp_tool_common import mcp_field
from tools.mcp_tool_schema import mcp_prefixed_tool_name
from tools.tool_output_truncate import truncate_head_tail

logger = logging.getLogger("tools.mcp_tool")


# Hard ceiling for one MCP text payload (chars), deliberately far ABOVE the budget layer's 50K
# spillover threshold so ordinary large results reach spillover intact; only floods are lossy.
# This is the FIRST line of defense against a buggy or malicious MCP server returning multi-megabyte text:
# without it the full payload is allocated, JSON-encoded and handed downstream before the budget/spillover
# layer ever sees it (#56059). Distilled from #56060 (Stoltemberg), #56072 (AlexFucuson9) and #56511
# (Tranquil-Flow), which capped at get_max_bytes() (50K) — correct protection, but at that level it would
# truncate before spillover could preserve the data. The 40% head / 60% tail split is #56511's shape.
_MCP_HARD_RESULT_CAP_CHARS = 2_000_000
# Cap on decoded resource bytes per block (a misbehaving server can't fill the cache disk).
# Base64 expands ~4/3; oversized payloads are rejected BEFORE decoding (never doubled in memory).
_MCP_RESOURCE_MAX_BYTES = 50 * 1024 * 1024
_MCP_RESOURCE_MAX_B64_CHARS = _MCP_RESOURCE_MAX_BYTES * 4 // 3 + 4


def _truncate_mcp_text_result(text: str, max_chars: int = _MCP_HARD_RESULT_CAP_CHARS) -> str:
    """Bound pathological MCP text before it propagates (#56059)."""
    return truncate_head_tail(text, max_chars, label="MCP RESULT")


def _is_reserved_mcp_meta_key(key: str) -> bool:
    """True if an MCP ``_meta`` key uses a protocol-reserved prefix: a ``modelcontextprotocol``
    or ``mcp`` label followed by at least one more label. A trailing one
    (``com.example.mcp/...``) is a vendor namespace.

    Ported from MoonshotAI/kimi-code#2600.
    """
    slash = key.find("/")
    if slash <= 0:
        return False
    labels = key[:slash].split(".")
    return any(label in ("modelcontextprotocol", "mcp") and i < len(labels) - 1 for i, label in enumerate(labels))


def _strip_reserved_meta_keys(meta) -> Optional[dict[str, Any]]:
    """Drop protocol-reserved keys from ``_meta``; None if nothing model-facing remains or the
    input wasn't a mapping."""
    if not isinstance(meta, dict):
        return None
    out = {k: v for k, v in meta.items() if isinstance(k, str) and (not _is_reserved_mcp_meta_key(k))}
    return out or None


def _base_mime(mime_type) -> str:
    """``type/subtype`` of a MIME string, lower-cased, parameters dropped."""
    return str(mime_type or "").split(";", 1)[0].strip().lower()


def _mcp_image_extension_for_mime_type(mime_type: str) -> str:
    """File extension for an MCP image MIME type (``.png`` fallback)."""
    normalized = _base_mime(mime_type)
    if normalized in {"image/jpeg", "image/jpg"}:
        return ".jpg"
    return mimetypes.guess_extension(normalized) or ".png"


def _decode_block_b64(data, what: str, label: str, *, cap_what: Optional[str] = None,
                      cap_suffix: str = "", decode_fail: str = "") -> tuple[Optional[bytes], str]:
    """Base64-decode one block payload: ``(bytes, "")`` or ``(None, inline_marker)``. With
    ``cap_what`` the payload is rejected on b64 length BEFORE decoding and on decoded size
    after. Decode failures warn and return ``decode_fail`` ("" = drop the block)."""
    if cap_what and len(data) > _MCP_RESOURCE_MAX_B64_CHARS:
        return None, f"[MCP {cap_what} too large to cache: ~{len(data) * 3 // 4} bytes{cap_suffix}]"
    try:
        raw_bytes = base64.b64decode(data)
    except (TypeError, ValueError) as exc:
        logger.warning("MCP %s decode failed (%s): %s", what, label, exc)
        return None, decode_fail
    if cap_what and len(raw_bytes) > _MCP_RESOURCE_MAX_BYTES:
        return None, f"[MCP {cap_what} too large to cache: {len(raw_bytes)} bytes{cap_suffix}]"
    return raw_bytes, ""


def _write_block_cache(writer: str, what: str, skip_label: str, *args,
                       unavailable: str = "", failed: str = "", **kwargs) -> tuple[Optional[str], str]:
    """Call ``gateway.platforms.base.<writer>(*args, **kwargs)``: ``(path, "")`` or ``(None,
    marker)``. Fail-open so one bad block never kills the tool result: gateway deps missing
    (cron without gateway) → ``unavailable``; any other cache error → warning + ``failed``."""
    try:
        import gateway.platforms.base as _base
        return getattr(_base, writer)(*args, **kwargs), ""
    except ImportError:
        logger.debug("MCP %s caching skipped — gateway.platforms.base unavailable", skip_label)
        return None, unavailable
    except Exception as exc:
        logger.warning("MCP %s cache failed: %s", what, exc)
        return None, failed


_WAV_MIME_EXT = {"audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav"}


def _cache_mcp_media_block(block, kind: str, writer: str, ext_for, *, cap_what: Optional[str] = None) -> str:
    """Cache an image/audio block and return a ``MEDIA:<path>`` tag. "" (logging, not raising)
    when the block isn't ``kind`` media, the base64 is malformed, or the cache rejects the
    bytes: the caller falls through to any text blocks."""
    data = getattr(block, "data", None)
    mime = _base_mime(mcp_field(block, "mime_type", "mimeType"))
    if data is None or not mime.startswith(f"{kind}/"):
        return ""
    raw_bytes, err = _decode_block_b64(data, f"{kind} block", mime, cap_what=cap_what)
    if raw_bytes is None:
        return err
    path, err = _write_block_cache(writer, f"{kind} block", kind, raw_bytes, ext=ext_for(mime))
    return err if path is None else f"MEDIA:{path}"


def _cache_mcp_image_block(block) -> str:
    """Cache an ``ImageContent`` block and return a ``MEDIA:<path>`` tag ("" on any failure)."""
    return _cache_mcp_media_block(block, "image", "cache_image_from_bytes", _mcp_image_extension_for_mime_type)


_MCP_NATIVE_IMAGE_MAX = 4  # images attached natively from one tool result; the rest stay MEDIA: paths
_MCP_NATIVE_IMAGE_CANDIDATES = 16  # images prepared at most per result: skips refill the 4 slots, within a bound


def _mcp_native_image_part(path: str) -> Optional[tuple[dict[str, Any], Optional[str]]]:
    """``(image_url part, scale note)`` for one cached MCP image, sized like every other native embed (the
    result is re-sent each later turn: ``vision.embed_target_bytes``, 1568 px long edge, JPEG) and normalized
    to a provider-accepted format (BMP and friends → PNG). None when the file cannot be embedded safely. The
    note maps embedded coordinates back to the original image (a screenshot's pixels are the screen's)."""
    from pathlib import Path
    from PIL import Image
    from tools.vision_tools import (_EMBED_MAX_DIMENSION, _MAX_BASE64_BYTES, _build_scale_note,
                                    _resize_image_for_vision)
    from tools.vision_tools_history_budget import resolve_embed_target_bytes
    from tools.vision_tools_image_prep import (_detect_image_mime_type_from_bytes, _normalize_to_supported_image,
                                               _validate_raster_image_decodable)
    src = Path(path)
    mime = _detect_image_mime_type_from_bytes(src.read_bytes())
    if not mime:
        return None
    normalized, mime, err = _normalize_to_supported_image(src, mime)
    if err or normalized is None:
        return None
    scale: dict[str, int] = {}
    target = min(resolve_embed_target_bytes(), _MAX_BASE64_BYTES)
    try:
        # A valid header over a truncated pixel stream passes the sniff and the cache; one undecodable
        # part makes the provider reject the whole request, so decode every frame first (as vision_analyze).
        if _validate_raster_image_decodable(normalized):
            return None
        with Image.open(normalized) as image:
            dims = image.size
        url = _resize_image_for_vision(normalized, mime_type=mime, max_base64_bytes=target,
                                       max_dimension=_EMBED_MAX_DIMENSION, force_jpeg=True, scale_out=scale)
    finally:
        if normalized != src:
            normalized.unlink(missing_ok=True)
    # The resizer is best-effort on both caps (a 64 px short-edge floor keeps a 60000x64 strip at 60000 px; the
    # quality ladder can bottom out above a low vision.embed_target_bytes). The part is re-sent every later turn,
    # so attach only what fits both; the MEDIA: path still carries the rest (vision_analyze can read it).
    if max(scale.get("new_width", dims[0]), scale.get("new_height", dims[1])) > _EMBED_MAX_DIMENSION:
        return None
    if len(url) > target:
        return None
    return {"type": "image_url", "image_url": {"url": url}}, _build_scale_note(scale or None, None)


def _mcp_result_with_native_images(text: str, image_paths: list[str]) -> Any:
    """*text* as-is, or the ``_multimodal`` envelope carrying the call's cached images when the active route
    takes images inside tool results — the same gate as ``vision_analyze`` and ``computer_use`` captures
    (``agent.image_input_mode``, an explicit ``auxiliary.vision`` backend, catalog vision, provider support).
    The text half keeps the ``MEDIA:`` paths so sharing and full-resolution reads still work. Any failure
    keeps the text: the paths already carry the images."""
    if not image_paths:
        return text
    try:
        import contextvars
        from tools.vision_tools import _should_use_native_vision_fast_path, _vision_cpu_executor
        from tools.vision_tools_history_budget import repeat_refusal
        if not _should_use_native_vision_fast_path():
            return text
    except Exception:  # deliberate boundary: the MEDIA: paths already carry the images, so any failure keeps the text
        logger.debug("MCP native image gate failed, keeping MEDIA: paths", exc_info=True)
        return text
    attached, notes = [], ""
    candidates = image_paths[:_MCP_NATIVE_IMAGE_CANDIDATES]
    # Prepare in waves sized to the free slots, so a damaged or already-in-context image hands its slot to the
    # next one instead of hiding it. Decode/resize runs on the bounded vision pool (a parallel tool batch of
    # image-heavy calls must not decode dozens of large images at once on tool threads), each job in a copy of
    # the caller's context: the active runtime (a managed local model: no WebP) and the profile's vision
    # settings are ContextVars a bare pool thread would not see.
    while candidates and len(attached) < _MCP_NATIVE_IMAGE_MAX:
        wave, candidates = candidates[:_MCP_NATIVE_IMAGE_MAX - len(attached)], candidates[_MCP_NATIVE_IMAGE_MAX - len(attached):]
        jobs = [(p, _vision_cpu_executor.submit(contextvars.copy_context().run, _mcp_native_image_part, p)) for p in wave]
        for path, job in jobs:
            try:
                ready = job.result()
            except Exception:  # one bad file keeps its MEDIA: path; the others still attach
                logger.debug("MCP native image prep failed for %s", path, exc_info=True)
                continue
            if not ready:
                continue
            part, note = ready
            # vision.max_calls_per_image: a polled screenshot tool re-sends the same pixels under a fresh cache
            # path each call, so the reservation keys on the resized data URL (identical pixels, identical key).
            if repeat_refusal(part["image_url"]["url"]):
                notes += f"\n- MEDIA:{path}: not attached; this image is already in context (vision.max_calls_per_image)."
                continue
            attached.append(part)
            if note:
                notes += f"\n- MEDIA:{path}: {note}"
    if not attached:
        if not notes:
            return text
        # Keep the result valid JSON: the refusal lines ride inside the envelope, not after its closing brace.
        try:
            payload = json.loads(text)
            payload["result"] = f"{payload.get('result') or ''}{notes}"
            return json.dumps(payload, ensure_ascii=False)
        except (TypeError, ValueError, AttributeError):
            return text + notes
    # The header and the scale notes are their own short part: an oversized tool text gets spilled to a file and
    # replaced by its head, and the coordinate map must survive that next to the resized screenshot it describes.
    header = "The image(s) from this call are attached — inspect them with your native vision."
    return {"_multimodal": True,
            "content": [{"type": "text", "text": text}, {"type": "text", "text": header + notes}, *attached],
            "text_summary": text}


def _cache_mcp_audio_block(block) -> str:
    """Cache an ``AudioContent`` block and return a ``MEDIA:<path>`` tag ("" on any failure)."""
    return _cache_mcp_media_block(
        block, "audio", "cache_audio_from_bytes",
        lambda mime: _WAV_MIME_EXT.get(mime) or mimetypes.guess_extension(mime) or ".ogg",
        cap_what="audio resource")


def _mcp_resource_filename(uri: str, mime_type: str) -> str:
    """Safe display filename from the URI's last path segment, used only as a name hint:
    ``cache_document_from_bytes`` re-sanitizes and prefixes it, so remote path components
    can't steer the cache location."""
    import re as _re
    from pathlib import Path
    from urllib.parse import urlparse, unquote
    name = ""
    if uri:
        try:
            name = Path(unquote(urlparse(str(uri)).path or "")).name
        except (ValueError, TypeError):
            pass
    # Strip control chars (hostile URIs could inject newlines/ANSI into the filename and
    # transcript marker) and cap length, preserving the extension.
    name = _re.sub(r"[\x00-\x1f\x7f]", "", name).strip()
    if len(name) > 150:
        stem, dot, ext = name.rpartition(".")
        name = stem[: 150 - len(ext) - 1] + "." + ext if dot and 0 < len(ext) <= 12 else name[:150]
    if not name or name in {".", ".."}:
        ext = mimetypes.guess_extension(_base_mime(mime_type)) or ".bin"
        name = f"resource{ext}"
    return name


def _render_mcp_dropped_block_notice(block, block_type: str) -> str:
    """Inline notice for an unsupported MCP content block (kimi-code#3227): silently dropping it
    leaves the model unaware content went missing. Carries whatever handles the block exposes —
    mime type, uri, size, name — so the agent can fetch or reason about the missing content."""
    details = [f"type={block_type}"]
    mime = mcp_field(block, "mime_type", "mimeType", None)
    if mime:
        details.append(f"mimeType={mime}")
    uri = getattr(block, "uri", None) or getattr(getattr(block, "resource", None), "uri", None)
    if uri:
        details.append(f"uri={uri}")
    for size_attr in ("size", "sizeInBytes"):
        size = getattr(block, size_attr, None)
        if isinstance(size, int):
            details.append(f"size={size}")
            break
    name = getattr(block, "name", None)
    if name and isinstance(name, str):
        details.append(f"name={name}")
    return f"[MCP content dropped: unsupported block ({', '.join(details)})]"


def _render_mcp_resource_block(block, server_name: str = "") -> str:
    """Render a ``ResourceLink`` or ``EmbeddedResource`` block as text: embedded text → the
    text; embedded blob → decoded (size-capped) into the document cache with a path marker;
    link → the URI plus a pointer at the server's read_resource tool (no fetch here — links
    are only readable via the originating session). "" for non-resource blocks; failures are
    reported inline rather than silently dropped."""
    block_type = getattr(block, "type", "")
    if block_type == "resource_link" or (hasattr(block, "uri") and not hasattr(block, "resource") and block_type != "text"):
        uri = getattr(block, "uri", None)
        if not uri:
            return ""
        name = getattr(block, "name", "") or ""
        mime = mcp_field(block, "mime_type", "mimeType", "") or ""
        details = f"uri={uri}" + (f", name={name}" if name else "") + (f", mimeType={mime}" if mime else "")
        reader = mcp_prefixed_tool_name(server_name, "read_resource") if server_name else "the MCP server's read_resource tool"
        return f"[MCP resource link: {details} — fetch it with {reader}]"
    resource = getattr(block, "resource", None)
    if resource is None:
        return ""
    text = getattr(resource, "text", None)
    if text is not None:
        return strip_unicode_tags(str(text))
    blob = getattr(resource, "blob", None)
    if blob is None:
        return ""
    uri = str(getattr(resource, "uri", "") or "")
    mime = str(mcp_field(resource, "mime_type", "mimeType", "") or "")
    raw_bytes, err = _decode_block_b64(
        blob, "embedded resource", mime or uri, cap_what="embedded resource", cap_suffix=f", uri={uri}",
        decode_fail=f"[MCP embedded resource could not be decoded: {mime or uri}]")
    if raw_bytes is None:
        return err
    kind = mime or "unknown type"
    path, err = _write_block_cache(
        "cache_document_from_bytes", "embedded resource", "resource", raw_bytes, _mcp_resource_filename(uri, mime),
        unavailable=f"[MCP embedded resource received ({len(raw_bytes)} bytes, {kind}) but document cache unavailable in this process]",
        failed=f"[MCP embedded resource could not be cached: {mime or uri}]")
    if path is None:
        return err
    return f"[MCP resource saved to {path} ({kind}, {len(raw_bytes)} bytes) — read it with read_file or terminal tools]"
