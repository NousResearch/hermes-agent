"""Rendering of MCP tool-result content blocks into model-facing text: size
capping, _meta filtering, image/audio caching to MEDIA tags, resource links and
embedded resources."""

import base64
import logging
import mimetypes
from typing import Any, Dict, Optional, Tuple
from tools.ansi_strip import strip_unicode_tags
from tools.mcp_tool_common import mcp_field
from tools.mcp_tool_schema import mcp_prefixed_tool_name

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
    """Pass text at or under ``max_chars`` unchanged; otherwise keep a 40% head / 60% tail
    split with an omission notice between.

    Bound pathological MCP text before it propagates (#56059).
    """
    if len(text) <= max_chars:
        return text
    head_chars = int(max_chars * 0.4)
    tail_chars = max_chars - head_chars
    omitted = len(text) - head_chars - tail_chars
    return (text[:head_chars] + f"\n\n... [MCP RESULT TRUNCATED - {omitted:,} chars omitted "
            f"out of {len(text):,} total] ...\n\n" + text[-tail_chars:])


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


def _strip_reserved_meta_keys(meta) -> Optional[Dict[str, Any]]:
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
                      cap_suffix: str = "", decode_fail: str = "") -> Tuple[Optional[bytes], str]:
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
                       unavailable: str = "", failed: str = "", **kwargs) -> Tuple[Optional[str], str]:
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
