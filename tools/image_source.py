"""Single resolver for every vision_analyze image source -> bytes + mime.

All source handling (data:/http(s)/file/local/container) funnels through
:func:`resolve_image_source` so size and magic-byte checks are enforced exactly
once.  Returns raw bytes (not a path): the downstream step is base64 -> data URL
(RFC 2397) and provider base64 content blocks.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any, Optional

# Decoded payload cap, mirroring tools/vision_tools._MAX_BASE64_BYTES
# (Gemini inline-data limit, the most restrictive major provider).
_MAX_BYTES = 20 * 1024 * 1024


class ImageResolutionError(Exception):
    def __init__(self, message: str, *, src: str = "", origin: str = ""):
        super().__init__(message)
        self.src, self.origin = src, origin


class UnsupportedScheme(ImageResolutionError):
    pass


class SourceUnsafe(ImageResolutionError):  # SSRF / path-allowlist
    pass


class SourceTooLarge(ImageResolutionError):
    pass


class SourceNotFound(ImageResolutionError):
    pass


class NotAnImage(ImageResolutionError):
    pass


@dataclass
class ResolveContext:
    task_id: Optional[str] = None
    cfg: Optional[dict[str, Any]] = None
    extra_roots: tuple = field(default_factory=tuple)


@dataclass
class ResolvedImage:
    data: bytes
    mime: str
    origin: str  # one of: data | http | file | local | container


async def resolve_image_source(src: str, ctx: ResolveContext) -> ResolvedImage:
    if not isinstance(src, str) or not src.strip():
        raise SourceNotFound("image_url is required", src=str(src))
    s = src.strip()
    if s.startswith("data:"):
        data, mime = _resolve_data_url(s)
        return _finalize(data, mime, "data", s)
    # http / file / local / container branches added in Tasks 5-7.
    raise UnsupportedScheme(
        "Unrecognized image source. Use an http(s) URL, a local file path, "
        "a file:// URI, or a data: URL.",
        src=s,
    )


def _resolve_data_url(s: str) -> tuple[bytes, str]:
    header, _, payload = s.partition(",")
    if ";base64" not in header:
        raise NotAnImage("data: URL must be base64-encoded", src=s[:64])
    declared = header[len("data:"):].split(";", 1)[0].strip() or "application/octet-stream"
    # Cheap pre-decode size gate on the encoded length (~4/3 expansion).
    if (len(payload) * 3) // 4 > _MAX_BYTES:
        raise SourceTooLarge("data: URL exceeds size limit", src=s[:64])
    try:
        data = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise NotAnImage(f"invalid base64 in data: URL: {exc}", src=s[:64])
    return data, declared  # real mime verified in _finalize via magic bytes


def _finalize(data: bytes, declared_mime: str, origin: str, src: str) -> ResolvedImage:
    """Intrinsic-correctness chokepoint: hard byte cap + magic-byte sniff."""
    from tools.vision_tools import _detect_image_mime_type_from_bytes

    if len(data) > _MAX_BYTES:
        raise SourceTooLarge("image exceeds size limit", src=src, origin=origin)
    sniffed = _detect_image_mime_type_from_bytes(data)
    if sniffed is None:
        raise NotAnImage("source is not a recognized image", src=src, origin=origin)
    return ResolvedImage(data=data, mime=sniffed, origin=origin)
