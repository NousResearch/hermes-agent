"""Single resolver for every vision_analyze image source -> bytes + mime.

All source handling (data:/http(s)/file/local/container) funnels through
:func:`resolve_image_source` so size and magic-byte checks are enforced exactly
once.  Returns raw bytes (not a path): the downstream step is base64 -> data URL
(RFC 2397) and provider base64 content blocks.
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from pathlib import Path
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
    if s.startswith(("http://", "https://")):
        if not _is_safe_http(s):
            raise SourceUnsafe("blocked: unsafe or private URL", src=s)
        return _finalize(await _download_to_bytes(s), "", "http", s)

    candidate = s[len("file://"):] if s.startswith("file://") else s
    if s.startswith("file://") or _looks_like_path(candidate):
        p = Path(os.path.expanduser(candidate))
        real = _maybe_translate_container_path(p, ctx).resolve()
        if not _within_allowed_roots(real, ctx):  # SEAM: returns True in PR 1
            raise SourceUnsafe(
                f"reading '{real}' is not permitted (outside allowed image roots)", src=s)
        if real.is_file():
            return _finalize(real.read_bytes(), "", "file", s)
        # No host file (tmpfs / root-owned / container-only) -> read inside the sandbox.
        return await _resolve_container_fallback(p, ctx, s)

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


def _is_safe_http(url: str) -> bool:
    from tools.url_safety import is_safe_url
    from tools.website_policy import check_website_access

    return is_safe_url(url) and not check_website_access(url)


async def _download_to_bytes(url: str) -> bytes:
    import tempfile

    from tools.vision_tools import _download_image

    with tempfile.NamedTemporaryFile(suffix=".img", delete=False) as tf:
        tmp = Path(tf.name)
    try:
        await _download_image(url, tmp)  # enforces the 50MB stream cap + redirect SSRF guard
        return tmp.read_bytes()
    finally:
        tmp.unlink(missing_ok=True)


def _looks_like_path(s: str) -> bool:
    return s.startswith(("/", "~", "./", "../")) or (len(s) > 1 and s[1] == ":")


def _within_allowed_roots(real: Path, ctx: ResolveContext) -> bool:
    """SEAM. PR 1 is permissive (preserves today's behavior). PR 2 replaces the
    body with the readable-root allowlist + its threat model + migration note."""
    return True


def _maybe_translate_container_path(p: Path, ctx: ResolveContext) -> Path:
    # Cache-dir reverse map only. Every other container path (tmpfs /workspace,
    # docker_volumes, root-owned) returns unchanged and falls through to the
    # exec-read fallback — the universal mechanism.
    from tools.credential_files import from_agent_visible_cache_path

    return Path(from_agent_visible_cache_path(str(p)))


def _get_active_env(task_id: Optional[str]):
    if not task_id:
        return None
    try:
        from tools.terminal_tool import get_active_env

        return get_active_env(task_id)
    except Exception:
        return None


async def _resolve_container_fallback(p: Path, ctx: ResolveContext, src: str) -> ResolvedImage:
    """Host file is absent/unreadable. Read the bytes inside the container.

    The agent can already ``cat`` any container file (file_operations.py reads
    root-owned mode-600 files this way); this reads a strict subset bounded by
    the same sandbox, so it introduces no new exposure. ``base64 -w0`` is
    GNU-only, so pipe through ``tr -d`` for BusyBox/Alpine.
    """
    import shlex

    env = _get_active_env(ctx.task_id)
    if env is None:
        raise SourceNotFound(
            f"'{p}' is not reachable on the host and no active sandbox is available "
            f"to read it", src=src, origin="container")

    res = env.execute(f"base64 {shlex.quote(str(p))} | tr -d '\\n'")
    if res.get("returncode", 1) != 0:
        raise SourceNotFound(f"could not read '{p}' inside the sandbox", src=src, origin="container")
    try:
        data = base64.b64decode(res.get("output", ""), validate=True)
    except Exception as exc:
        raise NotAnImage(f"sandbox returned non-image data for '{p}': {exc}", src=src)
    return _finalize(data, "", "container", src)


def _finalize(data: bytes, declared_mime: str, origin: str, src: str) -> ResolvedImage:
    """Intrinsic-correctness chokepoint: hard byte cap + magic-byte sniff."""
    from tools.vision_tools import _detect_image_mime_type_from_bytes

    if len(data) > _MAX_BYTES:
        raise SourceTooLarge("image exceeds size limit", src=src, origin=origin)
    sniffed = _detect_image_mime_type_from_bytes(data)
    if sniffed is None:
        raise NotAnImage("source is not a recognized image", src=src, origin=origin)
    return ResolvedImage(data=data, mime=sniffed, origin=origin)
