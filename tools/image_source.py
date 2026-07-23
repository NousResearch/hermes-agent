"""Single resolver for every vision_analyze image source -> bytes + mime.

All source handling (data:/http(s)/file/local/container) funnels through
:func:`resolve_image_source` so size and magic-byte checks are enforced exactly
once.  Returns raw bytes (not a path): the downstream step is base64 -> data URL
(RFC 2397) and provider base64 content blocks.

Security (terminal-backend confinement, GHSA-gpxw-6wxv-w3qq): under a non-local
terminal backend the file tools are confined to the sandbox (SECURITY.md 2.2),
but vision read images host-side. This resolver enforces the same boundary:

  * local backend            -> read any host path (chosen posture, unchanged)
  * non-local backend:
      path in a media cache   -> host-read (the gateway/download caches live on
                                 the host and are bind-mounted into the sandbox)
      path anywhere else      -> read the bytes *inside the sandbox* via exec-read
                                 (the agent can already ``cat`` any container file;
                                 this stays within the sandbox boundary and never
                                 reaches the host's ``/etc/passwd`` / ``~/.ssh``).

So a prompt-injected ``vision_analyze('/etc/passwd')`` under Docker reads the
*container's* file (what every other tool sees), not the host's — no escape —
while container-only images (tmpfs ``/workspace``, root-owned) are still
deliverable. This is the unified delivery + confinement model: the same
mechanism that fixes "vision can't see container files" also closes the escape.
"""
from __future__ import annotations

import asyncio
import base64
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Raw-bytes INGEST budget — what the resolver will load before handing off.
# This is deliberately the 50MB download cap (tools/vision_tools._VISION_MAX_DOWNLOAD_BYTES),
# NOT the 20MB provider payload cap. The 20MB cap (_MAX_BASE64_BYTES) is a
# *post-resize* limit enforced at the call sites: an oversized raw image must
# still reach the resizer so it can be downscaled under the payload cap. Capping
# raw bytes at 20MB here would reject every 20-50MB photo before resize can run.
_MAX_INGEST_BYTES = 50 * 1024 * 1024


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


@dataclass
class ResolvedImage:
    data: bytes
    mime: str
    origin: str  # one of: data | http | file | local | container


# Explicit URL scheme, e.g. "ftp://", "s3://". Bare Windows drive paths
# ("C:\x.png") don't match because they lack the "//".
_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")


async def resolve_image_source(src: str, ctx: ResolveContext) -> ResolvedImage:
    if not isinstance(src, str) or not src.strip():
        raise SourceNotFound("image_url is required", src=str(src))
    s = src.strip()
    if s.startswith("data:"):
        data, mime = _resolve_data_url(s)
        return _finalize(data, mime, "data", s)
    if s.startswith(("http://", "https://")):
        reason = _http_block_reason(s)
        if reason:
            raise SourceUnsafe(reason, src=s)
        return _finalize(await _download_to_bytes(s), "", "http", s)

    if _SCHEME_RE.match(s) and not s.lower().startswith("file://"):
        raise UnsupportedScheme(
            "Unrecognized image source scheme. Use an http(s) URL, a local "
            "file path, a file:// URI, or a data: URL.",
            src=s,
        )

    # Everything else is a filesystem path — including bare relative names
    # like "pic.png" (accepted on main; a path-shape gate here regressed them).
    candidate = s[len("file://"):] if s.lower().startswith("file://") else s
    p = Path(os.path.expanduser(candidate))
    # Confinement decision (see module docstring). Under a non-local backend
    # a path is host-readable ONLY if it lands in a media cache (after
    # translating a container-visible cache path back to its host mount);
    # every other path is read inside the sandbox via exec-read, so a host
    # path outside the caches never yields the host's bytes.
    host_target = _permitted_host_read_target(p, ctx)
    if host_target is not None and host_target.is_file():
        # Shared credential-read guard (agent.file_safety, #57698): refuse
        # secret-bearing files (.env, auth.json, ...) with an intentional,
        # specific error instead of relying on the magic-byte sniff to
        # reject them incidentally. Same chokepoint the image-gen/video-gen
        # provider plugins enforce on model-supplied local paths. Import is
        # best-effort (guard unavailability must not break image loading);
        # a real block always propagates.
        try:
            from agent.file_safety import raise_if_read_blocked
        except Exception:  # noqa: BLE001 — guard unavailable: proceed
            raise_if_read_blocked = None
        if raise_if_read_blocked is not None:
            try:
                raise_if_read_blocked(str(host_target))
            except ValueError as exc:
                raise SourceUnsafe(str(exc), src=s, origin="file")
        data = await asyncio.to_thread(host_target.read_bytes)
        return _finalize(data, "", "file", s)
    if _is_local_terminal_backend():
        # Local backend: any path was host-readable, so a miss simply means
        # the file doesn't exist — no sandbox to fall back to.
        raise SourceNotFound(f"image file not found: '{p}'", src=s, origin="file")
    # Not a permitted host read (or the host file is absent) -> read the
    # bytes inside the sandbox. Under a sandbox this reads the container's
    # filesystem, never the host's.
    return await _resolve_container_fallback(p, ctx, s)


def _resolve_data_url(s: str) -> tuple[bytes, str]:
    header, _, payload = s.partition(",")
    if ";base64" not in header:
        raise NotAnImage("data: URL must be base64-encoded", src=s[:64])
    declared = header[len("data:"):].split(";", 1)[0].strip() or "application/octet-stream"
    # Cheap pre-decode size gate on the encoded length (~4/3 expansion).
    if (len(payload) * 3) // 4 > _MAX_INGEST_BYTES:
        raise SourceTooLarge("data: URL exceeds size limit", src=s[:64])
    try:
        data = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise NotAnImage(f"invalid base64 in data: URL: {exc}", src=s[:64])
    return data, declared  # real mime verified in _finalize via magic bytes


def _http_block_reason(url: str) -> Optional[str]:
    """Return a human-readable block reason, or None when the URL is allowed.

    Pre-flight short-circuit: policy-blocked URLs are refused BEFORE any
    network I/O. ``_download_image`` re-checks policy internally (per attempt
    and against the final redirect target) — that second evaluation is
    intentional, not redundant: this one guarantees no bytes move for a
    blocked URL; the inner one covers redirects and non-resolver callers.
    Preserves the specific website-policy message so the agent sees *why*.
    """
    from tools.url_safety import is_safe_url
    from tools.website_policy import check_website_access

    if not is_safe_url(url):
        return "blocked: unsafe or private URL"
    blocked = check_website_access(url)
    if blocked:
        return blocked.get("message") or "blocked by website policy"
    return None


async def _download_to_bytes(url: str) -> bytes:
    import tempfile

    from tools.vision_tools import _download_image

    with tempfile.NamedTemporaryFile(suffix=".img", delete=False) as tf:
        tmp = Path(tf.name)
    try:
        # Enforces the 50MB stream cap, redirect SSRF guard, and website policy.
        await _download_image(url, tmp)
        return await asyncio.to_thread(tmp.read_bytes)
    except PermissionError as exc:  # website policy block
        raise SourceUnsafe(str(exc), src=url, origin="http")
    finally:
        tmp.unlink(missing_ok=True)


def _is_local_terminal_backend() -> bool:
    """True when the terminal backend runs directly on the host.

    Mirrors ``tools.browser_tool._is_local_backend`` and terminal_tool's own
    dispatch, which key off ``TERMINAL_ENV``.
    """
    return os.getenv("TERMINAL_ENV", "local").strip().lower() in ("local", "")


def _media_cache_roots() -> list:
    """Agent-managed media cache directories under HERMES_HOME (host side).

    The only host paths vision may read under a non-local backend: gateway-
    downloaded inbound media and the tools' own URL-download temp dirs. Covers
    the consolidated ``cache/`` layout and the legacy flat directories.
    """
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    return [
        home / "cache",  # cache/images, cache/vision, cache/video(s), cache/audio
        home / "image_cache",
        home / "audio_cache",
        home / "video_cache",
        home / "temp_vision_images",
        home / "temp_video_files",
    ]


def _permitted_host_read_target(p: Path, ctx: ResolveContext) -> Optional[Path]:
    """Return the host path to read, or ``None`` if a host read is not permitted.

    - Local backend: any path is permitted (chosen posture). Returns ``p``.
    - Non-local backend: permitted only if the path resolves inside a media
      cache root. A container-visible cache path (e.g. ``/root/.hermes/cache/
      images/x.png``) is first translated back to its host mount; anything that
      is not under a cache returns ``None`` so the caller routes it to the
      in-sandbox exec-read instead of reading the host filesystem.
    """
    if _is_local_terminal_backend():
        try:
            return p.resolve()
        except Exception:  # noqa: BLE001 — unresolved path: let is_file() fail downstream
            return p

    from tools.credential_files import from_agent_visible_cache_path

    host_candidate = Path(from_agent_visible_cache_path(str(p)))
    try:
        real = host_candidate.resolve()
    except Exception:  # noqa: BLE001 — cannot resolve -> not a safe host read
        return None
    for root in _media_cache_roots():
        try:
            real.relative_to(root.resolve())
            return real
        except ValueError:
            continue
    return None


def _get_active_env(task_id: Optional[str]):
    if not task_id:
        return None
    try:
        from tools.terminal_tool import get_active_env

        return get_active_env(task_id)
    except Exception:
        return None


def _ensure_container_env(task_id: Optional[str]) -> None:
    """Lazily bring up the sandbox (SSH/Docker/…) before an in-sandbox read.

    Unlike the terminal tool, vision never triggered environment creation, so a
    session whose first action is ``vision_analyze`` on a container-only path
    under a non-local backend found no active env and failed — until a terminal
    command happened to create one (issue #62825). Best-effort: any failure just
    leaves the env absent and the caller hits the existing fail-closed error.
    """
    if not task_id:
        return
    try:
        from tools.terminal_tool import ensure_task_env

        ensure_task_env(task_id)
    except Exception:
        pass


async def _resolve_container_fallback(p: Path, ctx: ResolveContext, src: str) -> ResolvedImage:
    """Read the image bytes inside the sandbox (fail-closed when none exists).

    Reached when a host read is not permitted or the host file is absent. The
    agent can already ``cat`` any container file (file_operations.py reads
    root-owned mode-600 files this way), so this stays within the same sandbox
    boundary and never touches the host filesystem. ``--`` stops a leading-dash
    path from being parsed as a ``base64`` option; ``base64 -w0`` is GNU-only,
    so pipe through ``tr -d`` for BusyBox.

    Fail-closed: if there is no active sandbox env we refuse rather than falling
    back to a host read, so a non-cache host path under a sandbox never leaks.
    """
    import asyncio
    import shlex

    # Bring the sandbox up on demand: without this, the first vision_analyze of
    # a session (before any terminal command) has no active env to read from
    # under a non-local backend (issue #62825).
    _ensure_container_env(ctx.task_id)

    env = _get_active_env(ctx.task_id)
    if env is None:
        raise SourceNotFound(
            f"'{p}' is not reachable inside the sandbox and no active sandbox "
            f"session is available to read it",
            src=src, origin="container")

    # Bound the read INSIDE the sandbox: head -c caps at ingest-limit+1 bytes
    # so a huge file (or /dev/zero) can't stream unbounded base64 into host
    # memory — the +1 byte lets us distinguish "exactly at the cap" from
    # "over the cap" after decode. The input redirect (< path) avoids argv
    # entirely, so leading-dash paths can't be parsed as options; base64
    # -w0 is GNU-only, so pipe through tr -d for BusyBox.
    # env.execute is a blocking backend exec; keep it off the event loop so a
    # multi-MB base64 read doesn't stall every other coroutine.
    qp = shlex.quote(str(p))
    res = await asyncio.to_thread(
        env.execute,
        f"head -c {_MAX_INGEST_BYTES + 1} < {qp} | base64 | tr -d '\\n'")
    if res.get("returncode", 1) != 0:
        raise SourceNotFound(f"could not read '{p}' inside the sandbox", src=src, origin="container")
    try:
        data = base64.b64decode(res.get("output", ""), validate=True)
    except Exception as exc:
        raise NotAnImage(f"sandbox returned non-image data for '{p}': {exc}", src=src)
    if len(data) > _MAX_INGEST_BYTES:
        raise SourceTooLarge("image exceeds size limit", src=src, origin="container")
    return _finalize(data, "", "container", src)


def _finalize(data: bytes, declared_mime: str, origin: str, src: str) -> ResolvedImage:
    """Intrinsic-correctness chokepoint: ingest byte cap + magic-byte sniff.

    The cap here is the generous 50MB *ingest* budget, not the 20MB provider
    payload cap — a 20-50MB image must survive this step so the call site can
    resize it under the payload cap. See ``_MAX_INGEST_BYTES``.
    """
    from tools.vision_tools import _detect_image_mime_type_from_bytes

    if len(data) > _MAX_INGEST_BYTES:
        raise SourceTooLarge("image exceeds size limit", src=src, origin=origin)
    sniffed = _detect_image_mime_type_from_bytes(data)
    if sniffed is None:
        if b"<svg" in data[:4096].lower():
            # Pass SVG through — the vision call sites rasterize it to PNG
            # via _normalize_to_supported_image before embedding (providers
            # only ingest raster images).
            return ResolvedImage(data=data, mime="image/svg+xml", origin=origin)
        raise NotAnImage("source is not a recognized image", src=src, origin=origin)
    return ResolvedImage(data=data, mime=sniffed, origin=origin)
