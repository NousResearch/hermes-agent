#!/usr/bin/env python3
"""
Vision Tools Module

This module provides vision analysis tools that work with image URLs.
Uses the centralized auxiliary vision router, which can select OpenRouter,
Nous, Codex, native Anthropic, or a custom OpenAI-compatible endpoint.

Available tools:
- vision_analyze_tool: Analyze images from URLs with custom prompts

Features:
- Downloads images from URLs and converts to base64 for API compatibility
- Comprehensive image description
- Context-aware analysis based on user queries
- Automatic temporary file cleanup
- Proper error handling and validation
- Debug logging support

Usage:
    from vision_tools import vision_analyze_tool
    import asyncio
    
    # Analyze an image
    result = await vision_analyze_tool(
        image_url="https://example.com/image.jpg",
        user_prompt="What architectural style is this building?"
    )
"""

import base64
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Awaitable, Dict, Optional
from urllib.parse import urlparse
import httpx
from agent.auxiliary_client import async_call_llm, extract_content_or_reasoning
from hermes_constants import get_hermes_dir
from tools.debug_helpers import DebugSession
from tools.website_policy import check_website_access
import sys

logger = logging.getLogger(__name__)

_debug = DebugSession("vision_tools", env_var="VISION_TOOLS_DEBUG")

# Configurable HTTP download timeout for _download_image().
# Separate from auxiliary.vision.timeout which governs the LLM API call.
# Resolution: config.yaml auxiliary.vision.download_timeout → env var → 30s default.
def _resolve_download_timeout() -> float:
    env_val = os.getenv("HERMES_VISION_DOWNLOAD_TIMEOUT", "").strip()
    if env_val:
        try:
            return float(env_val)
        except ValueError:
            pass
    try:
        from hermes_cli.config import cfg_get, load_config
        cfg = load_config()
        val = cfg_get(cfg, "auxiliary", "vision", "download_timeout")
        if val is not None:
            return float(val)
    except Exception:
        pass
    return 30.0

_VISION_DOWNLOAD_TIMEOUT = _resolve_download_timeout()

# Hard cap on downloaded image file size (50 MB). Prevents OOM from
# attacker-hosted multi-gigabyte files or decompression bombs.
_VISION_MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024


def _image_url_shape_ok(url: str) -> bool:
    """HTTP(S) shape check only (scheme, netloc). No DNS."""
    if not url or not isinstance(url, str):
        return False
    # Basic HTTP/HTTPS URL check
    if not url.startswith(("http://", "https://")):
        return False
    # Parse to ensure we at least have a network location; still allow URLs
    # without file extensions (e.g. CDN endpoints that redirect to images).
    parsed = urlparse(url)
    if not parsed.netloc:
        return False
    return True


def _validate_image_url(url: str) -> bool:
    """Validate image URL for sync callers and tests (SSRF via sync DNS check)."""
    if not _image_url_shape_ok(url):
        return False
    # Block private/internal addresses to prevent SSRF
    from tools.url_safety import is_safe_url
    return is_safe_url(url)


async def _validate_image_url_async(url: str) -> bool:
    """Validate remote image URL without blocking the event loop on DNS."""
    if not _image_url_shape_ok(url):
        return False
    from tools.url_safety import async_is_safe_url
    return await async_is_safe_url(url)


def _detect_image_mime_type(image_path: Path) -> Optional[str]:
    """Return a MIME type when the file looks like a supported image."""
    with image_path.open("rb") as f:
        header = f.read(64)

    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if header.startswith(b"BM"):
        return "image/bmp"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    if image_path.suffix.lower() == ".svg":
        head = image_path.read_text(encoding="utf-8", errors="ignore")[:4096].lower()
        if "<svg" in head:
            return "image/svg+xml"
    return None


def _is_retryable_download_error(error: Exception) -> bool:
    """Return True only for transient image-download failures worth retrying.

    Non-retryable (fail-fast):
      - httpx.HTTPStatusError with a 4xx status other than 429 (404/403/410/...):
        the resource is missing or forbidden; retrying can't change that.
      - PermissionError: blocked by website policy / SSRF guard.
      - ValueError: image too large or blocked redirect — deterministic.

    Retryable (transient):
      - httpx 429 (rate limited) and 5xx (server-side) errors.
      - Connection/timeout/transport errors (httpx.TransportError) and any
        other unclassified exception, which may be a flaky network blip.
    """
    if isinstance(error, (PermissionError, ValueError)):
        return False
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        if 400 <= status < 500 and status != 429:
            return False
        return True
    return True


async def _download_image(image_url: str, destination: Path, max_retries: int = 3) -> Path:
    """
    Download an image from a URL to a local destination (async) with retry logic.
    
    Args:
        image_url (str): The URL of the image to download
        destination (Path): The path where the image should be saved
        max_retries (int): Maximum number of retry attempts (default: 3)
        
    Returns:
        Path: The path to the downloaded image
        
    Raises:
        Exception: If download fails after all retries
    """
    import asyncio
    
    # Create parent directories if they don't exist
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    async def _ssrf_redirect_guard(response):
        """Re-validate each redirect target to prevent redirect-based SSRF.

        Without this, an attacker can host a public URL that 302-redirects
        to http://169.254.169.254/ and bypass the pre-flight is_safe_url check.

        Must be async because httpx.AsyncClient awaits event hooks.
        """
        if response.is_redirect and response.next_request:
            redirect_url = str(response.next_request.url)
            from tools.url_safety import async_is_safe_url
            if not await async_is_safe_url(redirect_url):
                raise ValueError(
                    f"Blocked redirect to private/internal address: {redirect_url}"
                )

    last_error = None
    for attempt in range(max_retries):
        try:
            blocked = check_website_access(image_url)
            if blocked:
                raise PermissionError(blocked["message"])

            # Download the image with appropriate headers using async httpx
            # Enable follow_redirects to handle image CDNs that redirect (e.g., Imgur, Picsum)
            # SSRF: event_hooks validates each redirect target against private IP ranges
            async with httpx.AsyncClient(
                timeout=_VISION_DOWNLOAD_TIMEOUT,
                follow_redirects=True,
                event_hooks={"response": [_ssrf_redirect_guard]},
            ) as client:
                response = await client.get(
                    image_url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        "Accept": "image/*,*/*;q=0.8",
                    },
                )
                response.raise_for_status()

                # Reject overly large images early via Content-Length header.
                cl = response.headers.get("content-length")
                if cl and int(cl) > _VISION_MAX_DOWNLOAD_BYTES:
                    raise ValueError(
                        f"Image too large ({int(cl)} bytes, max {_VISION_MAX_DOWNLOAD_BYTES})"
                    )

                final_url = str(response.url)
                blocked = check_website_access(final_url)
                if blocked:
                    raise PermissionError(blocked["message"])
                
                # Save the image content (double-check actual size)
                body = response.content
                if len(body) > _VISION_MAX_DOWNLOAD_BYTES:
                    raise ValueError(
                        f"Image too large ({len(body)} bytes, max {_VISION_MAX_DOWNLOAD_BYTES})"
                    )
                destination.write_bytes(body)
            
            return destination
        except Exception as e:
            last_error = e
            # Error-class-aware retry: only retry transient failures. A 4xx
            # client error (404/403/410, etc.) will never succeed on retry —
            # the resource isn't there or we're not allowed — so burning 3
            # attempts with 2s/4s/8s backoff just inflates latency. 429 (rate
            # limit) and 5xx remain retryable. PermissionError (policy block)
            # and ValueError (too-large / SSRF redirect) are also terminal.
            if not _is_retryable_download_error(e) or attempt >= max_retries - 1:
                logger.error(
                    "Image download failed after %s attempt(s): %s",
                    attempt + 1,
                    str(e)[:100],
                    exc_info=True,
                )
                raise
            wait_time = 2 ** (attempt + 1)  # 2s, 4s, 8s
            logger.warning("Image download failed (attempt %s/%s): %s", attempt + 1, max_retries, str(e)[:50])
            logger.warning("Retrying in %ss...", wait_time)
            await asyncio.sleep(wait_time)

    # The loop always returns on success or re-raises on the final/non-retryable
    # attempt, so reaching here means max_retries was non-positive.
    if last_error is not None:
        raise last_error
    raise RuntimeError(
        f"_download_image exited retry loop without attempting (max_retries={max_retries})"
    )


def _determine_mime_type(image_path: Path) -> str:
    """
    Determine the MIME type of an image based on its file extension.
    
    Args:
        image_path (Path): Path to the image file
        
    Returns:
        str: The MIME type (defaults to image/jpeg if unknown)
    """
    extension = image_path.suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp',
        '.svg': 'image/svg+xml'
    }
    return mime_types.get(extension, 'image/jpeg')


def _image_to_base64_data_url(image_path: Path, mime_type: Optional[str] = None) -> str:
    """
    Convert an image file to a base64-encoded data URL.
    
    Args:
        image_path (Path): Path to the image file
        mime_type (Optional[str]): MIME type of the image (auto-detected if None)
        
    Returns:
        str: Base64-encoded data URL (e.g., "data:image/jpeg;base64,...")
    """
    # Read the image as bytes
    data = image_path.read_bytes()
    
    # Encode to base64
    encoded = base64.b64encode(data).decode("ascii")
    
    # Determine MIME type
    mime = mime_type or _determine_mime_type(image_path)
    
    # Create data URL
    data_url = f"data:{mime};base64,{encoded}"
    
    return data_url


# Absolute hard ceiling for vision API payloads (20 MB) — above this, no major
# provider accepts the image and we reject outright.
_MAX_BASE64_BYTES = 20 * 1024 * 1024

# Proactive embed cap (4 MB).  This is the size we resize an image DOWN to
# before embedding it into conversation history, regardless of the 20 MB hard
# ceiling.  Anthropic's per-image base64 limit is 5 MB; once an oversized image
# is baked into history (e.g. a vision tool-result), it is re-sent on every
# subsequent turn and permanently wedges the session with a 400 that retries
# can't clear (the bad bytes are immutable history).  Capping at embed time —
# with headroom under 5 MB — is the only durable fix.  Matches the post-failure
# shrink target in agent.conversation_compression so behaviour is consistent
# whether we resize proactively or reactively.
_EMBED_TARGET_BYTES = 4 * 1024 * 1024

# Proactive embed dimension cap (px, longest side).  Anthropic enforces an
# 8000px per-side ceiling INDEPENDENTLY of the 5 MB byte cap — a tall full-page
# screenshot can be well under 5 MB yet far over 8000px (e.g. 1200×12000 at
# 0.06 MB), so the byte-only embed check above lets it slip into immutable
# history un-resized and the session bricks on a non-retryable 400.  We cap at
# 7900 (headroom under 8000) so the proactive resize shrinks tall small-byte
# images before they are embedded.
_EMBED_MAX_DIMENSION = 7900

# Target size when auto-resizing on API failure (5 MB).  After a provider
# rejects an image, we downscale to this target and retry once.
_RESIZE_TARGET_BYTES = 5 * 1024 * 1024


def _is_image_size_error(error: Exception) -> bool:
    """Detect if an API error is related to image or payload size."""
    err_str = str(error).lower()
    return any(hint in err_str for hint in (
        "too large", "payload", "413", "content_too_large",
        "request_too_large", "image_url", "invalid_request",
        "exceeds", "size limit",
    ))


def _image_exceeds_dimension(image_path: Path, max_dimension: int) -> bool:
    """True if the image's longest side exceeds ``max_dimension`` px.

    Anthropic enforces an 8000px per-side cap independently of the 5 MB byte
    cap, so a tall small-byte screenshot can pass every byte check yet trip a
    non-retryable 400.  Returns False (don't force a resize) when Pillow is
    unavailable or the file can't be read as an image — the byte-based checks
    still apply, and we never want a missing soft dependency to break the
    embed path.
    """
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(image_path) as _img:
            return max(_img.size) > max_dimension
    except Exception:
        return False


def _resize_image_for_vision(image_path: Path, mime_type: Optional[str] = None,
                              max_base64_bytes: int = _RESIZE_TARGET_BYTES,
                              max_dimension: Optional[int] = None) -> str:
    """Convert an image to a base64 data URL, auto-resizing if too large.

    Tries Pillow first to progressively downscale oversized images.  If Pillow
    is not installed or resizing still exceeds the limit, falls back to the raw
    bytes and lets the caller handle the size check.

    Args:
        max_dimension: If set, images whose longest side exceeds this pixel
            count are forcibly downscaled even if they're under the byte
            budget.  Anthropic enforces an 8000 px per-side cap independently
            of the 5 MB byte cap.

    Returns the base64 data URL string.
    """
    # Quick file-size estimate: base64 expands by ~4/3, plus data URL header.
    # Skip the expensive full-read + encode if Pillow can resize directly.
    file_size = image_path.stat().st_size
    estimated_b64 = (file_size * 4) // 3 + 100  # ~header overhead
    needs_resize_for_bytes = estimated_b64 > max_base64_bytes

    # Check pixel dimensions even if bytes are fine.
    needs_resize_for_dims = False
    if max_dimension is not None:
        try:
            from PIL import Image as _PILQuick
            with _PILQuick.open(image_path) as _quick_img:
                if max(_quick_img.size) > max_dimension:
                    needs_resize_for_dims = True
        except Exception:
            pass  # can't check; Pillow path below will handle or skip

    if not needs_resize_for_bytes and not needs_resize_for_dims:
        # Small enough — just encode directly.
        data_url = _image_to_base64_data_url(image_path, mime_type=mime_type)
        if len(data_url) <= max_base64_bytes:
            return data_url
    else:
        data_url = None  # defer full encode; try Pillow resize first

    # Attempt auto-resize with Pillow (soft dependency)
    try:
        from PIL import Image
        import io as _io
    except ImportError:
        # Pillow is a lazy-installable soft dependency. Try a best-effort
        # install (respects security.allow_lazy_installs; no-op if disabled or
        # offline), then re-import. If it still isn't importable, fall back to
        # the raw bytes and let the caller raise the size error.
        try:
            from tools.lazy_deps import ensure as _ensure_dep
            # prompt=False: never raise a blocking input() prompt mid-session.
            # Under the interactive CLI prompt_toolkit owns stdin, so a bare
            # input() deadlocks the terminal (#40490). The install is already
            # gated by security.allow_lazy_installs, so reaching here is opt-in.
            _ensure_dep("tool.vision", prompt=False)
            from PIL import Image
            import io as _io
        except Exception:
            logger.info("Pillow not installed — cannot auto-resize oversized image")
            if data_url is None:
                data_url = _image_to_base64_data_url(image_path, mime_type=mime_type)
            return data_url  # caller will raise the size error

    logger.info("Image file is %.1f MB (estimated base64 %.1f MB, limit %.1f MB, max_dimension=%s), auto-resizing...",
                file_size / (1024 * 1024), estimated_b64 / (1024 * 1024),
                max_base64_bytes / (1024 * 1024), max_dimension)

    mime = mime_type or _determine_mime_type(image_path)
    # Choose output format: JPEG for photos (smaller), PNG for transparency
    pil_format = "PNG" if mime == "image/png" else "JPEG"
    out_mime = "image/png" if pil_format == "PNG" else "image/jpeg"

    try:
        img = Image.open(image_path)
    except Exception as exc:
        logger.info("Pillow cannot open image for resizing: %s", exc)
        if data_url is None:
            data_url = _image_to_base64_data_url(image_path, mime_type=mime_type)
        return data_url  # fall through to size-check in caller
    # Convert RGBA to RGB for JPEG output
    if pil_format == "JPEG" and img.mode in {"RGBA", "P"}:
        img = img.convert("RGB")

    # Strategy: halve dimensions until both base64 fits AND pixel dimensions
    # are within limits, up to 4 rounds.
    # For JPEG, also try reducing quality at each size step.
    # For PNG, quality is irrelevant — only dimension reduction helps.
    quality_steps = (85, 70, 50) if pil_format == "JPEG" else (None,)
    prev_dims = (img.width, img.height)
    candidate = None  # will be set on first loop iteration

    def _dims_ok(w: int, h: int) -> bool:
        """True if both pixel dimensions are within the limit."""
        if max_dimension is None:
            return True
        return max(w, h) <= max_dimension

    for attempt in range(5):
        if attempt > 0:
            # Proportional scaling: halve the longer side and scale the
            # shorter side to preserve aspect ratio (min dimension 64).
            scale = 0.5
            new_w = max(int(img.width * scale), 64)
            new_h = max(int(img.height * scale), 64)
            # Re-derive the scale from whichever dimension hit the floor
            # so both axes shrink by the same factor.
            if new_w == 64 and img.width > 0:
                effective_scale = 64 / img.width
                new_h = max(int(img.height * effective_scale), 64)
            elif new_h == 64 and img.height > 0:
                effective_scale = 64 / img.height
                new_w = max(int(img.width * effective_scale), 64)
            # Stop if dimensions can't shrink further
            if (new_w, new_h) == prev_dims:
                break
            img = img.resize((new_w, new_h), Image.LANCZOS)
            prev_dims = (new_w, new_h)
            logger.info("Resized to %dx%d (attempt %d)", new_w, new_h, attempt)

        for q in quality_steps:
            buf = _io.BytesIO()
            save_kwargs = {"format": pil_format}
            if q is not None:
                save_kwargs["quality"] = q
            img.save(buf, **save_kwargs)
            encoded = base64.b64encode(buf.getvalue()).decode("ascii")
            candidate = f"data:{out_mime};base64,{encoded}"
            if len(candidate) <= max_base64_bytes and _dims_ok(img.width, img.height):
                logger.info("Auto-resized image fits: %.1f MB (quality=%s, %dx%d)",
                            len(candidate) / (1024 * 1024), q,
                            img.width, img.height)
                return candidate

    # If we still can't get it small enough, return the best attempt
    # and let the caller decide
    if candidate is not None:
        logger.warning("Auto-resize could not fit image under %.1f MB (best: %.1f MB)",
                       max_base64_bytes / (1024 * 1024), len(candidate) / (1024 * 1024))
        return candidate

    # Shouldn't reach here, but fall back to full encode
    return data_url or _image_to_base64_data_url(image_path, mime_type=mime_type)


# ---------------------------------------------------------------------------
# Native fast path: short-circuit the auxiliary LLM when the active main model
# supports native vision. Instead of asking a separate LLM to describe the
# image and returning text, we load the image, base64-encode it, and return a
# multimodal tool-result envelope. The agent loop unwraps the envelope into an
# OpenAI-style content list on the `tool` role; provider adapters (anthropic,
# codex_responses, chat_completions) translate that into Anthropic
# tool_result image blocks / Responses input_image / OpenAI image_url tool
# content. The main model then "sees" the pixels directly on its next turn.
# ---------------------------------------------------------------------------


def _supports_media_in_tool_results(provider: str, model: str) -> bool:
    """Whether the given provider+model combination accepts image content
    inside a tool-result message.

    Providers covered today (per spec docs verified Apr-2026):

      * Anthropic Messages API (``anthropic`` provider, plus aggregators that
        proxy Claude — ``openrouter``, ``nous``, ``vertex``, ``bedrock``):
        ``tool_result`` blocks accept ``image`` content blocks.
      * OpenAI Chat Completions: tool messages accept array content with
        ``image_url`` parts.
      * OpenAI Responses (``openai-codex``): ``function_call_output.output``
        accepts an array of ``input_text``/``input_image`` items.
      * Gemini 3 (and proxied via aggregators): supports multimodal tool
        results. Older Gemini does NOT.

    For unknown / legacy providers we conservatively return False — the
    caller falls back to the legacy aux-LLM text path.  The check is relaxed
    when the provider's ``ProviderProfile`` declares ``supports_vision=True``.
    """
    if not isinstance(provider, str):
        return False
    p = provider.strip().lower()
    if not p:
        return False

    # Aggregators that route to multiple vendors — assume support since
    # users on these aggregators are typically using vision-capable
    # frontier models. Falling back to text would be a regression for
    # them.
    _AGGREGATORS = {
        "openrouter", "nous", "vertex", "bedrock", "anthropic-vertex",
        "google-vertex",
    }
    if p in _AGGREGATORS:
        return True

    # Native Anthropic
    if p in {"anthropic", "claude", "anthropic-direct"}:
        return True

    # OpenAI Chat Completions and Responses
    if p in {"openai", "openai-chat", "openai-codex", "azure-openai"}:
        return True

    # Gemini — gate on model name; older Gemini variants did not support
    # multimodal functionResponse. Gemini 3.x does.
    if p in {"google", "gemini", "google-gemini", "google-vertex-gemini"}:
        if not isinstance(model, str):
            return False
        m = model.strip().lower()
        if "gemini-3" in m or "gemini-pro-3" in m or "gemini-flash-3" in m:
            return True
        return False

    # Check the provider's registered profile for the supports_vision flag.
    # This covers vision-capable providers like xiaomi, minimax, etc. that
    # aren't in the hardcoded list above.
    try:
        from providers import get_provider_profile
        profile = get_provider_profile(p)
        if profile is not None and profile.supports_vision:
            return True
    except Exception:
        pass

    # Other vision-capable provider stacks. Conservative default: False.
    # Add explicit entries here as we verify each provider's tool-result
    # multimodal support empirically.
    return False


def _should_use_native_vision_fast_path() -> bool:
    """Whether vision tools should attach the image to the main model directly
    instead of routing through the auxiliary vision LLM.

    True when image routing resolves to ``native`` AND either the provider is
    known to accept images inside tool results, or the user explicitly declared
    the model vision-capable via the ``model.supports_vision`` config override.
    The override is the escape hatch for custom/local providers that aren't in
    the static allowlist. Best-effort: any resolution failure returns False so
    the caller falls back to the legacy aux-LLM path.
    """
    try:
        from agent.auxiliary_client import _read_main_provider, _read_main_model
        from agent.image_routing import decide_image_input_mode, _lookup_supports_vision
        from hermes_cli.config import load_config

        provider = _read_main_provider()
        model = _read_main_model()
        cfg = load_config()
        if decide_image_input_mode(provider, model, cfg) != "native":
            return False
        return (
            _supports_media_in_tool_results(provider, model)
            or _lookup_supports_vision(provider, model, cfg) is True
        )
    except Exception as exc:
        logger.debug("Native vision fast-path check failed: %s", exc)
        return False


def _build_native_vision_tool_result(
    image_url: str,
    question: str,
    image_data_url: str,
    image_size_bytes: int,
) -> Dict[str, Any]:
    """Build the multimodal tool-result envelope returned by the fast path.

    Shape:
      {
        "_multimodal": True,
        "content": [
          {"type": "text", "text": "<short note + the user's question>"},
          {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ],
        "text_summary": "<plain-text fallback>",
        "meta": {"image_url": ..., "size_bytes": N},
      }

    The text part exists for two reasons: (1) it gives the model an
    instruction to act on now that the pixels are in context, and
    (2) providers that don't support multimodal tool results can fall back
    to ``text_summary``.
    """
    # The tool-result text part is intentionally minimal. The model already
    # has the user's original question in context; this just acknowledges
    # the image is now visible and reminds it what it was asked.
    text_part = (
        "Image loaded into your context — you can see it natively now. "
        "Use your built-in vision to answer the user."
    )
    if isinstance(question, str) and question.strip():
        text_part += f"\n\nQuestion: {question.strip()}"

    summary = (
        f"Image attached natively for the main model "
        f"({image_size_bytes / 1024:.1f} KB). "
        "Answer using built-in vision."
    )

    return {
        "_multimodal": True,
        "content": [
            {"type": "text", "text": text_part},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ],
        "text_summary": summary,
        "meta": {
            "image_url": image_url[:200],
            "size_bytes": image_size_bytes,
            "native_vision": True,
        },
    }


async def _vision_analyze_native(
    image_url: str,
    question: str,
) -> Any:
    """Fast path for vision-capable main models.

    Loads the image (local file OR remote URL), base64-encodes it, and
    returns a multimodal tool-result envelope. The agent loop unwraps it;
    provider adapters serialize it into the right tool-result-with-image
    shape for each backend.

    Returns:
        A ``_multimodal`` envelope dict on success.
        A JSON error string on failure (matches the existing tool-result
        contract so the agent loop displays errors normally).
    """
    if not isinstance(image_url, str) or not image_url.strip():
        return tool_error("image_url is required", success=False)

    temp_image_path: Optional[Path] = None
    should_cleanup = False
    try:
        from tools.interrupt import is_interrupted
        if is_interrupted():
            return tool_error("Interrupted", success=False)

        # Resolve the image source (mirrors vision_analyze_tool's logic
        # exactly so behaviour is consistent).
        resolved_url = image_url
        if resolved_url.startswith("file://"):
            resolved_url = resolved_url[len("file://"):]
        local_path = Path(os.path.expanduser(resolved_url))

        if local_path.is_file():
            temp_image_path = local_path
            should_cleanup = False
        elif await _validate_image_url_async(image_url):
            blocked = check_website_access(image_url)
            if blocked:
                return tool_error(blocked["message"], success=False)
            temp_dir = get_hermes_dir("cache/vision", "temp_vision_images")
            temp_image_path = temp_dir / f"temp_image_{uuid.uuid4()}.jpg"
            await _download_image(image_url, temp_image_path)
            should_cleanup = True
        else:
            return tool_error(
                "Invalid image source. Provide an HTTP/HTTPS URL or a "
                "valid local file path.",
                success=False,
            )

        image_size_bytes = temp_image_path.stat().st_size
        detected_mime_type = _detect_image_mime_type(temp_image_path)
        if not detected_mime_type:
            return tool_error(
                "Only real image files are supported for vision analysis.",
                success=False,
            )

        image_data_url = _image_to_base64_data_url(
            temp_image_path, mime_type=detected_mime_type,
        )

        # Proactive embed cap: this image gets baked into conversation
        # history and re-sent on every subsequent turn.  Anthropic rejects
        # any single base64 image over 5 MB OR over 8000px per side with a
        # 400, and because history is immutable, an oversized embed
        # permanently wedges the session — retries can't clear bytes (or
        # pixels) that are already in the request.  Resize DOWN to the embed
        # target (4 MB / 7900px, headroom under both ceilings) whenever the
        # payload exceeds either limit, not just at the 20 MB hard ceiling.
        _over_bytes = len(image_data_url) > _EMBED_TARGET_BYTES
        _over_dims = _image_exceeds_dimension(temp_image_path, _EMBED_MAX_DIMENSION)
        if _over_bytes or _over_dims:
            image_data_url = _resize_image_for_vision(
                temp_image_path, mime_type=detected_mime_type,
                max_base64_bytes=_EMBED_TARGET_BYTES,
                max_dimension=_EMBED_MAX_DIMENSION,
            )
            # If even resizing can't get under the absolute hard ceiling,
            # there's nothing more we can do — reject rather than embed a
            # session-wedging payload.
            if len(image_data_url) > _MAX_BASE64_BYTES:
                return tool_error(
                    f"Image too large for vision API: base64 payload is "
                    f"{len(image_data_url) / (1024 * 1024):.1f} MB "
                    f"(limit {_MAX_BASE64_BYTES / (1024 * 1024):.0f} MB) "
                    f"even after resizing. Install Pillow "
                    f"(`pip install Pillow`) for better auto-resize, "
                    f"or compress the image manually.",
                    success=False,
                )

        return _build_native_vision_tool_result(
            image_url=image_url,
            question=question,
            image_data_url=image_data_url,
            image_size_bytes=image_size_bytes,
        )

    except Exception as exc:
        logger.warning("Native vision fast path failed: %s", exc)
        return tool_error(f"Native vision failed: {exc}", success=False)
    finally:
        # Only delete temp files we created — never user-provided paths.
        if should_cleanup and temp_image_path is not None:
            try:
                if temp_image_path.exists():
                    temp_image_path.unlink()
            except Exception:
                pass


async def vision_analyze_tool(
    image_url: str,
    user_prompt: str,
    model: str = None,
) -> str:
    """
    Analyze an image from a URL or local file path using vision AI.
    
    This tool accepts either an HTTP/HTTPS URL or a local file path. For URLs,
    it downloads the image first. In both cases, the image is converted to base64
    and processed using Gemini 3 Flash Preview via OpenRouter API.
    
    The user_prompt parameter is expected to be pre-formatted by the calling
    function (typically model_tools.py) to include both full description
    requests and specific questions.
    
    Args:
        image_url (str): The URL or local file path of the image to analyze.
                         Accepts http://, https:// URLs or absolute/relative file paths.
        user_prompt (str): The pre-formatted prompt for the vision model
        model (str): The vision model to use (default: google/gemini-3-flash-preview)
    
    Returns:
        str: JSON string containing the analysis results with the following structure:
             {
                 "success": bool,
                 "analysis": str (defaults to error message if None)
             }
    
    Raises:
        Exception: If download fails, analysis fails, or API key is not set
        
    Note:
        - For URLs, temporary images are stored under $HERMES_HOME/cache/vision/ and cleaned up
        - For local file paths, the file is used directly and NOT deleted
        - Supports common image formats (JPEG, PNG, GIF, WebP, etc.)
    """
    if not isinstance(user_prompt, str):
        user_prompt = str(user_prompt) if user_prompt is not None else ""
    debug_call_data = {
        "parameters": {
            "image_url": image_url,
            "user_prompt": user_prompt[:200] + "..." if len(user_prompt) > 200 else user_prompt,
            "model": model
        },
        "error": None,
        "success": False,
        "analysis_length": 0,
        "model_used": model,
        "image_size_bytes": 0
    }
    
    temp_image_path = None
    # Track whether we should clean up the file after processing.
    # Local files (e.g. from the image cache) should NOT be deleted.
    should_cleanup = True
    detected_mime_type = None
    
    try:
        from tools.interrupt import is_interrupted
        if is_interrupted():
            return tool_error("Interrupted", success=False)

        logger.info("Analyzing image: %s", image_url[:60])
        logger.info("User prompt: %s", user_prompt[:100])
        
        # Determine if this is a local file path or a remote URL
        # Strip file:// scheme so file URIs resolve as local paths.
        resolved_url = image_url
        if resolved_url.startswith("file://"):
            resolved_url = resolved_url[len("file://"):]
        local_path = Path(os.path.expanduser(resolved_url))
        if local_path.is_file():
            # Local file path (e.g. from platform image cache) -- skip download
            logger.info("Using local image file: %s", image_url)
            temp_image_path = local_path
            should_cleanup = False  # Don't delete cached/local files
        elif await _validate_image_url_async(image_url):
            # Remote URL -- download to a temporary location
            blocked = check_website_access(image_url)
            if blocked:
                raise PermissionError(blocked["message"])
            logger.info("Downloading image from URL...")
            temp_dir = get_hermes_dir("cache/vision", "temp_vision_images")
            temp_image_path = temp_dir / f"temp_image_{uuid.uuid4()}.jpg"
            await _download_image(image_url, temp_image_path)
            should_cleanup = True
        else:
            raise ValueError(
                "Invalid image source. Provide an HTTP/HTTPS URL or a valid local file path."
            )
        
        # Get image file size for logging
        image_size_bytes = temp_image_path.stat().st_size
        image_size_kb = image_size_bytes / 1024
        logger.info("Image ready (%.1f KB)", image_size_kb)

        detected_mime_type = _detect_image_mime_type(temp_image_path)
        if not detected_mime_type:
            raise ValueError("Only real image files are supported for vision analysis.")
        
        # Convert image to base64 — send at full resolution first.
        # If the provider rejects it as too large, we auto-resize and retry.
        logger.info("Converting image to base64...")
        image_data_url = _image_to_base64_data_url(temp_image_path, mime_type=detected_mime_type)
        data_size_kb = len(image_data_url) / 1024
        logger.info("Image converted to base64 (%.1f KB)", data_size_kb)

        # Hard limit (20 MB) — no provider accepts payloads this large.
        if len(image_data_url) > _MAX_BASE64_BYTES:
            # Try to resize down to 5 MB before giving up.
            image_data_url = _resize_image_for_vision(
                temp_image_path, mime_type=detected_mime_type)
            if len(image_data_url) > _MAX_BASE64_BYTES:
                raise ValueError(
                    f"Image too large for vision API: base64 payload is "
                    f"{len(image_data_url) / (1024 * 1024):.1f} MB "
                    f"(limit {_MAX_BASE64_BYTES / (1024 * 1024):.0f} MB) "
                    f"even after resizing. "
                    f"Install Pillow (`pip install Pillow`) for better auto-resize, "
                    f"or compress the image manually."
                )

        debug_call_data["image_size_bytes"] = image_size_bytes
        
        # Use the prompt as provided (model_tools.py now handles full description formatting)
        comprehensive_prompt = user_prompt
        
        # Prepare the message with base64-encoded image
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": comprehensive_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    }
                ]
            }
        ]
        
        logger.info("Processing image with vision model...")
        
        # Call the vision API via centralized router.
        # Read timeout from config.yaml (auxiliary.vision.timeout), default 120s.
        # Local vision models (llama.cpp, ollama) can take well over 30s.
        vision_timeout = 120.0
        vision_temperature = 0.1
        try:
            from hermes_cli.config import cfg_get, load_config
            _cfg = load_config()
            _vision_cfg = cfg_get(_cfg, "auxiliary", "vision", default={})
            _vt = _vision_cfg.get("timeout")
            if _vt is not None:
                vision_timeout = float(_vt)
            _vtemp = _vision_cfg.get("temperature")
            if _vtemp is not None:
                vision_temperature = float(_vtemp)
        except Exception:
            pass
        call_kwargs = {
            "task": "vision",
            "messages": messages,
            "temperature": vision_temperature,
            "max_tokens": 2000,
            "timeout": vision_timeout,
        }
        if model:
            call_kwargs["model"] = model
        # Try full-size image first; on size-related rejection, downscale and retry.
        try:
            response = await async_call_llm(**call_kwargs)
        # On failure (SSL, timeout, etc.), try the configured fallback_chain
        # before giving up. Useful when the primary provider is blocked (e.g.
        # Gemini behind GFW) and local/cloud fallbacks are configured.
        except Exception as _api_err:
            if (_is_image_size_error(_api_err)
                    and len(image_data_url) > _RESIZE_TARGET_BYTES):
                logger.info(
                    "API rejected image (%.1f MB, likely too large); "
                    "auto-resizing to ~%.0f MB and retrying...",
                    len(image_data_url) / (1024 * 1024),
                    _RESIZE_TARGET_BYTES / (1024 * 1024),
                )
                image_data_url = _resize_image_for_vision(
                    temp_image_path, mime_type=detected_mime_type)
                messages[0]["content"][1]["image_url"]["url"] = image_data_url
                response = await async_call_llm(**call_kwargs)
            else:
                # Primary failed — try configured fallback_chain
                try:
                    import agent.auxiliary_client as _aux
                    import importlib
                    importlib.reload(_aux)
                    from agent.auxiliary_client import _try_configured_fallback_chain as _fb
                    _fb_client, _fb_model, _fb_label = _fb("vision", "auto", reason="primary failed")
                    if _fb_client is not None:
                        logger.warning(
                            "Primary vision provider failed, using fallback %s", _fb_label)
                        call_kwargs["model"] = _fb_model or call_kwargs.get("model")
                        response = _fb_client.chat.completions.create(
                            model=call_kwargs["model"],
                            messages=call_kwargs["messages"],
                            max_tokens=call_kwargs.get("max_tokens", 512),
                            temperature=0.0,
                        )
                        logger.info("Primary vision fallback (%s) succeeded", _fb_label)
                    else:
                        raise
                except Exception as _fb_err:
                    raise _api_err from _fb_err
        
        # Extract the analysis — fall back to reasoning if content is empty
        analysis = extract_content_or_reasoning(response)

        # Retry once on empty content (reasoning-only response)
        if not analysis:
            logger.warning("Vision LLM returned empty content, retrying once")
            response = await async_call_llm(**call_kwargs)
            analysis = extract_content_or_reasoning(response)

        analysis_length = len(analysis)
        
        logger.info("Image analysis completed (%s characters)", analysis_length)
        
        # Prepare successful response
        result = {
            "success": True,
            "analysis": analysis or "There was a problem with the request and the image could not be analyzed."
        }
        
        debug_call_data["success"] = True
        debug_call_data["analysis_length"] = analysis_length
        
        # Log debug information
        _debug.log_call("vision_analyze_tool", debug_call_data)
        _debug.save()
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"Error analyzing image: {str(e)}"
        logger.error("%s", error_msg, exc_info=True)
        
        # Detect vision capability errors — give the model a clear message
        # so it can inform the user instead of a cryptic API error.
        err_str = str(e).lower()
        if any(hint in err_str for hint in (
            "402", "insufficient", "payment required", "credits", "billing",
        )):
            analysis = (
                "Insufficient credits or payment required. Please top up your "
                f"API provider account and try again. Error: {e}"
            )
        elif any(hint in err_str for hint in (
            "does not support", "not support image",
            "content_policy", "multimodal",
            "unrecognized request argument", "image input",
        )):
            analysis = (
                f"{model} does not support vision or our request was not "
                f"accepted by the server. Error: {e}"
            )
        elif "invalid_request" in err_str or "image_url" in err_str:
            analysis = (
                "The vision API rejected the image. This can happen when the "
                "image is in an unsupported format, corrupted, or still too "
                "large after auto-resize. Try a smaller JPEG/PNG and retry. "
                f"Error: {e}"
            )
        else:
            analysis = (
                "There was a problem with the request and the image could not "
                f"be analyzed. Error: {e}"
            )
        
        # Prepare error response
        result = {
            "success": False,
            "error": error_msg,
            "analysis": analysis,
        }
        
        debug_call_data["error"] = error_msg
        _debug.log_call("vision_analyze_tool", debug_call_data)
        _debug.save()
        
        return json.dumps(result, indent=2, ensure_ascii=False)
    
    finally:
        # Clean up temporary image file (but NOT local/cached files)
        if should_cleanup and temp_image_path and temp_image_path.exists():
            try:
                temp_image_path.unlink()
                logger.debug("Cleaned up temporary image file")
            except Exception as cleanup_error:
                logger.warning(
                    "Could not delete temporary file: %s", cleanup_error, exc_info=True
                )


def check_vision_requirements() -> bool:
    """Check if the configured runtime vision path can resolve a client.

    Mirrors the fallback chain that ``call_llm(task="vision")`` actually uses
    at runtime: first the explicit ``auxiliary.vision.provider`` (if any),
    and if that fails, the auto chain (main provider → openrouter → nous).
    Without the auto-fallback step the tool would disappear from the model's
    tool list whenever the explicit provider name was unresolvable, even
    when the auto chain would have served the request (issue #31179).
    """
    try:
        from agent.auxiliary_client import resolve_vision_provider_client
    except ImportError:
        return False
    try:
        _provider, client, _model = resolve_vision_provider_client()
        if client is not None:
            return True
        # Same fallback to "auto" that call_llm performs when the configured
        # provider can't be resolved.
        _provider, client, _model = resolve_vision_provider_client(provider="auto")
        return client is not None
    except Exception:
        return False



if __name__ == "__main__":
    """
    Simple test/demo when run directly
    """
    print("👁️ Vision Tools Module")
    print("=" * 40)
    
    # Check if vision model is available
    api_available = check_vision_requirements()
    
    if not api_available:
        print("❌ No auxiliary vision model available")
        print("Configure a supported multimodal backend (OpenRouter, Nous, Codex, Anthropic, or a custom OpenAI-compatible endpoint).")
        sys.exit(1)
    else:
        print("✅ Vision model available")
    
    print("🛠️ Vision tools ready for use!")
    
    # Show debug mode status
    if _debug.active:
        print(f"🐛 Debug mode ENABLED - Session ID: {_debug.session_id}")
        print(f"   Debug logs will be saved to: ./logs/vision_tools_debug_{_debug.session_id}.json")
    else:
        print("🐛 Debug mode disabled (set VISION_TOOLS_DEBUG=true to enable)")
    
    print("\nBasic usage:")
    print("  from vision_tools import vision_analyze_tool")
    print("  import asyncio")
    print("")
    print("  async def main():")
    print("      result = await vision_analyze_tool(")
    print("          image_url='https://example.com/image.jpg',")
    print("          user_prompt='What do you see in this image?'")
    print("      )")
    print("      print(result)")
    print("  asyncio.run(main())")
    
    print("\nExample prompts:")
    print("  - 'What architectural style is this building?'")
    print("  - 'Describe the emotions and mood in this image'")
    print("  - 'What text can you read in this image?'")
    print("  - 'Identify any safety hazards visible'")
    print("  - 'What products or brands are shown?'")
    
    print("\nDebug mode:")
    print("  # Enable debug logging")
    print("  export VISION_TOOLS_DEBUG=true")
    print("  # Debug logs capture all vision analysis calls and results")
    print("  # Logs saved to: ./logs/vision_tools_debug_UUID.json")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error

VISION_ANALYZE_SCHEMA = {
    "name": "vision_analyze",
    "description": (
        "Load an image into the conversation so you can see it. Accepts a "
        "URL, local file path, or data URL. When your active model has "
        "native vision, the image is attached to your context directly "
        "and you read the pixels yourself on the next turn — call this "
        "any time the user references an image (filepath in their message, "
        "URL in tool output, screenshot from the browser, etc.). For "
        "non-vision models, falls back to an auxiliary vision model that "
        "returns a text description."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_url": {
                "type": "string",
                "description": "Image URL (http/https), local file path, or data: URL to load."
            },
            "question": {
                "type": "string",
                "description": "Your specific question or request about the image. Optional context the model uses on the next turn after seeing the image."
            }
        },
        "required": ["image_url", "question"]
    }
}


def _handle_vision_analyze(args: Dict[str, Any], **kw: Any) -> Awaitable[str]:
    image_url = args.get("image_url", "")
    question = args.get("question", "")

    # Fast path: when native image routing is in effect for the active main
    # model (provider accepts images in tool results, or the user set the
    # model.supports_vision override), short-circuit the auxiliary LLM and
    # return the image bytes as a multimodal tool-result envelope. The main
    # model sees the pixels directly on its next turn — no aux call, no
    # information loss, no extra latency.
    if _should_use_native_vision_fast_path():
        logger.info("vision_analyze: native fast path")
        return _vision_analyze_native(image_url, question)

    # Legacy path: aux LLM describes the image and we return its text.
    full_prompt = (
        "Fully describe and explain everything about this image, then answer the "
        f"following question:\n\n{question}"
    )
    model = os.getenv("AUXILIARY_VISION_MODEL", "").strip() or None
    return vision_analyze_tool(image_url, full_prompt, model)


registry.register(
    name="vision_analyze",
    toolset="vision",
    schema=VISION_ANALYZE_SCHEMA,
    handler=_handle_vision_analyze,
    check_fn=check_vision_requirements,
    is_async=True,
    emoji="👁️",
)


# ---------------------------------------------------------------------------
# Video Analysis Tool
# ---------------------------------------------------------------------------

# Extension → MIME. avi/mkv fall back to mp4.
_VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/mov",
    ".avi": "video/mp4",
    ".mkv": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
}

_MAX_VIDEO_BASE64_BYTES = 50 * 1024 * 1024  # 50 MB hard cap
_VIDEO_SIZE_WARN_BYTES = 20 * 1024 * 1024


def _detect_video_mime_type(video_path: Path) -> Optional[str]:
    """Return a video MIME type based on file extension, or None if unsupported."""
    ext = video_path.suffix.lower()
    return _VIDEO_MIME_TYPES.get(ext)


def _video_to_base64_data_url(video_path: Path, mime_type: Optional[str] = None) -> str:
    """Convert a video file to a base64-encoded data URL."""
    data = video_path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    mime = mime_type or _VIDEO_MIME_TYPES.get(video_path.suffix.lower(), "video/mp4")
    return f"data:{mime};base64,{encoded}"


async def _download_video(video_url: str, destination: Path, max_retries: int = 3) -> Path:
    """Download video from URL with SSRF protection and retry."""
    import asyncio

    destination.parent.mkdir(parents=True, exist_ok=True)

    async def _ssrf_redirect_guard(response):
        if response.is_redirect and response.next_request:
            redirect_url = str(response.next_request.url)
            from tools.url_safety import async_is_safe_url
            if not await async_is_safe_url(redirect_url):
                raise ValueError(
                    f"Blocked redirect to private/internal address: {redirect_url}"
                )

    last_error = None
    for attempt in range(max_retries):
        try:
            blocked = check_website_access(video_url)
            if blocked:
                raise PermissionError(blocked["message"])

            async with httpx.AsyncClient(
                timeout=60.0,
                follow_redirects=True,
                event_hooks={"response": [_ssrf_redirect_guard]},
            ) as client:
                response = await client.get(
                    video_url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        "Accept": "video/*,*/*;q=0.8",
                    },
                )
                response.raise_for_status()

                cl = response.headers.get("content-length")
                if cl and int(cl) > _MAX_VIDEO_BASE64_BYTES:
                    raise ValueError(
                        f"Video too large ({int(cl)} bytes, max {_MAX_VIDEO_BASE64_BYTES})"
                    )

                final_url = str(response.url)
                blocked = check_website_access(final_url)
                if blocked:
                    raise PermissionError(blocked["message"])

                body = response.content
                if len(body) > _MAX_VIDEO_BASE64_BYTES:
                    raise ValueError(
                        f"Video too large ({len(body)} bytes, max {_MAX_VIDEO_BASE64_BYTES})"
                    )
                destination.write_bytes(body)

            return destination
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                logger.warning("Video download failed (attempt %s/%s): %s", attempt + 1, max_retries, str(e)[:50])
                await asyncio.sleep(wait_time)
            else:
                logger.error(
                    "Video download failed after %s attempts: %s",
                    max_retries, str(e)[:100], exc_info=True,
                )

    if last_error is None:
        raise RuntimeError(
            f"_download_video exited retry loop without attempting (max_retries={max_retries})"
        )
    raise last_error


async def video_analyze_tool(
    video_url: str,
    user_prompt: str,
    model: str = None,
) -> str:
    """Analyze a video via multimodal LLM. Returns JSON {success, analysis}."""
    if not isinstance(user_prompt, str):
        user_prompt = str(user_prompt) if user_prompt is not None else ""
    debug_call_data = {
        "parameters": {
            "video_url": video_url,
            "user_prompt": user_prompt[:200] + "..." if len(user_prompt) > 200 else user_prompt,
            "model": model,
        },
        "error": None,
        "success": False,
        "analysis_length": 0,
        "model_used": model,
        "video_size_bytes": 0,
    }

    temp_video_path = None
    should_cleanup = True

    try:
        from tools.interrupt import is_interrupted
        if is_interrupted():
            return tool_error("Interrupted", success=False)

        logger.info("Analyzing video: %s", video_url[:60])
        logger.info("User prompt: %s", user_prompt[:100])

        # Resolve local path vs remote URL
        resolved_url = video_url
        if resolved_url.startswith("file://"):
            resolved_url = resolved_url[len("file://"):]
        local_path = Path(os.path.expanduser(resolved_url))

        if local_path.is_file():
            logger.info("Using local video file: %s", video_url)
            temp_video_path = local_path
            should_cleanup = False
        elif await _validate_image_url_async(video_url):
            blocked = check_website_access(video_url)
            if blocked:
                raise PermissionError(blocked["message"])
            temp_dir = get_hermes_dir("cache/video", "temp_video_files")
            temp_video_path = temp_dir / f"temp_video_{uuid.uuid4()}.mp4"
            await _download_video(video_url, temp_video_path)
            should_cleanup = True
        else:
            raise ValueError(
                "Invalid video source. Provide an HTTP/HTTPS URL or a valid local file path."
            )

        video_size_bytes = temp_video_path.stat().st_size
        video_size_mb = video_size_bytes / (1024 * 1024)
        logger.info("Video ready (%.1f MB)", video_size_mb)

        detected_mime = _detect_video_mime_type(temp_video_path)
        if not detected_mime:
            raise ValueError(
                f"Unsupported video format: '{temp_video_path.suffix}'. "
                f"Supported: {', '.join(sorted(_VIDEO_MIME_TYPES.keys()))}"
            )

        if video_size_bytes > _VIDEO_SIZE_WARN_BYTES:
            logger.warning("Video is %.1f MB — may be slow or rejected", video_size_mb)

        video_data_url = _video_to_base64_data_url(temp_video_path, mime_type=detected_mime)
        data_size_mb = len(video_data_url) / (1024 * 1024)

        if len(video_data_url) > _MAX_VIDEO_BASE64_BYTES:
            raise ValueError(
                f"Video too large for API: base64 payload is {data_size_mb:.1f} MB "
                f"(limit {_MAX_VIDEO_BASE64_BYTES / (1024 * 1024):.0f} MB). "
                f"Compress or trim the video and retry."
            )

        debug_call_data["video_size_bytes"] = video_size_bytes

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": video_data_url,
                        },
                    },
                ],
            }
        ]

        vision_timeout = 180.0
        vision_temperature = 0.1
        try:
            from hermes_cli.config import cfg_get, load_config
            _cfg = load_config()
            _vision_cfg = cfg_get(_cfg, "auxiliary", "vision", default={})
            _vt = _vision_cfg.get("timeout")
            if _vt is not None:
                vision_timeout = max(float(_vt), 180.0)
            _vtemp = _vision_cfg.get("temperature")
            if _vtemp is not None:
                vision_temperature = float(_vtemp)
        except Exception:
            pass

        call_kwargs = {
            "task": "vision",
            "messages": messages,
            "temperature": vision_temperature,
            "max_tokens": 4000,
            "timeout": vision_timeout,
        }
        if model:
            call_kwargs["model"] = model

        response = await async_call_llm(**call_kwargs)
        analysis = extract_content_or_reasoning(response)

        if not analysis:
            logger.warning("Empty video response, retrying once")
            response = await async_call_llm(**call_kwargs)
            analysis = extract_content_or_reasoning(response)

        analysis_length = len(analysis) if analysis else 0
        logger.info("Video analysis completed (%s characters)", analysis_length)

        result = {
            "success": True,
            "analysis": analysis or "There was a problem with the request and the video could not be analyzed.",
        }

        debug_call_data["success"] = True
        debug_call_data["analysis_length"] = analysis_length
        _debug.log_call("video_analyze_tool", debug_call_data)
        _debug.save()

        return json.dumps(result, indent=2, ensure_ascii=False)

    except Exception as e:
        error_msg = f"Error analyzing video: {str(e)}"
        logger.error("%s", error_msg, exc_info=True)

        err_str = str(e).lower()
        if any(hint in err_str for hint in (
            "402", "insufficient", "payment required", "credits", "billing",
        )):
            analysis = (
                "Insufficient credits or payment required. Please top up your "
                f"API provider account and try again. Error: {e}"
            )
        elif any(hint in err_str for hint in (
            "does not support", "not support video",
            "content_policy", "multimodal",
            "unrecognized request argument", "video input",
            "video_url",
        )):
            analysis = (
                f"The model does not support video analysis or the request was "
                f"rejected. Ensure you're using a video-capable model "
                f"(e.g. google/gemini-2.5-flash). Error: {e}"
            )
        elif any(hint in err_str for hint in (
            "too large", "payload", "413", "content_too_large",
            "request_too_large", "exceeds", "size limit",
        )):
            analysis = (
                "The video is too large for the API. Try compressing or trimming "
                f"the video (max ~50 MB). Error: {e}"
            )
        else:
            analysis = (
                "There was a problem with the request and the video could not "
                f"be analyzed. Error: {e}"
            )

        result = {
            "success": False,
            "error": error_msg,
            "analysis": analysis,
        }

        debug_call_data["error"] = error_msg
        _debug.log_call("video_analyze_tool", debug_call_data)
        _debug.save()

        return json.dumps(result, indent=2, ensure_ascii=False)

    finally:
        if should_cleanup and temp_video_path and temp_video_path.exists():
            try:
                temp_video_path.unlink()
                logger.debug("Cleaned up temporary video file")
            except Exception as cleanup_error:
                logger.warning(
                    "Could not delete temporary file: %s", cleanup_error, exc_info=True
                )


VIDEO_ANALYZE_SCHEMA = {
    "name": "video_analyze",
    "description": (
        "Analyze a video from a URL or local file path using a multimodal AI model. "
        "Sends the video to a video-capable model (e.g. Gemini) for understanding. "
        "Use this for video files — for images, use vision_analyze instead. "
        "Supports mp4, webm, mov, avi, mkv, mpeg formats. "
        "Note: large videos (>20 MB) may be slow; max ~50 MB."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "video_url": {
                "type": "string",
                "description": "Video URL (http/https) or local file path to analyze.",
            },
            "question": {
                "type": "string",
                "description": "Your specific question about the video. The AI will describe what happens in the video and answer your question.",
            },
        },
        "required": ["video_url", "question"],
    },
}


def _handle_video_analyze(args: Dict[str, Any], **kw: Any) -> Awaitable[str]:
    video_url = args.get("video_url", "")
    question = args.get("question", "")
    full_prompt = (
        "Fully describe and explain everything happening in this video, "
        "including visual content, motion, audio cues, text overlays, and scene "
        f"transitions. Then answer the following question:\n\n{question}"
    )
    model = os.getenv("AUXILIARY_VIDEO_MODEL", "").strip() or os.getenv("AUXILIARY_VISION_MODEL", "").strip() or None
    return video_analyze_tool(video_url, full_prompt, model)


registry.register(
    name="video_analyze",
    toolset="video",
    schema=VIDEO_ANALYZE_SCHEMA,
    handler=_handle_video_analyze,
    check_fn=check_vision_requirements,
    is_async=True,
    emoji="🎬",
)
