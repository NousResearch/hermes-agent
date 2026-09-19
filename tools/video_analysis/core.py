"""Video analysis transport negotiation and frame fallback.

The public video tool remains re-exported from :mod:`tools.vision_tools` for
backward compatibility, while the implementation lives here so the image and
video paths can evolve independently.

Providers disagree on chat-content dialects for native video. Hermes keeps the
existing OpenAI-style ``video_url`` request as the first attempt, retries the
llama.cpp ``input_video`` shape only when the provider rejects the first wire
format, then falls back to bounded chronological JPEG frames for image-only
endpoints. A provider's actual response authorizes each transition; endpoint
location or a guessed model name never substitutes for capability evidence.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Awaitable, Dict, Optional, Sequence

from hermes_constants import get_hermes_dir
from tools.registry import tool_error
from tools.video_analysis.frames import (
    VideoFrame,
    VideoFrameExtractionError,
    extract_video_frames as _extract_video_frames,
    extract_video_frames_async as _extract_video_frames_async,
)
from tools.website_policy import check_website_access

logger = logging.getLogger(__name__)

# Extension -> MIME. avi/mkv retain the historical mp4 fallback.
_VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/mov",
    ".avi": "video/mp4",
    ".mkv": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
}

_MAX_VIDEO_BASE64_BYTES = 50 * 1024 * 1024
# Keep network and local inputs below the largest raw payload that can fit in
# the 50 MiB data-URL budget after base64 expansion and a MIME prefix.
_MAX_VIDEO_DATA_PREFIX_BUDGET = 64
_MAX_VIDEO_RAW_BYTES = (
    (_MAX_VIDEO_BASE64_BYTES - _MAX_VIDEO_DATA_PREFIX_BUDGET) // 4
) * 3
_VIDEO_SIZE_WARN_BYTES = 20 * 1024 * 1024


class VideoInputNegotiationError(RuntimeError):
    """Raised after every authorized video transport has failed."""


def _vision_module():
    """Return the compatibility module without importing it during bootstrap."""
    from tools import vision_tools

    return vision_tools


def _auxiliary_callables():
    """Reuse vision_tools' lazy auxiliary-client seam and patch targets."""
    vision_tools = _vision_module()
    vision_tools._load_auxiliary_client()
    return vision_tools.async_call_llm, vision_tools.extract_content_or_reasoning


def _debug_session():
    """Keep video calls in the existing vision debug session."""
    return _vision_module()._debug


def _detect_video_mime_type(video_path: Path) -> Optional[str]:
    """Return a video MIME type based on file extension, or None if unsupported."""
    return _VIDEO_MIME_TYPES.get(video_path.suffix.lower())


def _max_raw_video_bytes(mime_type: str) -> int:
    prefix_bytes = len(f"data:{mime_type};base64,".encode("ascii"))
    payload_budget = _MAX_VIDEO_BASE64_BYTES - prefix_bytes
    if payload_budget <= 0:
        return 0
    return (payload_budget // 4) * 3


def _video_to_base64_data_url(
    video_path: Path, mime_type: Optional[str] = None
) -> str:
    """Convert a bounded video file to a base64-encoded data URL."""
    mime = mime_type or _VIDEO_MIME_TYPES.get(
        video_path.suffix.lower(), "video/mp4"
    )
    max_raw_bytes = _max_raw_video_bytes(mime)
    raw_size = video_path.stat().st_size
    if raw_size > max_raw_bytes:
        raise ValueError(
            "Video too large for API: raw payload is "
            f"{raw_size / (1024 * 1024):.1f} MB "
            f"(max {max_raw_bytes / (1024 * 1024):.1f} MB before base64). "
            "Compress or trim the video and retry."
        )

    data = video_path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    data_url = f"data:{mime};base64,{encoded}"
    if len(data_url.encode("ascii")) > _MAX_VIDEO_BASE64_BYTES:
        raise ValueError(
            "Video too large for API after base64 encoding: payload exceeds "
            f"{_MAX_VIDEO_BASE64_BYTES / (1024 * 1024):.0f} MB. "
            "Compress or trim the video and retry."
        )
    return data_url


def _terminal_backend_is_local() -> bool:
    backend = os.getenv("TERMINAL_ENV", "local").strip().lower()
    return backend in ("", "local")


def _is_path_like_video_source(value: str) -> bool:
    lowered = (value or "").strip().lower()
    if not lowered:
        return False
    return not lowered.startswith(("http://", "https://", "data:"))


async def _materialize_video_from_terminal_backend(
    video_source: str, task_id: Optional[str]
) -> Path:
    """Read a sandbox path through the shared media resolver into a temp file."""
    from tools.image_source import (
        ImageResolutionError,
        ResolveContext,
        resolve_image_source,
    )

    source = video_source
    if source.startswith("file://"):
        source = source[len("file://") :]
    suffix = Path(source).suffix.lower()
    if suffix not in _VIDEO_MIME_TYPES:
        raise ValueError(
            f"Unsupported video format: '{suffix}'. "
            f"Supported: {', '.join(sorted(_VIDEO_MIME_TYPES.keys()))}"
        )

    try:
        resolved = await resolve_image_source(
            video_source,
            ResolveContext(task_id=task_id),
            permitted=("video",),
        )
    except ImageResolutionError as exc:
        raise ValueError(
            f"Could not read video from terminal backend: {exc}"
        ) from exc

    temp_dir = get_hermes_dir("cache/video", "temp_video_files")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"terminal_video_{uuid.uuid4()}{suffix}"
    temp_path.write_bytes(resolved.data)
    return temp_path


async def _download_video(
    video_url: str, destination: Path, max_retries: int = 3
) -> Path:
    """Download video from URL with website policy, SSRF, and byte caps."""
    from tools.url_safety import (
        async_is_safe_url,
        create_ssrf_safe_async_client,
        redirect_target_from_response,
    )

    # Shared implementation keeps the stream cap and partial-file cleanup in
    # one place without importing vision_tools at module-import time.
    stream_download = _vision_module()._stream_download_to_file

    destination.parent.mkdir(parents=True, exist_ok=True)

    async def _ssrf_redirect_guard(response):
        redirect_url = redirect_target_from_response(response)
        if redirect_url and not await async_is_safe_url(redirect_url):
            raise ValueError(
                f"Blocked redirect to private/internal address: {redirect_url}"
            )

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            blocked = check_website_access(video_url)
            if blocked:
                raise PermissionError(blocked["message"])

            async with create_ssrf_safe_async_client(
                timeout=60.0,
                follow_redirects=True,
                event_hooks={"response": [_ssrf_redirect_guard]},
            ) as client:
                await stream_download(
                    client,
                    video_url,
                    destination,
                    _MAX_VIDEO_RAW_BYTES,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/120.0.0.0 Safari/537.36"
                        ),
                        "Accept": "video/*,*/*;q=0.8",
                    },
                    media_label="Video",
                )
            return destination
        except Exception as exc:  # noqa: BLE001 - retain provider diagnostics
            last_error = exc
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                logger.warning(
                    "Video download failed (attempt %s/%s): %s",
                    attempt + 1,
                    max_retries,
                    str(exc)[:80],
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(
                    "Video download failed after %s attempts: %s",
                    max_retries,
                    str(exc)[:160],
                    exc_info=True,
                )

    if last_error is None:
        raise RuntimeError(
            "_download_video exited retry loop without attempting "
            f"(max_retries={max_retries})"
        )
    raise last_error


def _raw_base64_from_data_url(video_data_url: str) -> str:
    header, separator, payload = video_data_url.partition(",")
    if not separator or ";base64" not in header:
        raise ValueError("video payload is not a base64 data URL")
    return payload


def _video_url_messages(user_prompt: str, video_data_url: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "video_url",
                    "video_url": {"url": video_data_url},
                },
            ],
        }
    ]


def _input_video_messages(
    user_prompt: str, video_data_url: str
) -> list[dict[str, Any]]:
    # llama.cpp's OpenAI-compatible server expects raw base64 here, not a data
    # URL. This transport is attempted only after the provider rejects the
    # generic video_url dialect, so existing providers keep their wire shape.
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "input_video",
                    "input_video": {
                        "data": _raw_base64_from_data_url(video_data_url)
                    },
                },
            ],
        }
    ]


def _frame_messages(
    user_prompt: str, frames: Sequence[VideoFrame]
) -> list[dict[str, Any]]:
    sample_times = ", ".join(
        f"{frame.timestamp_seconds:.2f}s" for frame in frames
    )
    evidence_note = (
        f"{user_prompt}\n\n"
        "Native video content was rejected by the provider. The following "
        f"{len(frames)} JPEG frames were sampled in chronological order at "
        f"{sample_times}. Use only visible evidence from these samples. Do not "
        "claim audio content, continuous motion, or events between samples "
        "unless the frames themselves support the inference."
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": evidence_note}]
    content.extend(
        {
            "type": "image_url",
            "image_url": {"url": frame.data_url},
        }
        for frame in frames
    )
    return [{"role": "user", "content": content}]


def _status_code(exc: Exception) -> Optional[int]:
    for candidate in (
        getattr(exc, "status_code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        if isinstance(candidate, int):
            return candidate
    match = re.match(r"^\s*(?:http\s*)?(\d{3})\b", str(exc), re.IGNORECASE)
    return int(match.group(1)) if match else None


def _error_text(exc: Exception) -> str:
    return str(exc).strip().lower()


def _compact_error(exc: Exception) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    text = re.sub(
        r"data:video/[^,\s]+,[a-zA-Z0-9+/=]+",
        "<video data omitted>",
        text,
    )
    text = re.sub(r"\s+", " ", text)
    return text[:280]


def _is_wire_dialect_rejection(exc: Exception) -> bool:
    text = _error_text(exc)
    return any(
        hint in text
        for hint in (
            "video_url",
            "input_video",
            "unsupported content[].type",
            "unsupported content type",
            "unknown content type",
            "unrecognized content type",
            "invalid content type",
            "content type is not supported",
            "expected content type",
            "unknown variant",
        )
    )


def _is_video_capability_rejection(exc: Exception) -> bool:
    text = _error_text(exc)
    return any(
        hint in text
        for hint in (
            "does not support video",
            "doesn't support video",
            "not support video",
            "video input is not supported",
            "video inputs are not supported",
            "unsupported video input",
            "video modality",
            "image input only",
            "only supports image",
            "only support image",
            "multimodal input",
        )
    )


def _is_fatal_request_error(exc: Exception) -> bool:
    status = _status_code(exc)
    if status in {401, 403, 404, 408, 409, 413, 429} or (
        status is not None and status >= 500
    ):
        return True
    text = _error_text(exc)
    return any(
        hint in text
        for hint in (
            "insufficient credits",
            "payment required",
            "billing",
            "authentication",
            "unauthorized",
            "forbidden",
            "rate limit",
            "too many requests",
            "timed out",
            "timeout",
            "connection error",
            "connection refused",
            "payload too large",
            "request too large",
            "content_too_large",
            "context length",
            "model not found",
        )
    )


def _is_ambiguous_bad_request(exc: Exception) -> bool:
    if _is_fatal_request_error(exc):
        return False
    status = _status_code(exc)
    if status in {400, 422}:
        return True
    text = _error_text(exc)
    return "bad request" in text or "invalid_request" in text


def _should_try_input_video(exc: Exception) -> bool:
    if _is_fatal_request_error(exc):
        return False
    if _is_wire_dialect_rejection(exc):
        return True
    if _is_video_capability_rejection(exc):
        return False
    return _is_ambiguous_bad_request(exc)


def _should_try_frames(exc: Exception) -> bool:
    if _is_fatal_request_error(exc):
        return False
    return (
        _is_wire_dialect_rejection(exc)
        or _is_video_capability_rejection(exc)
        or _is_ambiguous_bad_request(exc)
    )


def _attempt_summary(attempts: Sequence[tuple[str, Exception]]) -> str:
    return "; ".join(
        f"{transport}: {_compact_error(exc)}" for transport, exc in attempts
    )


async def _call_with_video_negotiation(
    *,
    async_call_llm,
    base_call_kwargs: dict[str, Any],
    user_prompt: str,
    video_data_url: str,
    video_path: Path,
) -> tuple[Any, list[dict[str, Any]], str, int]:
    """Call the provider through a bounded, evidence-driven transport ladder."""
    attempts: list[tuple[str, Exception]] = []

    video_url_messages = _video_url_messages(user_prompt, video_data_url)
    try:
        response = await async_call_llm(
            messages=video_url_messages, **base_call_kwargs
        )
        return response, video_url_messages, "video_url", 0
    except Exception as exc:  # noqa: BLE001 - provider error is classified below
        attempts.append(("video_url", exc))
        if not _should_try_input_video(exc) and not _should_try_frames(exc):
            raise
        first_error = exc

    last_error = first_error
    if _should_try_input_video(first_error):
        input_video_messages = _input_video_messages(user_prompt, video_data_url)
        logger.info(
            "Provider rejected video_url; retrying llama.cpp input_video dialect"
        )
        try:
            response = await async_call_llm(
                messages=input_video_messages, **base_call_kwargs
            )
            return response, input_video_messages, "input_video", 0
        except Exception as exc:  # noqa: BLE001 - classified below
            attempts.append(("input_video", exc))
            last_error = exc
            if not _should_try_frames(exc):
                raise VideoInputNegotiationError(
                    "The provider rejected the native video_url dialect, then "
                    "the input_video retry failed with a non-negotiable error. "
                    + _attempt_summary(attempts)
                ) from exc

    if not _should_try_frames(last_error):
        raise VideoInputNegotiationError(
            "No safe video transport fallback was authorized. "
            + _attempt_summary(attempts)
        ) from last_error

    logger.info(
        "Native video dialects were rejected; extracting chronological JPEG frames"
    )
    try:
        frames = await _extract_video_frames_async(video_path)
    except VideoFrameExtractionError as exc:
        raise VideoInputNegotiationError(
            "The provider rejected native video content and frame fallback "
            f"could not be produced: {exc}. Native attempts: "
            + _attempt_summary(attempts)
        ) from exc

    frame_messages = _frame_messages(user_prompt, frames)
    try:
        response = await async_call_llm(
            messages=frame_messages, **base_call_kwargs
        )
        return response, frame_messages, "image_frames", len(frames)
    except Exception as exc:  # noqa: BLE001 - include final provider receipt
        attempts.append(("image_frames", exc))
        raise VideoInputNegotiationError(
            "The provider rejected both native video dialects and the "
            "chronological JPEG-frame fallback. " + _attempt_summary(attempts)
        ) from exc


def _resolve_video_call_settings() -> tuple[float, float]:
    timeout = 180.0
    temperature = 0.1
    try:
        from hermes_cli.config import cfg_get, load_config

        cfg = load_config()
        video_cfg = cfg_get(cfg, "auxiliary", "video", default={}) or {}
        vision_cfg = cfg_get(cfg, "auxiliary", "vision", default={}) or {}
        configured_timeout = video_cfg.get("timeout", vision_cfg.get("timeout"))
        if configured_timeout is not None:
            timeout = float(configured_timeout)
        configured_temperature = video_cfg.get(
            "temperature", vision_cfg.get("temperature")
        )
        if configured_temperature is not None:
            temperature = float(configured_temperature)
    except Exception:  # noqa: BLE001 - config fallback is intentionally soft
        pass
    return timeout, temperature


async def video_analyze_tool(
    video_url: str,
    user_prompt: str,
    model: str = None,
    task_id: Optional[str] = None,
) -> str:
    """Analyze video with native-dialect negotiation and real-frame fallback."""
    if not isinstance(user_prompt, str):
        user_prompt = str(user_prompt) if user_prompt is not None else ""
    debug_call_data: dict[str, Any] = {
        "parameters": {
            "video_url": video_url,
            "user_prompt": (
                user_prompt[:200] + "..."
                if len(user_prompt) > 200
                else user_prompt
            ),
            "model": model,
        },
        "error": None,
        "success": False,
        "analysis_length": 0,
        "model_used": model,
        "video_size_bytes": 0,
        "transport": None,
        "frame_count": 0,
    }

    temp_video_path: Optional[Path] = None
    should_cleanup = True

    try:
        from tools.interrupt import is_interrupted

        if is_interrupted():
            return tool_error("Interrupted", success=False)

        logger.info("Analyzing video: %s", video_url[:60])
        logger.info("User prompt: %s", user_prompt[:100])

        resolved_url = video_url
        if resolved_url.startswith("file://"):
            resolved_url = resolved_url[len("file://") :]
        local_path = Path(os.path.expanduser(resolved_url))

        if not _terminal_backend_is_local() and _is_path_like_video_source(
            video_url
        ):
            logger.info("Reading video source via terminal backend: %s", video_url)
            temp_video_path = await _materialize_video_from_terminal_backend(
                video_url, task_id
            )
            should_cleanup = True
        elif local_path.is_file():
            from agent.file_safety import raise_if_read_blocked

            raise_if_read_blocked(str(local_path))
            logger.info("Using local video file: %s", video_url)
            temp_video_path = local_path
            should_cleanup = False
        elif await _vision_module()._validate_image_url_async(video_url):
            blocked = check_website_access(video_url)
            if blocked:
                raise PermissionError(blocked["message"])
            temp_dir = get_hermes_dir("cache/video", "temp_video_files")
            temp_video_path = temp_dir / f"temp_video_{uuid.uuid4()}.mp4"
            await _download_video(video_url, temp_video_path)
            should_cleanup = True
        else:
            raise ValueError(
                "Invalid video source. Provide an HTTP/HTTPS URL or a valid "
                "local file path."
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
            logger.warning(
                "Video is %.1f MB - may be slow or rejected", video_size_mb
            )

        video_data_url = await asyncio.to_thread(
            _video_to_base64_data_url,
            temp_video_path,
            detected_mime,
        )
        data_size_mb = len(video_data_url) / (1024 * 1024)
        if len(video_data_url) > _MAX_VIDEO_BASE64_BYTES:
            raise ValueError(
                f"Video too large for API: base64 payload is {data_size_mb:.1f} MB "
                f"(limit {_MAX_VIDEO_BASE64_BYTES / (1024 * 1024):.0f} MB). "
                "Compress or trim the video and retry."
            )

        debug_call_data["video_size_bytes"] = video_size_bytes
        timeout, temperature = _resolve_video_call_settings()
        base_call_kwargs: dict[str, Any] = {
            "task": "vision",
            "temperature": temperature,
            "timeout": timeout,
        }
        if model:
            base_call_kwargs["model"] = model

        async_call_llm, extract_content_or_reasoning = _auxiliary_callables()
        response, used_messages, transport, frame_count = (
            await _call_with_video_negotiation(
                async_call_llm=async_call_llm,
                base_call_kwargs=base_call_kwargs,
                user_prompt=user_prompt,
                video_data_url=video_data_url,
                video_path=temp_video_path,
            )
        )
        analysis = extract_content_or_reasoning(response)

        if not analysis:
            logger.warning(
                "Empty video response over %s, retrying once", transport
            )
            response = await async_call_llm(
                messages=used_messages, **base_call_kwargs
            )
            analysis = extract_content_or_reasoning(response)

        analysis_length = len(analysis) if analysis else 0
        logger.info(
            "Video analysis completed via %s (%s characters)",
            transport,
            analysis_length,
        )

        result = {
            "success": True,
            "analysis": analysis
            or (
                "There was a problem with the request and the video could not "
                "be analyzed."
            ),
            "transport": transport,
        }
        if frame_count:
            result["frame_count"] = frame_count

        debug_call_data["success"] = True
        debug_call_data["analysis_length"] = analysis_length
        debug_call_data["transport"] = transport
        debug_call_data["frame_count"] = frame_count
        debug = _debug_session()
        debug.log_call("video_analyze_tool", debug_call_data)
        debug.save()
        return json.dumps(result, indent=2, ensure_ascii=False)

    except Exception as exc:  # noqa: BLE001 - tool returns structured errors
        error_msg = f"Error analyzing video: {exc}"
        logger.error("%s", error_msg, exc_info=True)

        err_str = str(exc).lower()
        if isinstance(exc, VideoInputNegotiationError):
            analysis = str(exc)
        elif any(
            hint in err_str
            for hint in (
                "402",
                "insufficient",
                "payment required",
                "credits",
                "billing",
            )
        ):
            analysis = (
                "Insufficient credits or payment required. Please top up your "
                f"API provider account and try again. Error: {exc}"
            )
        elif any(
            hint in err_str
            for hint in (
                "too large",
                "payload",
                "413",
                "content_too_large",
                "request_too_large",
                "exceeds",
                "size limit",
            )
        ):
            analysis = (
                "The video is too large for the API. Try compressing or "
                f"trimming the video (raw max ~{_MAX_VIDEO_RAW_BYTES / (1024 * 1024):.1f} MB). "
                f"Error: {exc}"
            )
        elif _is_video_capability_rejection(exc) or _is_wire_dialect_rejection(
            exc
        ):
            analysis = (
                "The provider rejected the video payload before a compatible "
                "transport could settle. Error: "
                f"{exc}"
            )
        else:
            analysis = (
                "There was a problem with the request and the video could not "
                f"be analyzed. Error: {exc}"
            )

        result = {
            "success": False,
            "error": error_msg,
            "analysis": analysis,
        }
        debug_call_data["error"] = error_msg
        debug = _debug_session()
        debug.log_call("video_analyze_tool", debug_call_data)
        debug.save()
        return json.dumps(result, indent=2, ensure_ascii=False)

    finally:
        if should_cleanup and temp_video_path and temp_video_path.exists():
            try:
                temp_video_path.unlink()
                logger.debug("Cleaned up temporary video file")
            except Exception as cleanup_error:  # noqa: BLE001 - best effort
                logger.warning(
                    "Could not delete temporary file: %s",
                    cleanup_error,
                    exc_info=True,
                )


VIDEO_ANALYZE_SCHEMA = {
    "name": "video_analyze",
    "description": (
        "Analyze a video from a URL or local file path using a multimodal AI "
        "model. Negotiates native provider video formats and falls back to "
        "chronological image frames when the endpoint is image-only. Use this "
        "for video files; for images, use vision_analyze instead. Supports "
        "mp4, webm, mov, avi, mkv, and mpeg formats. Large videos (>20 MB) "
        f"may be slow; raw input max ~{_MAX_VIDEO_RAW_BYTES / (1024 * 1024):.1f} MB "
        "under the 50 MB encoded-payload cap."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "video_url": {
                "type": "string",
                "description": (
                    "Video URL (http/https) or local file path to analyze."
                ),
            },
            "question": {
                "type": "string",
                "description": (
                    "Your specific question about the video. The AI will "
                    "describe what happens and answer the question."
                ),
            },
        },
        "required": ["video_url", "question"],
    },
}


def _resolve_video_model() -> Optional[str]:
    model = None
    try:
        from hermes_cli.config import cfg_get, load_config

        cfg = load_config()
        configured = cfg_get(
            cfg, "auxiliary", "video", "model"
        ) or cfg_get(cfg, "auxiliary", "vision", "model")
        if configured:
            model = str(configured).strip() or None
    except Exception:  # noqa: BLE001 - env fallback is intentional
        pass
    if not model:
        model = (
            os.getenv("AUXILIARY_VIDEO_MODEL", "").strip()
            or os.getenv("AUXILIARY_VISION_MODEL", "").strip()
            or None
        )
    return model


def _handle_video_analyze(
    args: Dict[str, Any], **kw: Any
) -> Awaitable[str]:
    video_url = args.get("video_url", "")
    question = args.get("question", "")
    full_prompt = (
        "Fully describe and explain everything happening in this video, "
        "including visual content, motion, audio cues, text overlays, and scene "
        f"transitions. Then answer the following question:\n\n{question}"
    )
    # Resolve through the public module so existing callers/tests that patch
    # tools.vision_tools.video_analyze_tool keep working after the shard.
    public_tool = _vision_module().video_analyze_tool
    return public_tool(
        video_url,
        full_prompt,
        _resolve_video_model(),
        task_id=kw.get("task_id"),
    )
