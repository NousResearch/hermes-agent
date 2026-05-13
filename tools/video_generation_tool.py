#!/usr/bin/env python3
"""
Video Generation Tool -- Text-to-Video via FAL.ai

Provides video generation from text prompts using FAL.ai's video models.
Currently supports:
  - Seedance 2.0 (ByteDance) -- cinematic, native audio, multi-shot
  - Kling v3 Pro -- cinematic visuals, fluid motion, native audio

Architecture mirrors image_generation_tool.py:
  - FAL_VIDEO_MODELS catalog with per-model metadata
  - _build_fal_video_payload() translates unified inputs to model-specific payloads
  - Async polling for FAL's video generation (queued → processing → completed)
  - Unified duration/aspect_ratio interface filtered per-model via supports whitelist
  - Managed gateway support for Nous subscribers (no FAL_KEY required)

Usage:
    from tools.video_generation_tool import video_generate_tool
    result = video_generate_tool(prompt="A cat walking in a garden", duration=5)
"""

import json
import logging
import os
import datetime
import threading
import time
from typing import Any, Dict, Optional

# fal_client is imported lazily — same pattern as image_generation_tool.py
fal_client: Any = None


def _load_fal_client() -> Any:
    """Lazily import fal_client and rebind the module global on first use."""
    global fal_client
    if fal_client is not None:
        return fal_client
    try:
        from tools.lazy_deps import ensure as _lazy_ensure
        _lazy_ensure("image.fal", prompt=False)
    except ImportError:
        pass
    except Exception as e:
        raise ImportError(str(e))
    import fal_client as _fal_client
    fal_client = _fal_client
    return fal_client


from tools.tool_backend_helpers import (
    fal_key_is_configured,
    managed_nous_tools_enabled,
    prefers_gateway,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Managed FAL Gateway (for Nous subscribers)
# ---------------------------------------------------------------------------
_managed_fal_client = None
_managed_fal_client_config = None
_managed_fal_client_lock = threading.Lock()


def _resolve_managed_fal_gateway():
    """Return managed fal-queue gateway config when the user prefers the gateway
    or doesn't have a direct FAL key configured.
    """
    if fal_key_is_configured() and not prefers_gateway("video_gen"):
        return None
    from tools.managed_tool_gateway import resolve_managed_tool_gateway
    return resolve_managed_tool_gateway("fal-queue")


class _ManagedFalSyncClient:
    """Small per-instance wrapper around fal_client.SyncClient for managed queue hosts."""

    def __init__(self, managed_gateway):
        _load_fal_client()
        sync_client_class = getattr(fal_client, "SyncClient", None)
        if sync_client_class is None:
            raise RuntimeError("fal_client.SyncClient is required for managed FAL gateway mode")

        client_module = getattr(fal_client, "client", None)
        if client_module is None:
            raise RuntimeError("fal_client.client is required for managed FAL gateway mode")

        self._gateway_origin = managed_gateway.gateway_origin
        self._nous_user_token = managed_gateway.nous_user_token
        self._client = sync_client_class(
            base_url=managed_gateway.gateway_origin,
            credentials=self._nous_user_token,
        )

    def submit(self, model, arguments, headers=None):
        request_headers = dict(headers or {})
        request_headers["Authorization"] = f"Key {self._nous_user_token}"
        return self._client.submit(model, arguments=arguments, headers=request_headers)


def _get_managed_fal_client(managed_gateway):
    global _managed_fal_client, _managed_fal_client_config
    client_config = (managed_gateway.gateway_origin, managed_gateway.nous_user_token)

    with _managed_fal_client_lock:
        if _managed_fal_client is not None and _managed_fal_client_config == client_config:
            return _managed_fal_client

        _managed_fal_client = _ManagedFalSyncClient(managed_gateway)
        _managed_fal_client_config = client_config
        return _managed_fal_client


def _submit_fal_request(model, arguments, headers=None):
    """Submit a FAL request — direct or via managed gateway."""
    managed_gateway = _resolve_managed_fal_gateway()

    if managed_gateway is None:
        _load_fal_client()
        return fal_client.submit(model, arguments=arguments, headers=headers)

    managed_client = _get_managed_fal_client(managed_gateway)
    return managed_client.submit(model, arguments=arguments, headers=headers)


# ---------------------------------------------------------------------------
# FAL video model catalog
# ---------------------------------------------------------------------------

FAL_VIDEO_MODELS: Dict[str, Dict[str, Any]] = {
    "bytedance/seedance-2.0/text-to-video": {
        "display": "Seedance 2.0 (ByteDance)",
        "speed": "~10s",
        "strengths": "Cinematic, native audio, multi-shot editing",
        "price": "$0.30/s (720p with audio)",
        "duration_range": (3, 15),
        "aspect_ratios": {
            "landscape": "16:9",
            "square": "1:1",
            "portrait": "9:16",
            "cinematic": "21:9",
        },
        "defaults": {
            "prompt": "",
            "duration": 5,
            "aspect_ratio": "16:9",
            "enable_safety_checker": False,
        },
        "supports": {
            "prompt", "duration", "aspect_ratio", "seed",
            "negative_prompt", "enable_safety_checker",
        },
    },
    "bytedance/seedance-2.0/fast/text-to-video": {
        "display": "Seedance 2.0 Fast (ByteDance)",
        "speed": "~5s",
        "strengths": "Lower latency, cost-effective",
        "price": "$0.24/s (720p fast with audio)",
        "duration_range": (3, 10),
        "aspect_ratios": {
            "landscape": "16:9",
            "square": "1:1",
            "portrait": "9:16",
        },
        "defaults": {
            "prompt": "",
            "duration": 5,
            "aspect_ratio": "16:9",
            "enable_safety_checker": False,
        },
        "supports": {
            "prompt", "duration", "aspect_ratio", "seed",
            "negative_prompt", "enable_safety_checker",
        },
    },
    "kling-video/v3/pro/text-to-video": {
        "display": "Kling v3 Pro",
        "speed": "~30s",
        "strengths": "Professional-grade output with native audio",
        "price": "$0.17/s (with audio)",
        "duration_range": (5, 10),
        "aspect_ratios": {
            "landscape": "16:9",
            "portrait": "9:16",
        },
        "defaults": {
            "prompt": "",
            "duration": 5,
            "aspect_ratio": "16:9",
        },
        "supports": {
            "prompt", "duration", "aspect_ratio", "seed",
        },
    },
}

DEFAULT_VIDEO_MODEL = "bytedance/seedance-2.0/text-to-video"
VALID_ASPECT_RATIOS = frozenset({"16:9", "9:16", "1:1", "4:3", "3:4", "21:9"})


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------
def _resolve_fal_video_model() -> tuple[str, Dict[str, Any]]:
    """Resolve the active video model from config or env."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        video_cfg = cfg.get("video_gen", {}) if isinstance(cfg, dict) else {}
    except Exception:
        video_cfg = {}

    model_id = os.environ.get("FAL_VIDEO_MODEL") or video_cfg.get("model", DEFAULT_VIDEO_MODEL)

    if model_id not in FAL_VIDEO_MODELS:
        logger.warning(
            "Unknown FAL video model '%s' in config; falling back to %s",
            model_id, DEFAULT_VIDEO_MODEL,
        )
        return DEFAULT_VIDEO_MODEL, FAL_VIDEO_MODELS[DEFAULT_VIDEO_MODEL]

    return model_id, FAL_VIDEO_MODELS[model_id]


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------
def _build_fal_video_payload(
    model_id: str,
    prompt: str,
    duration: int = 5,
    aspect_ratio: str = "16:9",
    seed: Optional[int] = None,
    negative_prompt: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a FAL video request payload from unified inputs."""
    meta = FAL_VIDEO_MODELS[model_id]

    aspect_map = meta.get("aspect_ratios", {})
    aspect_lc = (aspect_ratio or "16:9").lower().strip()

    # If it's a direct ratio value (e.g., "9:16"), use it directly
    if aspect_lc in VALID_ASPECT_RATIOS and aspect_lc in {v for v in aspect_map.values()}:
        model_aspect = aspect_lc
    elif aspect_lc in aspect_map:
        # It's a unified name (e.g., "portrait")
        model_aspect = aspect_map[aspect_lc]
    else:
        # Fallback to landscape
        model_aspect = aspect_map.get("landscape", "16:9")

    payload: Dict[str, Any] = dict(meta.get("defaults", {}))
    payload["prompt"] = (prompt or "").strip()
    payload["duration"] = duration
    payload["aspect_ratio"] = model_aspect

    if seed is not None and isinstance(seed, int):
        payload["seed"] = seed
    if negative_prompt and isinstance(negative_prompt, str):
        payload["negative_prompt"] = negative_prompt.strip()

    if overrides:
        for k, v in overrides.items():
            if v is not None:
                payload[k] = v

    supports = meta["supports"]
    return {k: v for k, v in payload.items() if k in supports}


# ---------------------------------------------------------------------------
# Async submission + polling
# ---------------------------------------------------------------------------
def _poll_video_result(handler, timeout: int = 120, poll_interval: float = 2.0) -> Dict[str, Any]:
    """Poll FAL video generation until complete or timeout."""
    start = time.monotonic()

    while True:
        elapsed = time.monotonic() - start
        if elapsed > timeout:
            raise TimeoutError(f"Video generation timed out after {timeout}s")

        try:
            status_response = handler.status()
            status = status_response.get("status", "unknown") if isinstance(status_response, dict) else "unknown"

            if status == "COMPLETED":
                return handler.get()
            elif status in ("FAILED", "ERROR"):
                error_msg = status_response.get("error", "Unknown error") if isinstance(status_response, dict) else "Unknown error"
                raise RuntimeError(f"Video generation failed: {error_msg}")
            elif status in ("IN_QUEUE", "IN_PROGRESS", "PROCESSING"):
                time.sleep(poll_interval)
                continue
            else:
                time.sleep(poll_interval)
                continue

        except Exception as e:
            if "not found" in str(e).lower() or "404" in str(e):
                try:
                    return handler.get()
                except Exception:
                    raise
            raise


# ---------------------------------------------------------------------------
# Tool entry point
# ---------------------------------------------------------------------------
def video_generate_tool(
    prompt: str,
    duration: int = 5,
    aspect_ratio: str = "16:9",
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
) -> str:
    """Generate a video from a text prompt using the configured FAL video model.

    Parameters:
        prompt: Text description of the video to generate (required)
        duration: Video length in seconds (default 5, model-dependent range)
        aspect_ratio: Output aspect ratio (default "16:9")
        negative_prompt: Things to avoid in the video (optional)
        seed: Random seed for reproducibility (optional)

    Returns a JSON string with:
        {"success": bool, "video": {"url": str, "duration": int}, "error": str}
    """
    model_id, meta = _resolve_fal_video_model()

    start_time = datetime.datetime.now()

    try:
        # Validation
        if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
            raise ValueError("Prompt is required and must be a non-empty string")

        # Check authentication: FAL_KEY or managed gateway
        if not (fal_key_is_configured() or _resolve_managed_fal_gateway()):
            message = "FAL_KEY environment variable not set"
            if managed_nous_tools_enabled():
                message += " and managed FAL gateway is unavailable"
            raise ValueError(message)

        # Duration validation
        min_dur, max_dur = meta.get("duration_range", (3, 15))
        if not isinstance(duration, int) or duration < min_dur or duration > max_dur:
            raise ValueError(f"Duration must be between {min_dur} and {max_dur} seconds for this model (got {duration})")

        # Aspect ratio validation
        aspect_lc = (aspect_ratio or "16:9").lower().strip()
        if aspect_lc not in VALID_ASPECT_RATIOS:
            logger.warning(
                "Invalid aspect_ratio '%s', defaulting to '16:9'",
                aspect_ratio,
            )
            aspect_lc = "16:9"

        arguments = _build_fal_video_payload(
            model_id, prompt, duration=duration, aspect_ratio=aspect_lc,
            seed=seed, negative_prompt=negative_prompt,
        )

        logger.info(
            "Generating video with %s (%s) — prompt: %s, duration: %ds",
            meta.get("display", model_id), model_id, prompt[:80], duration,
        )

        # Submit request
        handler = _submit_fal_request(model_id, arguments=arguments)

        # Poll for completion
        timeout = max(meta.get("duration_range", (3, 15))[1] * 3, 60)
        result = _poll_video_result(handler, timeout=timeout)

        generation_time = (datetime.datetime.now() - start_time).total_seconds()

        if not result or "video" not in result and "url" not in result:
            raise ValueError("Invalid response from FAL.ai API — no video returned")

        video_data = result.get("video", result)
        video_url = video_data.get("url") if isinstance(video_data, dict) else result.get("url")

        if not video_url:
            raise ValueError("No video URL in response")

        return json.dumps({
            "success": True,
            "video": {
                "url": video_url,
                "duration": duration,
                "model": meta.get("display", model_id),
                "generation_time_seconds": round(generation_time, 2),
            },
        }, ensure_ascii=False)

    except Exception as e:
        generation_time = (datetime.datetime.now() - start_time).total_seconds()
        logger.error("Video generation failed: %s", e, exc_info=True)
        return json.dumps({
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------
def check_requirements() -> bool:
    """Return True if FAL.ai is configured for video generation."""
    return bool(fal_key_is_configured() or _resolve_managed_fal_gateway())


from tools.registry import registry

registry.register(
    name="video_generate",
    toolset="video",
    schema={
        "type": "function",
        "function": {
            "name": "video_generate",
            "description": (
                "Generate a video from a text prompt using AI. "
                "Creates a short video clip (3-15 seconds) based on your description. "
                "Specify duration (seconds) and aspect ratio. "
                "Returns a URL to the generated video file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Detailed description of the video to generate. "
                            "Include subject, action, setting, lighting, camera movement, and style. "
                            "Example: 'A golden retriever running through a sunlit meadow, slow motion, cinematic lighting'"
                        ),
                    },
                    "duration": {
                        "type": "integer",
                        "description": "Video length in seconds (3-15, model-dependent). Default: 5",
                        "default": 5,
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "description": "Output aspect ratio. Options: 16:9 (landscape), 9:16 (portrait), 1:1 (square), 4:3, 3:4, 21:9 (cinematic). Default: 16:9",
                        "default": "16:9",
                        "enum": ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],
                    },
                    "negative_prompt": {
                        "type": "string",
                        "description": "Optional: Describe what to avoid in the video (e.g., 'blurry, low quality, text, watermark')",
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Optional: Random seed for reproducible results",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    handler=lambda args, **kw: video_generate_tool(
        prompt=args.get("prompt", ""),
        duration=args.get("duration", 5),
        aspect_ratio=args.get("aspect_ratio", "16:9"),
        negative_prompt=args.get("negative_prompt"),
        seed=args.get("seed"),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_requirements,
    requires_env=["FAL_KEY"],
)
