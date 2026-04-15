"""
Video Generation Tool — Lumenfall Provider

Generates videos using Lumenfall's unified API, which routes to providers
like Google Veo, Wan, Kling, and more with automatic fallback.

Video generation is asynchronous: the tool submits a job and polls for
completion, blocking until the video is ready or a timeout is reached.

Available tool:
  lumenfall_video_generate — Generate videos from text prompts via Lumenfall

Requires:
  LUMENFALL_API_KEY environment variable (get one at https://lumenfall.ai)

Optional:
  LUMENFALL_BASE_URL — Override the API base URL (default: https://api.lumenfall.ai/openai/v1)
"""

import json
import logging
import datetime
from typing import Optional

from tools.lumenfall_client import (
    check_lumenfall_available,
    submit_video,
    poll_video,
    LumenfallAuthError,
    LumenfallBalanceError,
    LumenfallError,
)

logger = logging.getLogger(__name__)

# Aspect ratio mapping — same as image tool for consistency
ASPECT_RATIO_MAP = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
}

# Default video duration in seconds
DEFAULT_DURATION = 5

# Maximum time to wait for video generation (seconds)
MAX_POLL_WAIT = 600  # 10 minutes


def lumenfall_video_generate_tool(
    prompt: str,
    model: Optional[str] = None,
    duration: float = DEFAULT_DURATION,
    aspect_ratio: str = "landscape",
    image_url: Optional[str] = None,
) -> str:
    """Generate a video from a text prompt using Lumenfall.

    Submits an async job and polls until completion. Uses synchronous HTTP
    to avoid event-loop issues in threaded gateway contexts.

    Args:
        prompt: Text prompt describing the desired video.
        model: Video model to use (e.g. "wan-2.7-pro", "sora-2-pro", "kling-v3").
               If not specified, the server picks the best available default.
        duration: Desired video duration in seconds (default: 5).
        aspect_ratio: "landscape" (16:9), "square" (1:1), or "portrait" (9:16).
        image_url: Optional URL of a source image for image-to-video generation.
                   When provided, the video animates from this image.

    Returns:
        JSON string with {"success": bool, "video_url": str|null, ...}
    """
    start_time = datetime.datetime.now()

    try:
        # Validate prompt
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt is required and must be a non-empty string")

        # Validate aspect_ratio
        aspect_ratio_lower = aspect_ratio.lower().strip() if aspect_ratio else "landscape"
        if aspect_ratio_lower not in ASPECT_RATIO_MAP:
            logger.warning(
                "Invalid aspect_ratio '%s', defaulting to 'landscape'", aspect_ratio
            )
            aspect_ratio_lower = "landscape"

        ratio = ASPECT_RATIO_MAP[aspect_ratio_lower]

        # Validate duration
        if not isinstance(duration, (int, float)) or duration <= 0:
            duration = DEFAULT_DURATION
        duration = min(duration, 60)  # cap at 60s

        logger.info(
            "Lumenfall video generation: model=%s, duration=%.1fs, ratio=%s, prompt=%.80s",
            model or "(default)",
            duration,
            ratio,
            prompt,
        )

        # Step 1: Submit async video job
        submit_result = submit_video(
            prompt=prompt,
            model=model,
            seconds=duration,
            aspect_ratio=ratio,
            image_url=image_url,
        )

        video_id = submit_result.get("id")
        if not video_id:
            raise LumenfallError("No video ID returned from submit", error_code="no_id")

        submit_metadata = submit_result.get("metadata", {})
        logger.info(
            "Video job submitted: id=%s, model=%s, provider=%s",
            video_id,
            submit_metadata.get("executed_model", "unknown"),
            submit_metadata.get("provider_name", "unknown"),
        )

        # Step 2: Poll until completion
        final = poll_video(
            video_id=video_id,
            poll_interval=5,
            max_wait=MAX_POLL_WAIT,
        )

        generation_time = (datetime.datetime.now() - start_time).total_seconds()

        status = final.get("status", "")
        if status == "failed":
            error_obj = final.get("error", {})
            error_msg = error_obj.get("message", "Video generation failed")
            raise LumenfallError(
                f"Video generation failed: {error_msg}",
                error_code=error_obj.get("code", "generation_failed"),
            )

        if status != "completed":
            raise LumenfallError(
                f"Unexpected final status: {status}",
                error_code="unexpected_status",
            )

        # Extract video URL
        output = final.get("output", {})
        video_url = output.get("url") if output else None
        if not video_url:
            raise LumenfallError(
                "Video completed but no URL in response",
                error_code="missing_url",
            )

        # Extract metadata
        metadata = final.get("metadata", {})
        provider = metadata.get("provider_name", "unknown")
        executed_model = metadata.get("executed_model", model or "default")
        cost = metadata.get("cost")

        logger.info(
            "Video generated in %.1fs via %s (model=%s, cost=%s)",
            generation_time,
            provider,
            executed_model,
            f"${cost:.4f}" if cost else "n/a",
        )

        response_data = {
            "success": True,
            "video_url": video_url,
            "duration": final.get("seconds") or duration,
            "model": executed_model,
            "provider": provider,
        }

        return json.dumps(response_data, indent=2, ensure_ascii=False)

    except LumenfallAuthError as e:
        logger.error("Lumenfall auth error: %s", e)
        return json.dumps({
            "success": False,
            "video_url": None,
            "error": str(e),
            "error_type": "auth_error",
        }, indent=2)

    except LumenfallBalanceError as e:
        logger.error("Lumenfall balance error: %s", e)
        return json.dumps({
            "success": False,
            "video_url": None,
            "error": str(e),
            "error_type": "balance_error",
        }, indent=2)

    except (LumenfallError, ValueError) as e:
        logger.error("Video generation error: %s", e, exc_info=True)
        return json.dumps({
            "success": False,
            "video_url": None,
            "error": str(e),
            "error_type": type(e).__name__,
            "hint": "Use lumenfall_list_models to discover available models and their capabilities.",
        }, indent=2)

    except Exception as e:
        logger.error("Unexpected error in video generation: %s", e, exc_info=True)
        return json.dumps({
            "success": False,
            "video_url": None,
            "error": str(e),
            "error_type": type(e).__name__,
        }, indent=2)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error  # noqa: E402

LUMENFALL_VIDEO_GENERATE_SCHEMA = {
    "name": "lumenfall_video_generate",
    "description": (
        "Generate videos from text prompts using Lumenfall. "
        "Supports many video models across providers: Veo, Wan, Kling, "
        "and more. Optionally specify a model or let the server pick the "
        "best default. Video generation takes 30 seconds to 5 minutes "
        "depending on the model — let the user know before calling. "
        "Returns a video URL. "
        "Display it using markdown: [Watch video](URL)"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "Text prompt describing the desired video. "
                    "Describe the scene, action, style, and camera movement."
                ),
            },
            "model": {
                "type": "string",
                "description": (
                    "Video model to use. Examples: "
                    "wan-2.7-pro, sora-2-pro, kling-v3. "
                    "Leave empty for the best available default."
                ),
            },
            "duration": {
                "type": "number",
                "description": (
                    "Desired video duration in seconds. "
                    "Typical range: 3-10 seconds. Default: 5."
                ),
                "default": 5,
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["landscape", "square", "portrait"],
                "description": (
                    "Video aspect ratio. "
                    "'landscape' is 16:9 wide, "
                    "'portrait' is 9:16 tall, "
                    "'square' is 1:1."
                ),
                "default": "landscape",
            },
            "image_url": {
                "type": "string",
                "description": (
                    "URL of a source image for image-to-video generation. "
                    "When provided, the video will animate from this image "
                    "instead of generating from text alone. The image should "
                    "be a publicly accessible URL (PNG, JPEG, or WebP)."
                ),
            },
        },
        "required": ["prompt"],
    },
}


def _handle_lumenfall_video_generate(args, **kw):
    prompt = args.get("prompt", "")
    if not prompt:
        return tool_error("prompt is required for video generation")
    return lumenfall_video_generate_tool(
        prompt=prompt,
        model=args.get("model"),
        duration=args.get("duration", DEFAULT_DURATION),
        aspect_ratio=args.get("aspect_ratio", "landscape"),
        image_url=args.get("image_url"),
    )


registry.register(
    name="lumenfall_video_generate",
    toolset="lumenfall",
    schema=LUMENFALL_VIDEO_GENERATE_SCHEMA,
    handler=_handle_lumenfall_video_generate,
    check_fn=check_lumenfall_available,
    requires_env=[],
    is_async=False,
    emoji="🎬",
)
