"""
Lumenfall API client for Hermes Agent tools.

Thin synchronous HTTP client for Lumenfall's OpenAI-compatible API.
Uses httpx (already a hermes-agent dependency) with sync calls to avoid
event-loop lifecycle issues in the gateway's thread-pool pattern.

Endpoints used:
  - POST /v1/images/generations  (image generation)
  - POST /v1/videos              (video generation submit)
  - GET  /v1/videos/:id          (video generation poll)
  - GET  /v1/models              (model catalog)
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.lumenfall.ai/openai/v1"
DEFAULT_IMAGE_TIMEOUT = 120  # seconds
DEFAULT_VIDEO_SUBMIT_TIMEOUT = 30  # seconds
DEFAULT_VIDEO_POLL_TIMEOUT = 600  # 10 minutes max wait
DEFAULT_VIDEO_POLL_INTERVAL = 5  # seconds between polls


class LumenfallError(Exception):
    """Base error for Lumenfall API calls."""

    def __init__(self, message: str, status_code: int = 0, error_code: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


class LumenfallAuthError(LumenfallError):
    """Raised on 401 Unauthorized."""
    pass


class LumenfallBalanceError(LumenfallError):
    """Raised on 402 Payment Required (insufficient balance)."""
    pass


def _get_api_key() -> Optional[str]:
    key = os.environ.get("LUMENFALL_API_KEY")
    if key:
        return key
    # Fallback: load directly from HERMES_HOME/.env if the env var
    # wasn't propagated by the dotenv loader at startup.
    from hermes_constants import get_hermes_home
    env_file = get_hermes_home() / ".env"
    try:
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() == "LUMENFALL_API_KEY":
                    v = v.strip().strip("'\"")
                    if v:
                        os.environ["LUMENFALL_API_KEY"] = v
                        return v
    except FileNotFoundError:
        pass
    return None


def _get_base_url() -> str:
    return os.environ.get("LUMENFALL_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _build_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "hermes-agent/lumenfall-integration",
    }


def _handle_error_response(response: httpx.Response) -> None:
    """Raise a typed error from an HTTP error response."""
    status = response.status_code
    try:
        body = response.json()
        error_obj = body.get("error", {})
        message = error_obj.get("message", response.text[:500])
        code = error_obj.get("code", "")
    except Exception:
        message = response.text[:500]
        code = ""

    if status == 401:
        raise LumenfallAuthError(
            "Unauthorized: Invalid LUMENFALL_API_KEY. "
            "Check your API key at https://lumenfall.ai",
            status_code=status,
            error_code=code,
        )
    if status == 402:
        raise LumenfallBalanceError(
            "Insufficient balance. Top up at https://lumenfall.ai",
            status_code=status,
            error_code=code,
        )
    raise LumenfallError(
        f"Lumenfall API error ({status}): {message}",
        status_code=status,
        error_code=code,
    )


def check_lumenfall_available() -> bool:
    """Check if LUMENFALL_API_KEY is set."""
    return bool(_get_api_key())


def generate_image(
    prompt: str,
    model: Optional[str] = None,
    n: int = 1,
    aspect_ratio: Optional[str] = None,
    size: Optional[str] = None,
    output_format: str = "png",
    response_format: str = "url",
    timeout: int = DEFAULT_IMAGE_TIMEOUT,
    **extra_params,
) -> Dict[str, Any]:
    """Generate images via POST /v1/images/generations.

    Args:
        prompt: Text description of the desired image.
        model: Model ID (e.g. "gemini-3.1-flash-image-preview", "gpt-image-1.5", "flux.2-max").
               If None, the server picks a default.
        n: Number of images to generate (1-4).
        aspect_ratio: Aspect ratio string (e.g. "16:9", "1:1", "9:16").
        size: Size string (e.g. "1024x1024"). Mutually exclusive with aspect_ratio.
        output_format: Image format — "png", "jpeg", or "webp".
        response_format: "url" (default) or "b64_json".
        timeout: Request timeout in seconds.
        **extra_params: Additional provider-specific parameters passed through.

    Returns:
        Full API response dict with "data" array of image objects.

    Raises:
        LumenfallAuthError: Invalid API key.
        LumenfallBalanceError: Insufficient balance.
        LumenfallError: Any other API error.
        ValueError: Missing API key.
    """
    api_key = _get_api_key()
    if not api_key:
        raise ValueError(
            "LUMENFALL_API_KEY environment variable not set. "
            "Get your API key at https://lumenfall.ai"
        )

    base_url = _get_base_url()
    url = f"{base_url}/images/generations"
    headers = _build_headers(api_key)

    payload: Dict[str, Any] = {
        "prompt": prompt.strip(),
        "n": n,
        "output_format": output_format,
        "response_format": response_format,
    }
    if model:
        payload["model"] = model
    if aspect_ratio:
        payload["aspect_ratio"] = aspect_ratio
    elif size:
        payload["size"] = size
    payload.update(extra_params)

    logger.info("Lumenfall image generation: model=%s, prompt=%.60s...", model, prompt)

    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, headers=headers, json=payload)

    if response.status_code >= 400:
        _handle_error_response(response)

    return response.json()


def edit_image(
    image_url: str,
    prompt: Optional[str] = None,
    model: Optional[str] = None,
    mask_url: Optional[str] = None,
    n: int = 1,
    output_format: str = "png",
    response_format: str = "url",
    timeout: int = DEFAULT_IMAGE_TIMEOUT,
    **extra_params,
) -> Dict[str, Any]:
    """Edit an image via POST /v1/images/edits (multipart form).

    Supports inpainting, background removal, upscaling, and general edits.

    Args:
        image_url: URL (or raw bytes placeholder) of the source image.
        prompt: Text description of the desired edit.
        model: Model ID (e.g. "gpt-image-1.5").  If None, server picks default.
        mask_url: Optional mask URL for inpainting (white = edit region).
        n: Number of output images (1-4).
        output_format: Image format — "png", "jpeg", or "webp".
        response_format: "url" (default) or "b64_json".
        timeout: Request timeout in seconds.
        **extra_params: Additional provider-specific parameters passed through.

    Returns:
        Full API response dict with "data" array of image objects.

    Raises:
        LumenfallAuthError: Invalid API key.
        LumenfallBalanceError: Insufficient balance.
        LumenfallError: Any other API error.
        ValueError: Missing API key.
    """
    api_key = _get_api_key()
    if not api_key:
        raise ValueError(
            "LUMENFALL_API_KEY environment variable not set. "
            "Get your API key at https://lumenfall.ai"
        )

    base_url = _get_base_url()
    url = f"{base_url}/images/edits"
    headers = _build_headers(api_key)
    # Remove Content-Type — httpx sets it automatically for multipart uploads
    headers.pop("Content-Type", None)

    data: Dict[str, Any] = {
        "n": n,
        "output_format": output_format,
        "response_format": response_format,
    }
    if model:
        data["model"] = model
    if prompt:
        data["prompt"] = prompt
    data.update(extra_params)

    files: Dict[str, Any] = {
        "image": ("image_url", image_url),
    }
    if mask_url:
        files["mask"] = ("mask_url", mask_url)

    logger.info("Lumenfall image edit: model=%s, prompt=%.60s...", model, prompt)

    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, headers=headers, data=data, files=files)

    if response.status_code >= 400:
        _handle_error_response(response)

    return response.json()


def submit_video(
    prompt: str,
    model: Optional[str] = None,
    seconds: Optional[float] = None,
    aspect_ratio: Optional[str] = None,
    size: Optional[str] = None,
    n: int = 1,
    image_url: Optional[str] = None,
    timeout: int = DEFAULT_VIDEO_SUBMIT_TIMEOUT,
    **extra_params,
) -> Dict[str, Any]:
    """Submit an async video generation job via POST /v1/videos.

    Args:
        prompt: Text description of the desired video.
        model: Video model ID (e.g. "wan-2.7-pro", "sora-2-pro", "kling-v3").
        seconds: Desired video duration in seconds.
        aspect_ratio: Aspect ratio string (e.g. "16:9").
        size: Size string (e.g. "1280x720"). Mutually exclusive with aspect_ratio.
        n: Number of videos (1-4).
        image_url: Optional URL of a source image for image-to-video generation.
                   When provided, the API uses this image as the starting frame.
        timeout: Submit request timeout in seconds.
        **extra_params: Additional provider-specific parameters passed through.

    Returns:
        Job submission response with "id" and "status" fields.

    Raises:
        LumenfallError subclasses on API errors.
        ValueError: Missing API key.
    """
    api_key = _get_api_key()
    if not api_key:
        raise ValueError(
            "LUMENFALL_API_KEY environment variable not set. "
            "Get your API key at https://lumenfall.ai"
        )

    base_url = _get_base_url()
    url = f"{base_url}/videos"
    headers = _build_headers(api_key)

    payload: Dict[str, Any] = {
        "prompt": prompt.strip(),
        "n": n,
    }
    if model:
        payload["model"] = model
    if seconds is not None:
        payload["seconds"] = seconds
    if aspect_ratio:
        payload["aspect_ratio"] = aspect_ratio
    elif size:
        payload["size"] = size
    if image_url:
        payload["input_reference"] = {"image_url": image_url}
    payload.update(extra_params)

    logger.info("Lumenfall video submit: model=%s, prompt=%.60s...", model, prompt)

    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, headers=headers, json=payload)

    if response.status_code >= 400:
        _handle_error_response(response)

    return response.json()


def poll_video(
    video_id: str,
    poll_interval: int = DEFAULT_VIDEO_POLL_INTERVAL,
    max_wait: int = DEFAULT_VIDEO_POLL_TIMEOUT,
) -> Dict[str, Any]:
    """Poll GET /v1/videos/:id until terminal status.

    Args:
        video_id: The video request ID from submit_video().
        poll_interval: Seconds between poll attempts.
        max_wait: Maximum total seconds to wait before giving up.

    Returns:
        Final video status response (status "completed" or "failed").

    Raises:
        LumenfallError: On API errors or timeout.
    """
    api_key = _get_api_key()
    if not api_key:
        raise ValueError("LUMENFALL_API_KEY environment variable not set.")

    base_url = _get_base_url()
    url = f"{base_url}/videos/{video_id}"
    headers = _build_headers(api_key)

    start = time.monotonic()

    with httpx.Client(timeout=30) as client:
        while True:
            elapsed = time.monotonic() - start
            if elapsed > max_wait:
                raise LumenfallError(
                    f"Video generation timed out after {max_wait}s "
                    f"(video_id={video_id})",
                    error_code="timeout",
                )

            try:
                response = client.get(url, headers=headers)
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                logger.warning(
                    "Transient error polling video %s (%.0fs elapsed): %s",
                    video_id, elapsed, exc,
                )
                time.sleep(poll_interval)
                continue

            if response.status_code >= 400:
                _handle_error_response(response)

            data = response.json()
            status = data.get("status", "")
            logger.info(
                "Video %s status: %s (%.0fs elapsed)", video_id, status, elapsed
            )

            if status in ("completed", "failed"):
                return data

            time.sleep(poll_interval)


def list_models(
    timeout: int = 10,
) -> List[Dict[str, Any]]:
    """Fetch available models via GET /v1/models.

    Returns:
        List of model dicts with "id", "name", etc.

    Raises:
        LumenfallError subclasses on API errors.
    """
    api_key = _get_api_key()
    if not api_key:
        raise ValueError("LUMENFALL_API_KEY environment variable not set.")

    base_url = _get_base_url()
    url = f"{base_url}/models"
    headers = _build_headers(api_key)

    with httpx.Client(timeout=timeout) as client:
        response = client.get(url, headers=headers)

    if response.status_code >= 400:
        _handle_error_response(response)

    data = response.json()
    return data.get("data", [])
