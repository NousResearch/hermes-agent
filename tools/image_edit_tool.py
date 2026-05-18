#!/usr/bin/env python3
"""Image editing tools (Venice-backed).

Four agent-facing tools wrap Venice's image-editing endpoints:

    image_upscale            POST /images/upscale
    image_edit               POST /images/edit             (inpaint / restyle)
    image_compose            POST /images/multi-edit       (1-3 layered images)
    image_remove_background  POST /images/background-remove

Authentication: ``VENICE_API_KEY`` (same key as Venice chat / image gen /
video gen / TTS / STT). Base URL is overridable via ``VENICE_BASE_URL``
so the same code targets either Venice direct or the HermesOS
managed-Venice proxy.

All four return the standard
``{success, image, model, provider, ...}`` envelope shared with
``image_generate`` so display/markdown rendering is uniform across the
toolset.
"""

from __future__ import annotations

import base64
import datetime
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.registry import registry

logger = logging.getLogger(__name__)


DEFAULT_VENICE_BASE_URL = "https://api.venice.ai/api/v1"
DEFAULT_EDIT_MODEL = "firered-image-edit"


def _resolve_credentials() -> Tuple[str, str]:
    """Return ``(api_key, base_url)``. api_key is "" when not configured."""
    api_key = os.environ.get("VENICE_API_KEY", "").strip()
    base_url = (
        os.environ.get("VENICE_BASE_URL", "").strip()
        or DEFAULT_VENICE_BASE_URL
    ).rstrip("/")
    return api_key, base_url


def _images_cache_dir() -> Path:
    """Return ``$HERMES_HOME/cache/images/``, creating parents as needed."""
    from hermes_constants import get_hermes_home

    path = get_hermes_home() / "cache" / "images"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_binary_image(data: bytes, *, prefix: str, extension: str = "png") -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    path = _images_cache_dir() / f"{prefix}_{ts}_{short}.{extension}"
    path.write_bytes(data)
    return path


def _content_type_extension(content_type: str) -> str:
    ct = (content_type or "").lower()
    if "png" in ct:
        return "png"
    if "webp" in ct:
        return "webp"
    if "jpeg" in ct or "jpg" in ct:
        return "jpg"
    return "png"


def _is_http_url(value: str) -> bool:
    v = value.strip().lower()
    return v.startswith("http://") or v.startswith("https://")


def _is_data_url(value: str) -> bool:
    return value.strip().lower().startswith("data:")


def _decode_data_url_or_base64(value: str) -> Optional[bytes]:
    """Decode ``data:<mime>;base64,...`` or a raw base64 string to bytes.

    Returns None when the input doesn't look like base64 (e.g. when it's
    actually a file path).
    """
    if _is_data_url(value):
        # Strip everything up to the first comma.
        comma = value.find(",")
        if comma < 0:
            return None
        payload = value[comma + 1 :]
    else:
        payload = value
    try:
        return base64.b64decode(payload, validate=False)
    except Exception:
        return None


def _open_image_for_upload(image: str) -> Tuple[bytes, str]:
    """Read *image* into ``(bytes, filename)`` for multipart upload.

    Accepts:
    - Absolute or relative file path
    - HTTPS URL (fetched once and re-uploaded)
    - ``data:<mime>;base64,...`` data URL
    - Raw base64 string

    Raises ``FileNotFoundError`` / ``ValueError`` on bad input.
    """
    import requests

    candidate = image.strip()
    if _is_http_url(candidate):
        resp = requests.get(candidate, timeout=60)
        resp.raise_for_status()
        return resp.content, Path(candidate).name or "image.png"

    if _is_data_url(candidate):
        data = _decode_data_url_or_base64(candidate)
        if data is None:
            raise ValueError("Could not decode data: URL")
        return data, "image.png"

    path = Path(os.path.expanduser(candidate))
    if path.exists() and path.is_file():
        return path.read_bytes(), path.name

    # Fall back to raw base64.
    data = _decode_data_url_or_base64(candidate)
    if data is None:
        raise FileNotFoundError(
            f"Could not resolve image input: not a file path, URL, or base64 string"
        )
    return data, "image.png"


def _post_multipart(
    endpoint_path: str,
    *,
    image_bytes: bytes,
    image_name: str,
    data_fields: Dict[str, str],
    api_key: str,
    base_url: str,
    timeout: int = 180,
):
    """POST ``image_bytes`` + form fields to Venice. Returns the requests.Response."""
    import requests

    return requests.post(
        f"{base_url}{endpoint_path}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "hermes-agent/image-edit-venice",
        },
        files={"image": (image_name, image_bytes)},
        data=data_fields,
        timeout=timeout,
    )


def _success(image_path: str, *, tool: str, model: str = "", extra: Optional[Dict[str, Any]] = None) -> str:
    payload: Dict[str, Any] = {
        "success": True,
        "image": image_path,
        "tool": tool,
        "provider": "venice",
        "model": model,
    }
    if extra:
        for k, v in extra.items():
            payload.setdefault(k, v)
    return json.dumps(payload, ensure_ascii=False)


def _error(message: str, *, tool: str, error_type: str = "provider_error") -> str:
    return json.dumps(
        {
            "success": False,
            "image": None,
            "tool": tool,
            "provider": "venice",
            "error": message,
            "error_type": error_type,
        },
        ensure_ascii=False,
    )


def _decode_error_message(response) -> str:
    try:
        body = response.json()
        return (body.get("error") or {}).get("message") or response.text[:300]
    except Exception:
        return response.text[:300] if hasattr(response, "text") else "unknown error"


def _save_response_image(response, *, prefix: str) -> Path:
    content_type = response.headers.get("content-type", "")
    ext = _content_type_extension(content_type)
    return _save_binary_image(response.content, prefix=prefix, extension=ext)


def check_image_edit_requirements() -> bool:
    """Tool-availability gate. Available iff VENICE_API_KEY is set."""
    api_key, _ = _resolve_credentials()
    return bool(api_key)


# ---------------------------------------------------------------------------
# image_upscale
# ---------------------------------------------------------------------------


def image_upscale_tool(
    image: str,
    scale: int = 2,
    enhance: bool = False,
    enhance_prompt: Optional[str] = None,
    enhance_creativity: Optional[float] = None,
    replication: Optional[float] = None,
) -> str:
    """Upscale or enhance an image via Venice ``/images/upscale``."""
    api_key, base_url = _resolve_credentials()
    if not api_key:
        return _error(
            "VENICE_API_KEY not set. Configure Venice in `hermes model` first.",
            tool="image_upscale",
            error_type="missing_api_key",
        )

    try:
        image_bytes, image_name = _open_image_for_upload(image)
    except Exception as exc:
        return _error(f"Could not read image: {exc}", tool="image_upscale", error_type="bad_input")

    if scale < 1 or scale > 4:
        scale = max(1, min(4, scale))
    if scale == 1 and not enhance:
        enhance = True
    if scale > 1 and enhance:
        # Venice rejects scale>1 + enhance=true; force enhance off for upscale-only.
        enhance = False

    data: Dict[str, str] = {"scale": str(scale), "enhance": "true" if enhance else "false"}
    if enhance:
        if isinstance(enhance_prompt, str) and enhance_prompt.strip():
            data["enhancePrompt"] = enhance_prompt.strip()[:1500]
        if isinstance(enhance_creativity, (int, float)):
            data["enhanceCreativity"] = f"{max(0.0, min(1.0, float(enhance_creativity))):.3f}"
    if isinstance(replication, (int, float)):
        data["replication"] = f"{max(0.0, min(1.0, float(replication))):.3f}"

    try:
        response = _post_multipart(
            "/images/upscale",
            image_bytes=image_bytes, image_name=image_name,
            data_fields=data, api_key=api_key, base_url=base_url,
        )
    except Exception as exc:
        return _error(f"Venice request failed: {exc}", tool="image_upscale", error_type="connection_error")

    if response.status_code != 200:
        return _error(
            f"Venice upscale failed ({response.status_code}): {_decode_error_message(response)}",
            tool="image_upscale", error_type="api_error",
        )

    try:
        saved = _save_response_image(response, prefix="venice_upscale")
    except Exception as exc:
        return _error(f"Could not save image: {exc}", tool="image_upscale", error_type="io_error")

    return _success(
        str(saved),
        tool="image_upscale",
        extra={"scale": scale, "enhance": enhance},
    )


IMAGE_UPSCALE_SCHEMA = {
    "name": "image_upscale",
    "description": (
        "Upscale or enhance an existing image. Returns the saved upscaled "
        "image at an absolute path. Use scale=2 to double resolution; "
        "scale=1 with enhance=true to enhance without upsizing."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": "Image input: absolute file path, HTTPS URL, or base64-encoded bytes.",
            },
            "scale": {
                "type": "integer",
                "description": "Upscale factor (1-4). scale=1 requires enhance=true.",
                "default": 2,
            },
            "enhance": {
                "type": "boolean",
                "description": "Apply Venice's enhancement pass. Required when scale=1.",
                "default": False,
            },
            "enhance_prompt": {
                "type": "string",
                "description": "Optional style descriptor (e.g. 'gold', 'marble'). Max 1500 chars.",
            },
            "enhance_creativity": {
                "type": "number",
                "description": "0-1, how much enhancement modifies the image. Default 0.5.",
            },
            "replication": {
                "type": "number",
                "description": "0-1, preserve original noise/lines. Default 0.35.",
            },
        },
        "required": ["image"],
    },
}


registry.register(
    name="image_upscale",
    toolset="image_gen",
    schema=IMAGE_UPSCALE_SCHEMA,
    handler=lambda **kw: image_upscale_tool(**kw),
    check_fn=check_image_edit_requirements,
    requires_env=["VENICE_API_KEY"],
    is_async=False,
    emoji="🔍",
)


# ---------------------------------------------------------------------------
# image_edit  (inpaint / restyle)
# ---------------------------------------------------------------------------


def image_edit_tool(
    image: str,
    prompt: str,
    model: Optional[str] = None,
    aspect_ratio: str = "auto",
    resolution: Optional[str] = None,
    output_format: Optional[str] = None,
) -> str:
    """Edit / restyle an image with a text prompt via Venice ``/images/edit``."""
    api_key, base_url = _resolve_credentials()
    if not api_key:
        return _error(
            "VENICE_API_KEY not set. Configure Venice in `hermes model` first.",
            tool="image_edit", error_type="missing_api_key",
        )

    if not isinstance(prompt, str) or not prompt.strip():
        return _error("prompt is required", tool="image_edit", error_type="missing_prompt")

    try:
        image_bytes, image_name = _open_image_for_upload(image)
    except Exception as exc:
        return _error(f"Could not read image: {exc}", tool="image_edit", error_type="bad_input")

    chosen_model = (model or DEFAULT_EDIT_MODEL).strip() or DEFAULT_EDIT_MODEL

    data: Dict[str, str] = {
        "model": chosen_model,
        "prompt": prompt.strip()[:32_000],
        "aspect_ratio": (aspect_ratio or "auto").strip() or "auto",
    }
    if isinstance(resolution, str) and resolution.strip():
        data["resolution"] = resolution.strip()
    if isinstance(output_format, str) and output_format.strip():
        data["output_format"] = output_format.strip()

    try:
        response = _post_multipart(
            "/images/edit",
            image_bytes=image_bytes, image_name=image_name,
            data_fields=data, api_key=api_key, base_url=base_url,
        )
    except Exception as exc:
        return _error(f"Venice request failed: {exc}", tool="image_edit", error_type="connection_error")

    if response.status_code != 200:
        return _error(
            f"Venice edit failed ({response.status_code}): {_decode_error_message(response)}",
            tool="image_edit", error_type="api_error",
        )

    try:
        saved = _save_response_image(response, prefix=f"venice_edit_{chosen_model}")
    except Exception as exc:
        return _error(f"Could not save image: {exc}", tool="image_edit", error_type="io_error")

    return _success(
        str(saved),
        tool="image_edit",
        model=chosen_model,
        extra={"aspect_ratio": data["aspect_ratio"]},
    )


IMAGE_EDIT_SCHEMA = {
    "name": "image_edit",
    "description": (
        "Edit, inpaint, or restyle an existing image guided by a text prompt. "
        "Use for 'make the sky purple', 'replace the background with a forest', "
        "'change the person's hair color', etc. Returns the saved edited image."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": "Source image: file path, HTTPS URL, or base64.",
            },
            "prompt": {
                "type": "string",
                "description": "Text instructions for the edit. Short and descriptive works best.",
            },
            "model": {
                "type": "string",
                "description": "Venice edit model (defaults to firered-image-edit).",
            },
            "aspect_ratio": {
                "type": "string",
                "description": "auto | 1:1 | 3:2 | 16:9 | 21:9 | 9:16 | 2:3 | 3:4 | 4:5",
                "default": "auto",
            },
            "resolution": {
                "type": "string",
                "description": "1K | 2K | 4K (tier models only).",
            },
            "output_format": {
                "type": "string",
                "description": "jpeg | png | webp (default: webp).",
            },
        },
        "required": ["image", "prompt"],
    },
}


registry.register(
    name="image_edit",
    toolset="image_gen",
    schema=IMAGE_EDIT_SCHEMA,
    handler=lambda **kw: image_edit_tool(**kw),
    check_fn=check_image_edit_requirements,
    requires_env=["VENICE_API_KEY"],
    is_async=False,
    emoji="🖌️",
)


# ---------------------------------------------------------------------------
# image_compose  (multi-edit: 1-3 layered images)
# ---------------------------------------------------------------------------


def image_compose_tool(
    images: List[str],
    prompt: str,
    model: Optional[str] = None,
    aspect_ratio: str = "auto",
    resolution: Optional[str] = None,
    output_format: Optional[str] = None,
) -> str:
    """Compose 1-3 images into a single output via Venice ``/images/multi-edit``.

    The first image is the base; subsequent ones are edit layers. Venice
    accepts base64 strings or HTTP URLs (NOT file uploads here), so we
    encode local paths into base64 before sending.
    """
    api_key, base_url = _resolve_credentials()
    if not api_key:
        return _error(
            "VENICE_API_KEY not set. Configure Venice in `hermes model` first.",
            tool="image_compose", error_type="missing_api_key",
        )

    if not isinstance(images, list) or not images:
        return _error("images must be a non-empty list (1-3 entries)", tool="image_compose", error_type="bad_input")
    if len(images) > 3:
        return _error("Venice multi-edit supports at most 3 images", tool="image_compose", error_type="too_many_images")

    if not isinstance(prompt, str) or not prompt.strip():
        return _error("prompt is required", tool="image_compose", error_type="missing_prompt")

    encoded: List[str] = []
    for entry in images:
        if not isinstance(entry, str) or not entry.strip():
            return _error("Each image entry must be a non-empty string", tool="image_compose", error_type="bad_input")
        candidate = entry.strip()
        if _is_http_url(candidate):
            encoded.append(candidate)
            continue
        try:
            data_bytes, _ = _open_image_for_upload(candidate)
        except Exception as exc:
            return _error(f"Could not read image '{entry}': {exc}", tool="image_compose", error_type="bad_input")
        encoded.append(base64.b64encode(data_bytes).decode("ascii"))

    chosen_model = (model or DEFAULT_EDIT_MODEL).strip() or DEFAULT_EDIT_MODEL

    payload: Dict[str, Any] = {
        "modelId": chosen_model,
        "prompt": prompt.strip()[:32_000],
        "images": encoded,
        "aspect_ratio": (aspect_ratio or "auto").strip() or "auto",
    }
    if isinstance(resolution, str) and resolution.strip():
        payload["resolution"] = resolution.strip()
    if isinstance(output_format, str) and output_format.strip():
        payload["output_format"] = output_format.strip()

    try:
        import requests

        response = requests.post(
            f"{base_url}/images/multi-edit",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "hermes-agent/image-compose-venice",
            },
            json=payload,
            timeout=240,
        )
    except Exception as exc:
        return _error(f"Venice request failed: {exc}", tool="image_compose", error_type="connection_error")

    if response.status_code != 200:
        return _error(
            f"Venice multi-edit failed ({response.status_code}): {_decode_error_message(response)}",
            tool="image_compose", error_type="api_error",
        )

    try:
        saved = _save_response_image(response, prefix=f"venice_compose_{chosen_model}")
    except Exception as exc:
        return _error(f"Could not save image: {exc}", tool="image_compose", error_type="io_error")

    return _success(
        str(saved),
        tool="image_compose",
        model=chosen_model,
        extra={"image_count": len(encoded), "aspect_ratio": payload["aspect_ratio"]},
    )


IMAGE_COMPOSE_SCHEMA = {
    "name": "image_compose",
    "description": (
        "Combine 1-3 images into a single composition guided by a text prompt. "
        "The first image is the base; additional images are layered edits. "
        "Use for 'put this product on this background', 'merge style of A with subject of B', etc."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "images": {
                "type": "array",
                "description": "1-3 source images. Each is a file path, HTTPS URL, or base64.",
                "items": {"type": "string"},
                "maxItems": 3,
                "minItems": 1,
            },
            "prompt": {
                "type": "string",
                "description": "How to combine / edit the images.",
            },
            "model": {
                "type": "string",
                "description": "Venice edit model (defaults to firered-image-edit).",
            },
            "aspect_ratio": {
                "type": "string",
                "description": "auto | 1:1 | 3:2 | 16:9 | 21:9 | 9:16 | 2:3 | 3:4 | 4:5",
                "default": "auto",
            },
            "resolution": {
                "type": "string",
                "description": "1K | 2K | 4K (tier models only).",
            },
            "output_format": {
                "type": "string",
                "description": "jpeg | png | webp.",
            },
        },
        "required": ["images", "prompt"],
    },
}


registry.register(
    name="image_compose",
    toolset="image_gen",
    schema=IMAGE_COMPOSE_SCHEMA,
    handler=lambda **kw: image_compose_tool(**kw),
    check_fn=check_image_edit_requirements,
    requires_env=["VENICE_API_KEY"],
    is_async=False,
    emoji="🧩",
)


# ---------------------------------------------------------------------------
# image_remove_background
# ---------------------------------------------------------------------------


def image_remove_background_tool(image: str) -> str:
    """Remove the background from an image via Venice ``/images/background-remove``.

    Returns a PNG with a transparent background.
    """
    api_key, base_url = _resolve_credentials()
    if not api_key:
        return _error(
            "VENICE_API_KEY not set. Configure Venice in `hermes model` first.",
            tool="image_remove_background", error_type="missing_api_key",
        )

    candidate = (image or "").strip()
    if not candidate:
        return _error("image is required", tool="image_remove_background", error_type="bad_input")

    # Venice accepts either `image` (multipart/base64) OR `image_url`. Prefer
    # image_url for HTTPS inputs to avoid re-downloading.
    if _is_http_url(candidate):
        try:
            import requests

            response = requests.post(
                f"{base_url}/images/background-remove",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "hermes-agent/image-bg-remove-venice",
                },
                json={"image_url": candidate},
                timeout=180,
            )
        except Exception as exc:
            return _error(f"Venice request failed: {exc}", tool="image_remove_background", error_type="connection_error")
    else:
        try:
            image_bytes, image_name = _open_image_for_upload(candidate)
        except Exception as exc:
            return _error(f"Could not read image: {exc}", tool="image_remove_background", error_type="bad_input")
        try:
            response = _post_multipart(
                "/images/background-remove",
                image_bytes=image_bytes, image_name=image_name,
                data_fields={}, api_key=api_key, base_url=base_url,
            )
        except Exception as exc:
            return _error(f"Venice request failed: {exc}", tool="image_remove_background", error_type="connection_error")

    if response.status_code != 200:
        return _error(
            f"Venice background-remove failed ({response.status_code}): {_decode_error_message(response)}",
            tool="image_remove_background", error_type="api_error",
        )

    try:
        # Always saved as .png since Venice returns transparent PNG.
        saved = _save_binary_image(response.content, prefix="venice_bgremove", extension="png")
    except Exception as exc:
        return _error(f"Could not save image: {exc}", tool="image_remove_background", error_type="io_error")

    return _success(str(saved), tool="image_remove_background")


IMAGE_REMOVE_BACKGROUND_SCHEMA = {
    "name": "image_remove_background",
    "description": (
        "Remove the background from an image. Returns a PNG with a "
        "transparent background, saved as an absolute file path."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": "Image input: file path, HTTPS URL, or base64.",
            },
        },
        "required": ["image"],
    },
}


registry.register(
    name="image_remove_background",
    toolset="image_gen",
    schema=IMAGE_REMOVE_BACKGROUND_SCHEMA,
    handler=lambda **kw: image_remove_background_tool(**kw),
    check_fn=check_image_edit_requirements,
    requires_env=["VENICE_API_KEY"],
    is_async=False,
    emoji="✂️",
)
