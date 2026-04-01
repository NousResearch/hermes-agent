#!/usr/bin/env python3
"""
Image generation tools.

Supports provider-routed image generation with OpenAI as the preferred path.
Generated images are stored locally so Hermes can deliver them as native
attachments on messaging platforms via MEDIA:/absolute/path references.
"""

import base64
import datetime
import json
import logging
import os
import subprocess
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI

try:
    import fal_client
except ImportError:  # pragma: no cover - optional dependency
    fal_client = None

from hermes_cli.config import get_hermes_home
from tools.debug_helpers import DebugSession

logger = logging.getLogger(__name__)

DEFAULT_PROVIDER = "openai"
DEFAULT_OPENAI_MODEL = "gpt-image-1.5"
DEFAULT_OPENAI_QUALITY = "high"
DEFAULT_OPENAI_BACKGROUND = "auto"
DEFAULT_FAL_MODEL = "fal-ai/flux-2-pro"
DEFAULT_ASPECT_RATIO = "landscape"
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 4.5
DEFAULT_NUM_IMAGES = 1
DEFAULT_OUTPUT_FORMAT = "png"

# FAL safety settings
ENABLE_SAFETY_CHECKER = False
SAFETY_TOLERANCE = "5"

# Aspect ratio mapping
FAL_ASPECT_RATIO_MAP = {
    "landscape": "landscape_16_9",
    "square": "square_hd",
    "portrait": "portrait_16_9",
}
OPENAI_ASPECT_RATIO_MAP = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}
VALID_ASPECT_RATIOS = list(FAL_ASPECT_RATIO_MAP.keys())

# FAL upscaling configuration
UPSCALER_MODEL = "fal-ai/clarity-upscaler"
UPSCALER_FACTOR = 2
UPSCALER_SAFETY_CHECKER = False
UPSCALER_DEFAULT_PROMPT = "masterpiece, best quality, highres"
UPSCALER_NEGATIVE_PROMPT = "(worst quality, low quality, normal quality:2)"
UPSCALER_CREATIVITY = 0.35
UPSCALER_RESEMBLANCE = 0.6
UPSCALER_GUIDANCE_SCALE = 4
UPSCALER_NUM_INFERENCE_STEPS = 18

VALID_IMAGE_SIZES = [
    "square_hd",
    "square",
    "portrait_4_3",
    "portrait_16_9",
    "landscape_4_3",
    "landscape_16_9",
]
VALID_OUTPUT_FORMATS = ["jpeg", "png", "webp"]
VALID_ACCELERATION_MODES = ["none", "regular", "high"]

_debug = DebugSession("image_tools", env_var="IMAGE_TOOLS_DEBUG")


def _generated_image_dir() -> Path:
    out_dir = get_hermes_home() / "cache" / "generated_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _prepared_edit_image_dir() -> Path:
    out_dir = get_hermes_home() / "cache" / "prepared_edit_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_generated_image_bytes(data: bytes, output_format: str) -> str:
    ext = (output_format or DEFAULT_OUTPUT_FORMAT).lower().strip()
    if ext == "jpg":
        ext = "jpeg"
    if ext not in {"png", "jpeg", "webp", "gif"}:
        ext = DEFAULT_OUTPUT_FORMAT
    filename = f"generated_{uuid.uuid4().hex[:12]}.{ext}"
    path = _generated_image_dir() / filename
    path.write_bytes(data)
    return str(path.resolve())


def _prepare_dalle_edit_image(source_path: Path) -> Path:
    """Convert any supported image into a square PNG suitable for dall-e-2 edits."""
    output_path = _prepared_edit_image_dir() / f"edit_input_{uuid.uuid4().hex[:12]}.png"
    ffmpeg_cmd = [
        "/usr/bin/ffmpeg",
        "-y",
        "-i",
        str(source_path),
        "-vf",
        "pad=max(iw\\,ih):max(iw\\,ih):(ow-iw)/2:(oh-ih)/2:color=white,"
        "scale=1024:1024:flags=lanczos,format=rgba",
        "-pix_fmt",
        "rgba",
        "-frames:v",
        "1",
        str(output_path),
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise ValueError("ffmpeg is required for the OpenAI image edit fallback path") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise ValueError(f"Failed to prepare edit image with ffmpeg: {stderr}") from exc

    if not output_path.exists():
        raise ValueError("Prepared edit image was not created")
    if output_path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Prepared edit image exceeds the 4MB limit for dall-e-2 edits")
    return output_path


def _detect_local_image_mime_type(image_path: Path) -> Optional[str]:
    """Return a MIME type when the file looks like a supported image."""
    with image_path.open("rb") as f:
        header = f.read(64)

    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    if image_path.suffix.lower() == ".svg":
        head = image_path.read_text(encoding="utf-8", errors="ignore")[:4096].lower()
        if "<svg" in head:
            return "image/svg+xml"
    return None


def _resolve_existing_file(path_value: Any, label: str) -> Path:
    """Validate and normalize a local file path passed to the image tool."""
    if path_value in (None, ""):
        raise ValueError(f"{label} is required")
    if not isinstance(path_value, str):
        raise ValueError(f"{label} must be a string path")
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    if not path.exists():
        raise ValueError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    return path


def _resolve_edit_image_paths(
    image_path: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
) -> List[Path]:
    """Collect and validate one or more local image paths for OpenAI edits."""
    raw_paths: List[str] = []
    if image_path:
        raw_paths.append(image_path)
    if image_paths:
        if not isinstance(image_paths, list):
            raise ValueError("image_paths must be a list of local file paths")
        raw_paths.extend(str(item) for item in image_paths if item)

    resolved: List[Path] = []
    seen = set()
    for idx, raw in enumerate(raw_paths):
        path = _resolve_existing_file(raw, f"image_path[{idx}]")
        mime = _detect_local_image_mime_type(path)
        if mime not in {"image/png", "image/jpeg", "image/webp"}:
            raise ValueError(
                f"Unsupported edit image type for {path}. Supported inputs: PNG, JPEG, WEBP."
            )
        key = str(path.resolve())
        if key not in seen:
            resolved.append(path)
            seen.add(key)
    return resolved


def _resolve_image_provider() -> str:
    configured = os.getenv("IMAGE_GEN_PROVIDER", "").strip().lower()
    openai_ready = check_openai_image_api_key()
    fal_ready = check_fal_api_key() and fal_client is not None

    if configured == "openai":
        return "openai" if openai_ready else "none"
    if configured == "fal":
        return "fal" if fal_ready else "none"

    if openai_ready:
        return "openai"
    if fal_ready:
        return "fal"
    return "none"


def _validate_parameters(
    image_size: Union[str, Dict[str, int]],
    num_inference_steps: int,
    guidance_scale: float,
    num_images: int,
    output_format: str,
    acceleration: str = "none",
) -> Dict[str, Any]:
    validated = {}

    if isinstance(image_size, str):
        if image_size not in VALID_IMAGE_SIZES:
            raise ValueError(f"Invalid image_size '{image_size}'. Must be one of: {VALID_IMAGE_SIZES}")
        validated["image_size"] = image_size
    elif isinstance(image_size, dict):
        if "width" not in image_size or "height" not in image_size:
            raise ValueError("Custom image_size must contain 'width' and 'height' keys")
        if not isinstance(image_size["width"], int) or not isinstance(image_size["height"], int):
            raise ValueError("Custom image_size width and height must be integers")
        if image_size["width"] < 64 or image_size["height"] < 64:
            raise ValueError("Custom image_size dimensions must be at least 64x64")
        if image_size["width"] > 2048 or image_size["height"] > 2048:
            raise ValueError("Custom image_size dimensions must not exceed 2048x2048")
        validated["image_size"] = image_size
    else:
        raise ValueError("image_size must be either a preset string or a dict with width/height")

    if not isinstance(num_inference_steps, int) or num_inference_steps < 1 or num_inference_steps > 100:
        raise ValueError("num_inference_steps must be an integer between 1 and 100")
    validated["num_inference_steps"] = num_inference_steps

    if not isinstance(guidance_scale, (int, float)) or guidance_scale < 0.1 or guidance_scale > 20.0:
        raise ValueError("guidance_scale must be a number between 0.1 and 20.0")
    validated["guidance_scale"] = float(guidance_scale)

    if not isinstance(num_images, int) or num_images < 1 or num_images > 4:
        raise ValueError("num_images must be an integer between 1 and 4")
    validated["num_images"] = num_images

    output_format = (output_format or DEFAULT_OUTPUT_FORMAT).lower().strip()
    if output_format == "jpg":
        output_format = "jpeg"
    if output_format not in VALID_OUTPUT_FORMATS:
        raise ValueError(f"Invalid output_format '{output_format}'. Must be one of: {VALID_OUTPUT_FORMATS}")
    validated["output_format"] = output_format

    if acceleration not in VALID_ACCELERATION_MODES:
        raise ValueError(f"Invalid acceleration '{acceleration}'. Must be one of: {VALID_ACCELERATION_MODES}")
    validated["acceleration"] = acceleration

    return validated


def _upscale_image(image_url: str, original_prompt: str) -> Optional[Dict[str, Any]]:
    if fal_client is None:
        return None
    try:
        logger.info("Upscaling image with Clarity Upscaler...")
        upscaler_arguments = {
            "image_url": image_url,
            "prompt": f"{UPSCALER_DEFAULT_PROMPT}, {original_prompt}",
            "upscale_factor": UPSCALER_FACTOR,
            "negative_prompt": UPSCALER_NEGATIVE_PROMPT,
            "creativity": UPSCALER_CREATIVITY,
            "resemblance": UPSCALER_RESEMBLANCE,
            "guidance_scale": UPSCALER_GUIDANCE_SCALE,
            "num_inference_steps": UPSCALER_NUM_INFERENCE_STEPS,
            "enable_safety_checker": UPSCALER_SAFETY_CHECKER,
        }
        handler = fal_client.submit(UPSCALER_MODEL, arguments=upscaler_arguments)
        result = handler.get()
        if result and "image" in result:
            upscaled_image = result["image"]
            logger.info(
                "Image upscaled successfully to %sx%s",
                upscaled_image.get("width", "unknown"),
                upscaled_image.get("height", "unknown"),
            )
            return {
                "url": upscaled_image["url"],
                "width": upscaled_image.get("width", 0),
                "height": upscaled_image.get("height", 0),
                "upscaled": True,
                "upscale_factor": UPSCALER_FACTOR,
            }
        logger.error("Upscaler returned invalid response")
        return None
    except Exception as e:  # pragma: no cover - network/provider behavior
        logger.error("Error upscaling image: %s", e, exc_info=True)
        return None


def _generate_via_fal(
    prompt: str,
    aspect_ratio_lower: str,
    num_inference_steps: int,
    guidance_scale: float,
    num_images: int,
    output_format: str,
    seed: Optional[int],
) -> Dict[str, Any]:
    if fal_client is None:
        raise ValueError("fal_client is not installed")
    if not os.getenv("FAL_KEY"):
        raise ValueError("FAL_KEY environment variable not set")

    image_size = FAL_ASPECT_RATIO_MAP[aspect_ratio_lower]
    validated_params = _validate_parameters(
        image_size, num_inference_steps, guidance_scale, num_images, output_format, "none"
    )
    arguments = {
        "prompt": prompt.strip(),
        "image_size": validated_params["image_size"],
        "num_inference_steps": validated_params["num_inference_steps"],
        "guidance_scale": validated_params["guidance_scale"],
        "num_images": validated_params["num_images"],
        "output_format": validated_params["output_format"],
        "enable_safety_checker": ENABLE_SAFETY_CHECKER,
        "safety_tolerance": SAFETY_TOLERANCE,
        "sync_mode": True,
    }
    if seed is not None and isinstance(seed, int):
        arguments["seed"] = seed

    logger.info("Submitting generation request to FAL.ai model %s", DEFAULT_FAL_MODEL)
    handler = fal_client.submit(DEFAULT_FAL_MODEL, arguments=arguments)
    result = handler.get()
    if not result or "images" not in result:
        raise ValueError("Invalid response from FAL.ai API - no images returned")

    images = result.get("images", [])
    if not images:
        raise ValueError("No images were generated")

    formatted_images = []
    for img in images:
        if isinstance(img, dict) and "url" in img:
            original_image = {
                "url": img["url"],
                "width": img.get("width", 0),
                "height": img.get("height", 0),
            }
            upscaled_image = _upscale_image(img["url"], prompt.strip())
            if upscaled_image:
                formatted_images.append(upscaled_image)
            else:
                original_image["upscaled"] = False
                formatted_images.append(original_image)

    if not formatted_images:
        raise ValueError("No valid image URLs returned from FAL")

    first = formatted_images[0]
    return {
        "provider": "fal",
        "model": DEFAULT_FAL_MODEL,
        "image": first["url"],
        "media_tag": None,
        "image_mode": "remote_url",
        "images_generated": len(formatted_images),
        "upscaled_count": sum(1 for img in formatted_images if img.get("upscaled", False)),
    }


def _generate_via_openai(
    prompt: str,
    aspect_ratio_lower: str,
    output_format: str,
    edit_image_paths: Optional[List[Path]] = None,
    mask_path: Optional[Path] = None,
    input_fidelity: Optional[str] = None,
) -> Dict[str, Any]:
    api_key = (os.getenv("OPENAI_API_KEY", "").strip() or os.getenv("VOICE_TOOLS_OPENAI_KEY", "").strip())
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if base_url:
        client_kwargs["base_url"] = base_url.rstrip("/")

    client = OpenAI(**client_kwargs)
    model = os.getenv("OPENAI_IMAGE_MODEL", DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL
    edit_model = os.getenv("OPENAI_IMAGE_EDIT_MODEL", "").strip() or model
    quality = os.getenv("OPENAI_IMAGE_QUALITY", DEFAULT_OPENAI_QUALITY).strip() or DEFAULT_OPENAI_QUALITY
    background = os.getenv("OPENAI_IMAGE_BACKGROUND", DEFAULT_OPENAI_BACKGROUND).strip() or DEFAULT_OPENAI_BACKGROUND
    size = OPENAI_ASPECT_RATIO_MAP[aspect_ratio_lower]
    normalized_format = (output_format or DEFAULT_OUTPUT_FORMAT).lower().strip()
    if normalized_format == "jpg":
        normalized_format = "jpeg"
    fidelity = (input_fidelity or "high").strip().lower()
    if fidelity not in {"high", "low"}:
        fidelity = "high"

    with ExitStack() as stack:
        if edit_image_paths:
            logger.info(
                "Submitting image edit request to OpenAI image model %s using %d input image(s)",
                edit_model,
                len(edit_image_paths),
            )
            images = [stack.enter_context(path.open("rb")) for path in edit_image_paths]
            image_input: Union[Any, List[Any]]
            if len(images) == 1:
                image_input = images[0]
            else:
                image_input = images
            edit_kwargs: Dict[str, Any] = {
                "model": edit_model,
                "image": image_input,
                "prompt": prompt.strip(),
                "size": size,
                "response_format": "b64_json",
                "n": 1,
            }
            if mask_path is not None:
                edit_kwargs["mask"] = stack.enter_context(mask_path.open("rb"))
            try:
                result = client.images.edit(**edit_kwargs)
                model = edit_model
            except Exception as exc:
                if edit_model == "dall-e-2":
                    raise
                logger.warning(
                    "Primary OpenAI image edit model %s failed (%s). Falling back to dall-e-2 edit mode.",
                    edit_model,
                    exc,
                )
                with ExitStack() as fallback_stack:
                    prepared_image = _prepare_dalle_edit_image(edit_image_paths[0])
                    fallback_kwargs: Dict[str, Any] = {
                        "model": "dall-e-2",
                        "image": fallback_stack.enter_context(prepared_image.open("rb")),
                        "prompt": prompt.strip(),
                        "size": "1024x1024",
                        "response_format": "b64_json",
                        "n": 1,
                    }
                    if mask_path is not None:
                        prepared_mask = _prepare_dalle_edit_image(mask_path)
                        fallback_kwargs["mask"] = fallback_stack.enter_context(prepared_mask.open("rb"))
                    result = client.images.edit(**fallback_kwargs)
                    model = "dall-e-2"
                    normalized_format = "png"
                    size = "1024x1024"
                    quality = "standard"
                    background = "auto"
        else:
            logger.info("Submitting generation request to OpenAI image model %s", model)
            result = client.images.generate(
                model=model,
                prompt=prompt.strip(),
                size=size,
                quality=quality,
                background=background,
                output_format=normalized_format,
                response_format="b64_json",
            )

    data = getattr(result, "data", None) or []
    if not data:
        raise ValueError("OpenAI image generation returned no images")

    first = data[0]
    b64_json = getattr(first, "b64_json", None)
    image_url = getattr(first, "url", None)
    if not b64_json and isinstance(first, dict):
        b64_json = first.get("b64_json")
        image_url = first.get("url")

    if b64_json:
        image_bytes = base64.b64decode(b64_json)
        image_path = _save_generated_image_bytes(image_bytes, normalized_format)
        return {
            "provider": "openai",
            "model": model,
            "image": image_path,
            "media_tag": f"MEDIA:{image_path}",
            "image_mode": "local_file",
            "images_generated": 1,
            "quality": quality,
            "background": background,
            "size": size,
            "operation": "edit" if edit_image_paths else "generate",
            "source_images": [str(path.resolve()) for path in edit_image_paths] if edit_image_paths else [],
            "input_fidelity": fidelity if edit_image_paths else None,
        }

    if image_url:
        return {
            "provider": "openai",
            "model": model,
            "image": image_url,
            "media_tag": None,
            "image_mode": "remote_url",
            "images_generated": 1,
            "quality": quality,
            "background": background,
            "size": size,
            "operation": "edit" if edit_image_paths else "generate",
            "source_images": [str(path.resolve()) for path in edit_image_paths] if edit_image_paths else [],
            "input_fidelity": fidelity if edit_image_paths else None,
        }

    raise ValueError("OpenAI image generation returned neither b64_json nor url")


def image_generate_tool(
    prompt: str,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    num_images: int = DEFAULT_NUM_IMAGES,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    seed: Optional[int] = None,
    image_path: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
    mask_path: Optional[str] = None,
    input_fidelity: str = "high",
) -> str:
    aspect_ratio_lower = aspect_ratio.lower().strip() if aspect_ratio else DEFAULT_ASPECT_RATIO
    if aspect_ratio_lower not in VALID_ASPECT_RATIOS:
        logger.warning("Invalid aspect_ratio '%s', defaulting to '%s'", aspect_ratio, DEFAULT_ASPECT_RATIO)
        aspect_ratio_lower = DEFAULT_ASPECT_RATIO

    debug_call_data = {
        "parameters": {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio_lower,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "output_format": output_format,
            "seed": seed,
            "image_path": image_path,
            "image_paths": image_paths,
            "mask_path": mask_path,
            "input_fidelity": input_fidelity,
        },
        "provider": None,
        "model": None,
        "error": None,
        "success": False,
        "images_generated": 0,
        "generation_time": 0,
    }

    start_time = datetime.datetime.now()
    try:
        if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
            raise ValueError("Prompt is required and must be a non-empty string")

        provider = _resolve_image_provider()
        if provider == "none":
            raise ValueError("No configured image provider available (set OPENAI_API_KEY or FAL_KEY)")

        output_format = (output_format or DEFAULT_OUTPUT_FORMAT).lower().strip()
        if output_format == "jpg":
            output_format = "jpeg"

        edit_image_paths = _resolve_edit_image_paths(image_path=image_path, image_paths=image_paths)
        resolved_mask_path = None
        if mask_path:
            resolved_mask_path = _resolve_existing_file(mask_path, "mask_path")
            mask_mime = _detect_local_image_mime_type(resolved_mask_path)
            if mask_mime not in {"image/png", "image/jpeg", "image/webp"}:
                raise ValueError("mask_path must point to a PNG, JPEG, or WEBP image")

        logger.info(
            "%s image with provider %s: %s",
            "Editing" if edit_image_paths else "Generating",
            provider,
            prompt[:80],
        )
        if provider == "openai":
            result_data = _generate_via_openai(
                prompt=prompt,
                aspect_ratio_lower=aspect_ratio_lower,
                output_format=output_format,
                edit_image_paths=edit_image_paths,
                mask_path=resolved_mask_path,
                input_fidelity=input_fidelity,
            )
        else:
            if edit_image_paths:
                raise ValueError("Editing an existing image currently requires the OpenAI image provider")
            result_data = _generate_via_fal(
                prompt=prompt,
                aspect_ratio_lower=aspect_ratio_lower,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images=num_images,
                output_format=output_format,
                seed=seed,
            )

        generation_time = (datetime.datetime.now() - start_time).total_seconds()
        response_data = {
            "success": True,
            "provider": result_data["provider"],
            "model": result_data["model"],
            "image": result_data["image"],
            "media_tag": result_data.get("media_tag"),
        }
        if result_data.get("image_mode"):
            response_data["image_mode"] = result_data["image_mode"]
        if result_data.get("operation"):
            response_data["operation"] = result_data["operation"]
        if result_data.get("source_images"):
            response_data["source_images"] = result_data["source_images"]

        debug_call_data["provider"] = result_data["provider"]
        debug_call_data["model"] = result_data["model"]
        debug_call_data["success"] = True
        debug_call_data["images_generated"] = int(result_data.get("images_generated", 1))
        debug_call_data["generation_time"] = generation_time
        _debug.log_call("image_generate_tool", debug_call_data)
        _debug.save()
        return json.dumps(response_data, indent=2, ensure_ascii=False)

    except Exception as e:
        generation_time = (datetime.datetime.now() - start_time).total_seconds()
        error_msg = f"Error generating image: {str(e)}"
        logger.error("%s", error_msg, exc_info=True)
        debug_call_data["error"] = error_msg
        debug_call_data["generation_time"] = generation_time
        _debug.log_call("image_generate_tool", debug_call_data)
        _debug.save()
        return json.dumps({"success": False, "image": None}, indent=2, ensure_ascii=False)


def check_openai_image_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("VOICE_TOOLS_OPENAI_KEY"))


def check_fal_api_key() -> bool:
    return bool(os.getenv("FAL_KEY"))


def check_image_generation_requirements() -> bool:
    provider = _resolve_image_provider()
    if provider == "openai":
        return check_openai_image_api_key()
    if provider == "fal":
        return check_fal_api_key() and fal_client is not None
    return False


def get_debug_session_info() -> Dict[str, Any]:
    return _debug.get_session_info()


if __name__ == "__main__":
    provider = _resolve_image_provider()
    print("🎨 Image Generation Tools")
    print("=" * 40)
    print(f"Provider: {provider or 'none'}")
    if provider == "openai":
        print(f"Model: {os.getenv('OPENAI_IMAGE_MODEL', DEFAULT_OPENAI_MODEL)}")
        print(f"Output dir: {_generated_image_dir()}")
    elif provider == "fal":
        print(f"Model: {DEFAULT_FAL_MODEL}")
    else:
        print("No configured provider. Set IMAGE_GEN_PROVIDER=openai and OPENAI_API_KEY.")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry

IMAGE_GENERATE_SCHEMA = {
    "name": "image_generate",
    "description": (
        "Generate a new image or edit an existing local image from a text prompt. "
        "For edits, pass image_path or image_paths pointing to local cached images "
        "(for example Telegram/Discord cached image paths that Hermes already saw). "
        "OpenAI image generation/editing is preferred when OPENAI_API_KEY is configured. "
        "Returns JSON with either a hosted image URL or a local absolute image file path plus "
        "a MEDIA:/path tag. If a local media_tag is returned, include it verbatim in your "
        "response so Hermes can send the image as an attachment."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The text prompt describing the desired image. Be detailed and descriptive.",
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["landscape", "square", "portrait"],
                "description": "The aspect ratio of the generated image. landscape is wide, portrait is tall, square is 1:1.",
                "default": "landscape",
            },
            "image_path": {
                "type": "string",
                "description": "Optional local absolute path to an existing image to edit instead of generating from scratch.",
            },
            "image_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of local absolute image paths to use as edit/reference inputs.",
            },
            "mask_path": {
                "type": "string",
                "description": "Optional local absolute path to a mask image for selective edits.",
            },
            "input_fidelity": {
                "type": "string",
                "enum": ["high", "low"],
                "description": "How closely OpenAI should preserve the uploaded image(s) during edits.",
                "default": "high",
            },
        },
        "required": ["prompt"],
    },
}


def _handle_image_generate(args, **kw):
    prompt = args.get("prompt", "")
    if not prompt:
        return json.dumps({"error": "prompt is required for image generation"})
    return image_generate_tool(
        prompt=prompt,
        aspect_ratio=args.get("aspect_ratio", "landscape"),
        num_inference_steps=50,
        guidance_scale=4.5,
        num_images=1,
        output_format="png",
        seed=None,
        image_path=args.get("image_path"),
        image_paths=args.get("image_paths"),
        mask_path=args.get("mask_path"),
        input_fidelity=args.get("input_fidelity", "high"),
    )


registry.register(
    name="image_generate",
    toolset="image_gen",
    schema=IMAGE_GENERATE_SCHEMA,
    handler=_handle_image_generate,
    check_fn=check_image_generation_requirements,
    requires_env=["OPENAI_API_KEY"],
    is_async=False,
    emoji="🎨",
)
