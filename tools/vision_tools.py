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
from tools.debug_helpers import DebugSession
from tools.website_policy import check_website_access

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
        from hermes_cli.config import load_config
        cfg = load_config()
        val = cfg.get("auxiliary", {}).get("vision", {}).get("download_timeout")
        if val is not None:
            return float(val)
    except Exception:
        pass
    return 30.0

_VISION_DOWNLOAD_TIMEOUT = _resolve_download_timeout()

# Hard cap on downloaded image file size (50 MB). Prevents OOM from
# attacker-hosted multi-gigabyte files or decompression bombs.
_VISION_MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024


def _validate_image_url(url: str) -> bool:
    """
    Basic validation of image URL format.
    
    Args:
        url (str): The URL to validate
        
    Returns:
        bool: True if URL appears to be valid, False otherwise
    """
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

    # Block private/internal addresses to prevent SSRF
    from tools.url_safety import is_safe_url
    if not is_safe_url(url):
        return False

    return True


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
            from tools.url_safety import is_safe_url
            if not is_safe_url(redirect_url):
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
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)  # 2s, 4s, 8s
                logger.warning("Image download failed (attempt %s/%s): %s", attempt + 1, max_retries, str(e)[:50])
                logger.warning("Retrying in %ss...", wait_time)
                await asyncio.sleep(wait_time)
            else:
                logger.error(
                    "Image download failed after %s attempts: %s",
                    max_retries,
                    str(e)[:100],
                    exc_info=True,
                )
    
    if last_error is None:
        raise RuntimeError(
            f"_download_image exited retry loop without attempting (max_retries={max_retries})"
        )
    raise last_error


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
        - For URLs, temporary images are stored in ./temp_vision_images/ and cleaned up
        - For local file paths, the file is used directly and NOT deleted
        - Supports common image formats (JPEG, PNG, GIF, WebP, etc.)
    """
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
        elif _validate_image_url(image_url):
            # Remote URL -- download to a temporary location
            blocked = check_website_access(image_url)
            if blocked:
                raise PermissionError(blocked["message"])
            logger.info("Downloading image from URL...")
            temp_dir = Path("./temp_vision_images")
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
        
        # Convert image to base64 data URL
        logger.info("Converting image to base64...")
        image_data_url = _image_to_base64_data_url(temp_image_path, mime_type=detected_mime_type)
        # Calculate size in KB for better readability
        data_size_kb = len(image_data_url) / 1024
        logger.info("Image converted to base64 (%.1f KB)", data_size_kb)

        # Pre-flight size check: most vision APIs cap base64 payloads at 5 MB.
        # Reject early with a clear message instead of a cryptic provider 400.
        _MAX_BASE64_BYTES = 5 * 1024 * 1024  # 5 MB
        # The data URL includes the header (e.g. "data:image/jpeg;base64,") which
        # is negligible, but measure the full string to be safe.
        if len(image_data_url) > _MAX_BASE64_BYTES:
            raise ValueError(
                f"Image too large for vision API: base64 payload is "
                f"{len(image_data_url) / (1024 * 1024):.1f} MB (limit 5 MB). "
                f"Resize or compress the image and try again."
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
        try:
            from hermes_cli.config import load_config
            _cfg = load_config()
            _vt = _cfg.get("auxiliary", {}).get("vision", {}).get("timeout")
            if _vt is not None:
                vision_timeout = float(_vt)
        except Exception:
            pass
        call_kwargs = {
            "task": "vision",
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2000,
            "timeout": vision_timeout,
        }
        if model:
            call_kwargs["model"] = model
        response = await async_call_llm(**call_kwargs)
        
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
                "image is too large, in an unsupported format, or corrupted. "
                "Try a smaller JPEG/PNG (under 3.5 MB) and retry. "
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
    """Check if the configured runtime vision path can resolve a client."""
    try:
        from agent.auxiliary_client import resolve_vision_provider_client

        _provider, client, _model = resolve_vision_provider_client()
        return client is not None
    except Exception:
        return False


def get_debug_session_info() -> Dict[str, Any]:
    """
    Get information about the current debug session.
    
    Returns:
        Dict[str, Any]: Dictionary containing debug session information
    """
    return _debug.get_session_info()


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
        exit(1)
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
    "description": "Analyze images using AI vision. Provides a comprehensive description and answers a specific question about the image content.",
    "parameters": {
        "type": "object",
        "properties": {
            "image_url": {
                "type": "string",
                "description": "Image URL (http/https) or local file path to analyze."
            },
            "question": {
                "type": "string",
                "description": "Your specific question or request about the image to resolve. The AI will automatically provide a complete image description AND answer your specific question."
            }
        },
        "required": ["image_url", "question"]
    }
}


def _handle_vision_analyze(args: Dict[str, Any], **kw: Any) -> Awaitable[str]:
    image_url = args.get("image_url", "")
    question = args.get("question", "")
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
