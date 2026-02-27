#!/usr/bin/env python3
"""
Vision Tools Module

This module provides vision analysis tools that work with image URLs.
Uses Gemini 3 Flash Preview via OpenRouter API for intelligent image understanding.

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

import json
import logging
import os
import asyncio
import uuid
import base64
from pathlib import Path
from typing import Dict, Any, Optional
import httpx
from openai import AsyncOpenAI
from agent.auxiliary_client import get_vision_auxiliary_client
from tools.debug_helpers import DebugSession

logger = logging.getLogger(__name__)

# Resolve vision auxiliary client at module level; build an async wrapper.
_aux_sync_client, DEFAULT_VISION_MODEL = get_vision_auxiliary_client()
_aux_async_client: AsyncOpenAI | None = None
if _aux_sync_client is not None:
    _async_kwargs = {
        "api_key": _aux_sync_client.api_key,
        "base_url": str(_aux_sync_client.base_url),
    }
    if "openrouter" in str(_aux_sync_client.base_url).lower():
        _async_kwargs["default_headers"] = {
            "HTTP-Referer": "https://github.com/NousResearch/hermes-agent",
            "X-OpenRouter-Title": "Hermes Agent",
            "X-OpenRouter-Categories": "cli-agent",
        }
    _aux_async_client = AsyncOpenAI(**_async_kwargs)

_debug = DebugSession("vision_tools", env_var="VISION_TOOLS_DEBUG")


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
    
    # Check if it's a valid URL format
    if not (url.startswith('http://') or url.startswith('https://')):
        return False
    
    # Check for common image extensions (optional, as URLs may not have extensions)
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg']
    
    return True  # Allow all HTTP/HTTPS URLs for flexibility


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
    
    last_error = None
    for attempt in range(max_retries):
        try:
            # Download the image with appropriate headers using async httpx
            # Enable follow_redirects to handle image CDNs that redirect (e.g., Imgur, Picsum)
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(
                    image_url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        "Accept": "image/*,*/*;q=0.8",
                    },
                )
                response.raise_for_status()
                
                # Save the image content
                destination.write_bytes(response.content)
            
            return destination
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)  # 2s, 4s, 8s
                logger.warning("Image download failed (attempt %s/%s): %s", attempt + 1, max_retries, str(e)[:50])
                logger.warning("Retrying in %ss...", wait_time)
                await asyncio.sleep(wait_time)
            else:
                logger.error("Image download failed after %s attempts: %s", max_retries, str(e)[:100])
    
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
    model: str = DEFAULT_VISION_MODEL
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
    
    try:
        from tools.interrupt import is_interrupted
        if is_interrupted():
            return json.dumps({"success": False, "error": "Interrupted"})

        logger.info("Analyzing image: %s", image_url[:60])
        logger.info("User prompt: %s", user_prompt[:100])
        
        # Check auxiliary vision client availability
        if _aux_async_client is None or DEFAULT_VISION_MODEL is None:
            return json.dumps({
                "success": False,
                "analysis": "Vision analysis unavailable: no auxiliary vision model configured. "
                            "Set OPENROUTER_API_KEY or configure Nous Portal to enable vision tools."
            }, indent=2, ensure_ascii=False)
        
        # Determine if this is a local file path or a remote URL
        local_path = Path(image_url)
        if local_path.is_file():
            # Local file path (e.g. from platform image cache) -- skip download
            logger.info("Using local image file: %s", image_url)
            temp_image_path = local_path
            should_cleanup = False  # Don't delete cached/local files
        elif _validate_image_url(image_url):
            # Remote URL -- download to a temporary location
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
        
        # Convert image to base64 data URL
        logger.info("Converting image to base64...")
        image_data_url = _image_to_base64_data_url(temp_image_path)
        # Calculate size in KB for better readability
        data_size_kb = len(image_data_url) / 1024
        logger.info("Image converted to base64 (%.1f KB)", data_size_kb)
        
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
        
        logger.info("Processing image with %s...", model)
        
        # Call the vision API
        from agent.auxiliary_client import get_auxiliary_extra_body, auxiliary_max_tokens_param
        _extra = get_auxiliary_extra_body()
        response = await _aux_async_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            **auxiliary_max_tokens_param(2000),
            **({} if not _extra else {"extra_body": _extra}),
        )
        
        # Extract the analysis
        analysis = response.choices[0].message.content.strip()
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
        logger.error("%s", error_msg)
        
        # Prepare error response
        result = {
            "success": False,
            "analysis": "There was a problem with the request and the image could not be analyzed."
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
                logger.warning("Could not delete temporary file: %s", cleanup_error)


def check_vision_requirements() -> bool:
    """Check if an auxiliary vision model is available."""
    return _aux_async_client is not None


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
        print("Set OPENROUTER_API_KEY or configure Nous Portal to enable vision tools.")
        exit(1)
    else:
        print(f"✅ Vision model available: {DEFAULT_VISION_MODEL}")
    
    print("🛠️ Vision tools ready for use!")
    print(f"🧠 Using model: {DEFAULT_VISION_MODEL}")
    
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
from tools.registry import registry

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


def _handle_vision_analyze(args, **kw):
    image_url = args.get("image_url", "")
    question = args.get("question", "")
    full_prompt = f"Fully describe and explain everything about this image, then answer the following question:\n\n{question}"
    model = DEFAULT_VISION_MODEL or "google/gemini-3-flash-preview"
    return vision_analyze_tool(image_url, full_prompt, model)


registry.register(
    name="vision_analyze",
    toolset="vision",
    schema=VISION_ANALYZE_SCHEMA,
    handler=_handle_vision_analyze,
    check_fn=check_vision_requirements,
    is_async=True,
)
