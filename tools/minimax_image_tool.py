#!/usr/bin/env python3
"""
MiniMax Image Generation Tool

This module provides image generation using MiniMax's Image-01 model via their API.
Uses the same MINIMAX_API_KEY that is used for text/chat models.

API Docs: https://platform.minimax.io/docs/guides/image-generation

Features:
- Text-to-Image generation using MiniMax Image-01 model
- Multiple aspect ratios supported (1:1, 16:9, 9:16, etc.)
- Returns base64 encoded images
- Sync implementation for gateway thread-pool compatibility

Usage:
    from tools.minimax_image_tool import minimax_image_generate_tool
    
    result = minimax_image_generate_tool(
        prompt="a cute kawaii cat with flower crown",
        aspect_ratio="1:1"
    )
"""

import json
import logging
import os
import base64
import datetime
import tempfile
from typing import Optional
import requests

from tools.debug_helpers import DebugSession

logger = logging.getLogger(__name__)

# MiniMax API configuration
MINIMAX_IMAGE_API_URL = "https://api.minimax.io/v1/image_generation"
DEFAULT_MODEL = "image-01"
DEFAULT_ASPECT_RATIO = "1:1"
DEFAULT_RESPONSE_FORMAT = "base64"

# Aspect ratio mapping for MiniMax API
ASPECT_RATIO_MAP = {
    "1:1": "1:1",
    "16:9": "16:9",
    "9:16": "9:16",
    "4:3": "4:3",
    "3:4": "3:4",
}
VALID_ASPECT_RATIOS = list(ASPECT_RATIO_MAP.keys())

_debug = DebugSession("minimax_image_tools", env_var="MINIMAX_IMAGE_TOOLS_DEBUG")


def _get_api_key() -> Optional[str]:
    """Get MiniMax API key from environment or .env file."""
    api_key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if api_key:
        return api_key
    
    # Try reading from .env file
    env_path = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("MINIMAX_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    
    # Try MINIMAX_CN_API_KEY for China endpoint users
    api_key = os.environ.get("MINIMAX_CN_API_KEY", "").strip()
    if api_key:
        return api_key
    
    return None


def _validate_aspect_ratio(aspect_ratio: str) -> str:
    """Validate and normalize aspect ratio."""
    if not aspect_ratio:
        return DEFAULT_ASPECT_RATIO
    aspect_ratio = aspect_ratio.strip()
    if aspect_ratio not in ASPECT_RATIO_MAP:
        logger.warning("Invalid aspect_ratio '%s', defaulting to '%s'", aspect_ratio, DEFAULT_ASPECT_RATIO)
        return DEFAULT_ASPECT_RATIO
    return aspect_ratio


def minimax_image_generate_tool(
    prompt: str,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    num_images: int = 1,
    seed: Optional[int] = None,
) -> str:
    """
    Generate images from text prompts using MiniMax's Image-01 model.
    
    Args:
        prompt (str): The text prompt describing the desired image
        aspect_ratio (str): Image aspect ratio - "1:1", "16:9", "9:16", "4:3", "3:4" (default: "1:1")
        num_images (int): Number of images to generate (1-4, default: 1)
        seed (Optional[int]): Random seed for reproducible results
    
    Returns:
        str: JSON string containing:
            {
                "success": bool,
                "image": str or None,  # Path to saved image file
                "image_base64": str or None,  # Base64 encoded image
                "error": str or None
            }
    """
    debug_data = {
        "parameters": {
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "aspect_ratio": aspect_ratio,
            "num_images": num_images,
            "seed": seed,
        },
        "error": None,
        "success": False,
        "images_generated": 0,
        "generation_time": 0,
    }
    
    start_time = datetime.datetime.now()
    
    try:
        # Validate prompt
        if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
            raise ValueError("Prompt is required and must be a non-empty string")
        
        # Get API key
        api_key = _get_api_key()
        if not api_key:
            raise ValueError("MINIMAX_API_KEY environment variable not set. Get your key at: https://platform.minimax.io/")
        
        # Validate aspect ratio
        validated_aspect_ratio = _validate_aspect_ratio(aspect_ratio)
        
        # Validate num_images
        if not isinstance(num_images, int) or num_images < 1 or num_images > 4:
            raise ValueError("num_images must be an integer between 1 and 4")
        
        # Prepare API request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": DEFAULT_MODEL,
            "prompt": prompt.strip(),
            "aspect_ratio": validated_aspect_ratio,
            "response_format": DEFAULT_RESPONSE_FORMAT,
        }
        
        if num_images > 1:
            payload["num_images"] = num_images
        
        if seed is not None and isinstance(seed, int):
            payload["seed"] = seed
        
        logger.info("Generating %d image(s) with MiniMax Image-01: %s", num_images, prompt[:80])
        
        # Make API request
        response = requests.post(
            MINIMAX_IMAGE_API_URL,
            headers=headers,
            json=payload,
            timeout=120  # Image generation can take a while
        )
        
        generation_time = (datetime.datetime.now() - start_time).total_seconds()
        
        if response.status_code != 200:
            error_msg = response.text[:500] if response.text else "Unknown error"
            raise ValueError(f"MiniMax API error (status {response.status_code}): {error_msg}")
        
        # Parse response
        data = response.json()
        
        # Check for API errors
        if "base_resp" in data and data["base_resp"].get("status_code") != 0:
            error_msg = data["base_resp"].get("status_msg", "Unknown API error")
            raise ValueError(f"MiniMax API error: {error_msg}")
        
        # Extract images from base64
        images_base64 = data.get("data", {}).get("image_base64", [])
        if not images_base64:
            raise ValueError("No images returned from MiniMax API")
        
        # Save images to temp files
        saved_paths = []
        for i, img_b64 in enumerate(images_base64):
            try:
                img_bytes = base64.b64decode(img_b64)
                
                # Determine file extension from base64 header if possible
                # Standard JPEG/PNG signatures
                if img_b64.startswith("/9j/"):
                    ext = "jpg"
                elif img_b64.startswith("iVBOR"):
                    ext = "png"
                else:
                    ext = "jpg"
                
                temp_path = f"/tmp/minimax_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.{ext}"
                with open(temp_path, "wb") as f:
                    f.write(img_bytes)
                saved_paths.append(temp_path)
                
            except Exception as e:
                logger.warning("Failed to decode/save image %d: %s", i, str(e))
                continue
        
        if not saved_paths:
            raise ValueError("Failed to save any images from API response")
        
        logger.info("Generated %d image(s) in %.1fs", len(saved_paths), generation_time)
        
        # Return result with first image path
        response_data = {
            "success": True,
            "image": saved_paths[0],  # Primary image path for display
            "images": saved_paths,  # All generated images
            "image_base64": images_base64[0] if images_base64 else None,
            "generation_time_seconds": generation_time,
        }
        
        debug_data["success"] = True
        debug_data["images_generated"] = len(saved_paths)
        debug_data["generation_time"] = generation_time
        
        _debug.log_call("minimax_image_generate_tool", debug_data)
        _debug.save()
        
        return json.dumps(response_data, indent=2, ensure_ascii=False)
        
    except Exception as e:
        generation_time = (datetime.datetime.now() - start_time).total_seconds()
        error_msg = str(e)
        logger.error("Error generating image with MiniMax: %s", error_msg, exc_info=True)
        
        response_data = {
            "success": False,
            "image": None,
            "images": [],
            "image_base64": None,
            "error": error_msg,
        }
        
        debug_data["error"] = error_msg
        debug_data["generation_time"] = generation_time
        _debug.log_call("minimax_image_generate_tool", debug_data)
        _debug.save()
        
        return json.dumps(response_data, indent=2, ensure_ascii=False)


def check_minimax_image_requirements() -> bool:
    """
    Check if MiniMax API key is available for image generation.
    
    Returns:
        bool: True if API key is set, False otherwise
    """
    return _get_api_key() is not None


def check_minimax_image_library() -> bool:
    """
    Check if the requests library is available.
    
    Returns:
        bool: True if requests is available, False otherwise
    """
    try:
        import requests
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry

MINIMAX_IMAGE_GENERATE_SCHEMA = {
    "name": "minimax_image_generate",
    "description": "Generate images from text prompts using MiniMax's Image-01 model. Uses the same MINIMAX_API_KEY as text/chat models. Returns a local file path to the generated image. Display it using markdown: ![description](file_path) or send it directly to the user. Aspect ratios: 1:1 (square), 16:9 (wide), 9:16 (tall), 4:3, 3:4.",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The text prompt describing the desired image. Be detailed and descriptive for best results."
            },
            "aspect_ratio": {
                "type": "string",
                "enum": VALID_ASPECT_RATIOS,
                "description": f"The aspect ratio of the generated image. Options: {', '.join(VALID_ASPECT_RATIOS)}. Default: {DEFAULT_ASPECT_RATIO}",
                "default": DEFAULT_ASPECT_RATIO
            },
            "num_images": {
                "type": "integer",
                "description": "Number of images to generate (1-4). Default: 1",
                "default": 1
            },
            "seed": {
                "type": "integer",
                "description": "Random seed for reproducible results. Optional."
            }
        },
        "required": ["prompt"]
    }
}


def _handle_minimax_image_generate(args, **kw):
    """Handler function for the registry."""
    prompt = args.get("prompt", "")
    if not prompt:
        return json.dumps({"success": False, "error": "prompt is required"})
    
    return minimax_image_generate_tool(
        prompt=prompt,
        aspect_ratio=args.get("aspect_ratio", DEFAULT_ASPECT_RATIO),
        num_images=args.get("num_images", 1),
        seed=args.get("seed"),
    )


registry.register(
    name="minimax_image_generate",
    toolset="image_gen",
    schema=MINIMAX_IMAGE_GENERATE_SCHEMA,
    handler=_handle_minimax_image_generate,
    check_fn=check_minimax_image_requirements,
    requires_env=["MINIMAX_API_KEY"],
    is_async=False,
    emoji="🎨",
)


if __name__ == "__main__":
    print("🎨 MiniMax Image Generation Tool - Image-01 Model")
    print("=" * 60)
    
    # Check if API key is available
    api_key = _get_api_key()
    if not api_key:
        print("❌ MINIMAX_API_KEY environment variable not set")
        print("Please set your API key: export MINIMAX_API_KEY='your-key-here'")
        print("Get API key at: https://platform.minimax.io/")
        exit(1)
    else:
        print(f"✅ MiniMax API key found: {api_key[:12]}...")
    
    # Check if requests library is available
    try:
        import requests
        print("✅ requests library available")
    except ImportError:
        print("❌ requests library not found")
        exit(1)
    
    print("\n🛠️ MiniMax image generation tool ready!")
    print(f"   API URL: {MINIMAX_IMAGE_API_URL}")
    print(f"   Model: {DEFAULT_MODEL}")
    print(f"   Default aspect ratio: {DEFAULT_ASPECT_RATIO}")
    
    print("\nUsage:")
    print("  from tools.minimax_image_tool import minimax_image_generate_tool")
    print("")
    print("  result = minimax_image_generate_tool(")
    print("      prompt='a cute kawaii cat with flower crown',")
    print("      aspect_ratio='1:1'")
    print("  )")
    print("")
    print("  print(result)  # Returns JSON with image path")
