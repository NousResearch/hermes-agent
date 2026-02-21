#!/usr/bin/env python3
"""
Image Generation Tools Module

This module provides image generation tools using FAL.ai's FLUX 2 Pro model with 
automatic upscaling via FAL.ai's Clarity Upscaler for enhanced image quality.

Available tools:
- image_generate_tool: Generate images from text prompts with automatic upscaling

Features:
- High-quality image generation using FLUX 2 Pro model
- Automatic 2x upscaling using Clarity Upscaler for enhanced quality
- Comprehensive parameter control (size, steps, guidance, etc.)
- Proper error handling and validation with fallback to original images
- Debug logging support
- Sync mode for immediate results

Usage:
    from image_generation_tool import image_generate_tool
    import asyncio
    
    # Generate and automatically upscale an image
    result = await image_generate_tool(
        prompt="A serene mountain landscape with cherry blossoms",
        image_size="landscape_4_3",
        num_images=1
    )
"""

import json
import logging
import os
import asyncio
import datetime
from typing import Dict, Any, Optional, Union
import fal_client
from tools.debug_helpers import DebugSession

logger = logging.getLogger(__name__)

# Configuration for image generation
DEFAULT_MODEL = "fal-ai/flux-2-pro"
DEFAULT_ASPECT_RATIO = "landscape"
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 4.5
DEFAULT_NUM_IMAGES = 1
DEFAULT_OUTPUT_FORMAT = "png"

# Safety settings
ENABLE_SAFETY_CHECKER = False
SAFETY_TOLERANCE = "5"  # Maximum tolerance (1-5, where 5 is most permissive)

# Aspect ratio mapping - simplified choices for model to select
ASPECT_RATIO_MAP = {
    "landscape": "landscape_16_9",
    "square": "square_hd",
    "portrait": "portrait_16_9"
}
VALID_ASPECT_RATIOS = list(ASPECT_RATIO_MAP.keys())

# Configuration for automatic upscaling
UPSCALER_MODEL = "fal-ai/clarity-upscaler"
UPSCALER_FACTOR = 2
UPSCALER_SAFETY_CHECKER = False
UPSCALER_DEFAULT_PROMPT = "masterpiece, best quality, highres"
UPSCALER_NEGATIVE_PROMPT = "(worst quality, low quality, normal quality:2)"
UPSCALER_CREATIVITY = 0.35
UPSCALER_RESEMBLANCE = 0.6
UPSCALER_GUIDANCE_SCALE = 4
UPSCALER_NUM_INFERENCE_STEPS = 18

# Valid parameter values for validation based on FLUX 2 Pro documentation
VALID_IMAGE_SIZES = [
    "square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"
]
VALID_OUTPUT_FORMATS = ["jpeg", "png"]
VALID_ACCELERATION_MODES = ["none", "regular", "high"]

_debug = DebugSession("image_tools", env_var="IMAGE_TOOLS_DEBUG")


def _validate_parameters(
    image_size: Union[str, Dict[str, int]], 
    num_inference_steps: int,
    guidance_scale: float,
    num_images: int,
    output_format: str,
    acceleration: str = "none"
) -> Dict[str, Any]:
    """
    Validate and normalize image generation parameters for FLUX 2 Pro model.
    
    Args:
        image_size: Either a preset string or custom size dict
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale value
        num_images: Number of images to generate
        output_format: Output format for images
        acceleration: Acceleration mode for generation speed
    
    Returns:
        Dict[str, Any]: Validated and normalized parameters
    
    Raises:
        ValueError: If any parameter is invalid
    """
    validated = {}
    
    # Validate image_size
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
    
    # Validate num_inference_steps
    if not isinstance(num_inference_steps, int) or num_inference_steps < 1 or num_inference_steps > 100:
        raise ValueError("num_inference_steps must be an integer between 1 and 100")
    validated["num_inference_steps"] = num_inference_steps
    
    # Validate guidance_scale (FLUX 2 Pro default is 4.5)
    if not isinstance(guidance_scale, (int, float)) or guidance_scale < 0.1 or guidance_scale > 20.0:
        raise ValueError("guidance_scale must be a number between 0.1 and 20.0")
    validated["guidance_scale"] = float(guidance_scale)
    
    # Validate num_images
    if not isinstance(num_images, int) or num_images < 1 or num_images > 4:
        raise ValueError("num_images must be an integer between 1 and 4")
    validated["num_images"] = num_images
    
    # Validate output_format
    if output_format not in VALID_OUTPUT_FORMATS:
        raise ValueError(f"Invalid output_format '{output_format}'. Must be one of: {VALID_OUTPUT_FORMATS}")
    validated["output_format"] = output_format
    
    # Validate acceleration
    if acceleration not in VALID_ACCELERATION_MODES:
        raise ValueError(f"Invalid acceleration '{acceleration}'. Must be one of: {VALID_ACCELERATION_MODES}")
    validated["acceleration"] = acceleration
    
    return validated


async def _upscale_image(image_url: str, original_prompt: str) -> Dict[str, Any]:
    """
    Upscale an image using FAL.ai's Clarity Upscaler.
    
    Args:
        image_url (str): URL of the image to upscale
        original_prompt (str): Original prompt used to generate the image
    
    Returns:
        Dict[str, Any]: Upscaled image data or None if upscaling fails
    """
    try:
        logger.info("Upscaling image with Clarity Upscaler...")
        
        # Prepare arguments for upscaler
        upscaler_arguments = {
            "image_url": image_url,
            "prompt": f"{UPSCALER_DEFAULT_PROMPT}, {original_prompt}",
            "upscale_factor": UPSCALER_FACTOR,
            "negative_prompt": UPSCALER_NEGATIVE_PROMPT,
            "creativity": UPSCALER_CREATIVITY,
            "resemblance": UPSCALER_RESEMBLANCE,
            "guidance_scale": UPSCALER_GUIDANCE_SCALE,
            "num_inference_steps": UPSCALER_NUM_INFERENCE_STEPS,
            "enable_safety_checker": UPSCALER_SAFETY_CHECKER
        }
        
        # Submit upscaler request
        handler = await fal_client.submit_async(
            UPSCALER_MODEL,
            arguments=upscaler_arguments
        )
        
        # Get the upscaled result
        result = await handler.get()
        
        if result and "image" in result:
            upscaled_image = result["image"]
            logger.info("Image upscaled successfully to %sx%s", upscaled_image.get('width', 'unknown'), upscaled_image.get('height', 'unknown'))
            return {
                "url": upscaled_image["url"],
                "width": upscaled_image.get("width", 0),
                "height": upscaled_image.get("height", 0),
                "upscaled": True,
                "upscale_factor": UPSCALER_FACTOR
            }
        else:
            logger.error("Upscaler returned invalid response")
            return None
            
    except Exception as e:
        logger.error("Error upscaling image: %s", e)
        return None


async def image_generate_tool(
    prompt: str,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    num_images: int = DEFAULT_NUM_IMAGES,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    seed: Optional[int] = None
) -> str:
    """
    Generate images from text prompts using FAL.ai's FLUX 2 Pro model with automatic upscaling.
    
    This tool uses FAL.ai's FLUX 2 Pro model for high-quality text-to-image generation 
    with extensive customization options. Generated images are automatically upscaled 2x 
    using FAL.ai's Clarity Upscaler for enhanced quality. The final upscaled images are 
    returned as URLs that can be displayed using <img src="{URL}"></img> tags.
    
    Args:
        prompt (str): The text prompt describing the desired image
        aspect_ratio (str): Image aspect ratio - "landscape", "square", or "portrait" (default: "landscape")
        num_inference_steps (int): Number of denoising steps (1-50, default: 50)
        guidance_scale (float): How closely to follow prompt (0.1-20.0, default: 4.5)
        num_images (int): Number of images to generate (1-4, default: 1)
        output_format (str): Image format "jpeg" or "png" (default: "png")
        seed (Optional[int]): Random seed for reproducible results (optional)
    
    Returns:
        str: JSON string containing minimal generation results:
             {
                 "success": bool,
                 "image": str or None  # URL of the upscaled image, or None if failed
             }
    """
    # Validate and map aspect_ratio to actual image_size
    aspect_ratio_lower = aspect_ratio.lower().strip() if aspect_ratio else DEFAULT_ASPECT_RATIO
    if aspect_ratio_lower not in ASPECT_RATIO_MAP:
        logger.warning("Invalid aspect_ratio '%s', defaulting to '%s'", aspect_ratio, DEFAULT_ASPECT_RATIO)
        aspect_ratio_lower = DEFAULT_ASPECT_RATIO
    image_size = ASPECT_RATIO_MAP[aspect_ratio_lower]
    
    debug_call_data = {
        "parameters": {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "output_format": output_format,
            "seed": seed
        },
        "error": None,
        "success": False,
        "images_generated": 0,
        "generation_time": 0
    }
    
    start_time = datetime.datetime.now()
    
    try:
        logger.info("Generating %s image(s) with FLUX 2 Pro: %s", num_images, prompt[:80])
        
        # Validate prompt
        if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
            raise ValueError("Prompt is required and must be a non-empty string")
        
        # Check API key availability
        if not os.getenv("FAL_KEY"):
            raise ValueError("FAL_KEY environment variable not set")
        
        # Validate other parameters
        validated_params = _validate_parameters(
            image_size, num_inference_steps, guidance_scale, num_images, output_format, "none"
        )
        
        # Prepare arguments for FAL.ai FLUX 2 Pro API
        arguments = {
            "prompt": prompt.strip(),
            "image_size": validated_params["image_size"],
            "num_inference_steps": validated_params["num_inference_steps"],
            "guidance_scale": validated_params["guidance_scale"],
            "num_images": validated_params["num_images"],
            "output_format": validated_params["output_format"],
            "enable_safety_checker": ENABLE_SAFETY_CHECKER,
            "safety_tolerance": SAFETY_TOLERANCE,
            "sync_mode": True  # Use sync mode for immediate results
        }
        
        # Add seed if provided
        if seed is not None and isinstance(seed, int):
            arguments["seed"] = seed
        
        logger.info("Submitting generation request to FAL.ai FLUX 2 Pro...")
        logger.info("  Model: %s", DEFAULT_MODEL)
        logger.info("  Aspect Ratio: %s -> %s", aspect_ratio_lower, image_size)
        logger.info("  Steps: %s", validated_params['num_inference_steps'])
        logger.info("  Guidance: %s", validated_params['guidance_scale'])
        
        # Submit request to FAL.ai
        handler = await fal_client.submit_async(
            DEFAULT_MODEL,
            arguments=arguments
        )
        
        # Get the result
        result = await handler.get()
        
        generation_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # Process the response
        if not result or "images" not in result:
            raise ValueError("Invalid response from FAL.ai API - no images returned")
        
        images = result.get("images", [])
        if not images:
            raise ValueError("No images were generated")
        
        # Format image data and upscale images
        formatted_images = []
        for img in images:
            if isinstance(img, dict) and "url" in img:
                original_image = {
                    "url": img["url"],
                    "width": img.get("width", 0),
                    "height": img.get("height", 0)
                }
                
                # Attempt to upscale the image
                upscaled_image = await _upscale_image(img["url"], prompt.strip())
                
                if upscaled_image:
                    # Use upscaled image if successful
                    formatted_images.append(upscaled_image)
                else:
                    # Fall back to original image if upscaling fails
                    logger.warning("Using original image as fallback")
                    original_image["upscaled"] = False
                    formatted_images.append(original_image)
        
        if not formatted_images:
            raise ValueError("No valid image URLs returned from API")
        
        upscaled_count = sum(1 for img in formatted_images if img.get("upscaled", False))
        logger.info("Generated %s image(s) in %.1fs (%s upscaled)", len(formatted_images), generation_time, upscaled_count)
        
        # Prepare successful response - minimal format
        response_data = {
            "success": True,
            "image": formatted_images[0]["url"] if formatted_images else None
        }
        
        debug_call_data["success"] = True
        debug_call_data["images_generated"] = len(formatted_images)
        debug_call_data["generation_time"] = generation_time
        
        # Log debug information
        _debug.log_call("image_generate_tool", debug_call_data)
        _debug.save()
        
        return json.dumps(response_data, indent=2, ensure_ascii=False)
        
    except Exception as e:
        generation_time = (datetime.datetime.now() - start_time).total_seconds()
        error_msg = f"Error generating image: {str(e)}"
        logger.error("%s", error_msg)
        
        # Prepare error response - minimal format
        response_data = {
            "success": False,
            "image": None
        }
        
        debug_call_data["error"] = error_msg
        debug_call_data["generation_time"] = generation_time
        _debug.log_call("image_generate_tool", debug_call_data)
        _debug.save()
        
        return json.dumps(response_data, indent=2, ensure_ascii=False)


def check_fal_api_key() -> bool:
    """
    Check if the FAL.ai API key is available in environment variables.
    
    Returns:
        bool: True if API key is set, False otherwise
    """
    return bool(os.getenv("FAL_KEY"))


def check_image_generation_requirements() -> bool:
    """
    Check if all requirements for image generation tools are met.
    
    Returns:
        bool: True if requirements are met, False otherwise
    """
    try:
        # Check API key
        if not check_fal_api_key():
            return False
        
        # Check if fal_client is available
        import fal_client
        return True
        
    except ImportError:
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
    print("🎨 Image Generation Tools Module - FLUX 2 Pro + Auto Upscaling")
    print("=" * 60)
    
    # Check if API key is available
    api_available = check_fal_api_key()
    
    if not api_available:
        print("❌ FAL_KEY environment variable not set")
        print("Please set your API key: export FAL_KEY='your-key-here'")
        print("Get API key at: https://fal.ai/")
        exit(1)
    else:
        print("✅ FAL.ai API key found")
    
    # Check if fal_client is available
    try:
        import fal_client
        print("✅ fal_client library available")
    except ImportError:
        print("❌ fal_client library not found")
        print("Please install: pip install fal-client")
        exit(1)
    
    print("🛠️ Image generation tools ready for use!")
    print(f"🤖 Using model: {DEFAULT_MODEL}")
    print(f"🔍 Auto-upscaling with: {UPSCALER_MODEL} ({UPSCALER_FACTOR}x)")
    
    # Show debug mode status
    if _debug.active:
        print(f"🐛 Debug mode ENABLED - Session ID: {_debug.session_id}")
        print(f"   Debug logs will be saved to: ./logs/image_tools_debug_{_debug.session_id}.json")
    else:
        print("🐛 Debug mode disabled (set IMAGE_TOOLS_DEBUG=true to enable)")
    
    print("\nBasic usage:")
    print("  from image_generation_tool import image_generate_tool")
    print("  import asyncio")
    print("")
    print("  async def main():")
    print("      # Generate image with automatic 2x upscaling")
    print("      result = await image_generate_tool(")
    print("          prompt='A serene mountain landscape with cherry blossoms',")
    print("          image_size='landscape_4_3',")
    print("          num_images=1")
    print("      )")
    print("      print(result)")
    print("  asyncio.run(main())")
    
    print("\nSupported image sizes:")
    for size in VALID_IMAGE_SIZES:
        print(f"  - {size}")
    print("  - Custom: {'width': 512, 'height': 768} (if needed)")
    
    print("\nAcceleration modes:")
    for mode in VALID_ACCELERATION_MODES:
        print(f"  - {mode}")
    
    print("\nExample prompts:")
    print("  - 'A candid street photo of a woman with a pink bob and bold eyeliner'")
    print("  - 'Modern architecture building with glass facade, sunset lighting'")
    print("  - 'Abstract art with vibrant colors and geometric patterns'")
    print("  - 'Portrait of a wise old owl perched on ancient tree branch'")
    print("  - 'Futuristic cityscape with flying cars and neon lights'")
    
    print("\nDebug mode:")
    print("  # Enable debug logging")
    print("  export IMAGE_TOOLS_DEBUG=true")
    print("  # Debug logs capture all image generation calls and results")
    print("  # Logs saved to: ./logs/image_tools_debug_UUID.json")
