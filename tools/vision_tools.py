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
import os
import asyncio
import uuid
import datetime
import base64
from pathlib import Path
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
import httpx  # Use httpx for async HTTP requests

# Initialize OpenRouter API client for vision processing
openrouter_client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Configuration for vision processing
DEFAULT_VISION_MODEL = "google/gemini-3-flash-preview"

# Debug mode configuration
DEBUG_MODE = os.getenv("VISION_TOOLS_DEBUG", "false").lower() == "true"
DEBUG_SESSION_ID = str(uuid.uuid4())
DEBUG_LOG_PATH = Path("./logs")
DEBUG_DATA = {
    "session_id": DEBUG_SESSION_ID,
    "start_time": datetime.datetime.now().isoformat(),
    "debug_enabled": DEBUG_MODE,
    "tool_calls": []
} if DEBUG_MODE else None

# Create logs directory if debug mode is enabled
if DEBUG_MODE:
    DEBUG_LOG_PATH.mkdir(exist_ok=True)
    print(f"🐛 Vision debug mode enabled - Session ID: {DEBUG_SESSION_ID}")


def _log_debug_call(tool_name: str, call_data: Dict[str, Any]) -> None:
    """
    Log a debug call entry to the global debug data structure.
    
    Args:
        tool_name (str): Name of the tool being called
        call_data (Dict[str, Any]): Data about the call including parameters and results
    """
    if not DEBUG_MODE or not DEBUG_DATA:
        return
    
    call_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "tool_name": tool_name,
        **call_data
    }
    
    DEBUG_DATA["tool_calls"].append(call_entry)


def _save_debug_log() -> None:
    """
    Save the current debug data to a JSON file in the logs directory.
    """
    if not DEBUG_MODE or not DEBUG_DATA:
        return
    
    try:
        debug_filename = f"vision_tools_debug_{DEBUG_SESSION_ID}.json"
        debug_filepath = DEBUG_LOG_PATH / debug_filename
        
        # Update end time
        DEBUG_DATA["end_time"] = datetime.datetime.now().isoformat()
        DEBUG_DATA["total_calls"] = len(DEBUG_DATA["tool_calls"])
        
        with open(debug_filepath, 'w', encoding='utf-8') as f:
            json.dump(DEBUG_DATA, f, indent=2, ensure_ascii=False)
        
        print(f"🐛 Vision debug log saved: {debug_filepath}")
        
    except Exception as e:
        print(f"❌ Error saving vision debug log: {str(e)}")


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
                print(f"⚠️  Image download failed (attempt {attempt + 1}/{max_retries}): {str(e)[:50]}")
                print(f"   Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"❌ Image download failed after {max_retries} attempts: {str(e)[:100]}")
    
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
    Analyze an image from a URL using vision AI.
    
    This tool downloads images from URLs, converts them to base64, and processes
    them using Gemini 3 Flash Preview via OpenRouter API. The image is downloaded to a
    temporary location and automatically cleaned up after processing.
    
    The user_prompt parameter is expected to be pre-formatted by the calling
    function (typically model_tools.py) to include both full description
    requests and specific questions.
    
    Args:
        image_url (str): The URL of the image to analyze (must be http:// or https://)
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
        - Temporary images are stored in ./temp_vision_images/
        - Images are automatically deleted after processing
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
    
    try:
        print(f"🔍 Analyzing image from URL: {image_url[:60]}{'...' if len(image_url) > 60 else ''}", flush=True)
        print(f"📝 User prompt: {user_prompt[:100]}{'...' if len(user_prompt) > 100 else ''}", flush=True)
        
        # Validate image URL
        if not _validate_image_url(image_url):
            raise ValueError("Invalid image URL format. Must start with http:// or https://")
        
        # Check API key availability
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        # Download the image to a temporary location
        print(f"⬇️  Downloading image from URL...", flush=True)
        temp_dir = Path("./temp_vision_images")
        temp_image_path = temp_dir / f"temp_image_{uuid.uuid4()}.jpg"
        
        await _download_image(image_url, temp_image_path)
        
        # Get image file size for logging
        image_size_bytes = temp_image_path.stat().st_size
        image_size_kb = image_size_bytes / 1024
        print(f"✅ Image downloaded successfully ({image_size_kb:.1f} KB)", flush=True)
        
        # Convert image to base64 data URL
        print(f"🔄 Converting image to base64...", flush=True)
        image_data_url = _image_to_base64_data_url(temp_image_path)
        # Calculate size in KB for better readability
        data_size_kb = len(image_data_url) / 1024
        print(f"✅ Image converted to base64 ({data_size_kb:.1f} KB)", flush=True)
        
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
        
        print(f"🧠 Processing image with {model}...", flush=True)
        
        # Call the vision API with reasoning enabled
        response = await openrouter_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,  # Low temperature for consistent analysis
            max_tokens=2000,  # Generous limit for detailed analysis
            extra_body={
                "reasoning": {
                    "enabled": True,
                    "effort": "xhigh"
                }
            }
        )
        
        # Extract the analysis
        analysis = response.choices[0].message.content.strip()
        analysis_length = len(analysis)
        
        print(f"✅ Image analysis completed ({analysis_length} characters)", flush=True)
        
        # Prepare successful response
        result = {
            "success": True,
            "analysis": analysis or "There was a problem with the request and the image could not be analyzed."
        }
        
        debug_call_data["success"] = True
        debug_call_data["analysis_length"] = analysis_length
        
        # Log debug information
        _log_debug_call("vision_analyze_tool", debug_call_data)
        _save_debug_log()
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"Error analyzing image: {str(e)}"
        print(f"❌ {error_msg}", flush=True)
        
        # Prepare error response
        result = {
            "success": False,
            "analysis": "There was a problem with the request and the image could not be analyzed."
        }
        
        debug_call_data["error"] = error_msg
        _log_debug_call("vision_analyze_tool", debug_call_data)
        _save_debug_log()
        
        return json.dumps(result, indent=2, ensure_ascii=False)
    
    finally:
        # Clean up temporary image file
        if temp_image_path and temp_image_path.exists():
            try:
                temp_image_path.unlink()
                print(f"🧹 Cleaned up temporary image file", flush=True)
            except Exception as cleanup_error:
                print(f"⚠️  Warning: Could not delete temporary file: {cleanup_error}", flush=True)


def check_openrouter_api_key() -> bool:
    """
    Check if the OpenRouter API key is available in environment variables.
    
    Returns:
        bool: True if API key is set, False otherwise
    """
    return bool(os.getenv("OPENROUTER_API_KEY"))


def check_vision_requirements() -> bool:
    """
    Check if all requirements for vision tools are met.
    
    Returns:
        bool: True if requirements are met, False otherwise
    """
    return check_openrouter_api_key()


def get_debug_session_info() -> Dict[str, Any]:
    """
    Get information about the current debug session.
    
    Returns:
        Dict[str, Any]: Dictionary containing debug session information
    """
    if not DEBUG_MODE or not DEBUG_DATA:
        return {
            "enabled": False,
            "session_id": None,
            "log_path": None,
            "total_calls": 0
        }
    
    return {
        "enabled": True,
        "session_id": DEBUG_SESSION_ID,
        "log_path": str(DEBUG_LOG_PATH / f"vision_tools_debug_{DEBUG_SESSION_ID}.json"),
        "total_calls": len(DEBUG_DATA["tool_calls"])
    }


if __name__ == "__main__":
    """
    Simple test/demo when run directly
    """
    print("👁️ Vision Tools Module")
    print("=" * 40)
    
    # Check if API key is available
    api_available = check_openrouter_api_key()
    
    if not api_available:
        print("❌ OPENROUTER_API_KEY environment variable not set")
        print("Please set your API key: export OPENROUTER_API_KEY='your-key-here'")
        print("Get API key at: https://openrouter.ai/")
        exit(1)
    else:
        print("✅ OpenRouter API key found")
    
    print("🛠️ Vision tools ready for use!")
    print(f"🧠 Using model: {DEFAULT_VISION_MODEL}")
    
    # Show debug mode status
    if DEBUG_MODE:
        print(f"🐛 Debug mode ENABLED - Session ID: {DEBUG_SESSION_ID}")
        print(f"   Debug logs will be saved to: ./logs/vision_tools_debug_{DEBUG_SESSION_ID}.json")
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
