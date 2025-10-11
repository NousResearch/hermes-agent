#!/usr/bin/env python3
"""
Vision Tools Module

This module provides vision analysis tools that work with image URLs.
Uses Gemini Flash via Nous Research API for intelligent image understanding.

Available tools:
- vision_analyze_tool: Analyze images from URLs with custom prompts

Features:
- Comprehensive image description
- Context-aware analysis based on user queries
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
from dotenv import load_dotenv
import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from openai import AsyncOpenAI

load_dotenv()

# Initialize Nous Research API client for vision processing
nous_client = AsyncOpenAI(
    api_key=os.getenv("NOUS_API_KEY"),
    base_url="https://inference-api.nousresearch.com/v1"
)

# Configuration for vision processing
DEFAULT_VISION_MODEL = "gemini-2.5-flash"

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


async def vision_analyze_tool(
    image_url: str,
    user_prompt: str,
    model: str = DEFAULT_VISION_MODEL
) -> str:
    """
    Analyze an image from a URL using vision AI.
    
    This tool processes images using Gemini Flash via Nous Research API.
    The user_prompt parameter is expected to be pre-formatted by the calling
    function (typically model_tools.py) to include both full description
    requests and specific questions.
    
    Args:
        image_url (str): The URL of the image to analyze
        user_prompt (str): The pre-formatted prompt for the vision model
        model (str): The vision model to use (default: gemini-2.5-flash)
    
    Returns:
        str: JSON string containing the analysis results with the following structure:
             {
                 "success": bool,
                 "analysis": str (defaults to error message if None)
             }
    
    Raises:
        Exception: If analysis fails or API key is not set
    """
    debug_call_data = {
        "parameters": {
            "image_url": image_url,
            "user_prompt": user_prompt,
            "model": model
        },
        "error": None,
        "success": False,
        "analysis_length": 0,
        "model_used": model
    }
    
    try:
        print(f"🔍 Analyzing image from URL: {image_url[:60]}{'...' if len(image_url) > 60 else ''}")
        print(f"📝 User prompt: {user_prompt[:100]}{'...' if len(user_prompt) > 100 else ''}")
        
        # Validate image URL
        if not _validate_image_url(image_url):
            raise ValueError("Invalid image URL format. Must start with http:// or https://")
        
        # Check API key availability
        if not os.getenv("NOUS_API_KEY"):
            raise ValueError("NOUS_API_KEY environment variable not set")
        
        # Use the prompt as provided (model_tools.py now handles full description formatting)
        comprehensive_prompt = user_prompt
        
        # Prepare the message with image URL format
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
                            "url": image_url
                        }
                    }
                ]
            }
        ]
        
        print(f"🧠 Processing image with {model}...")
        
        # Call the vision API
        response = await nous_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,  # Low temperature for consistent analysis
            max_tokens=2000   # Generous limit for detailed analysis
        )
        
        # Extract the analysis
        analysis = response.choices[0].message.content.strip()
        analysis_length = len(analysis)
        
        print(f"✅ Image analysis completed ({analysis_length} characters)")
        
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
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_msg = f"Error analyzing image: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Prepare error response
        result = {
            "success": False,
            "analysis": "There was a problem with the request and the image could not be analyzed."
        }
        
        debug_call_data["error"] = error_msg
        _log_debug_call("vision_analyze_tool", debug_call_data)
        _save_debug_log()
        
        return json.dumps(result, indent=2)


def check_nous_api_key() -> bool:
    """
    Check if the Nous Research API key is available in environment variables.
    
    Returns:
        bool: True if API key is set, False otherwise
    """
    return bool(os.getenv("NOUS_API_KEY"))


def check_vision_requirements() -> bool:
    """
    Check if all requirements for vision tools are met.
    
    Returns:
        bool: True if requirements are met, False otherwise
    """
    return check_nous_api_key()


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
    api_available = check_nous_api_key()
    
    if not api_available:
        print("❌ NOUS_API_KEY environment variable not set")
        print("Please set your API key: export NOUS_API_KEY='your-key-here'")
        print("Get API key at: https://inference-api.nousresearch.com/")
        exit(1)
    else:
        print("✅ Nous Research API key found")
    
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
