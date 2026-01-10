#!/usr/bin/env python3
"""
Tools Package

This package contains all the specific tool implementations for the Hermes Agent.
Each module provides specialized functionality for different capabilities:

- web_tools: Web search, content extraction, and crawling
- simple_terminal_tool: Simple command execution on virtual machines (no session persistence)
- vision_tools: Image analysis and understanding
- mixture_of_agents_tool: Multi-model collaborative reasoning
- image_generation_tool: Text-to-image generation with upscaling

The tools are imported into model_tools.py which provides a unified interface
for the AI agent to access all capabilities.
"""

# Export all tools for easy importing
from .web_tools import (
    web_search_tool,
    web_extract_tool,
    web_crawl_tool,
    check_firecrawl_api_key
)

from .simple_terminal_tool import (
    simple_terminal_tool,
    check_requirements as check_terminal_requirements,
    cleanup_vm,
    SIMPLE_TERMINAL_TOOL_DESCRIPTION
)

from .vision_tools import (
    vision_analyze_tool,
    check_vision_requirements
)

from .mixture_of_agents_tool import (
    mixture_of_agents_tool,
    check_moa_requirements
)

from .image_generation_tool import (
    image_generate_tool,
    check_image_generation_requirements
)

__all__ = [
    # Web tools
    'web_search_tool',
    'web_extract_tool',
    'web_crawl_tool',
    'check_firecrawl_api_key',
    # Terminal tools (simple - no session persistence)
    'simple_terminal_tool',
    'check_terminal_requirements',
    'cleanup_vm',
    'SIMPLE_TERMINAL_TOOL_DESCRIPTION',
    # Vision tools
    'vision_analyze_tool',
    'check_vision_requirements',
    # MoA tools
    'mixture_of_agents_tool',
    'check_moa_requirements',
    # Image generation tools
    'image_generate_tool',
    'check_image_generation_requirements',
]

