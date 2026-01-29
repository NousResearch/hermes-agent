#!/usr/bin/env python3
"""
Tools Package

This package contains all the specific tool implementations for the Hermes Agent.
Each module provides specialized functionality for different capabilities:

- web_tools: Web search, content extraction, and crawling
- terminal_tool: Command execution using mini-swe-agent (local/docker/modal backends)
- terminal_hecate: Command execution on MorphCloud/Hecate cloud VMs (alternative backend)
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

# Primary terminal tool (mini-swe-agent backend: local/docker/singularity/modal)
from .terminal_tool import (
    terminal_tool,
    check_terminal_requirements,
    cleanup_vm,
    cleanup_all_environments,
    get_active_environments_info,
    TERMINAL_TOOL_DESCRIPTION
)

# Alternative terminal tool (Hecate/MorphCloud cloud VMs)
from .terminal_hecate import (
    terminal_hecate_tool,
    check_hecate_requirements,
    TERMINAL_HECATE_DESCRIPTION
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

# Browser automation tools (agent-browser + Browserbase)
from .browser_tool import (
    browser_navigate,
    browser_snapshot,
    browser_click,
    browser_type,
    browser_scroll,
    browser_back,
    browser_press,
    browser_close,
    browser_get_images,
    browser_vision,
    cleanup_browser,
    cleanup_all_browsers,
    get_active_browser_sessions,
    check_browser_requirements,
    BROWSER_TOOL_SCHEMAS
)

__all__ = [
    # Web tools
    'web_search_tool',
    'web_extract_tool',
    'web_crawl_tool',
    'check_firecrawl_api_key',
    # Terminal tools (mini-swe-agent backend)
    'terminal_tool',
    'check_terminal_requirements',
    'cleanup_vm',
    'cleanup_all_environments',
    'get_active_environments_info',
    'TERMINAL_TOOL_DESCRIPTION',
    # Terminal tools (Hecate/MorphCloud backend)
    'terminal_hecate_tool',
    'check_hecate_requirements',
    'TERMINAL_HECATE_DESCRIPTION',
    # Vision tools
    'vision_analyze_tool',
    'check_vision_requirements',
    # MoA tools
    'mixture_of_agents_tool',
    'check_moa_requirements',
    # Image generation tools
    'image_generate_tool',
    'check_image_generation_requirements',
    # Browser automation tools
    'browser_navigate',
    'browser_snapshot',
    'browser_click',
    'browser_type',
    'browser_scroll',
    'browser_back',
    'browser_press',
    'browser_close',
    'browser_get_images',
    'browser_vision',
    'cleanup_browser',
    'cleanup_all_browsers',
    'get_active_browser_sessions',
    'check_browser_requirements',
    'BROWSER_TOOL_SCHEMAS',
]

