#!/usr/bin/env python3
"""
Tools Package

This package contains all the specific tool implementations for the Hermes Agent.
Each module provides specialized functionality for different capabilities:

- web_tools: Web search, content extraction, and crawling
- terminal_tool: Command execution (local/docker/modal/daytona/ssh/singularity backends)
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

# Backward-compat: some tests/consumers expect `import tools; tools.web_tools`
# to exist as a module attribute (not just exported functions).
from . import web_tools as web_tools  # noqa: F401

# Primary terminal tool (local/docker/singularity/modal/daytona/ssh)
from .terminal_tool import (
    terminal_tool,
    check_terminal_requirements,
    cleanup_vm,
    cleanup_all_environments,
    get_active_environments_info,
    register_task_env_overrides,
    clear_task_env_overrides,
    TERMINAL_TOOL_DESCRIPTION
)

from .vision_tools import (
    vision_analyze_tool,
    check_vision_requirements
)

# Backward-compat: expose the module as `tools.vision_tools`
from . import vision_tools as vision_tools  # noqa: F401

from .mixture_of_agents_tool import (
    mixture_of_agents_tool,
    check_moa_requirements
)

Keep package import side effects minimal. Importing ``tools`` should not
eagerly import the full tool stack, because several subsystems load tools while
``hermes_cli.config`` is still initializing.

Callers should import concrete submodules directly, for example:

    import tools.web_tools
    from tools import browser_tool

Python will resolve those submodules via the package path without needing them
to be re-exported here.
"""


def check_file_requirements():
    """File tools only require terminal backend availability."""
    from .terminal_tool import check_terminal_requirements

    return check_terminal_requirements()


__all__ = ["check_file_requirements"]
