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

# Primary terminal tool (mini-swe-agent backend: local/docker/singularity/modal)
from .terminal_tool import (
    terminal_tool,
    check_terminal_requirements,
    cleanup_vm,
    cleanup_all_environments,
    get_active_environments_info,
    TERMINAL_TOOL_DESCRIPTION
)

# Optional toolsets: keep imports soft so users can run subsets of tools without
# installing every dependency (requirements gating lives in model_tools.py).
try:
    from .web_tools import check_firecrawl_api_key, web_crawl_tool, web_extract_tool, web_search_tool
except ModuleNotFoundError:  # pragma: no cover
    web_search_tool = None  # type: ignore[assignment]
    web_extract_tool = None  # type: ignore[assignment]
    web_crawl_tool = None  # type: ignore[assignment]

    def check_firecrawl_api_key() -> bool:  # type: ignore[no-redef]
        return False

try:
    # Alternative terminal tool (Hecate/MorphCloud cloud VMs)
    from .terminal_hecate import TERMINAL_HECATE_DESCRIPTION, check_hecate_requirements, terminal_hecate_tool
except ModuleNotFoundError:  # pragma: no cover
    terminal_hecate_tool = None  # type: ignore[assignment]
    TERMINAL_HECATE_DESCRIPTION = ""

    def check_hecate_requirements() -> bool:  # type: ignore[no-redef]
        return False

try:
    from .vision_tools import check_vision_requirements, vision_analyze_tool
except ModuleNotFoundError:  # pragma: no cover
    vision_analyze_tool = None  # type: ignore[assignment]

    def check_vision_requirements() -> bool:  # type: ignore[no-redef]
        return False

try:
    from .mixture_of_agents_tool import check_moa_requirements, mixture_of_agents_tool
except ModuleNotFoundError:  # pragma: no cover
    mixture_of_agents_tool = None  # type: ignore[assignment]

    def check_moa_requirements() -> bool:  # type: ignore[no-redef]
        return False

try:
    from .image_generation_tool import check_image_generation_requirements, image_generate_tool
except ModuleNotFoundError:  # pragma: no cover
    image_generate_tool = None  # type: ignore[assignment]

    def check_image_generation_requirements() -> bool:  # type: ignore[no-redef]
        return False

try:
    from .skills_tool import (
        SKILLS_TOOL_DESCRIPTION,
        check_skills_requirements,
        skill_view,
        skills_categories,
        skills_list,
    )
except ModuleNotFoundError:  # pragma: no cover
    skills_categories = None  # type: ignore[assignment]
    skills_list = None  # type: ignore[assignment]
    skill_view = None  # type: ignore[assignment]
    SKILLS_TOOL_DESCRIPTION = ""

    def check_skills_requirements() -> bool:  # type: ignore[no-redef]
        return False

try:
    # Browser automation tools (agent-browser + Browserbase)
    from .browser_tool import (
        BROWSER_TOOL_SCHEMAS,
        browser_back,
        browser_click,
        browser_close,
        browser_get_images,
        browser_navigate,
        browser_press,
        browser_scroll,
        browser_snapshot,
        browser_type,
        browser_vision,
        check_browser_requirements,
        cleanup_all_browsers,
        cleanup_browser,
        get_active_browser_sessions,
    )
except ModuleNotFoundError:  # pragma: no cover
    browser_navigate = None  # type: ignore[assignment]
    browser_snapshot = None  # type: ignore[assignment]
    browser_click = None  # type: ignore[assignment]
    browser_type = None  # type: ignore[assignment]
    browser_scroll = None  # type: ignore[assignment]
    browser_back = None  # type: ignore[assignment]
    browser_press = None  # type: ignore[assignment]
    browser_close = None  # type: ignore[assignment]
    browser_get_images = None  # type: ignore[assignment]
    browser_vision = None  # type: ignore[assignment]
    cleanup_browser = None  # type: ignore[assignment]
    cleanup_all_browsers = None  # type: ignore[assignment]
    get_active_browser_sessions = None  # type: ignore[assignment]
    BROWSER_TOOL_SCHEMAS = []

    def check_browser_requirements() -> bool:  # type: ignore[no-redef]
        return False

# Cronjob management tools (CLI-only, hermes-cli toolset)
from .cronjob_tools import (
    schedule_cronjob,
    list_cronjobs,
    remove_cronjob,
    check_cronjob_requirements,
    get_cronjob_tool_definitions,
    SCHEDULE_CRONJOB_SCHEMA,
    LIST_CRONJOBS_SCHEMA,
    REMOVE_CRONJOB_SCHEMA
)

# RL Training tools (Tinker-Atropos)
from .rl_training_tool import (
    rl_list_environments,
    rl_select_environment,
    rl_get_current_config,
    rl_edit_config,
    rl_start_training,
    rl_check_status,
    rl_stop_training,
    rl_get_results,
    rl_list_runs,
    rl_test_inference,
    check_rl_api_keys,
    get_missing_keys,
)

# File manipulation tools (read, write, patch, search)
from .file_tools import (
    read_file_tool,
    write_file_tool,
    patch_tool,
    search_tool,
    get_file_tools,
    clear_file_ops_cache,
)

# File tools have no external requirements - they use the terminal backend
def check_file_requirements():
    """File tools only require terminal backend to be available."""
    from .terminal_tool import check_terminal_requirements
    return check_terminal_requirements()

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
    # Skills tools
    'skills_categories',
    'skills_list',
    'skill_view',
    'check_skills_requirements',
    'SKILLS_TOOL_DESCRIPTION',
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
    # Cronjob management tools (CLI-only)
    'schedule_cronjob',
    'list_cronjobs',
    'remove_cronjob',
    'check_cronjob_requirements',
    'get_cronjob_tool_definitions',
    'SCHEDULE_CRONJOB_SCHEMA',
    'LIST_CRONJOBS_SCHEMA',
    'REMOVE_CRONJOB_SCHEMA',
    # RL Training tools
    'rl_list_environments',
    'rl_select_environment',
    'rl_get_current_config',
    'rl_edit_config',
    'rl_start_training',
    'rl_check_status',
    'rl_stop_training',
    'rl_get_results',
    'rl_list_runs',
    'rl_test_inference',
    'check_rl_api_keys',
    'get_missing_keys',
    # File manipulation tools
    'read_file_tool',
    'write_file_tool',
    'patch_tool',
    'search_tool',
    'get_file_tools',
    'clear_file_ops_cache',
    'check_file_requirements',
]
