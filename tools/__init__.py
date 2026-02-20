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
    register_task_env_overrides,
    clear_task_env_overrides,
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

from .skills_tool import (
    skills_list,
    skill_view,
    check_skills_requirements,
    SKILLS_TOOL_DESCRIPTION
)

from .skill_manager_tool import (
    skill_manage,
    check_skill_manage_requirements,
    SKILL_MANAGE_SCHEMA
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

# Text-to-speech tools (Edge TTS / ElevenLabs / OpenAI)
from .tts_tool import (
    text_to_speech_tool,
    check_tts_requirements,
)

# Planning & task management tool
from .todo_tool import (
    todo_tool,
    check_todo_requirements,
    TODO_SCHEMA,
    TodoStore,
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
    'register_task_env_overrides',
    'clear_task_env_overrides',
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
    'skills_list',
    'skill_view',
    'check_skills_requirements',
    'SKILLS_TOOL_DESCRIPTION',
    # Skill management
    'skill_manage',
    'check_skill_manage_requirements',
    'SKILL_MANAGE_SCHEMA',
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
    # Text-to-speech tools
    'text_to_speech_tool',
    'check_tts_requirements',
    # Planning & task management tool
    'todo_tool',
    'check_todo_requirements',
    'TODO_SCHEMA',
    'TodoStore',
]

