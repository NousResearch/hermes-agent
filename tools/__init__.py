#!/usr/bin/env python3
"""
Tools Package

This package contains all the specific tool implementations for the Hermes Agent.
Each module provides specialized functionality for different capabilities:

- web_tools: Web search, content extraction, and crawling
- terminal_tool: Command execution using mini-swe-agent (local/docker/modal backends)
- vision_tools: Image analysis and understanding
- mixture_of_agents_tool: Multi-model collaborative reasoning
- image_generation_tool: Text-to-image generation with upscaling

The tools are imported into model_tools.py which provides a unified interface
for the AI agent to access all capabilities.
"""

# Export all tools for easy importing
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

# Clarifying questions tool (interactive Q&A with the user)
from .clarify_tool import (
    CLARIFY_SCHEMA,
    check_clarify_requirements,
    clarify_tool,
)

# Code execution sandbox (programmatic tool calling)
from .code_execution_tool import (
    EXECUTE_CODE_SCHEMA,
    check_sandbox_requirements,
    execute_code,
)

# Cronjob management tools (CLI-only, hermes-cli toolset)
from .cronjob_tools import (
    LIST_CRONJOBS_SCHEMA,
    REMOVE_CRONJOB_SCHEMA,
    SCHEDULE_CRONJOB_SCHEMA,
    check_cronjob_requirements,
    get_cronjob_tool_definitions,
    list_cronjobs,
    remove_cronjob,
    schedule_cronjob,
)

# Subagent delegation (spawn child agents with isolated context)
from .delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    check_delegate_requirements,
    delegate_task,
)

# File manipulation tools (read, write, patch, search)
from .file_tools import (
    clear_file_ops_cache,
    get_file_tools,
    patch_tool,
    read_file_tool,
    search_tool,
    write_file_tool,
)
from .image_generation_tool import check_image_generation_requirements, image_generate_tool
from .mixture_of_agents_tool import check_moa_requirements, mixture_of_agents_tool

# RL Training tools (Tinker-Atropos)
from .rl_training_tool import (
    check_rl_api_keys,
    get_missing_keys,
    rl_check_status,
    rl_edit_config,
    rl_get_current_config,
    rl_get_results,
    rl_list_environments,
    rl_list_runs,
    rl_select_environment,
    rl_start_training,
    rl_stop_training,
    rl_test_inference,
)
from .skill_manager_tool import SKILL_MANAGE_SCHEMA, check_skill_manage_requirements, skill_manage
from .skills_tool import SKILLS_TOOL_DESCRIPTION, check_skills_requirements, skill_view, skills_list

# Primary terminal tool (mini-swe-agent backend: local/docker/singularity/modal)
from .terminal_tool import (
    TERMINAL_TOOL_DESCRIPTION,
    check_terminal_requirements,
    cleanup_all_environments,
    cleanup_vm,
    clear_task_env_overrides,
    get_active_environments_info,
    register_task_env_overrides,
    terminal_tool,
)

# Planning & task management tool
from .todo_tool import (
    TODO_SCHEMA,
    TodoStore,
    check_todo_requirements,
    todo_tool,
)

# Text-to-speech tools (Edge TTS / ElevenLabs / OpenAI)
from .tts_tool import (
    check_tts_requirements,
    text_to_speech_tool,
)
from .vision_tools import check_vision_requirements, vision_analyze_tool
from .web_tools import check_firecrawl_api_key, web_crawl_tool, web_extract_tool, web_search_tool


# File tools have no external requirements - they use the terminal backend
def check_file_requirements():
    """File tools only require terminal backend to be available."""
    from .terminal_tool import check_terminal_requirements

    return check_terminal_requirements()


__all__ = [
    # Web tools
    "web_search_tool",
    "web_extract_tool",
    "web_crawl_tool",
    "check_firecrawl_api_key",
    # Terminal tools (mini-swe-agent backend)
    "terminal_tool",
    "check_terminal_requirements",
    "cleanup_vm",
    "cleanup_all_environments",
    "get_active_environments_info",
    "register_task_env_overrides",
    "clear_task_env_overrides",
    "TERMINAL_TOOL_DESCRIPTION",
    # Vision tools
    "vision_analyze_tool",
    "check_vision_requirements",
    # MoA tools
    "mixture_of_agents_tool",
    "check_moa_requirements",
    # Image generation tools
    "image_generate_tool",
    "check_image_generation_requirements",
    # Skills tools
    "skills_list",
    "skill_view",
    "check_skills_requirements",
    "SKILLS_TOOL_DESCRIPTION",
    # Skill management
    "skill_manage",
    "check_skill_manage_requirements",
    "SKILL_MANAGE_SCHEMA",
    # Browser automation tools
    "browser_navigate",
    "browser_snapshot",
    "browser_click",
    "browser_type",
    "browser_scroll",
    "browser_back",
    "browser_press",
    "browser_close",
    "browser_get_images",
    "browser_vision",
    "cleanup_browser",
    "cleanup_all_browsers",
    "get_active_browser_sessions",
    "check_browser_requirements",
    "BROWSER_TOOL_SCHEMAS",
    # Cronjob management tools (CLI-only)
    "schedule_cronjob",
    "list_cronjobs",
    "remove_cronjob",
    "check_cronjob_requirements",
    "get_cronjob_tool_definitions",
    "SCHEDULE_CRONJOB_SCHEMA",
    "LIST_CRONJOBS_SCHEMA",
    "REMOVE_CRONJOB_SCHEMA",
    # RL Training tools
    "rl_list_environments",
    "rl_select_environment",
    "rl_get_current_config",
    "rl_edit_config",
    "rl_start_training",
    "rl_check_status",
    "rl_stop_training",
    "rl_get_results",
    "rl_list_runs",
    "rl_test_inference",
    "check_rl_api_keys",
    "get_missing_keys",
    # File manipulation tools
    "read_file_tool",
    "write_file_tool",
    "patch_tool",
    "search_tool",
    "get_file_tools",
    "clear_file_ops_cache",
    "check_file_requirements",
    # Text-to-speech tools
    "text_to_speech_tool",
    "check_tts_requirements",
    # Planning & task management tool
    "todo_tool",
    "check_todo_requirements",
    "TODO_SCHEMA",
    "TodoStore",
    # Clarifying questions tool
    "clarify_tool",
    "check_clarify_requirements",
    "CLARIFY_SCHEMA",
    # Code execution sandbox
    "execute_code",
    "check_sandbox_requirements",
    "EXECUTE_CODE_SCHEMA",
    # Subagent delegation
    "delegate_task",
    "check_delegate_requirements",
    "DELEGATE_TASK_SCHEMA",
]
