#!/usr/bin/env python3
"""
Tools package.

The package avoids eager imports so optional dependencies do not break import
of unrelated tool modules.
"""

from __future__ import annotations

from importlib import import_module


_EXPORT_MAP = {
    # Web tools
    "web_search_tool": ("tools.web_tools", "web_search_tool"),
    "web_extract_tool": ("tools.web_tools", "web_extract_tool"),
    "web_crawl_tool": ("tools.web_tools", "web_crawl_tool"),
    "check_firecrawl_api_key": ("tools.web_tools", "check_firecrawl_api_key"),
    # Terminal tools
    "terminal_tool": ("tools.terminal_tool", "terminal_tool"),
    "check_terminal_requirements": ("tools.terminal_tool", "check_terminal_requirements"),
    "cleanup_vm": ("tools.terminal_tool", "cleanup_vm"),
    "cleanup_all_environments": ("tools.terminal_tool", "cleanup_all_environments"),
    "get_active_environments_info": ("tools.terminal_tool", "get_active_environments_info"),
    "register_task_env_overrides": ("tools.terminal_tool", "register_task_env_overrides"),
    "clear_task_env_overrides": ("tools.terminal_tool", "clear_task_env_overrides"),
    "TERMINAL_TOOL_DESCRIPTION": ("tools.terminal_tool", "TERMINAL_TOOL_DESCRIPTION"),
    # Vision + image generation
    "vision_analyze_tool": ("tools.vision_tools", "vision_analyze_tool"),
    "check_vision_requirements": ("tools.vision_tools", "check_vision_requirements"),
    "mixture_of_agents_tool": ("tools.mixture_of_agents_tool", "mixture_of_agents_tool"),
    "check_moa_requirements": ("tools.mixture_of_agents_tool", "check_moa_requirements"),
    "image_generate_tool": ("tools.image_generation_tool", "image_generate_tool"),
    "check_image_generation_requirements": ("tools.image_generation_tool", "check_image_generation_requirements"),
    # Skills
    "skills_list": ("tools.skills_tool", "skills_list"),
    "skill_view": ("tools.skills_tool", "skill_view"),
    "check_skills_requirements": ("tools.skills_tool", "check_skills_requirements"),
    "SKILLS_TOOL_DESCRIPTION": ("tools.skills_tool", "SKILLS_TOOL_DESCRIPTION"),
    "skill_manage": ("tools.skill_manager_tool", "skill_manage"),
    "check_skill_manage_requirements": ("tools.skill_manager_tool", "check_skill_manage_requirements"),
    "SKILL_MANAGE_SCHEMA": ("tools.skill_manager_tool", "SKILL_MANAGE_SCHEMA"),
    # Browser
    "browser_navigate": ("tools.browser_tool", "browser_navigate"),
    "browser_snapshot": ("tools.browser_tool", "browser_snapshot"),
    "browser_click": ("tools.browser_tool", "browser_click"),
    "browser_type": ("tools.browser_tool", "browser_type"),
    "browser_scroll": ("tools.browser_tool", "browser_scroll"),
    "browser_back": ("tools.browser_tool", "browser_back"),
    "browser_press": ("tools.browser_tool", "browser_press"),
    "browser_close": ("tools.browser_tool", "browser_close"),
    "browser_get_images": ("tools.browser_tool", "browser_get_images"),
    "browser_vision": ("tools.browser_tool", "browser_vision"),
    "cleanup_browser": ("tools.browser_tool", "cleanup_browser"),
    "cleanup_all_browsers": ("tools.browser_tool", "cleanup_all_browsers"),
    "get_active_browser_sessions": ("tools.browser_tool", "get_active_browser_sessions"),
    "check_browser_requirements": ("tools.browser_tool", "check_browser_requirements"),
    "BROWSER_TOOL_SCHEMAS": ("tools.browser_tool", "BROWSER_TOOL_SCHEMAS"),
    # Cron
    "schedule_cronjob": ("tools.cronjob_tools", "schedule_cronjob"),
    "list_cronjobs": ("tools.cronjob_tools", "list_cronjobs"),
    "remove_cronjob": ("tools.cronjob_tools", "remove_cronjob"),
    "check_cronjob_requirements": ("tools.cronjob_tools", "check_cronjob_requirements"),
    "get_cronjob_tool_definitions": ("tools.cronjob_tools", "get_cronjob_tool_definitions"),
    "SCHEDULE_CRONJOB_SCHEMA": ("tools.cronjob_tools", "SCHEDULE_CRONJOB_SCHEMA"),
    "LIST_CRONJOBS_SCHEMA": ("tools.cronjob_tools", "LIST_CRONJOBS_SCHEMA"),
    "REMOVE_CRONJOBS_SCHEMA": ("tools.cronjob_tools", "REMOVE_CRONJOBS_SCHEMA"),
    # RL
    "rl_list_environments": ("tools.rl_training_tool", "rl_list_environments"),
    "rl_select_environment": ("tools.rl_training_tool", "rl_select_environment"),
    "rl_get_current_config": ("tools.rl_training_tool", "rl_get_current_config"),
    "rl_edit_config": ("tools.rl_training_tool", "rl_edit_config"),
    "rl_start_training": ("tools.rl_training_tool", "rl_start_training"),
    "rl_check_status": ("tools.rl_training_tool", "rl_check_status"),
    "rl_stop_training": ("tools.rl_training_tool", "rl_stop_training"),
    "rl_get_results": ("tools.rl_training_tool", "rl_get_results"),
    "rl_list_runs": ("tools.rl_training_tool", "rl_list_runs"),
    "rl_test_inference": ("tools.rl_training_tool", "rl_test_inference"),
    "check_rl_api_keys": ("tools.rl_training_tool", "check_rl_api_keys"),
    "get_missing_keys": ("tools.rl_training_tool", "get_missing_keys"),
    # Files
    "read_file_tool": ("tools.file_tools", "read_file_tool"),
    "write_file_tool": ("tools.file_tools", "write_file_tool"),
    "patch_tool": ("tools.file_tools", "patch_tool"),
    "search_tool": ("tools.file_tools", "search_tool"),
    "get_file_tools": ("tools.file_tools", "get_file_tools"),
    "clear_file_ops_cache": ("tools.file_tools", "clear_file_ops_cache"),
    # TTS
    "text_to_speech_tool": ("tools.tts_tool", "text_to_speech_tool"),
    "check_tts_requirements": ("tools.tts_tool", "check_tts_requirements"),
    # Research tools
    "research_state_tool": ("tools.research_state_tool", "research_state_tool"),
    "check_research_state_requirements": ("tools.research_state_tool", "check_research_state_requirements"),
    "research_loop_tool": ("tools.research_loop_tool", "research_loop_tool"),
    "check_research_loop_requirements": ("tools.research_loop_tool", "check_research_loop_requirements"),
    "tinker_posttrain_tool": ("tools.tinker_posttrain_tool", "tinker_posttrain_tool"),
    "check_tinker_posttrain_requirements": ("tools.tinker_posttrain_tool", "check_tinker_posttrain_requirements"),
    "research_manager_tool": ("tools.research_manager_tool", "research_manager_tool"),
    "check_research_manager_requirements": ("tools.research_manager_tool", "check_research_manager_requirements"),
    "evaluation_tool": ("tools.evaluation_tool", "evaluation_tool"),
    "check_evaluation_requirements": ("tools.evaluation_tool", "check_evaluation_requirements"),
    "dataset_tool": ("tools.dataset_tool", "dataset_tool"),
    "check_dataset_requirements": ("tools.dataset_tool", "check_dataset_requirements"),
    "literature_tool": ("tools.literature_tool", "literature_tool"),
    "check_literature_requirements": ("tools.literature_tool", "check_literature_requirements"),
    "judge_tool": ("tools.judge_tool", "judge_tool"),
    "check_judge_requirements": ("tools.judge_tool", "check_judge_requirements"),
    # Planning + execution
    "todo_tool": ("tools.todo_tool", "todo_tool"),
    "check_todo_requirements": ("tools.todo_tool", "check_todo_requirements"),
    "TODO_SCHEMA": ("tools.todo_tool", "TODO_SCHEMA"),
    "TodoStore": ("tools.todo_tool", "TodoStore"),
    "clarify_tool": ("tools.clarify_tool", "clarify_tool"),
    "check_clarify_requirements": ("tools.clarify_tool", "check_clarify_requirements"),
    "CLARIFY_SCHEMA": ("tools.clarify_tool", "CLARIFY_SCHEMA"),
    "execute_code": ("tools.code_execution_tool", "execute_code"),
    "check_sandbox_requirements": ("tools.code_execution_tool", "check_sandbox_requirements"),
    "EXECUTE_CODE_SCHEMA": ("tools.code_execution_tool", "EXECUTE_CODE_SCHEMA"),
    "delegate_task": ("tools.delegate_tool", "delegate_task"),
    "check_delegate_requirements": ("tools.delegate_tool", "check_delegate_requirements"),
    "DELEGATE_TASK_SCHEMA": ("tools.delegate_tool", "DELEGATE_TASK_SCHEMA"),
}


def __getattr__(name: str):
    if name == "check_file_requirements":
        return check_file_requirements
    if name not in _EXPORT_MAP:
        raise AttributeError(name)
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def check_file_requirements():
    """File tools only require terminal backend to be available."""
    from .terminal_tool import check_terminal_requirements

    return check_terminal_requirements()


__all__ = sorted(list(_EXPORT_MAP.keys()) + ["check_file_requirements"])
