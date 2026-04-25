"""Tool group definitions for lazy loading.

This module defines which tool modules belong to which groups,
enabling selective/lazy loading of non-core tools on demand.
"""

# Tools that are always loaded eagerly (core group) at startup
CORE_TOOLS = [
    "terminal_tool",       # terminal, process
    "file_tools",         # read_file, write_file, patch, search_files
    "todo_tool",          # todo
    "memory_tool",        # memory
    "delegate_tool",      # delegate_task
    "cronjob_tools",      # cronjob
    "send_message_tool",  # send_message
]

# Non-core tool groups that can be lazy-loaded on demand
TOOL_GROUPS = {
    "core": CORE_TOOLS,
    "web": [
        "web_tools",        # web_search, web_extract
        "browser_tool",     # browser_navigate, browser_snapshot, etc.
        "vision_tools",     # vision_analyze
    ],
    "productivity": [
        "tts_tool",         # text_to_speech
        "image_generation_tool",  # image_generate
    ],
    "skills": [
        "skills_tool",      # skills_list, skill_manage
        "skill_manager_tool",  # skill_view
    ],
    "dev": [
        "code_execution_tool",  # execute_code
        "rl_training_tool",    # rl_list_environments, etc.
    ],
    "home": [
        "homeassistant_tool",  # ha_list_entities, etc.
    ],
    "session": [
        "session_search_tool",  # session_search
    ],
    "moa": [
        "mixture_of_agents_tool",  # mixture_of_agents
    ],
    "utility": [
        "clarify_tool",      # clarify
        "process_registry",  # process info
    ],
}

# Map of tool name -> group it belongs to (for lazy loading trigger)
TOOL_TO_GROUP: dict[str, str] = {}
for group_name, modules in TOOL_GROUPS.items():
    for module in modules:
        TOOL_TO_GROUP[module] = group_name

# Module name -> toolset name mapping (for discover_builtin_tools filtering)
# This is used to map file names to their toolset identifiers
MODULE_TO_TOOLSET = {
    "terminal_tool": "terminal",
    "file_tools": "file",
    "todo_tool": "todo",
    "memory_tool": "memory",
    "delegate_tool": "delegation",
    "cronjob_tools": "cronjob",
    "send_message_tool": "messaging",
    "web_tools": "web",
    "browser_tool": "browser",
    "vision_tools": "vision",
    "tts_tool": "tts",
    "image_generation_tool": "image_gen",
    "skills_tool": "skills",
    "skill_manager_tool": "skills",
    "code_execution_tool": "code_execution",
    "rl_training_tool": "rl",
    "homeassistant_tool": "homeassistant",
    "session_search_tool": "session_search",
    "mixture_of_agents_tool": "moa",
    "clarify_tool": "clarify",
    "process_registry": "terminal",
}
