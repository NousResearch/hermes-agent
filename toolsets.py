#!/usr/bin/env python3
"""
Toolsets Module
"""

from typing import List, Dict, Any, Set, Optional


_HERMES_CORE_TOOLS = [
    # Web
    "web_search", "web_extract",
    # Terminal + process management
    "terminal", "process",
    # File manipulation
    "read_file", "write_file", "patch", "search_files",
    # Vision + image generation
    "vision_analyze", "image_generate",
    # MoA
    "mixture_of_agents",
    # Skills
    "skills_list", "skill_view", "skill_manage",
    # Browser automation
    "browser_navigate", "browser_snapshot", "browser_click",
    "browser_type", "browser_scroll", "browser_back",
    "browser_press", "browser_close", "browser_get_images",
    "browser_vision",
    # Text-to-speech
    "text_to_speech",
    # Planning & memory
    "todo", "memory",
    # Session history search
    "session_search",
    # Clarifying questions
    "clarify",
    # Desktop notifications
    "notify", "notify_sound",
    # Pomodoro timer
    "pomodoro_start", "pomodoro_status", "pomodoro_stop", "pomodoro_history",
    # Code execution + delegation
    "execute_code", "delegate_task",
    # Cronjob management
    "schedule_cronjob", "list_cronjobs", "remove_cronjob",
    # Cross-platform messaging (gated on gateway running via check_fn)
    "send_message",
]


TOOLSETS = {
    "web": {
        "description": "Web research and content extraction tools",
        "tools": ["web_search", "web_extract"],
        "includes": []
    },
    "search": {
        "description": "Web search only (no content extraction/scraping)",
        "tools": ["web_search"],
        "includes": []
    },
    "vision": {
        "description": "Image analysis and vision tools",
        "tools": ["vision_analyze"],
        "includes": []
    },
    "image_gen": {
        "description": "Creative generation tools (images)",
        "tools": ["image_generate"],
        "includes": []
    },
    "terminal": {
        "description": "Terminal/command execution and process management tools",
        "tools": ["terminal", "process"],
        "includes": []
    },
    "moa": {
        "description": "Advanced reasoning and problem-solving tools",
        "tools": ["mixture_of_agents"],
        "includes": []
    },
    "skills": {
        "description": "Access, create, edit, and manage skill documents with specialized instructions and knowledge",
        "tools": ["skills_list", "skill_view", "skill_manage"],
        "includes": []
    },
    "browser": {
        "description": "Browser automation for web interaction (navigate, click, type, scroll, iframes, hold-click) with web search for finding URLs",
        "tools": [
            "browser_navigate", "browser_snapshot", "browser_click",
            "browser_type", "browser_scroll", "browser_back",
            "browser_press", "browser_close", "browser_get_images",
            "browser_vision", "web_search"
        ],
        "includes": []
    },
    "cronjob": {
        "description": "Cronjob management tools - schedule, list, and remove automated tasks",
        "tools": ["schedule_cronjob", "list_cronjobs", "remove_cronjob"],
        "includes": []
    },
    "rl": {
        "description": "RL training tools for running reinforcement learning on Tinker-Atropos",
        "tools": [
            "rl_list_environments", "rl_select_environment",
            "rl_get_current_config", "rl_edit_config",
            "rl_start_training", "rl_check_status",
            "rl_stop_training", "rl_get_results",
            "rl_list_runs", "rl_test_inference"
        ],
        "includes": []
    },
    "file": {
        "description": "File manipulation tools: read, write, patch (with fuzzy matching), and search (content + files)",
        "tools": ["read_file", "write_file", "patch", "search_files"],
        "includes": []
    },
    "tts": {
        "description": "Text-to-speech: convert text to audio with Edge TTS (free), ElevenLabs, or OpenAI",
        "tools": ["text_to_speech"],
        "includes": []
    },
    "todo": {
        "description": "Task planning and tracking for multi-step work",
        "tools": ["todo"],
        "includes": []
    },
    "memory": {
        "description": "Persistent memory across sessions (personal notes + user profile)",
        "tools": ["memory"],
        "includes": []
    },
    "session_search": {
        "description": "Search and recall past conversations with summarization",
        "tools": ["session_search"],
        "includes": []
    },
    "clarify": {
        "description": "Ask the user clarifying questions (multiple-choice or open-ended)",
        "tools": ["clarify"],
        "includes": []
    },
    "notification": {
        "description": "Desktop/system notifications — alert the user when tasks complete (Linux, macOS, Windows)",
        "tools": ["notify", "notify_sound"],
        "includes": []
    },
    "pomodoro": {
        "description": "Pomodoro focus timer — work/break sessions with desktop notifications",
        "tools": ["pomodoro_start", "pomodoro_status", "pomodoro_stop", "pomodoro_history"],
        "includes": []
    },
    "code_execution": {
        "description": "Run Python scripts that call tools programmatically (reduces LLM round trips)",
        "tools": ["execute_code"],
        "includes": []
    },
    "delegation": {
        "description": "Spawn subagents with isolated context for complex subtasks",
        "tools": ["delegate_task"],
        "includes": []
    },
    "debugging": {
        "description": "Debugging and troubleshooting toolkit",
        "tools": ["terminal", "process"],
        "includes": ["web", "file"]
    },
    "safe": {
        "description": "Safe toolkit without terminal access",
        "tools": ["mixture_of_agents"],
        "includes": ["web", "vision", "image_gen"]
    },
    "hermes-cli": {
        "description": "Full interactive CLI toolset - all default tools plus cronjob management",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },
    "hermes-telegram": {
        "description": "Telegram bot toolset - full access for personal use (terminal has safety checks)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },
    "hermes-discord": {
        "description": "Discord bot toolset - full access (terminal has safety checks via dangerous command approval)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },
    "hermes-whatsapp": {
        "description": "WhatsApp bot toolset - similar to Telegram (personal messaging, more trusted)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },
    "hermes-slack": {
        "description": "Slack bot toolset - full access for workspace use (terminal has safety checks)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },
    "hermes-gateway": {
        "description": "Gateway toolset - union of all messaging platform tools",
        "tools": [],
        "includes": ["hermes-telegram", "hermes-discord", "hermes-whatsapp", "hermes-slack"]
    }
}


def get_toolset(name: str) -> Optional[Dict[str, Any]]:
    return TOOLSETS.get(name)


def resolve_toolset(name: str, visited: Set[str] = None) -> List[str]:
    if visited is None:
        visited = set()
    if name in {"all", "*"}:
        all_tools: Set[str] = set()
        for toolset_name in get_toolset_names():
            resolved = resolve_toolset(toolset_name, visited.copy())
            all_tools.update(resolved)
        return list(all_tools)
    if name in visited:
        print(f"⚠️  Circular dependency detected in toolset '{name}'")
        return []
    visited.add(name)
    toolset = TOOLSETS.get(name)
    if not toolset:
        return []
    tools = set(toolset.get("tools", []))
    for included_name in toolset.get("includes", []):
        included_tools = resolve_toolset(included_name, visited.copy())
        tools.update(included_tools)
    return list(tools)


def resolve_multiple_toolsets(toolset_names: List[str]) -> List[str]:
    all_tools = set()
    for name in toolset_names:
        tools = resolve_toolset(name)
        all_tools.update(tools)
    return list(all_tools)


def get_all_toolsets() -> Dict[str, Dict[str, Any]]:
    return TOOLSETS.copy()


def get_toolset_names() -> List[str]:
    return list(TOOLSETS.keys())


def validate_toolset(name: str) -> bool:
    if name in {"all", "*"}:
        return True
    return name in TOOLSETS


def create_custom_toolset(name: str, description: str, tools: List[str] = None, includes: List[str] = None) -> None:
    TOOLSETS[name] = {
        "description": description,
        "tools": tools or [],
        "includes": includes or []
    }


def get_toolset_info(name: str) -> Dict[str, Any]:
    toolset = get_toolset(name)
    if not toolset:
        return None
    resolved_tools = resolve_toolset(name)
    return {
        "name": name,
        "description": toolset["description"],
        "direct_tools": toolset["tools"],
        "includes": toolset["includes"],
        "resolved_tools": resolved_tools,
        "tool_count": len(resolved_tools),
        "is_composite": len(toolset["includes"]) > 0
    }


def print_toolset_tree(name: str, indent: int = 0) -> None:
    prefix = "  " * indent
    toolset = get_toolset(name)
    if not toolset:
        print(f"{prefix}❌ Unknown toolset: {name}")
        return
    print(f"{prefix}📦 {name}: {toolset['description']}")
    if toolset["tools"]:
        print(f"{prefix}  🔧 Tools: {', '.join(toolset['tools'])}")
    if toolset["includes"]:
        print(f"{prefix}  📂 Includes:")
        for included in toolset["includes"]:
            print_toolset_tree(included, indent + 2)


if __name__ == "__main__":
    print("Toolsets System Demo")
    print("=" * 60)
    for name, toolset in get_all_toolsets().items():
        info = get_toolset_info(name)
        composite = "[composite]" if info["is_composite"] else "[leaf]"
        print(f"  {composite} {name:20} - {toolset['description']}")
