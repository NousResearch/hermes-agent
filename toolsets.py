#!/usr/bin/env python3
"""
Toolsets Module

Registry-owned tool membership lives in ``tools.registry``. This module adds
bundle/composition semantics on top of that live registry state so platform
presets, scenario bundles, and compatibility aliases do not need to duplicate
tool ownership.
"""

from typing import Any, Dict, List, Optional, Set

from tools.registry import registry


_HERMES_CORE_TOOLSETS = [
    "web",
    "terminal",
    "file",
    "vision",
    "image_gen",
    "skills",
    "browser",
    "tts",
    "todo",
    "memory",
    "session_search",
    "clarify",
    "code_execution",
    "delegation",
    "cronjob",
    "messaging",
    "homeassistant",
]


LEGACY_TOOLSET_ALIASES: Dict[str, List[str]] = {
    "web_tools": ["web_search", "web_extract"],
    "terminal_tools": ["terminal", "process"],
    "vision_tools": ["vision_analyze"],
    "moa_tools": ["mixture_of_agents"],
    "image_tools": ["image_generate"],
    "skills_tools": ["skills_list", "skill_view", "skill_manage"],
    "browser_tools": [
        "browser_back",
        "browser_click",
        "browser_console",
        "browser_get_images",
        "browser_navigate",
        "browser_press",
        "browser_scroll",
        "browser_snapshot",
        "browser_type",
        "browser_vision",
        "web_search",
    ],
    "cronjob_tools": ["cronjob"],
    "rl_tools": [
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
    ],
    "file_tools": ["read_file", "write_file", "patch", "search_files"],
    "tts_tools": ["text_to_speech"],
}


# Static bundle definitions only. Registry-owned toolsets resolve their direct
# membership live from ``tools.registry``; the ``tools`` lists here only contain
# extra bundle-specific additions that are not owned by the toolset itself.
TOOLSETS = {
    "web": {
        "description": "Web research and content extraction tools",
        "tools": [],
        "includes": [],
    },
    "search": {
        "description": "Web search only (no content extraction/scraping)",
        "tools": ["web_search"],
        "includes": [],
    },
    "vision": {
        "description": "Image analysis and vision tools",
        "tools": [],
        "includes": [],
    },
    "image_gen": {
        "description": "Creative generation tools (images)",
        "tools": [],
        "includes": [],
    },
    "terminal": {
        "description": "Terminal/command execution and process management tools",
        "tools": [],
        "includes": [],
    },
    "moa": {
        "description": "Advanced reasoning and problem-solving tools",
        "tools": [],
        "includes": [],
    },
    "skills": {
        "description": "Access, create, edit, and manage skill documents with specialized instructions and knowledge",
        "tools": [],
        "includes": [],
    },
    "browser": {
        "description": "Browser automation for web interaction (navigate, click, type, scroll, iframes, hold-click) with web search for finding URLs",
        "tools": [],
        "includes": ["search"],
    },
    "cronjob": {
        "description": "Cronjob management tool - create, list, update, pause, resume, remove, and trigger scheduled tasks",
        "tools": [],
        "includes": [],
    },
    "messaging": {
        "description": "Cross-platform messaging: send messages to Telegram, Discord, Slack, SMS, etc.",
        "tools": [],
        "includes": [],
    },
    "rl": {
        "description": "RL training tools for running reinforcement learning on Tinker-Atropos",
        "tools": [],
        "includes": [],
    },
    "file": {
        "description": "File manipulation tools: read, write, patch (with fuzzy matching), and search (content + files)",
        "tools": [],
        "includes": [],
    },
    "tts": {
        "description": "Text-to-speech: convert text to audio with Edge TTS (free), ElevenLabs, or OpenAI",
        "tools": [],
        "includes": [],
    },
    "todo": {
        "description": "Task planning and tracking for multi-step work",
        "tools": [],
        "includes": [],
    },
    "memory": {
        "description": "Persistent memory across sessions (personal notes + user profile)",
        "tools": [],
        "includes": [],
    },
    "session_search": {
        "description": "Search and recall past conversations with summarization",
        "tools": [],
        "includes": [],
    },
    "clarify": {
        "description": "Ask the user clarifying questions (multiple-choice or open-ended)",
        "tools": [],
        "includes": [],
    },
    "code_execution": {
        "description": "Run Python scripts that call tools programmatically (reduces LLM round trips)",
        "tools": [],
        "includes": [],
    },
    "delegation": {
        "description": "Spawn subagents with isolated context for complex subtasks",
        "tools": [],
        "includes": [],
    },
    # "honcho" toolset removed — Honcho is now a memory provider plugin.
    "homeassistant": {
        "description": "Home Assistant smart home control and monitoring",
        "tools": [],
        "includes": [],
    },
    "debugging": {
        "description": "Debugging and troubleshooting toolkit",
        "tools": [],
        "includes": ["terminal", "web", "file"],
    },
    "safe": {
        "description": "Safe toolkit without terminal access",
        "tools": [],
        "includes": ["web", "vision", "image_gen"],
    },
    "hermes-acp": {
        "description": "Editor integration (VS Code, Zed, JetBrains) — coding-focused tools without messaging, audio, or clarify UI",
        "tools": [],
        "includes": [
            "web",
            "terminal",
            "file",
            "vision",
            "skills",
            "browser",
            "todo",
            "memory",
            "session_search",
            "code_execution",
            "delegation",
        ],
    },
    "hermes-api-server": {
        "description": "OpenAI-compatible API server — full agent tools accessible via HTTP (no interactive UI tools like clarify or send_message)",
        "tools": [],
        "includes": [
            "web",
            "terminal",
            "file",
            "vision",
            "image_gen",
            "skills",
            "browser",
            "todo",
            "memory",
            "session_search",
            "code_execution",
            "delegation",
            "cronjob",
            "homeassistant",
        ],
    },
    "hermes-cli": {
        "description": "Full interactive CLI toolset - all default tools plus cronjob management",
        "tools": [],
        "includes": list(_HERMES_CORE_TOOLSETS),
    },
    "hermes-telegram": {
        "description": "Telegram bot toolset - full access for personal use (terminal has safety checks)",
        "tools": [],
        "includes": list(_HERMES_CORE_TOOLSETS),
    },
    "hermes-discord": {
        "description": "Discord bot toolset - full access (terminal has safety checks via dangerous command approval)",
        "tools": [],
        "includes": list(_HERMES_CORE_TOOLSETS),
    },
    "hermes-whatsapp": {
        "description": "WhatsApp bot toolset - similar to Telegram (personal messaging, more trusted)",
        "tools": [],
        "includes": list(_HERMES_CORE_TOOLSETS),
    },
    "hermes-slack": {
        "description": "Slack bot toolset - full access for workspace use (terminal has safety checks)",
        "tools": [],
        "includes": list(_HERMES_CORE_TOOLSETS),
    },
    "hermes-signal": {
        "description": "Signal bot toolset - encrypted messaging platform (full access)",
        "tools": [],
        "includes": list(_HERMES_CORE_TOOLSETS),
    },
    "hermes-bluebubbles": {
        "description": "BlueBubbles iMessage bot toolset - Apple iMessage via local BlueBubbles server",
        "tools": [],
        "includes": list(_HERMES_CORE_TOOLSETS),
    },
    "hermes-homeassistant": {
        "description": "Home Assistant bot toolset - smart home event monitoring and control",
        "tools": [],
        "includes": list(_HERMES_CORE_TOOLSETS),
    },
    "hermes-email": {
        "description": "Email bot toolset - interact with Hermes via email (IMAP/SMTP)",
        "tools": [],
        "includes": list(_HERMES_CORE_TOOLSETS),
    },
    "hermes-mattermost": {
        "description": "Mattermost bot toolset - self-hosted team messaging (full access)",
        "tools": [],
        "includes": list(_HERMES_CORE_TOOLSETS),
    },
    "hermes-matrix": {
        "description": "Matrix bot toolset - decentralized encrypted messaging (full access)",
        "tools": [],
        "includes": list(_HERMES_CORE_TOOLSETS),
    },
    "hermes-dingtalk": {
        "description": "DingTalk bot toolset - enterprise messaging platform (full access)",
        "tools": [],
        "includes": list(_HERMES_CORE_TOOLSETS),
    },
    "hermes-feishu": {
        "description": "Feishu/Lark bot toolset - enterprise messaging via Feishu/Lark (full access)",
        "tools": [],
        "includes": list(_HERMES_CORE_TOOLSETS),
    },
    "hermes-weixin": {
        "description": "Weixin bot toolset - personal WeChat messaging via iLink (full access)",
        "tools": [],
        "includes": list(_HERMES_CORE_TOOLSETS),
    },
    "hermes-wecom": {
        "description": "WeCom bot toolset - enterprise WeChat messaging (full access)",
        "tools": [],
        "includes": list(_HERMES_CORE_TOOLSETS),
    },
    "hermes-sms": {
        "description": "SMS bot toolset - interact with Hermes via SMS (Twilio)",
        "tools": [],
        "includes": list(_HERMES_CORE_TOOLSETS),
    },
    "hermes-webhook": {
        "description": "Webhook toolset - receive and process external webhook events",
        "tools": [],
        "includes": list(_HERMES_CORE_TOOLSETS),
    },
    "hermes-gateway": {
        "description": "Gateway toolset - union of all messaging platform tools",
        "tools": [],
        "includes": [
            "hermes-telegram",
            "hermes-discord",
            "hermes-whatsapp",
            "hermes-slack",
            "hermes-signal",
            "hermes-bluebubbles",
            "hermes-homeassistant",
            "hermes-email",
            "hermes-sms",
            "hermes-mattermost",
            "hermes-matrix",
            "hermes-dingtalk",
            "hermes-feishu",
            "hermes-wecom",
            "hermes-weixin",
            "hermes-webhook",
        ],
    },
}

def _get_registry_toolset_names() -> Set[str]:
    """Return live toolset names from the registry."""
    try:
        return set(registry.get_registered_toolset_names())
    except Exception:
        return set()


def _get_registry_tool_names(toolset_name: str) -> List[str]:
    """Return live direct tool names for a registry-owned toolset."""
    try:
        return registry.get_tools_for_toolset(toolset_name)
    except Exception:
        return []


def _get_builtin_registry_tool_names(toolset_name: str) -> List[str]:
    """Return only built-in direct tool names for a registry-owned toolset."""
    try:
        return registry.get_tools_for_toolset(toolset_name, builtin_only=True)
    except Exception:
        return []


def is_legacy_toolset(name: str) -> bool:
    """Return True when *name* is a backward-compatible toolset alias."""
    return name in LEGACY_TOOLSET_ALIASES


def resolve_legacy_toolset(name: str) -> List[str]:
    """Resolve a backward-compatible legacy toolset alias to live tool names."""
    tool_names = LEGACY_TOOLSET_ALIASES.get(name)
    if not tool_names:
        return []
    available = set(registry.get_all_tool_names())
    return [tool_name for tool_name in tool_names if tool_name in available]


def get_legacy_toolset_map() -> Dict[str, List[str]]:
    """Return live resolved tool names for every supported legacy alias."""
    return {
        name: sorted(resolve_legacy_toolset(name))
        for name in LEGACY_TOOLSET_ALIASES
    }


def _get_mcp_toolset_aliases() -> Dict[str, str]:
    """Map raw MCP server names to their internal registry toolset names."""
    aliases: Dict[str, str] = {}
    reserved_names = set(TOOLSETS) | set(LEGACY_TOOLSET_ALIASES)
    reserved_names.update(
        toolset_name
        for toolset_name in _get_registry_toolset_names()
        if not toolset_name.startswith("mcp-")
    )
    for toolset_name in _get_registry_toolset_names():
        if not toolset_name.startswith("mcp-"):
            continue
        alias = toolset_name[4:]
        if alias and alias not in reserved_names:
            aliases.setdefault(alias, toolset_name)
    return aliases


def _default_toolset_description(name: str, display_name: Optional[str] = None) -> str:
    """Return a synthetic description for dynamic toolsets."""
    label = display_name or name
    if name.startswith("mcp-"):
        return f"MCP server '{name[4:]}' tools"
    if display_name and name.startswith("mcp-"):
        return f"MCP server '{display_name}' tools"
    return f"Plugin toolset: {label}"


def _normalize_toolset_name(name: str) -> str:
    """Resolve user-facing aliases to the registry-owned toolset name."""
    return _get_mcp_toolset_aliases().get(name, name)


def get_toolset(name: str) -> Optional[Dict[str, Any]]:
    """
    Get a toolset definition by name.

    Returns a merged live view where direct tool ownership comes from the
    registry and static bundle metadata contributes descriptions/includes.
    """
    static_def = TOOLSETS.get(name)
    normalized_name = _normalize_toolset_name(name)
    registry_tools = (
        _get_builtin_registry_tool_names(normalized_name)
        if static_def is not None
        else _get_registry_tool_names(normalized_name)
    )

    if not static_def and not registry_tools:
        return None

    direct_tools = list(dict.fromkeys((static_def or {}).get("tools", []) + registry_tools))
    includes = list((static_def or {}).get("includes", []))
    description = (static_def or {}).get("description") or _default_toolset_description(
        normalized_name,
        display_name=name if name != normalized_name else None,
    )

    return {
        "description": description,
        "tools": direct_tools,
        "includes": includes,
    }


def resolve_toolset(name: str, visited: Set[str] = None) -> List[str]:
    """
    Recursively resolve a toolset to get all tool names.
    """
    if visited is None:
        visited = set()

    if name in {"all", "*"}:
        all_tools: Set[str] = set()
        for toolset_name in get_toolset_names():
            all_tools.update(resolve_toolset(toolset_name, visited.copy()))
        return list(all_tools)

    if name in visited:
        return []

    visited.add(name)

    toolset = get_toolset(name)
    if not toolset:
        return []

    tools = set(toolset.get("tools", []))
    for included_name in toolset.get("includes", []):
        tools.update(resolve_toolset(included_name, visited))

    return list(tools)


def resolve_multiple_toolsets(toolset_names: List[str]) -> List[str]:
    """
    Resolve multiple toolsets and combine their tools.
    """
    all_tools = set()
    for name in toolset_names:
        all_tools.update(resolve_toolset(name))
    return list(all_tools)


def get_all_toolsets() -> Dict[str, Dict[str, Any]]:
    """
    Get all available toolsets with their definitions.

    Includes static bundles, registry-owned leaf toolsets, and raw MCP server
    aliases that resolve to live ``mcp-*`` registry toolsets.
    """
    result: Dict[str, Dict[str, Any]] = {}
    for name in get_toolset_names():
        toolset = get_toolset(name)
        if toolset:
            result[name] = toolset
    return result


def get_toolset_names() -> List[str]:
    """
    Get names of all available toolsets (excluding the special all/* aliases).
    """
    names = set(TOOLSETS.keys())
    for toolset_name in _get_registry_toolset_names():
        if toolset_name.startswith("mcp-"):
            alias = toolset_name[4:]
            if (
                alias
                and alias not in TOOLSETS
                and alias not in LEGACY_TOOLSET_ALIASES
                and alias not in _get_registry_toolset_names()
            ):
                names.add(alias)
            else:
                names.add(toolset_name)
            continue
        names.add(toolset_name)
    return sorted(names)


def validate_toolset(name: str) -> bool:
    """
    Check if a toolset name is valid.
    """
    if name in {"all", "*"}:
        return True
    if name in TOOLSETS:
        return True
    if name in _get_mcp_toolset_aliases():
        return True
    return name in _get_registry_toolset_names()


def create_custom_toolset(
    name: str,
    description: str,
    tools: List[str] = None,
    includes: List[str] = None,
) -> None:
    """
    Create a custom bundle toolset at runtime.
    """
    TOOLSETS[name] = {
        "description": description,
        "tools": tools or [],
        "includes": includes or [],
    }


def get_toolset_info(name: str) -> Dict[str, Any]:
    """
    Get detailed information about a toolset including resolved tools.
    """
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
        "is_composite": bool(toolset["includes"]),
    }


def get_hermes_core_tools() -> List[str]:
    """Return a snapshot of the default Hermes CLI tool inventory."""
    return sorted(resolve_toolset("hermes-cli"))


_HERMES_CORE_TOOLS = get_hermes_core_tools()


if __name__ == "__main__":
    print("Toolsets System Demo")
    print("=" * 60)

    print("\nAvailable Toolsets:")
    print("-" * 40)
    for name, toolset in get_all_toolsets().items():
        info = get_toolset_info(name)
        composite = "[composite]" if info["is_composite"] else "[leaf]"
        print(f"  {composite} {name:20} - {toolset['description']}")
        print(f"     Tools: {len(info['resolved_tools'])} total")

    print("\nToolset Resolution Examples:")
    print("-" * 40)
    for name in ["web", "terminal", "safe", "debugging"]:
        tools = resolve_toolset(name)
        print(f"\n  {name}:")
        print(f"    Resolved to {len(tools)} tools: {', '.join(sorted(tools))}")

    print("\nMultiple Toolset Resolution:")
    print("-" * 40)
    combined = resolve_multiple_toolsets(["web", "vision", "terminal"])
    print("  Combining ['web', 'vision', 'terminal']:")
    print(f"    Result: {', '.join(sorted(combined))}")

    print("\nCustom Toolset Creation:")
    print("-" * 40)
    create_custom_toolset(
        name="my_custom",
        description="My custom toolset for specific tasks",
        tools=["web_search"],
        includes=["terminal", "vision"],
    )
    custom_info = get_toolset_info("my_custom")
    print("  Created 'my_custom' toolset:")
    print(f"    Description: {custom_info['description']}")
    print(f"    Resolved tools: {', '.join(custom_info['resolved_tools'])}")
