#!/usr/bin/env python3
"""
Toolsets Module

This module provides a flexible system for defining and managing tool aliases/toolsets.
Toolsets allow you to group tools together for specific scenarios and can be composed
from individual tools or other toolsets.

Features:
- Define custom toolsets with specific tools
- Compose toolsets from other toolsets
- Built-in common toolsets for typical use cases
- Easy extension for new toolsets
- Support for dynamic toolset resolution

Usage:
    from toolsets import get_toolset, resolve_toolset, get_all_toolsets
    
    # Get tools for a specific toolset
    tools = get_toolset("research")
    
    # Resolve a toolset to get all tool names (including from composed toolsets)
    all_tools = resolve_toolset("full_stack")
"""

from typing import List, Dict, Any, Set, Optional
import json


# Core toolset definitions
# These can include individual tools or reference other toolsets
TOOLSETS = {
    # Basic toolsets - individual tool categories
    "web": {
        "description": "Web research and content extraction tools",
        "tools": ["web_search", "web_extract"],
        "includes": []  # No other toolsets included
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
        "description": "Terminal/command execution tools",
        "tools": ["terminal"],
        "includes": []
    },
    
    "moa": {
        "description": "Advanced reasoning and problem-solving tools",
        "tools": ["mixture_of_agents"],
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
    
    # Scenario-specific toolsets
    
    "debugging": {
        "description": "Debugging and troubleshooting toolkit",
        "tools": ["terminal"],
        "includes": ["web"]  # For searching error messages and solutions
    },
    
    "safe": {
        "description": "Safe toolkit without terminal access",
        "tools": ["mixture_of_agents"],
        "includes": ["web", "vision", "creative"]
    }
}



def get_toolset(name: str) -> Optional[Dict[str, Any]]:
    """
    Get a toolset definition by name.
    
    Args:
        name (str): Name of the toolset
        
    Returns:
        Dict: Toolset definition with description, tools, and includes
        None: If toolset not found
    """
    # Return toolset definition
    return TOOLSETS.get(name)


def resolve_toolset(name: str, visited: Set[str] = None) -> List[str]:
    """
    Recursively resolve a toolset to get all tool names.
    
    This function handles toolset composition by recursively resolving
    included toolsets and combining all tools.
    
    Args:
        name (str): Name of the toolset to resolve
        visited (Set[str]): Set of already visited toolsets (for cycle detection)
        
    Returns:
        List[str]: List of all tool names in the toolset
    """
    if visited is None:
        visited = set()
    
    # Special aliases that represent all tools across every toolset
    # This ensures future toolsets are automatically included without changes.
    if name in {"all", "*"}:
        all_tools: Set[str] = set()
        for toolset_name in get_toolset_names():
            # Use a fresh visited set per branch to avoid cross-branch contamination
            resolved = resolve_toolset(toolset_name, visited.copy())
            all_tools.update(resolved)
        return list(all_tools)

    # Check for cycles
    if name in visited:
        print(f"⚠️  Circular dependency detected in toolset '{name}'")
        return []
    
    visited.add(name)
    
    # Get toolset definition
    toolset = TOOLSETS.get(name)
    if not toolset:
        return []
    
    # Collect direct tools
    tools = set(toolset.get("tools", []))
    
    # Recursively resolve included toolsets
    for included_name in toolset.get("includes", []):
        included_tools = resolve_toolset(included_name, visited.copy())
        tools.update(included_tools)
    
    return list(tools)


def resolve_multiple_toolsets(toolset_names: List[str]) -> List[str]:
    """
    Resolve multiple toolsets and combine their tools.
    
    Args:
        toolset_names (List[str]): List of toolset names to resolve
        
    Returns:
        List[str]: Combined list of all tool names (deduplicated)
    """
    all_tools = set()
    
    for name in toolset_names:
        tools = resolve_toolset(name)
        all_tools.update(tools)
    
    return list(all_tools)


def get_all_toolsets() -> Dict[str, Dict[str, Any]]:
    """
    Get all available toolsets with their definitions.
    
    Returns:
        Dict: All toolset definitions
    """
    return TOOLSETS.copy()


def get_toolset_names() -> List[str]:
    """
    Get names of all available toolsets (excluding aliases).
    
    Returns:
        List[str]: List of toolset names
    """
    return list(TOOLSETS.keys())




def validate_toolset(name: str) -> bool:
    """
    Check if a toolset name is valid.
    
    Args:
        name (str): Toolset name to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Accept special alias names for convenience
    if name in {"all", "*"}:
        return True
    return name in TOOLSETS


def create_custom_toolset(
    name: str,
    description: str,
    tools: List[str] = None,
    includes: List[str] = None
) -> None:
    """
    Create a custom toolset at runtime.
    
    Args:
        name (str): Name for the new toolset
        description (str): Description of the toolset
        tools (List[str]): Direct tools to include
        includes (List[str]): Other toolsets to include
    """
    TOOLSETS[name] = {
        "description": description,
        "tools": tools or [],
        "includes": includes or []
    }




def get_toolset_info(name: str) -> Dict[str, Any]:
    """
    Get detailed information about a toolset including resolved tools.
    
    Args:
        name (str): Toolset name
        
    Returns:
        Dict: Detailed toolset information
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
        "is_composite": len(toolset["includes"]) > 0
    }


def print_toolset_tree(name: str, indent: int = 0) -> None:
    """
    Print a tree view of a toolset and its composition.
    
    Args:
        name (str): Toolset name
        indent (int): Current indentation level
    """
    prefix = "  " * indent
    toolset = get_toolset(name)
    
    if not toolset:
        print(f"{prefix}❌ Unknown toolset: {name}")
        return
    
    # Print toolset name and description
    print(f"{prefix}📦 {name}: {toolset['description']}")
    
    # Print direct tools
    if toolset["tools"]:
        print(f"{prefix}  🔧 Tools: {', '.join(toolset['tools'])}")
    
    # Print included toolsets
    if toolset["includes"]:
        print(f"{prefix}  📂 Includes:")
        for included in toolset["includes"]:
            print_toolset_tree(included, indent + 2)


if __name__ == "__main__":
    """
    Demo and testing of the toolsets system
    """
    print("🎯 Toolsets System Demo")
    print("=" * 60)
    
    # Show all available toolsets
    print("\n📦 Available Toolsets:")
    print("-" * 40)
    for name, toolset in get_all_toolsets().items():
        info = get_toolset_info(name)
        composite = "📂" if info["is_composite"] else "🔧"
        print(f"{composite} {name:20} - {toolset['description']}")
        print(f"   Tools: {len(info['resolved_tools'])} total")
    
    
    # Demo toolset resolution
    print("\n🔍 Toolset Resolution Examples:")
    print("-" * 40)
    
    examples = ["research", "development", "full_stack", "minimal", "safe"]
    for name in examples:
        tools = resolve_toolset(name)
        print(f"\n{name}:")
        print(f"  Resolved to {len(tools)} tools: {', '.join(sorted(tools))}")
    
    # Show toolset composition tree
    print("\n🌳 Toolset Composition Tree:")
    print("-" * 40)
    print("\nExample: 'content_creation' toolset:")
    print_toolset_tree("content_creation")
    
    print("\nExample: 'full_stack' toolset:")
    print_toolset_tree("full_stack")
    
    # Demo multiple toolset resolution
    print("\n🔗 Multiple Toolset Resolution:")
    print("-" * 40)
    combined = resolve_multiple_toolsets(["minimal", "vision", "reasoning"])
    print(f"Combining ['minimal', 'vision', 'reasoning']:")
    print(f"  Result: {', '.join(sorted(combined))}")
    
    # Demo custom toolset creation
    print("\n➕ Custom Toolset Creation:")
    print("-" * 40)
    create_custom_toolset(
        name="my_custom",
        description="My custom toolset for specific tasks",
        tools=["web_search"],
        includes=["terminal", "vision"]
    )
    
    custom_info = get_toolset_info("my_custom")
    print(f"Created 'my_custom' toolset:")
    print(f"  Description: {custom_info['description']}")
    print(f"  Resolved tools: {', '.join(custom_info['resolved_tools'])}")
