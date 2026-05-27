"""Tag-based tool discovery and invocation meta-tools.

Provides three tools that enable dynamic discovery and invocation of other tools:
1. list_tool_tags - List all unique tags across registered tools
2. search_tools_by_tag - Find tools matching given tags
3. invoke_tool - Execute any tool by name with arguments

This enables the agent to discover tools on-demand rather than having all tool
schemas injected into the LLM context at once, dramatically reducing context size
and improving scalability, especially for local models.
"""

import json
import logging
from typing import List

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)


def list_tool_tags(args: dict, **kwargs) -> str:
    """List all unique tool tags or search tags by keyword.
    
    Args:
        args: Dict with optional "query" (string to filter tags by substring)
        
    Returns:
        JSON with "tags" list or error
    """
    try:
        query = args.get("query", "").lower()
        all_tags = registry.get_all_tags()
        
        if query:
            all_tags = [t for t in all_tags if query in t.lower()]
        
        return tool_result({"tags": all_tags})
    except Exception as e:
        logger.exception("list_tool_tags error: %s", e)
        return tool_error(f"Failed to list tags: {e}")


def search_tools_by_tag(args: dict, **kwargs) -> str:
    """Search for tools matching given tags.
    
    Args:
        args: Dict with "tags" (string or list of tags)
        
    Returns:
        JSON with "tools" list containing {name, description} objects or error
    """
    try:
        tags = args.get("tags")
        
        # Support both single tag (string) and multiple tags (list)
        if isinstance(tags, str):
            tags = [tags]
        elif not isinstance(tags, list):
            return tool_error("'tags' must be a string or list of strings")
        
        if not tags:
            return tool_error("'tags' parameter cannot be empty")
        
        # Find tools matching any of the given tags (OR semantics)
        found_entries = registry.find_tools_by_tags(tags)
        
        tools = [
            {
                "name": entry.name,
                "description": entry.description or entry.schema.get("description", ""),
                "toolset": entry.toolset,
                "tags": entry.tags,
            }
            for entry in found_entries
        ]
        
        # Sort by name for consistent output
        tools.sort(key=lambda t: t["name"])
        
        return tool_result({"tools": tools, "count": len(tools)})
    except Exception as e:
        logger.exception("search_tools_by_tag error: %s", e)
        return tool_error(f"Failed to search tools: {e}")


def invoke_tool(args: dict, **kwargs) -> str:
    """Invoke a tool by name with given arguments.
    
    Args:
        args: Dict with:
            - "tool" (string): Tool name to invoke
            - "args" (dict): Arguments to pass to the tool
        
    Returns:
        JSON result from the tool handler or error
    """
    try:
        tool_name = args.get("tool")
        tool_args = args.get("args", {})
        
        if not tool_name:
            return tool_error("'tool' parameter is required")
        
        if not isinstance(tool_args, dict):
            return tool_error("'args' must be a JSON object")
        
        # Delegate to the registry dispatcher (handles exceptions and async)
        result = registry.dispatch(tool_name, tool_args, **kwargs)
        
        # Result is already a JSON string from the handler
        return result
    except Exception as e:
        logger.exception("invoke_tool error: %s", e)
        return tool_error(f"Failed to invoke tool: {e}")


# Register the three meta-tools
registry.register(
    name="list_tool_tags",
    toolset="tools_registry",
    schema={
        "name": "list_tool_tags",
        "description": "List all available tool tags or search tags by keyword",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional substring to filter tags (case-insensitive). "
                                  "If omitted, all tags are listed."
                }
            }
        }
    },
    handler=lambda args, **kw: list_tool_tags(
        {
            "query": args.get("query", "")
        },
        **kw
    ),
    description="List all available tool tags or search tags by keyword",
    emoji="🏷️",
    is_async=False,
)

registry.register(
    name="search_tools_by_tag",
    toolset="tools_registry",
    schema={
        "name": "search_tools_by_tag",
        "description": "Find tools that match the given tags",
        "parameters": {
            "type": "object",
            "properties": {
                "tags": {
                    "type": ["string", "array"],
                    "description": "Tag or list of tags to filter tools by. "
                                  "Tools matching ANY of the tags are returned (OR semantics).",
                    "items": {"type": "string"}
                }
            },
            "required": ["tags"]
        }
    },
    handler=lambda args, **kw: search_tools_by_tag(
        {
            "tags": args.get("tags", [])
        },
        **kw
    ),
    description="Find tools that match the given tags",
    emoji="🔍",
    is_async=False,
)

registry.register(
    name="invoke_tool",
    toolset="tools_registry",
    schema={
        "name": "invoke_tool",
        "description": "Execute a tool by name with specified arguments",
        "parameters": {
            "type": "object",
            "properties": {
                "tool": {
                    "type": "string",
                    "description": "The name of the tool to invoke"
                },
                "args": {
                    "type": "object",
                    "description": "JSON object of arguments for the tool",
                    "additionalProperties": True
                }
            },
            "required": ["tool", "args"]
        }
    },
    handler=lambda args, **kw: invoke_tool(
        {
            "tool": args.get("tool", ""),
            "args": args.get("args", {})
        },
        **kw
    ),
    description="Execute a tool by name with specified arguments",
    emoji="⚙️",
    is_async=False,
)