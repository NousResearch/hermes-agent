#!/usr/bin/env python3
"""
Project Context & Living Document Tool Module

Ported from Cortex Agent's Living Project Context Engine (MIT Licensed).
Provides automatic discovery, reading, scaffolding, and live auto-evolution of
HERMES.md (or CORTEX.md) project specification files in the workspace root.

Design:
- Single `project_context` tool: supply `action` ('read', 'update', 'init')
- 'read': Searches upward from CWD for HERMES.md / CORTEX.md and returns context
- 'update': Autonomously appends newly learned architectural patterns or conventions
- 'init': Scaffolds a new HERMES.md project context file
- 100% self-contained standard library implementation
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

PROJECT_CONTEXT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "project_context",
        "description": (
            "Read, scaffold, or auto-evolve the living HERMES.md project context file. "
            "Allows the agent to read workspace conventions or append newly discovered "
            "architectural patterns directly into project documentation. "
            "Ported from Cortex Agent (MIT Licensed)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "update", "init"],
                    "description": "Action to perform: 'read' (fetch context), 'update' (append insight), 'init' (scaffold file).",
                    "default": "read"
                },
                "insight": {
                    "type": "string",
                    "description": "Architectural insight or convention to append (required for 'update').",
                    "default": ""
                },
                "project_name": {
                    "type": "string",
                    "description": "Project name for initialization (optional for 'init').",
                    "default": ""
                }
            },
            "required": ["action"]
        }
    }
}


def check_project_context_requirements() -> bool:
    """Project context uses Python standard library file operations."""
    return True


def find_project_context_file(start_path: Optional[Path] = None) -> Optional[Path]:
    """Walk up directory tree from CWD to locate HERMES.md or CORTEX.md."""
    current = (start_path or Path.cwd()).resolve()
    for parent in [current] + list(current.parents):
        for filename in ("HERMES.md", "CORTEX.md"):
            candidate = parent / filename
            if candidate.exists() and candidate.is_file():
                return candidate
        # Stop at git repository boundary if no file found
        if (parent / ".git").exists() and parent != current:
            break
    return None


def project_context_tool(action: str = "read", insight: str = "", project_name: str = "") -> str:
    """Handler for the project_context tool."""
    from tools.registry import tool_error

    action = (action or "read").lower()

    if action == "read":
        ctx_file = find_project_context_file()
        if not ctx_file:
            return (
                "No HERMES.md or CORTEX.md project context file found in current workspace. "
                "Use `project_context` with action='init' to create one."
            )
        try:
            content = ctx_file.read_text(encoding="utf-8", errors="replace")
            return f"### 📋 Living Project Context (`{ctx_file.name}` at `{ctx_file.parent}`)\n\n{content}"
        except Exception as e:
            return tool_error(f"Failed to read project context file: {str(e)}")

    elif action == "update":
        if not insight or not insight.strip():
            return tool_error("Parameter 'insight' is required for action='update'.")

        ctx_file = find_project_context_file()
        if not ctx_file:
            ctx_file = Path.cwd() / "HERMES.md"
            ctx_file.write_text(f"# {Path.cwd().name} Project Context\n\n## Learned Architectural Insights\n\n", encoding="utf-8")

        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M")
            append_block = f"\n- [{timestamp}] {insight.strip()}\n"

            content = ctx_file.read_text(encoding="utf-8", errors="replace")
            if "## Learned Architectural Insights" not in content:
                content += "\n\n## Learned Architectural Insights\n"
            content += append_block

            ctx_file.write_text(content, encoding="utf-8")
            return f"Successfully appended insight to `{ctx_file.name}`:\n{append_block.strip()}"
        except Exception as e:
            return tool_error(f"Failed to update project context file: {str(e)}")

    elif action == "init":
        p_name = project_name.strip() or Path.cwd().name
        target_file = Path.cwd() / "HERMES.md"
        if target_file.exists():
            return f"HERMES.md already exists at `{target_file}`."

        template = f"""# {p_name} — Project Specification & Context

> Living project context managed autonomously by Agent.

---

## 🎯 Architecture Overview
Describe the core components, tech stack, and module boundaries here.

## 🛠️ Development & Testing Rules
- Unit tests command: `pytest`
- Code formatting & linting guidelines

## 🧠 Learned Architectural Insights
*The agent will auto-append newly discovered patterns and conventions here as it works.*
"""
        try:
            target_file.write_text(template, encoding="utf-8")
            return f"Successfully initialized living project context file at `{target_file}`."
        except Exception as e:
            return tool_error(f"Failed to initialize HERMES.md: {str(e)}")

    else:
        return tool_error(f"Unknown action '{action}'. Valid actions are: read, update, init.")


# --- Registry ---
from tools.registry import registry

registry.register(
    name="project_context",
    toolset="project_context",
    schema=PROJECT_CONTEXT_SCHEMA,
    handler=lambda args, **kw: project_context_tool(
        action=args.get("action", "read"),
        insight=args.get("insight", ""),
        project_name=args.get("project_name", "")
    ),
    check_fn=check_project_context_requirements,
    emoji="📋",
)
