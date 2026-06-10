#!/usr/bin/env python3
"""
Obsidian Vault Tool — Hermes-level filesystem-first knowledge management.

Provides vault operations as Hermes tools:
- vault_health: diagnostic summary of vault (note count, size, projects, recent notes)
- vault_read: read a specific note from the vault with line numbers
- vault_search: full-text search across vault notes
- vault_list: list notes in the vault or a subfolder
- vault_create: create a new note with YAML frontmatter and wikilinks

Vault path resolution priority:
1. OBSIDIAN_VAULT_PATH env var (set in ~/.hermes/.env)
2. ~/Documents/Obsidian Vault
3. /opt/data/vault (Linux server fallback)
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vault path resolution
# ---------------------------------------------------------------------------

def _resolve_vault_path() -> Optional[Path]:
    """Resolve the Obsidian vault path from environment or defaults."""
    env_path = os.getenv("OBSIDIAN_VAULT_PATH", "")
    if env_path and os.path.isdir(env_path):
        return Path(env_path).resolve()

    # Default fallbacks
    candidates = [
        Path.home() / "Documents" / "Obsidian Vault",
        Path("/opt/data/vault"),
    ]
    for p in candidates:
        if p.is_dir() and (p / ".obsidian").is_dir():
            return p
    # Accept any directory that looks like a vault (has .obsidian)
    for p in candidates:
        if p.is_dir():
            return p

    # Check env_path even if .obsidian is missing
    if env_path and os.path.isdir(env_path):
        return Path(env_path).resolve()

    return None


def _get_vault() -> Path:
    """Get vault path, raising if not found."""
    vault = _resolve_vault_path()
    if vault is None:
        raise FileNotFoundError(
            "Obsidian vault not found. Set OBSIDIAN_VAULT_PATH in ~/.hermes/.env "
            "or create a vault at ~/Documents/Obsidian Vault."
        )
    return vault


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate_text(text: str, max_len: int = 8000) -> str:
    """Truncate text to max_len with a note if cut."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n\n[...truncated at {max_len} chars]"


def _format_timestamp(path: Path) -> str:
    """Return human-readable modification time for a file."""
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return "unknown"


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def vault_health(task_id: Optional[str] = None) -> str:
    """Return a diagnostic summary of the Obsidian vault."""
    vault = _get_vault()

    notes = list(vault.rglob("*.md"))
    note_count = len(notes)
    total_size = sum(f.stat().st_size for f in notes if f.is_file())
    dir_count = len([d for d in vault.rglob("*") if d.is_dir()])

    # Config check
    config_ok = (vault / ".obsidian" / "app.json").exists()

    # Projects (top-level dirs that aren't .obsidian, assets, or hidden)
    projects = []
    for d in sorted(vault.iterdir()):
        if d.is_dir() and not d.name.startswith(".") and d.name != "assets":
            proj_notes = len(list(d.rglob("*.md")))
            if proj_notes > 0:
                projects.append(f"  {d.name}/ ({proj_notes} notes)")

    # Recent notes
    sorted_notes = sorted(notes, key=lambda p: p.stat().st_mtime, reverse=True)
    recent = []
    for n in sorted_notes[:10]:
        rel = str(n.relative_to(vault))
        ts = _format_timestamp(n)
        recent.append(f"  {ts}  {rel}")

    result = {
        "vault_path": str(vault),
        "note_count": note_count,
        "directory_count": dir_count - 1,  # minus vault root
        "total_size_bytes": total_size,
        "total_size_human": f"{total_size / 1024:.0f} KB" if total_size < 1024 * 1024 else f"{total_size / (1024 * 1024):.1f} MB",
        "config_ok": config_ok,
        "projects": projects,
        "recent_notes": recent,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


def vault_read(path: str, task_id: Optional[str] = None) -> str:
    """Read a vault note by relative path. Returns content with line numbers.

    Args:
        path: Relative path within the vault (e.g., 'project/README.md')
    """
    vault = _get_vault()
    note_path = (vault / path).resolve()

    # Security: ensure path stays within vault
    if not str(note_path).startswith(str(vault)):
        return json.dumps({"error": "Path escapes vault boundary", "path": path})

    if not note_path.exists():
        similar = []
        for f in vault.rglob("*.md"):
            if path.lower() in str(f.relative_to(vault)).lower():
                similar.append(str(f.relative_to(vault)))
        return json.dumps({
            "error": f"Note not found: {path}",
            "similar_notes": similar[:10],
        }, ensure_ascii=False)

    try:
        content = note_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return json.dumps({"error": f"Binary or unreadable file: {path}"})

    lines = content.split("\n")
    numbered = "\n".join(f"{i+1:4d}|{line}" for i, line in enumerate(lines))
    return json.dumps({
        "path": path,
        "content": _truncate_text(numbered, 8000),
        "total_lines": len(lines),
        "size_bytes": len(content),
        "modified": _format_timestamp(note_path),
    }, ensure_ascii=False)


def vault_search(query: str, max_results: int = 20, task_id: str = None) -> str:
    """Full-text search across all markdown notes in the vault.

    Args:
        query: Search term or regex pattern
        max_results: Max results to return (default 20)
    """
    vault = _get_vault()

    if not query:
        return json.dumps({"error": "query is required"})

    results = []
    try:
        pattern = re.compile(query, re.IGNORECASE)
    except re.error:
        # Fall back to simple substring search
        pattern = None

    for note in vault.rglob("*.md"):
        if len(results) >= max_results:
            break
        try:
            content = note.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        matched_lines = []
        if pattern:
            for i, line in enumerate(content.split("\n"), 1):
                if pattern.search(line):
                    matched_lines.append({"line": i, "text": line.strip()[:200]})
                    if len(matched_lines) >= 5:
                        break
        else:
            lower_query = query.lower()
            for i, line in enumerate(content.split("\n"), 1):
                if lower_query in line.lower():
                    matched_lines.append({"line": i, "text": line.strip()[:200]})
                    if len(matched_lines) >= 5:
                        break

        if matched_lines:
            results.append({
                "note": str(note.relative_to(vault)),
                "modified": _format_timestamp(note),
                "match_count": len(matched_lines),
                "matches": matched_lines,
            })

    return json.dumps({
        "query": query,
        "total_results": len(results),
        "results": results,
    }, ensure_ascii=False)


def vault_list(subfolder: str = "", task_id: str = None) -> str:
    """List markdown notes in the vault or a subfolder.

    Args:
        subfolder: Optional subfolder path relative to vault root
    """
    vault = _get_vault()
    target = vault
    if subfolder:
        target = (vault / subfolder).resolve()
        if not str(target).startswith(str(vault)):
            return json.dumps({"error": "Path escapes vault boundary"})

    if not target.exists():
        return json.dumps({"error": f"Path not found: {subfolder}"})

    notes = []
    for f in sorted(target.rglob("*.md")):
        rel = str(f.relative_to(vault))
        notes.append({
            "path": rel,
            "modified": _format_timestamp(f),
            "size_bytes": f.stat().st_size,
        })

    return json.dumps({
        "vault": str(vault),
        "subfolder": subfolder or "(root)",
        "note_count": len(notes),
        "notes": notes[:100],  # Cap at 100 to avoid excessive output
    }, ensure_ascii=False)


def vault_create(
    path: str,
    content: str = "",
    tags: List[str] = None,
    vertical: str = "",
    task_id: str = None,
) -> str:
    """Create a new note in the vault with YAML frontmatter.

    Args:
        path: Relative path within the vault (e.g., 'project/new-note.md')
        content: Markdown content for the note body
        tags: Optional list of tags for YAML frontmatter
        vertical: Optional vertical/category for frontmatter
    """
    vault = _get_vault()
    note_path = (vault / path).resolve()

    # Security check
    if not str(note_path).startswith(str(vault)):
        return json.dumps({"error": "Path escapes vault boundary"})

    if note_path.exists():
        return json.dumps({"error": f"Note already exists: {path}"})

    # Build YAML frontmatter
    frontmatter = ["---"]
    if tags:
        tag_list = ", ".join(tags)
        frontmatter.append(f"tags: [{tag_list}]")
    else:
        frontmatter.append("tags: []")
    if vertical:
        frontmatter.append(f"vertical: {vertical}")
    frontmatter.append(f"created: {datetime.now().strftime('%Y-%m-%d')}")
    frontmatter.append("---")

    full_content = "\n".join(frontmatter) + "\n\n" + content

    # Create parent directories
    note_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        note_path.write_text(full_content, encoding="utf-8")
    except OSError as e:
        return json.dumps({"error": f"Failed to write note: {e}"})

    return json.dumps({
        "created": path,
        "vault": str(vault),
        "size_bytes": len(full_content),
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------

def check_obsidian_requirements() -> bool:
    """Check if an Obsidian vault is available."""
    try:
        vault = _resolve_vault_path()
        return vault is not None and vault.is_dir()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

VAULT_HEALTH_SCHEMA = {
    "name": "vault_health",
    "description": (
        "Get a diagnostic summary of the Obsidian vault: note count, "
        "size, projects, recent notes, and config status. Use this before "
        "other vault operations to understand the vault structure."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}

VAULT_READ_SCHEMA = {
    "name": "vault_read",
    "description": (
        "Read a specific Obsidian vault note by its relative path. "
        "Returns content with line numbers and metadata. "
        "Path is relative to vault root (e.g., 'project/notes.md')."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path to the note within the vault (e.g., 'lead-qualification-os/README.md')",
            },
        },
        "required": ["path"],
    },
}

VAULT_SEARCH_SCHEMA = {
    "name": "vault_search",
    "description": (
        "Full-text search across all markdown notes in the Obsidian vault. "
        "Returns matching notes with line numbers and context. "
        "Accepts regex patterns or plain text."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search term or regex pattern to find in vault notes",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of matching notes to return (default 20)",
                "default": 20,
            },
        },
        "required": ["query"],
    },
}

VAULT_LIST_SCHEMA = {
    "name": "vault_list",
    "description": (
        "List all markdown notes in the Obsidian vault or a specific subfolder. "
        "Returns note paths, modification dates, and sizes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "subfolder": {
                "type": "string",
                "description": "Optional subfolder path relative to vault root (e.g., 'lead-qualification-os/verticals')",
                "default": "",
            },
        },
    },
}

VAULT_CREATE_SCHEMA = {
    "name": "vault_create",
    "description": (
        "Create a new markdown note in the Obsidian vault with YAML frontmatter. "
        "Parent directories are created automatically. "
        "Use this to persist research, generate dashboards, or store structured knowledge."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path for the new note (e.g., 'research/competitors.md')",
            },
            "content": {
                "type": "string",
                "description": "Markdown content for the note body (after frontmatter)",
                "default": "",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for YAML frontmatter",
            },
            "vertical": {
                "type": "string",
                "description": "Optional vertical/category for frontmatter",
                "default": "",
            },
        },
        "required": ["path"],
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error

registry.register(
    name="vault_health",
    toolset="file",
    schema=VAULT_HEALTH_SCHEMA,
    handler=lambda args, **kw: vault_health(task_id=kw.get("task_id")),
    check_fn=check_obsidian_requirements,
    emoji="🏥",
)

registry.register(
    name="vault_read",
    toolset="file",
    schema=VAULT_READ_SCHEMA,
    handler=lambda args, **kw: vault_read(
        path=args.get("path", ""),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_obsidian_requirements,
    emoji="📖",
)

registry.register(
    name="vault_search",
    toolset="file",
    schema=VAULT_SEARCH_SCHEMA,
    handler=lambda args, **kw: vault_search(
        query=args.get("query", ""),
        max_results=args.get("max_results", 20),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_obsidian_requirements,
    emoji="🔍",
)

registry.register(
    name="vault_list",
    toolset="file",
    schema=VAULT_LIST_SCHEMA,
    handler=lambda args, **kw: vault_list(
        subfolder=args.get("subfolder", ""),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_obsidian_requirements,
    emoji="📋",
)

registry.register(
    name="vault_create",
    toolset="file",
    schema=VAULT_CREATE_SCHEMA,
    handler=lambda args, **kw: vault_create(
        path=args.get("path", ""),
        content=args.get("content", ""),
        tags=args.get("tags"),
        vertical=args.get("vertical", ""),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_obsidian_requirements,
    emoji="✨",
)
