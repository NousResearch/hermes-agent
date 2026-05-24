"""Domain knowledge loader tool — loads domain KB notes for the agent.

Tool name: load_domain_knowledge
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from hermes_constants import display_hermes_home
from tools.registry import registry

_VALID_DOMAINS = [
    "frontend", "backend", "devops", "security", "testing",
    "data", "mobile", "infrastructure",
    "business", "marketing", "sales", "finance", "operations", "people",
]


def check_requirements() -> bool:
    """Check if the Obsidian vault path is accessible."""
    vault_path = os.environ.get("OBSIDIAN_VAULT_PATH")
    if vault_path:
        return Path(vault_path).exists()
    fallback = Path.home() / "Documents" / "Obsidian Vault"
    if fallback.exists():
        return True
    # Check the standalone vault
    standalone = Path.home() / "ObsidianVault" / "HermesAgent"
    return standalone.exists()


def _resolve_vault_path() -> Path:
    """Resolve the vault path from env or fallbacks."""
    vault_path = os.environ.get("OBSIDIAN_VAULT_PATH")
    if vault_path:
        return Path(vault_path)
    standalone = Path.home() / "ObsidianVault" / "HermesAgent"
    if standalone.exists():
        return standalone
    fallback = Path.home() / "Documents" / "Obsidian Vault"
    return fallback


def load_domain_knowledge(domains: List[str], task_id: str = None) -> str:
    """Load domain knowledge notes for the given domains.

    Args:
        domains: List of domain slugs to load notes from.
        task_id: Optional task identifier (ignored, for tool interface compatibility).

    Returns:
        JSON string with loaded notes or error.
    """
    if not isinstance(domains, list):
        return json.dumps({
            "success": False,
            "error": "domains must be a list of strings",
            "notes": [],
        })

    # Filter to valid domains
    valid = [d for d in domains if d in _VALID_DOMAINS]
    invalid = [d for d in domains if d not in _VALID_DOMAINS]

    vault = _resolve_vault_path()
    notes: List[Dict[str, Any]] = []

    for domain_slug in valid:
        domain_dir = vault / "domains" / domain_slug
        if not domain_dir.exists():
            notes.append({
                "domain": domain_slug,
                "status": "not_found",
                "path": str(domain_dir),
                "content": None,
            })
            continue

        for note_file in sorted(domain_dir.glob("*.md")):
            if note_file.name == "README.md":
                continue
            try:
                content = note_file.read_text(encoding="utf-8")
                notes.append({
                    "domain": domain_slug,
                    "status": "loaded",
                    "path": str(note_file),
                    "title": note_file.stem,
                    "content": content,
                })
            except Exception as e:
                notes.append({
                    "domain": domain_slug,
                    "status": "error",
                    "path": str(note_file),
                    "error": str(e),
                    "content": None,
                })

    result = {
        "success": True,
        "domains_requested": domains,
        "domains_valid": valid,
        "domains_invalid": invalid,
        "vault_path": str(vault),
        "notes_loaded": len([n for n in notes if n["status"] == "loaded"]),
        "notes": notes,
    }
    return json.dumps(result)


# Register the tool
registry.register(
    name="load_domain_knowledge",
    toolset="knowledge",
    schema={
        "name": "load_domain_knowledge",
        "description": (
            "Load shared domain knowledge notes for the given domain slugs. "
            "Valid domains: frontend, backend, devops, security, testing, data, mobile, infrastructure, "
            "business, marketing, sales, finance, operations, people. "
            "Returns loaded notes as JSON with content, path, and status."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of domain slugs to load notes from.",
                },
            },
            "required": ["domains"],
        },
    },
    handler=lambda args, **kw: load_domain_knowledge(
        domains=args.get("domains", []),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_requirements,
)
