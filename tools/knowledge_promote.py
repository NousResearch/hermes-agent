"""Knowledge promote tool — promotes knowledge from project-local to domain-shared.

Tool name: promote_knowledge
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home
from tools.registry import registry


def _resolve_vault_path() -> Path:
    vault_path = os.environ.get("OBSIDIAN_VAULT_PATH")
    if vault_path:
        return Path(vault_path)
    standalone = Path.home() / "ObsidianVault" / "HermesAgent"
    if standalone.exists():
        return standalone
    return Path.home() / "Documents" / "Obsidian Vault"


def _slugify(text: str) -> str:
    """Convert title to filename-safe slug."""
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip('-') or "untitled"


def _update_domain_index(vault: Path, domain: str, increment: int = 1) -> None:
    """Update the domain index note count for a domain."""
    index_path = vault / "domains" / "index.md"
    if not index_path.exists():
        return
    try:
        text = index_path.read_text(encoding="utf-8")
        # Find the row for this domain and update the note count
        pattern = rf'(\|.*{re.escape(domain)}.*\|)\s*(\d+)\s*\|'
        def increment_match(m: re.Match) -> str:
            prefix = m.group(1)
            count = int(m.group(2)) + increment
            return f"{prefix} {count} |"
        new_text = re.sub(pattern, increment_match, text)
        index_path.write_text(new_text, encoding="utf-8")
    except Exception:
        pass  # Non-critical — index update is best-effort


def _add_backlink(vault: Path, project_slug: str, note_title: str, domain: str) -> None:
    """Add a backlink to the source project note."""
    project_note = vault / "projects" / f"{project_slug}.md"
    if not project_note.exists():
        return
    try:
        text = project_note.read_text(encoding="utf-8")
        backlink = f"\n- Promoted: [[{note_title}]] → [[domains/{domain}/README|{domain}]] domain"
        # Check if backlink already exists
        if backlink.strip() not in text:
            # Append to the Notes section
            if "## Notes" in text:
                text = text.replace("## Notes", f"## Notes{backlink}")
            else:
                text = text.rstrip() + backlink + "\n"
            project_note.write_text(text, encoding="utf-8")
    except Exception:
        pass  # Non-critical


def promote_knowledge(
    title: str,
    content: str,
    source_project: str,
    target_domain: str,
    summary: Optional[str] = None,
    task_id: str = None,
) -> str:
    """Promote knowledge from project-local to domain-shared KB.

    Args:
        title: Title of the knowledge note
        content: Markdown content of the knowledge
        source_project: Project slug where knowledge originated
        target_domain: Target domain slug (frontend, backend, etc.)
        summary: Optional short summary for the note
        task_id: Optional task identifier

    Returns:
        JSON string with success status and note path
    """
    valid_domains = [
        "frontend", "backend", "devops", "security", "testing",
        "data", "mobile", "infrastructure",
        "business", "marketing", "sales", "finance", "operations", "people",
    ]

    if target_domain not in valid_domains:
        return json.dumps({
            "success": False,
            "error": f"Invalid domain: {target_domain}. Valid: {', '.join(valid_domains)}",
        })

    vault = _resolve_vault_path()
    domain_dir = vault / "domains" / target_domain
    if not domain_dir.exists():
        domain_dir.mkdir(parents=True, exist_ok=True)

    slug = _slugify(title)
    timestamp = datetime.now().strftime("%Y-%m-%d")
    note_path = domain_dir / f"{slug}.md"

    # Handle duplicate titles
    counter = 1
    while note_path.exists():
        slug = f"{_slugify(title)}-{counter}"
        note_path = domain_dir / f"{slug}.md"
        counter += 1

    # Build note with frontmatter
    frontmatter = (
        f"---\n"
        f"title: {title}\n"
        f"tags:\n"
        f"  - hermes-agent/knowledge\n"
        f"  - {target_domain}\n"
        f"  - promoted\n"
        f"status: approved\n"
        f"origin_project: {source_project}\n"
        f"promoted_at: {timestamp}\n"
        f"updated: {timestamp}\n"
        f"---\n\n"
    )

    if summary:
        frontmatter += f"> **Summary:** {summary}\n\n"

    full_content = frontmatter + content

    try:
        note_path.write_text(full_content, encoding="utf-8")
        # Update domain index (best-effort)
        _update_domain_index(vault, target_domain, increment=1)
        # Add backlink to source project (best-effort)
        _add_backlink(vault, source_project, title, target_domain)

        return json.dumps({
            "success": True,
            "note_path": str(note_path),
            "domain": target_domain,
            "source_project": source_project,
            "promoted_at": timestamp,
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
        })


def check_requirements() -> bool:
    """Check if vault path exists."""
    vault = _resolve_vault_path()
    return vault.exists()


# Register the tool
registry.register(
    name="promote_knowledge",
    toolset="knowledge",
    schema={
        "name": "promote_knowledge",
        "description": (
            "Promote knowledge from project-local KB to a shared domain KB. "
            "Valid domains: frontend, backend, devops, security, testing, data, mobile, infrastructure, "
            "business, marketing, sales, finance, operations, people. "
            "Creates a note in the domain directory with frontmatter tracking origin and timestamp."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Title of the knowledge note"},
                "content": {"type": "string", "description": "Markdown content of the knowledge"},
                "source_project": {"type": "string", "description": "Project slug where knowledge originated"},
                "target_domain": {"type": "string", "description": "Target domain slug"},
                "summary": {"type": "string", "description": "Optional short summary"},
            },
            "required": ["title", "content", "source_project", "target_domain"],
        },
    },
    handler=lambda args, **kw: promote_knowledge(
        title=args.get("title", ""),
        content=args.get("content", ""),
        source_project=args.get("source_project", ""),
        target_domain=args.get("target_domain", ""),
        summary=args.get("summary"),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_requirements,
)
