#!/usr/bin/env python3
"""Persistent markdown wiki tool for compounding personal knowledge."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from agent.wiki_paths import resolve_llm_wiki_path

logger = logging.getLogger(__name__)


PAGE_TYPES: Dict[str, tuple[str, str]] = {
    "entity": ("entities", "## Entities"),
    "concept": ("concepts", "## Concepts"),
    "comparison": ("comparisons", "## Comparisons"),
    "query": ("queries", "## Queries"),
    "article": ("articles", "## Articles"),
    "summary": ("articles", "## Articles"),
}


@dataclass(frozen=True)
class WikiLayout:
    root: Path
    pages_root: Path
    raw_root: Path
    schema_file: Path
    index_file: Path
    log_file: Path
    tags_file: Path
    legacy: bool

    @property
    def page_dirs(self) -> Dict[str, Path]:
        return {
            page_type: self.pages_root / dirname
            for page_type, (dirname, _section) in PAGE_TYPES.items()
        }


def _resolve_layout() -> WikiLayout:
    root = resolve_llm_wiki_path()
    legacy = (root / "wiki").exists() or (root / "wiki" / "_index.md").exists()
    pages_root = root / "wiki" if legacy else root
    return WikiLayout(
        root=root,
        pages_root=pages_root,
        raw_root=root / "raw",
        schema_file=root / "SCHEMA.md",
        index_file=pages_root / ("_index.md" if legacy else "index.md"),
        log_file=pages_root / ("_log.md" if legacy else "log.md"),
        tags_file=pages_root / "_tags.md",
        legacy=legacy,
    )


def _is_within(base: Path, candidate: Path) -> bool:
    try:
        return candidate.resolve().is_relative_to(base.resolve())
    except (OSError, ValueError):
        return False


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower().strip())
    return re.sub(r"-+", "-", slug).strip("-") or "untitled"


def _normalize_csv(value: str) -> List[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _relative_page_path(layout: WikiLayout, path: Path) -> str:
    return str(path.relative_to(layout.pages_root))


def _nav_files(layout: WikiLayout) -> set[Path]:
    return {
        layout.schema_file,
        layout.index_file,
        layout.log_file,
        layout.tags_file,
    }


def _collect_page_files(layout: WikiLayout) -> List[Path]:
    files: List[Path] = []
    seen: set[Path] = set()
    for path in layout.page_dirs.values():
        if not path.is_dir():
            continue
        for md_file in sorted(path.rglob("*.md")):
            if md_file in seen:
                continue
            seen.add(md_file)
            files.append(md_file)
    return files


def _parse_frontmatter(text: str) -> Dict[str, Any]:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?", text, re.DOTALL)
    if not match:
        return {}
    parsed = yaml.safe_load(match.group(1))
    return parsed if isinstance(parsed, dict) else {}


def _strip_frontmatter(text: str) -> str:
    return re.sub(r"^---\s*\n.*?\n---\s*\n?", "", text, flags=re.DOTALL)


def _build_frontmatter(data: Dict[str, Any]) -> str:
    return "---\n" + yaml.safe_dump(data, sort_keys=False, allow_unicode=False).strip() + "\n---\n\n"


def _first_heading(text: str) -> Optional[str]:
    for line in _strip_frontmatter(text).splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return None


def _summarize_content(text: str, fallback: str) -> str:
    body = _strip_frontmatter(text)
    for line in body.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith(">"):
            return line[:140]
    return fallback[:140]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _ensure_structure(layout: WikiLayout) -> None:
    layout.root.mkdir(parents=True, exist_ok=True)
    layout.pages_root.mkdir(parents=True, exist_ok=True)
    layout.raw_root.mkdir(parents=True, exist_ok=True)
    for subdir in ("articles", "papers", "transcripts", "assets"):
        (layout.raw_root / subdir).mkdir(parents=True, exist_ok=True)
    for page_dir in layout.page_dirs.values():
        page_dir.mkdir(parents=True, exist_ok=True)


def _default_schema(domain: str) -> str:
    return f"""# Wiki Schema

## Domain
{domain}

## Operating Rules
- Raw sources live in `raw/` and are immutable.
- Every wiki page must have YAML frontmatter with `title`, `created`, `updated`, `type`, and `tags`.
- Use lowercase, hyphenated filenames.
- Use `[[wikilinks]]` to connect related pages.
- Update `updated` whenever a page changes.
- Every new page must be listed in the index.
- Significant actions must be appended to the log.

## Page Types
- `entities/`: people, companies, products, tools, places
- `concepts/`: ideas, themes, frameworks, recurring topics
- `comparisons/`: side-by-side analyses, trade-offs, evaluations
- `queries/`: durable answers and investigations worth keeping
- `articles/`: synthesis pages and source summaries
"""


def _default_index(layout: WikiLayout) -> str:
    root_hint = _relative_page_path(layout, layout.index_file) if _is_within(layout.pages_root, layout.index_file) else layout.index_file.name
    return f"""# Wiki Index

> Content catalog for the persistent wiki.
> Read this first before answering questions from the compiled knowledge base.

## Entities
<!-- entity pages listed here -->

## Concepts
<!-- concept pages listed here -->

## Comparisons
<!-- comparison pages listed here -->

## Queries
<!-- filed query results listed here -->

## Articles
<!-- synthesis and summary pages listed here -->

<!-- Generated by Hermes. Index file: {root_hint} -->
"""


def _default_log() -> str:
    return """# Wiki Log

> Chronological record of wiki actions.
> Format: `## [YYYY-MM-DD] action | subject`
"""


def _ensure_navigation_files(layout: WikiLayout, domain: str = "General personal wiki") -> None:
    if not layout.schema_file.exists():
        layout.schema_file.write_text(_default_schema(domain), encoding="utf-8")
    if not layout.index_file.exists():
        layout.index_file.write_text(_default_index(layout), encoding="utf-8")
    if not layout.log_file.exists():
        layout.log_file.write_text(_default_log(), encoding="utf-8")


def _append_log(layout: WikiLayout, action: str, subject: str, details: Optional[Iterable[str]] = None) -> str:
    _ensure_structure(layout)
    _ensure_navigation_files(layout)

    today = datetime.now().strftime("%Y-%m-%d")
    lines = [f"## [{today}] {action} | {subject}"]
    for detail in details or ():
        lines.append(f"- {detail}")
    entry = "\n".join(lines) + "\n"

    existing = layout.log_file.read_text(encoding="utf-8").rstrip()
    updated = existing + "\n\n" + entry if existing else entry
    layout.log_file.write_text(updated.rstrip() + "\n", encoding="utf-8")
    return entry.strip()


def _insert_index_entry(index_text: str, section_header: str, entry: str) -> str:
    if entry in index_text:
        return index_text

    lines = index_text.splitlines()
    insert_at = len(lines)
    in_section = False

    for i, line in enumerate(lines):
        if line.strip() == section_header:
            in_section = True
            insert_at = i + 1
            continue
        if in_section and line.startswith("## "):
            insert_at = i
            break
        if in_section:
            insert_at = i + 1

    lines.insert(insert_at, entry)
    return "\n".join(lines).rstrip() + "\n"


def _update_index(layout: WikiLayout, page_path: Path, title: str, page_type: str, summary: str) -> None:
    _ensure_navigation_files(layout)
    section_header = PAGE_TYPES[page_type][1]
    relative_path = _relative_page_path(layout, page_path)
    entry = f"- [[{relative_path}|{title}]] — {summary}"
    current = layout.index_file.read_text(encoding="utf-8")
    updated = _insert_index_entry(current, section_header, entry)
    layout.index_file.write_text(updated, encoding="utf-8")


def _candidate_paths(layout: WikiLayout, page: str) -> List[Path]:
    page = page.strip()
    candidates = [
        layout.pages_root / page,
        layout.pages_root / f"{page}.md",
        layout.root / page,
        layout.root / f"{page}.md",
    ]
    for page_dir in layout.page_dirs.values():
        candidates.append(page_dir / page)
        candidates.append(page_dir / f"{page}.md")
    return candidates


def _search(query: str, max_results: int = 10) -> str:
    layout = _resolve_layout()
    files = _collect_page_files(layout)
    if not files:
        return json.dumps(
            {
                "results": [],
                "message": "Wiki is empty or not initialized. Run kb(action='init') first.",
                "wiki_path": str(layout.root),
            },
            indent=2,
        )

    query_lower = query.lower()
    query_terms = query_lower.split()
    results = []

    for md_file in files:
        try:
            content = _read_text(md_file)
        except (OSError, UnicodeDecodeError):
            continue

        content_lower = content.lower()
        score = sum(content_lower.count(term) for term in query_terms)
        if score == 0:
            continue

        frontmatter = _parse_frontmatter(content)
        title = frontmatter.get("title") or _first_heading(content) or md_file.stem.replace("-", " ").title()
        snippet = ""
        for line in _strip_frontmatter(content).splitlines():
            if any(term in line.lower() for term in query_terms):
                snippet = line.strip()[:200]
                break

        results.append(
            {
                "file": _relative_page_path(layout, md_file),
                "title": title,
                "type": frontmatter.get("type"),
                "tags": frontmatter.get("tags", []),
                "score": score,
                "snippet": snippet,
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return json.dumps(
        {
            "results": results[:max_results],
            "matches": min(len(results), max_results),
            "total_matches": len(results),
            "wiki_path": str(layout.root),
        },
        indent=2,
    )


def _list_pages() -> str:
    layout = _resolve_layout()
    if not layout.index_file.exists():
        return json.dumps(
            {
                "error": "Wiki index not found. Run kb(action='init') first.",
                "wiki_path": str(layout.root),
            }
        )
    return _read_text(layout.index_file)


def _read_page(page: str) -> str:
    layout = _resolve_layout()
    for candidate in _candidate_paths(layout, page):
        if not _is_within(layout.root, candidate):
            continue
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file():
            return _read_text(resolved)

    page_lower = page.lower().replace(" ", "-")
    for md_file in _collect_page_files(layout):
        if page_lower in md_file.stem.lower():
            return _read_text(md_file)

    return json.dumps({"error": f"Page not found: {page}", "wiki_path": str(layout.root)})


def _write_page(
    title: str,
    content: str,
    page_type: str = "concept",
    tags: str = "",
    sources: str = "",
) -> str:
    layout = _resolve_layout()
    if page_type not in PAGE_TYPES:
        return json.dumps({"error": f"page_type must be one of: {', '.join(sorted(PAGE_TYPES))}"})

    _ensure_structure(layout)
    _ensure_navigation_files(layout)

    slug = _slugify(title)
    target_dir = layout.page_dirs[page_type]
    target_file = target_dir / f"{slug}.md"
    now = datetime.now().strftime("%Y-%m-%d")
    tag_list = _normalize_csv(tags)
    source_list = _normalize_csv(sources)

    existing_frontmatter: Dict[str, Any] = {}
    created = now
    action = "create"
    if target_file.exists():
        existing_text = _read_text(target_file)
        existing_frontmatter = _parse_frontmatter(existing_text)
        created = str(existing_frontmatter.get("created", now))
        action = "update"

    frontmatter = {
        "title": title,
        "created": created,
        "updated": now,
        "type": page_type,
        "tags": tag_list,
        "sources": source_list,
    }
    full_content = _build_frontmatter(frontmatter) + content.strip() + "\n"
    target_file.write_text(full_content, encoding="utf-8")

    summary = _summarize_content(content, title)
    _update_index(layout, target_file, title, page_type, summary)
    _append_log(
        layout,
        action,
        title,
        details=[
            f"page: {_relative_page_path(layout, target_file)}",
            f"type: {page_type}",
            f"tags: {', '.join(tag_list) if tag_list else 'none'}",
        ],
    )

    return json.dumps(
        {
            "success": True,
            "action": action,
            "file": _relative_page_path(layout, target_file),
            "title": title,
            "wiki_path": str(layout.root),
        },
        indent=2,
    )


def _init_wiki(domain: str) -> str:
    layout = _resolve_layout()
    _ensure_structure(layout)
    _ensure_navigation_files(layout, domain=domain)
    _append_log(
        layout,
        "create",
        "Wiki initialized",
        details=[
            f"domain: {domain}",
            f"wiki_path: {layout.root}",
            "structure: raw/, entities/, concepts/, comparisons/, queries/, articles/",
        ],
    )
    return json.dumps(
        {
            "success": True,
            "wiki_path": str(layout.root),
            "layout": "legacy" if layout.legacy else "modern",
            "created": [
                str(layout.schema_file),
                str(layout.index_file),
                str(layout.log_file),
            ],
        },
        indent=2,
    )


def _parse_date(value: Any) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    for candidate in (text, text[:10]):
        try:
            return datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        except ValueError:
            continue
    return None


def _extract_wikilinks(text: str) -> List[str]:
    matches = re.findall(r"\[\[([^\]|#]+)", text)
    return [match.strip().lower().removesuffix(".md") for match in matches if match.strip()]


def _lint_wiki(stale_days: int = 180) -> str:
    layout = _resolve_layout()
    files = _collect_page_files(layout)
    if not files:
        return json.dumps(
            {
                "success": False,
                "error": "Wiki is empty or not initialized.",
                "wiki_path": str(layout.root),
            },
            indent=2,
        )

    index_text = layout.index_file.read_text(encoding="utf-8") if layout.index_file.exists() else ""
    now = datetime.now(timezone.utc)

    stem_map: Dict[str, Path] = {}
    rel_map: Dict[str, Path] = {}
    inbound = {path: 0 for path in files}
    missing_frontmatter: List[str] = []
    missing_index: List[str] = []
    stale_pages: List[str] = []
    broken_links: set[str] = set()

    for path in files:
        rel_path = _relative_page_path(layout, path).removesuffix(".md").lower()
        stem_map[path.stem.lower()] = path
        rel_map[rel_path] = path

    for path in files:
        text = _read_text(path)
        frontmatter = _parse_frontmatter(text)
        if not frontmatter:
            missing_frontmatter.append(_relative_page_path(layout, path))

        updated_at = _parse_date(frontmatter.get("updated"))
        if updated_at is not None:
            updated_utc = updated_at if updated_at.tzinfo else updated_at.replace(tzinfo=timezone.utc)
            age_days = (now - updated_utc).days
            if age_days >= stale_days:
                stale_pages.append(f"{_relative_page_path(layout, path)} ({age_days}d)")

        relative_path = _relative_page_path(layout, path)
        if relative_path not in index_text and path.stem not in index_text:
            missing_index.append(relative_path)

        for link in _extract_wikilinks(text):
            target = rel_map.get(link) or stem_map.get(Path(link).name)
            if target is None:
                broken_links.add(link)
                continue
            if target != path:
                inbound[target] += 1

    orphans = [
        _relative_page_path(layout, path)
        for path, count in inbound.items()
        if count == 0 and len(files) > 1
    ]

    issues = {
        "missing_frontmatter": missing_frontmatter,
        "missing_index_entries": missing_index,
        "orphan_pages": orphans,
        "stale_pages": stale_pages,
        "broken_wikilinks": sorted(broken_links),
    }
    issue_count = sum(len(items) for items in issues.values())

    _append_log(
        layout,
        "lint",
        f"{issue_count} issues found",
        details=[f"{key}: {len(value)}" for key, value in issues.items()],
    )

    try:
        import asyncio
        from agent.graph_manager import GraphManager
        from hermes_constants import get_hermes_dir
        gm = GraphManager(get_hermes_dir("context-graph/kuzu_db", "kuzu_db"))
        archived_facts = asyncio.run(gm.decay_knowledge_graph(half_life_days=365, threshold=0.2))
        if archived_facts > 0:
            issues["archived_facts"] = ["%d stale facts tombstoned" % archived_facts]
            issue_count += archived_facts
            _append_log(layout, "lint", f"Decayed memory graph. Archived {archived_facts} stale facts.")
    except Exception as e:
        logger.debug("Failed to run active memory decay during lint: %s", e)

    return json.dumps(
        {
            "success": True,
            "issue_count": issue_count,
            "wiki_path": str(layout.root),
            "issues": issues,
        },
        indent=2,
    )


def kb_tool(
    action: str,
    query: str = None,
    page: str = None,
    title: str = None,
    content: str = None,
    page_type: str = "concept",
    tags: str = "",
    message: str = None,
    max_results: int = 10,
    domain: str = "General personal wiki",
    sources: str = "",
    stale_days: int = 180,
) -> str:
    """Single entry point for the wiki knowledge base."""

    if action == "init":
        return _init_wiki(domain)
    if action == "search":
        if not query:
            return json.dumps({"error": "query is required for search action"})
        return _search(query, max_results=max_results)
    if action == "list":
        return _list_pages()
    if action == "read":
        if not page:
            return json.dumps({"error": "page is required for read action"})
        return _read_page(page)
    if action == "file":
        if not title or not content:
            return json.dumps({"error": "title and content are required for file action"})
        return _write_page(title=title, content=content, page_type=page_type, tags=tags, sources=sources)
    if action == "log":
        if not message:
            return json.dumps({"error": "message is required for log action"})
        layout = _resolve_layout()
        entry = _append_log(layout, "note", "manual log", details=[message])
        return json.dumps({"success": True, "entry": entry, "wiki_path": str(layout.root)}, indent=2)
    if action == "lint":
        return _lint_wiki(stale_days=stale_days)

    return json.dumps(
        {
            "error": f"Unknown action '{action}'. Use: init, search, list, read, file, log, lint"
        }
    )


def check_kb_requirements() -> bool:
    """Always register the wiki tool so it can initialize a wiki from scratch."""
    return True


KB_SCHEMA = {
    "name": "kb",
    "description": (
        "Operate Hermes's persistent markdown wiki for personal knowledge. The wiki can live "
        "inside Obsidian or at a standalone path. Use it to build a compounding knowledge base "
        "of entities, concepts, comparisons, durable query answers, and synthesis pages.\n\n"
        "Path resolution:\n"
        "- `LLM_WIKI_PATH`\n"
        "- `knowledge.wiki_path` in config\n"
        "- Existing legacy `~/hermes-kb`\n"
        "- Otherwise, if Obsidian is configured, `<vault>/<agent_prefix>/Wiki`\n\n"
        "Actions:\n"
        "- init: Create the wiki structure, schema, index, and log\n"
        "- search: Full-text search across wiki pages\n"
        "- list: Read the wiki index\n"
        "- read: Read a specific page or raw file\n"
        "- file: Create or update a wiki page\n"
        "- lint: Audit the wiki for orphans, broken links, stale pages, and missing index entries\n"
        "- log: Append a manual note to the wiki log"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["init", "search", "list", "read", "file", "lint", "log"],
                "description": "The action to perform.",
            },
            "query": {
                "type": "string",
                "description": "Search query for search.",
            },
            "page": {
                "type": "string",
                "description": "Page name or relative path for read.",
            },
            "title": {
                "type": "string",
                "description": "Page title for file.",
            },
            "content": {
                "type": "string",
                "description": "Markdown page content for file.",
            },
            "page_type": {
                "type": "string",
                "enum": ["entity", "concept", "comparison", "query", "article", "summary"],
                "description": "Page type for file.",
            },
            "tags": {
                "type": "string",
                "description": "Comma-separated tags for file.",
            },
            "sources": {
                "type": "string",
                "description": "Comma-separated source references for file.",
            },
            "message": {
                "type": "string",
                "description": "Manual log message for log.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of search results (default 10).",
                "default": 10,
            },
            "domain": {
                "type": "string",
                "description": "Domain description used when initializing a wiki.",
            },
            "stale_days": {
                "type": "integer",
                "description": "Age threshold for stale-page lint findings (default 180).",
                "default": 180,
            },
        },
        "required": ["action"],
    },
}


from tools.registry import registry

registry.register(
    name="kb",
    toolset="knowledge",
    schema=KB_SCHEMA,
    handler=lambda args, **kw: kb_tool(
        action=args.get("action", ""),
        query=args.get("query"),
        page=args.get("page"),
        title=args.get("title"),
        content=args.get("content"),
        page_type=args.get("page_type", "concept"),
        tags=args.get("tags", ""),
        message=args.get("message"),
        max_results=args.get("max_results", 10),
        domain=args.get("domain", "General personal wiki"),
        sources=args.get("sources", ""),
        stale_days=args.get("stale_days", 180),
    ),
    check_fn=check_kb_requirements,
    emoji="📖",
    description="Search, maintain, and lint the persistent markdown wiki knowledge base",
    mutates=True,
    requires_confirmation=False,
)
