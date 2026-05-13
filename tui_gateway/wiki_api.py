"""Wiki scanning API for the TUI gateway.

Provides filesystem-level wiki introspection for native clients that
render graph views or page detail. Reads from $WIKI_PATH (default ~/wiki).
"""
import os
import re
from pathlib import Path


from typing import Optional

def _default_wiki_path() -> str:
    env = os.environ.get("WIKI_PATH", "")
    if env:
        return env
    return os.path.expanduser("~/wiki")


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Simple frontmatter parser — returns (metadata, body)."""
    if not content.startswith("---"):
        return {}, content
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content
    metadata = {}
    for line in parts[1].strip().split("\n"):
        line = line.strip()
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            # strip outer quotes
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                val = val[1:-1]
            metadata[key] = val
    return metadata, parts[2]


def _extract_wikilinks(body: str) -> list[str]:
    """Extract [[wikilinks]] from markdown body."""
    pattern = r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]'
    matches = re.findall(pattern, body)
    return [m.strip().lower().replace(" ", "-") for m in matches]


def wiki_scan(wiki_path: Optional[str] = None) -> dict:
    """Scan wiki directory and return graph structure."""
    wiki = Path(wiki_path or _default_wiki_path())
    if not wiki.exists():
        return {"pages": [], "links": []}

    pages: list[dict] = []
    page_ids: set[str] = set()
    links: list[dict] = []
    subdirs = ["entities", "concepts", "comparisons", "queries", "raw"]

    # First pass: collect all pages
    for subdir in subdirs:
        dir_path = wiki / subdir
        if not dir_path.exists():
            continue
        for file in dir_path.iterdir():
            if file.suffix != ".md":
                continue
            try:
                content = file.read_text(encoding="utf-8")
            except Exception:
                continue
            fm, _ = _parse_frontmatter(content)
            slug = file.stem
            rel_path = f"{subdir}/{file.name}"

            # Parse tags (handles "[tag1, tag2]" or "tag1, tag2")
            raw_tags = fm.get("tags", "")
            tags: list[str] = []
            if raw_tags:
                cleaned = raw_tags.strip().strip("[]").replace("'", "").replace('"', "")
                tags = [t.strip() for t in cleaned.split(",") if t.strip()]

            pages.append(
                {
                    "id": slug,
                    "title": fm.get("title", slug),
                    "type": fm.get("type", "concept"),
                    "tags": tags,
                    "path": rel_path,
                    "created": fm.get("created", ""),
                    "updated": fm.get("updated", ""),
                    "confidence": fm.get("confidence", ""),
                    "contested": fm.get("contested", "").lower() == "true",
                }
            )
            page_ids.add(slug)

    # Second pass: extract wikilinks (only link to existing pages)
    for subdir in subdirs:
        dir_path = wiki / subdir
        if not dir_path.exists():
            continue
        for file in dir_path.iterdir():
            if file.suffix != ".md":
                continue
            try:
                content = file.read_text(encoding="utf-8")
            except Exception:
                continue
            _, body = _parse_frontmatter(content)
            slug = file.stem
            for target in _extract_wikilinks(body):
                if target in page_ids:
                    links.append({"source": slug, "target": target, "type": "wikilink"})

    return {"pages": pages, "links": links}


def wiki_page(path: str, wiki_path: Optional[str] = None) -> Optional[dict]:
    """Read a single wiki page by relative path (e.g. 'entities/dflash-mlx.md')."""
    wiki = Path(wiki_path or _default_wiki_path())
    target = wiki / path
    # Security: refuse to escape the wiki directory
    try:
        target = target.resolve()
        wiki = wiki.resolve()
    except Exception:
        return None
    if not str(target).startswith(str(wiki)):
        return None
    if not target.exists() or target.suffix != ".md":
        return None
    try:
        content = target.read_text(encoding="utf-8")
    except Exception:
        return None
    fm, body = _parse_frontmatter(content)
    return {"frontmatter": fm, "body": body, "path": path}
