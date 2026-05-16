"""
Pure-Python fallback for the knowledge store.
Used when the Rust binary is not available.

Reads/writes the same H2 markdown + directory tree format
that the Rust binary uses. No dependencies beyond stdlib.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Optional


def _knowledge_binary() -> Optional[Path]:
    """Return path to Rust binary, or None if not found."""
    candidates = [
        Path.home() / ".hermes" / "bin" / "knowledge",
        Path("/usr/local/bin/knowledge"),
        Path("/opt/hermes/bin/knowledge"),
    ]
    for p in candidates:
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


def _expand_home(path: str) -> str:
    if path.startswith("~"):
        return path.replace("~", str(Path.home()), 1)
    return path


def _store_path() -> Path:
    return Path.home() / ".hermes" / "knowledge"


# ── Binary interface ──

def append_binary(category: str, content: str) -> str:
    """Append via Rust binary. Returns stored path."""
    bin_path = _knowledge_binary()
    if not bin_path:
        raise FileNotFoundError("knowledge binary not found")

    result = subprocess.run(
        [str(bin_path), "append", category, content],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def read_binary(category: str) -> str:
    bin_path = _knowledge_binary()
    if not bin_path:
        raise FileNotFoundError("knowledge binary not found")

    result = subprocess.run(
        [str(bin_path), "read", category],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout


def search_binary(category: str, query: str) -> str:
    bin_path = _knowledge_binary()
    if not bin_path:
        raise FileNotFoundError("knowledge binary not found")

    result = subprocess.run(
        [str(bin_path), "search", category, query],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout


# ── Pure Python fallback ──

def parse_h2(content: str) -> dict:
    """Pure Python H2 parser. Same format as Rust version."""
    fields = {}
    current_field = None
    current_value = []

    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("## "):
            # Flush previous field
            if current_field:
                value = "\n".join(current_value).strip()
                if value:
                    fields[current_field] = value
                current_value = []

            field_name = stripped[3:].strip().lower()
            if field_name:
                current_field = field_name
        elif current_field is not None:
            current_value.append(line)

    # Flush final field
    if current_field:
        value = "\n".join(current_value).strip()
        if value:
            fields[current_field] = value

    return fields


def _slugify(text: str, max_len: int = 80) -> str:
    """Basic slug: lowercase, replace non-alnum with dash."""
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug[:max_len]


def _resolve_path(category: str, fields: dict) -> str:
    """Determine directory tree path from fields (matches Rust logic)."""
    path_fields = ["tool", "severity", "domain", "type"]
    parts = [category]
    for f in path_fields:
        if len(parts) >= 3:
            break
        if f in fields:
            slug = _slugify(fields[f])
            if slug:
                parts.append(slug)
    return "/".join(parts)


def _generate_slug(fields: dict) -> str:
    """Generate filename slug from fields."""
    candidates = ["title", "source", "fix", "summary"]
    for field in candidates:
        if field in fields:
            s = _slugify(fields[field])
            if s:
                return s
    for value in fields.values():
        if len(value) > 5:
            s = _slugify(value)
            if s:
                return s
    return "untitled"


def append_python(category: str, content: str) -> str:
    """Pure Python append. Writes to same file structure as Rust."""
    fields = parse_h2(content)
    if not fields:
        raise ValueError("no ## field headers found in content")

    store = _store_path()
    rel_dir = _resolve_path(category, fields)
    slug = _generate_slug(fields)
    filename = f"{slug}.md"
    rel_path = f"{rel_dir}/{filename}"

    abs_path = store / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(content)

    return rel_path


def read_python(category: str) -> str:
    """Pure Python read. Walks the tree, assembles markdown."""
    store = _store_path()
    cat_dir = store / category

    if not cat_dir.exists():
        return ""

    files = []
    for md_file in sorted(cat_dir.rglob("*.md")):
        if md_file.is_file():
            files.append(md_file)

    output = []
    for i, f in enumerate(files):
        content = f.read_text().strip()
        if i > 0:
            output.append("---")
        output.append(content)

    return "\n".join(output) + "\n" if output else ""


def search_python(category: str, query: str) -> str:
    """Pure Python search. Greps file contents and fields."""
    store = _store_path()
    cat_dir = store / category

    if not cat_dir.exists():
        return f"No matches for \"{query}\" in {category}\n"

    # Parse query: split field filters from tokens
    field_filters = {}
    tokens = []
    for part in query.split():
        if ":" in part:
            k, v = part.split(":", 1)
            field_filters[k.lower()] = v.lower()
        else:
            tokens.append(part.lower())

    matches = []
    for md_file in sorted(cat_dir.rglob("*.md")):
        if not md_file.is_file():
            continue
        content = md_file.read_text()
        fields = parse_h2(content)

        # Check field filters
        if field_filters:
            if not all(
                fields.get(k, "").lower() == v for k, v in field_filters.items()
            ):
                continue

        # Check tokens
        if tokens:
            lower_content = content.lower()
            hits = sum(
                1
                for t in tokens
                if t in lower_content
                or any(t in v.lower() for v in fields.values())
            )
            if hits > 0:
                matches.append((hits, content.strip()))
        else:
            matches.append((1, content.strip()))

    # Rank by hits
    matches.sort(key=lambda x: -x[0])

    if not matches:
        return f"No matches for \"{query}\" in {category}\n"

    return "\n---\n".join(m[1] for m in matches) + "\n"


# ── Unified interface (binary first, Python fallback) ──

def _try_binary_or_fallback(func_binary, func_python, *args):
    """Try binary first, fall back to Python."""
    try:
        return func_binary(*args)
    except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired):
        return func_python(*args)


def append(category: str, content: str) -> str:
    return _try_binary_or_fallback(append_binary, append_python, category, content)


def read(category: str) -> str:
    return _try_binary_or_fallback(read_binary, read_python, category)


def search(category: str, query: str) -> str:
    return _try_binary_or_fallback(search_binary, search_python, category, query)
