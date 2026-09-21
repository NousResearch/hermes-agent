"""
GBrain knowledge tools for Hermes Agent.

GBrain is Sahil's structured Markdown knowledge layer at ``~/brain``. These
built-in tools expose it directly to KenseiAgent alongside Mnemosyne's
conversational memory provider.

Search has two modes selected by ``BRAIN_SEARCH_MODE`` (env var):
  - ``keyword`` (default): grep/ripgrep over ~/brain/*.md files. Zero external
    deps, no API key, instant results.
  - ``embeddings``: semantic search via local Ollama (nomic-embed-text). No paid
    key required; runs entirely on the local Ollama daemon.

Get, put, graph, and status all operate directly on the markdown files in
``~/brain/`` -- graph parses ``[[wikilinks]]``. No gbrain CLI / PGLite dependency.

# Upstream: HermesAgent | tools/gbrain.py | Last checked: 2026-06-15 | Policy: ~/brain/conventions/infrastructure.md ## Source Code Policy
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import requests

from tools.registry import registry, tool_error, tool_result


GBRAIN_REPO = Path(os.environ.get("GBRAIN_REPO", "~/brain")).expanduser()
BRAIN_SEARCH_MODE = os.environ.get("BRAIN_SEARCH_MODE", "keyword").strip().lower()
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
_RIPGREP = bool(shutil.which("rg"))


# ── Availability checks ──────────────────────────────────────────────────────

def _check_repo_available() -> bool:
    """Repo must exist -- all GBrain ops read/write the markdown files directly."""
    return GBRAIN_REPO.is_dir()


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(minimum, min(maximum, parsed))


def _list_md_files() -> list[Path]:
    """All .md files under GBRAIN_REPO (excluding .git)."""
    files: list[Path] = []
    for root, dirs, fnames in os.walk(str(GBRAIN_REPO)):
        dirs[:] = [d for d in dirs if d != ".git"]
        for fn in fnames:
            if fn.endswith(".md"):
                files.append(Path(root) / fn)
    return files


def _path_to_slug(path: Path) -> str:
    """Convert a file path under GBRAIN_REPO into a brain page slug."""
    try:
        rel = path.relative_to(GBRAIN_REPO)
        return str(rel.with_suffix(""))
    except ValueError:
        return path.stem


_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[#|][^\]]*)?\]\]")


def _build_link_index() -> tuple[dict[str, Path], dict[str, str], dict[str, list[str]], int]:
    """Parse [[wikilinks]] from every brain page directly off the markdown.

    Returns (slug->path, stem->slug, slug->resolved-target-slugs, dangling_count).
    A link may reference a full slug ("people/sahil-saghir") or a bare basename
    ("sahil-saghir"); both resolve. Links to no known page are counted as dangling.
    """
    slug_to_path: dict[str, Path] = {}
    stem_to_slug: dict[str, str] = {}
    raw: dict[str, list[str]] = {}
    for fp in _list_md_files():
        slug = _path_to_slug(fp)
        slug_to_path[slug] = fp
        stem_to_slug.setdefault(Path(slug).name, slug)
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = ""
        raw[slug] = [m.strip() for m in _WIKILINK_RE.findall(text)]

    links: dict[str, list[str]] = {}
    dangling = 0
    for slug, targets in raw.items():
        resolved: list[str] = []
        for t in targets:
            if t in slug_to_path:
                resolved.append(t)
            elif Path(t).name in stem_to_slug:
                resolved.append(stem_to_slug[Path(t).name])
            else:
                dangling += 1
        links[slug] = resolved
    return slug_to_path, stem_to_slug, links, dangling


# ── Search (dual-mode: keyword / embeddings) ─────────────────────────────────

GBRAIN_SEARCH_SCHEMA = {
    "name": "gbrain_search",
    "description": (
        "Search Sahil's GBrain structured knowledge base at ~/brain. "
        "Use this for durable facts, project knowledge, people, accounts, "
        "infrastructure conventions, and timeline context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Keyword or natural-language search query."},
            "limit": {"type": "integer", "description": "Maximum result pages to return.", "default": 5},
        },
        "required": ["query"],
    },
}


def gbrain_search(args: dict, **_: Any) -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        return tool_error("query is required")
    limit = _bounded_int(args.get("limit"), default=5, minimum=1, maximum=25)

    if BRAIN_SEARCH_MODE == "embeddings":
        return _search_embeddings(query, limit)
    return _search_keyword(query, limit)


def _search_keyword(query: str, limit: int) -> str:
    """grep/ripgrep keyword search over ~/brain markdown files."""
    brain_dir = str(GBRAIN_REPO)
    if not GBRAIN_REPO.is_dir():
        return tool_error(f"brain dir not found: {brain_dir}")

    if _RIPGREP:
        cmd = [
            "rg", "-i", "--no-heading", "--with-filename", "-n", "-m", "3",
            "--glob", "*.md", "--", query, brain_dir,
        ]
    else:
        cmd = [
            "grep", "-r", "-i", "--no-heading", "--with-filename", "-n", "-m", "3",
            "--", query, os.path.join(brain_dir, "*.md"),
        ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL)
        if result.returncode not in (0, 1):  # 1 = no match (not an error)
            return tool_error(f"keyword search failed: {result.stderr.strip()[:200]}")
    except subprocess.TimeoutExpired:
        return tool_error("keyword search timed out")
    except FileNotFoundError:
        return tool_error("grep/rg not installed")

    # Parse results into structured list
    seen: set[str] = set()
    structured = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(":", 2)
        if len(parts) < 2:
            continue
        slug = _path_to_slug(Path(parts[0])) if parts[0] else ""
        if slug and slug not in seen:
            seen.add(slug)
            snippet = parts[2].strip()[:200] if len(parts) > 2 else ""
            structured.append({"slug": slug, "snippet": snippet})
        if len(structured) >= limit:
            break

    return tool_result({
        "query": query,
        "results": structured,
        "total": len(structured),
        "limit": limit,
        "mode": "keyword",
    })


def _search_embeddings(query: str, limit: int) -> str:
    """Semantic search via local Ollama embeddings."""
    if not GBRAIN_REPO.is_dir():
        return tool_error(f"brain dir not found: {str(GBRAIN_REPO)}")

    # 1. Embed the query
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": query},
            timeout=30,
        )
        resp.raise_for_status()
        query_vec = resp.json()["embedding"]
    except Exception as exc:
        return tool_error(f"Ollama embedding failed: {exc}")

    # 2. Walk ~/brain markdown files and embed each page
    md_files = _list_md_files()
    if not md_files:
        return tool_error("no markdown pages found in brain dir")

    page_vectors: list[tuple[str, str, list[float]]] = []
    for fp in md_files:
        try:
            content = fp.read_text(encoding="utf-8", errors="replace")[:2000]
        except Exception:
            continue
        slug = _path_to_slug(fp)
        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={"model": OLLAMA_EMBED_MODEL, "prompt": content[:512]},
                timeout=30,
            )
            resp.raise_for_status()
            vec = resp.json()["embedding"]
            page_vectors.append((slug, content[:200], vec))
        except Exception:
            continue

    # 3. Cosine similarity ranking
    def _cos_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    scored = [(_cos_sim(query_vec, pv), slug, snippet) for slug, snippet, pv in page_vectors]
    scored.sort(key=lambda x: x[0], reverse=True)

    results = [
        {"slug": slug, "snippet": snippet[:200], "score": round(score, 4)}
        for score, slug, snippet in scored[:limit]
        if score > 0
    ]

    return tool_result({
        "query": query,
        "results": results,
        "total": len(results),
        "limit": limit,
        "mode": "embeddings",
    })


# ── Get / Put (direct file ops, no gbrain CLI) ───────────────────────────────

GBRAIN_GET_SCHEMA = {
    "name": "gbrain_get",
    "description": "Read a specific GBrain page by slug, e.g. people/sahil-saghir.",
    "parameters": {
        "type": "object",
        "properties": {
            "slug": {"type": "string", "description": "Page slug without .md."},
        },
        "required": ["slug"],
    },
}


def gbrain_get(args: dict, **_: Any) -> str:
    slug = str(args.get("slug", "")).strip().removesuffix(".md")
    if not slug:
        return tool_error("slug is required")
    target = (GBRAIN_REPO / f"{slug}.md").resolve()
    if not target.is_file():
        return tool_error(f"page not found: {slug} (file {target} does not exist)")
    if GBRAIN_REPO.resolve() not in target.parents:
        return tool_error("slug must resolve under the GBrain repo")
    content = target.read_text(encoding="utf-8")
    return tool_result({"slug": slug, "content": content})


GBRAIN_PUT_SCHEMA = {
    "name": "gbrain_put",
    "description": (
        "Create or replace a GBrain Markdown page. "
        "Use only for durable structured knowledge; content should include YAML frontmatter."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "slug": {"type": "string", "description": "Target page slug without .md."},
            "content": {"type": "string", "description": "Full Markdown page content including frontmatter."},
        },
        "required": ["slug", "content"],
    },
}


def _ensure_owner_only_directory(root: Path, directory: Path) -> None:
    """Create missing descendants of ``root`` without following symlinks."""
    relative = directory.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        try:
            current.lstat()
        except FileNotFoundError:
            current.mkdir(mode=0o700)
            current.lstat()
        if current.is_symlink() or not current.is_dir():
            raise ValueError(f"unsafe GBrain directory component: {current}")
        # Existing Brain directories intentionally remain at the accepted 750
        # policy. Newly-created directories are owner-only 700.


def _atomic_write_owner_only(target: Path, content: str) -> None:
    """Durably replace ``target`` using a same-directory mode-600 temp file."""
    fd = -1
    temp_path: Path | None = None
    try:
        fd, temp_name = tempfile.mkstemp(
            prefix=".gbrain-", suffix=".tmp", dir=str(target.parent)
        )
        temp_path = Path(temp_name)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1  # ownership transferred to ``handle``
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
        temp_path = None
        directory_fd = os.open(
            target.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def gbrain_put(args: dict, **_: Any) -> str:
    slug = str(args.get("slug", "")).strip().removesuffix(".md")
    content = str(args.get("content", ""))
    if not slug:
        return tool_error("slug is required")
    if not content.strip():
        return tool_error("content is required")
    repo_root = GBRAIN_REPO.resolve()
    slug_path = Path(slug)
    if (
        not slug_path.parts
        or slug_path.is_absolute()
        or any(part in ("", ".", "..") for part in slug_path.parts)
    ):
        return tool_error("slug must resolve under the GBrain repo")
    target = repo_root / f"{slug}.md"
    try:
        target.relative_to(repo_root)
        _ensure_owner_only_directory(repo_root, target.parent)
    except (OSError, ValueError):
        return tool_error("slug must resolve under the GBrain repo without symlinks")
    if target.is_symlink():
        return tool_error("slug must not resolve through a symlink")
    if target.exists() and not target.is_file():
        return tool_error("target page must be a regular file")
    _atomic_write_owner_only(target, content)
    return tool_result({"status": "ok", "slug": slug, "path": str(target)})


# ── Graph / Status (markdown-native -- parse [[wikilinks]], no CLI/PGLite) ────

GBRAIN_GRAPH_SCHEMA = {
    "name": "gbrain_graph",
    "description": "Explore graph links from a GBrain page slug.",
    "parameters": {
        "type": "object",
        "properties": {
            "slug": {"type": "string", "description": "Starting page slug."},
            "depth": {"type": "integer", "description": "Traversal depth, 1-5.", "default": 2},
        },
        "required": ["slug"],
    },
}


def gbrain_graph(args: dict, **_: Any) -> str:
    slug = str(args.get("slug", "")).strip().removesuffix(".md")
    if not slug:
        return tool_error("slug is required")
    depth = _bounded_int(args.get("depth"), default=2, minimum=1, maximum=5)

    slug_to_path, stem_to_slug, links, _dangling = _build_link_index()

    # Resolve the starting slug (accept a full slug or a bare basename).
    if slug not in slug_to_path:
        if Path(slug).name in stem_to_slug:
            slug = stem_to_slug[Path(slug).name]
        else:
            return tool_error(f"page not found: {slug}")

    # Undirected adjacency: outbound links plus inbound backlinks.
    adjacency: dict[str, set[str]] = {s: set(t) for s, t in links.items()}
    for src, targets in links.items():
        for tgt in targets:
            adjacency.setdefault(tgt, set()).add(src)

    # Breadth-first walk out to the requested depth.
    visited = {slug: 0}
    frontier = [slug]
    edges: list[list[str]] = []
    while frontier:
        nxt: list[str] = []
        for node in frontier:
            d = visited[node]
            if d >= depth:
                continue
            for nb in sorted(adjacency.get(node, ())):
                pair = sorted([node, nb])
                if pair not in edges:
                    edges.append(pair)
                if nb not in visited:
                    visited[nb] = d + 1
                    nxt.append(nb)
        frontier = nxt

    nodes = [
        {"slug": s, "distance": dist}
        for s, dist in sorted(visited.items(), key=lambda kv: (kv[1], kv[0]))
        if s != slug
    ]
    return tool_result({
        "slug": slug,
        "depth": depth,
        "connected": len(nodes),
        "nodes": nodes,
        "edges": edges,
    })


GBRAIN_STATUS_SCHEMA = {
    "name": "gbrain_status",
    "description": "Return GBrain repo status, page count, and mode info.",
    "parameters": {"type": "object", "properties": {}},
}


def gbrain_status(args: dict, **_: Any) -> str:
    slug_to_path, _stem, links, dangling = _build_link_index()
    total_links = sum(len(v) for v in links.values())
    inbound: set[str] = {t for targets in links.values() for t in targets}
    outbound: set[str] = {s for s, targets in links.items() if targets}
    orphans = [s for s in slug_to_path if s not in inbound and s not in outbound]
    return tool_result({
        "available": _check_repo_available(),
        "repo": str(GBRAIN_REPO),
        "store": "markdown (single source of truth)",
        "page_count": len(slug_to_path),
        "link_count": total_links,
        "dangling_links": dangling,
        "orphan_pages": len(orphans),
        "search_mode": BRAIN_SEARCH_MODE,
        "embedding_model": OLLAMA_EMBED_MODEL if BRAIN_SEARCH_MODE == "embeddings" else "N/A",
    })


# ── Registration ─────────────────────────────────────────────────────────────

registry.register(
    name="gbrain_search",
    toolset="gbrain",
    schema=GBRAIN_SEARCH_SCHEMA,
    handler=gbrain_search,
    check_fn=_check_repo_available,
    requires_env=[],
    is_async=False,
    description=GBRAIN_SEARCH_SCHEMA["description"],
    emoji="🧠",
)
registry.register(
    name="gbrain_get",
    toolset="gbrain",
    schema=GBRAIN_GET_SCHEMA,
    handler=gbrain_get,
    check_fn=_check_repo_available,
    requires_env=[],
    is_async=False,
    description=GBRAIN_GET_SCHEMA["description"],
    emoji="📄",
)
registry.register(
    name="gbrain_put",
    toolset="gbrain",
    schema=GBRAIN_PUT_SCHEMA,
    handler=gbrain_put,
    check_fn=_check_repo_available,
    requires_env=[],
    is_async=False,
    description=GBRAIN_PUT_SCHEMA["description"],
    emoji="✍️",
)
registry.register(
    name="gbrain_graph",
    toolset="gbrain",
    schema=GBRAIN_GRAPH_SCHEMA,
    handler=gbrain_graph,
    check_fn=_check_repo_available,
    requires_env=[],
    is_async=False,
    description=GBRAIN_GRAPH_SCHEMA["description"],
    emoji="🔗",
)
registry.register(
    name="gbrain_status",
    toolset="gbrain",
    schema=GBRAIN_STATUS_SCHEMA,
    handler=gbrain_status,
    check_fn=_check_repo_available,
    requires_env=[],
    is_async=False,
    description=GBRAIN_STATUS_SCHEMA["description"],
    emoji="📊",
)
