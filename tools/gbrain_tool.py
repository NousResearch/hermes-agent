#!/usr/bin/env python3
"""
GBrain Tool — Native Hermes tool wrapping gbrain CLI.
Provides brain-first lookup, read/write, and ambient signal capture.

Install gbrain: git clone https://github.com/garrytan/gbrain ~/gbrain && cd ~/gbrain && bun install && bun link
Brain location: ~/.gbrain/brain.pglite
"""
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

GBRAIN_BIN = os.path.expanduser("~/.bun/bin/gbrain")
BRAIN_DIR = os.path.expanduser("~/.gbrain")


def _run_gbrain(args: list, timeout: int = 15) -> dict:
    """Run gbrain CLI and return parsed output."""
    env = os.environ.copy()
    env["PATH"] = f"{os.path.dirname(GBRAIN_BIN)}:{env.get('PATH', '')}"
    try:
        result = subprocess.run(
            [GBRAIN_BIN] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=BRAIN_DIR,
        )
        return {
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "timeout", "exit_code": -1}
    except FileNotFoundError:
        return {"stdout": "", "stderr": "gbrain not found", "exit_code": -1}


def _is_available() -> bool:
    return os.path.isfile(GBRAIN_BIN) and os.path.isdir(BRAIN_DIR)


# ---------------------------------------------------------------------------
# Tool Handlers
# ---------------------------------------------------------------------------


def brain_search(query: str, limit: int = 5) -> str:
    """Keyword search across brain pages (tsvector full-text)."""
    if not _is_available():
        return json.dumps({"error": "gbrain not installed or brain not initialized"})
    r = _run_gbrain(["search", query, "--limit", str(limit)])
    return json.dumps({"query": query, "results": r["stdout"], "exit_code": r["exit_code"]})


def brain_query(question: str, limit: int = 5, expand: bool = True) -> str:
    """Hybrid search (RRF + query expansion) — best for natural questions."""
    if not _is_available():
        return json.dumps({"error": "gbrain not installed or brain not initialized"})
    args = ["query", question, "--limit", str(limit)]
    if not expand:
        args.append("--no-expand")
    r = _run_gbrain(args)
    return json.dumps({"question": question, "results": r["stdout"], "exit_code": r["exit_code"]})


def brain_get(slug: str, fuzzy: bool = False) -> str:
    """Read a brain page by slug."""
    if not _is_available():
        return json.dumps({"error": "gbrain not installed or brain not initialized"})
    args = ["get", slug]
    if fuzzy:
        args.append("--fuzzy")
    r = _run_gbrain(args)
    return json.dumps({"slug": slug, "content": r["stdout"], "exit_code": r["exit_code"]})


def brain_put(slug: str, content: str) -> str:
    """Write/update a brain page."""
    if not _is_available():
        return json.dumps({"error": "gbrain not installed or brain not initialized"})
    args = ["put", slug]
    r = subprocess.run(
        [GBRAIN_BIN] + args,
        input=content,
        capture_output=True,
        text=True,
        timeout=15,
        env={**os.environ, "PATH": f"{os.path.dirname(GBRAIN_BIN)}:{os.environ.get('PATH','')}"},
        cwd=BRAIN_DIR,
    )
    return json.dumps({"slug": slug, "exit_code": r.returncode, "stdout": r.stdout.strip()})


def brain_backlinks(slug: str) -> str:
    """Get all pages that link to this slug."""
    if not _is_available():
        return json.dumps({"error": "gbrain not installed"})
    r = _run_gbrain(["backlinks", slug])
    return json.dumps({"slug": slug, "backlinks": r["stdout"], "exit_code": r["exit_code"]})


def brain_timeline(slug: Optional[str] = None) -> str:
    """Get timeline entries for a page, or global timeline if no slug."""
    if not _is_available():
        return json.dumps({"error": "gbrain not installed"})
    args = ["timeline"] if not slug else ["timeline", slug]
    r = _run_gbrain(args)
    return json.dumps({"slug": slug, "timeline": r["stdout"], "exit_code": r["exit_code"]})


def brain_stats() -> str:
    """Get brain statistics."""
    if not _is_available():
        return json.dumps({"error": "gbrain not installed"})
    r = _run_gbrain(["stats"])
    return json.dumps({"stats": r["stdout"], "exit_code": r["exit_code"]})


def brain_list(limit: int = 20, tag: Optional[str] = None) -> str:
    """List brain pages."""
    if not _is_available():
        return json.dumps({"error": "gbrain not installed"})
    args = ["list", "--limit", str(limit)]
    if tag:
        args += ["--tag", tag]
    r = _run_gbrain(args)
    return json.dumps({"pages": r["stdout"], "exit_code": r["exit_code"]})


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
from tools.registry import registry

registry.register(
    name="brain_search",
    toolset="brain",
    schema={
        "name": "brain_search",
        "description": "Keyword search across brain pages (tsvector full-text). Use this BEFORE external web searches when the user asks about people, companies, or topics that may already be in the brain.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "default": 5, "description": "Max results"},
            },
            "required": ["query"],
        },
    },
    handler=lambda args, **kw: brain_search(args["query"], args.get("limit", 5)),
    check_fn=_is_available,
)

registry.register(
    name="brain_query",
    toolset="brain",
    schema={
        "name": "brain_query",
        "description": "Hybrid search (vector + keyword + RRF expansion). Best for natural language questions about people, companies, or topics already in the brain.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Natural language question"},
                "limit": {"type": "integer", "default": 5},
                "expand": {"type": "boolean", "default": True},
            },
            "required": ["question"],
        },
    },
    handler=lambda args, **kw: brain_query(args["question"], args.get("limit", 5), args.get("expand", True)),
    check_fn=_is_available,
)

registry.register(
    name="brain_get",
    toolset="brain",
    schema={
        "name": "brain_get",
        "description": "Read a brain page by its slug (URL-safe identifier).",
        "parameters": {
            "type": "object",
            "properties": {
                "slug": {"type": "string", "description": "Page slug (e.g. 'jazper-ai-trader')"},
                "fuzzy": {"type": "boolean", "default": False},
            },
            "required": ["slug"],
        },
    },
    handler=lambda args, **kw: brain_get(args["slug"], args.get("fuzzy", False)),
    check_fn=_is_available,
)

registry.register(
    name="brain_put",
    toolset="brain",
    schema={
        "name": "brain_put",
        "description": "Write or update a brain page. Every fact must carry inline [Source: ...] citation.",
        "parameters": {
            "type": "object",
            "properties": {
                "slug": {"type": "string", "description": "Page slug"},
                "content": {"type": "string", "description": "Markdown content with frontmatter"},
            },
            "required": ["slug", "content"],
        },
    },
    handler=lambda args, **kw: brain_put(args["slug"], args["content"]),
    check_fn=_is_available,
)

registry.register(
    name="brain_backlinks",
    toolset="brain",
    schema={
        "name": "brain_backlinks",
        "description": "Get all pages that link to a given slug (who references this entity?).",
        "parameters": {
            "type": "object",
            "properties": {
                "slug": {"type": "string"},
            },
            "required": ["slug"],
        },
    },
    handler=lambda args, **kw: brain_backlinks(args["slug"]),
    check_fn=_is_available,
)

registry.register(
    name="brain_timeline",
    toolset="brain",
    schema={
        "name": "brain_timeline",
        "description": "Get timeline entries for a brain page (or global if no slug).",
        "parameters": {
            "type": "object",
            "properties": {
                "slug": {"type": "string"},
            },
        },
    },
    handler=lambda args, **kw: brain_timeline(args.get("slug")),
    check_fn=_is_available,
)

registry.register(
    name="brain_stats",
    toolset="brain",
    schema={
        "name": "brain_stats",
        "description": "Get brain statistics (page count, link count, etc.).",
        "parameters": {"type": "object", "properties": {}},
    },
    handler=lambda args, **kw: brain_stats(),
    check_fn=_is_available,
)

registry.register(
    name="brain_list",
    toolset="brain",
    schema={
        "name": "brain_list",
        "description": "List brain pages, optionally filtered by tag.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 20},
                "tag": {"type": "string"},
            },
        },
    },
    handler=lambda args, **kw: brain_list(args.get("limit", 20), args.get("tag")),
    check_fn=_is_available,
)
