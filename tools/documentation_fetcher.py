#!/usr/bin/env python3
"""
Documentation Fetcher Tool - Fetch documentation from llms.txt and Context7 API.

Provides documentation retrieval for libraries and frameworks:
  - llms.txt: Parse project-local llms.txt file (standard for LLM-friendly docs)
  - Context7 API: Fetch up-to-date documentation from context7.com

Features:
  - Caching: Popular docs cached in ~/.hermes/cache/docs/
  - Fallback: Try llms.txt first, then Context7 API
  - Search: Query-based search within documentation

llms.txt spec: https://llmstxt.org/
Context7: https://context7.com / https://github.com/upstash/context7
"""

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

# ─── Context7 MCP Config Generator ────────────────────────────────────────────

def generate_context7_mcp_config() -> dict:
    """Generate the Context7 MCP server configuration dict.

    Returns a dict suitable for merging into the ``mcp_servers`` key of
    ~/.hermes/config.yaml.  Users can call this programmatically or copy
    the returned config into their YAML.

    Example usage::

        from tools.documentation_fetcher import generate_context7_mcp_config
        config = generate_context7_mcp_config()
        # Merge into existing config:
        # existing_config.setdefault("mcp_servers", {}).update(config)

    Returns:
        dict: Context7 MCP server configuration with stdio transport settings.
    """
    return {
        "context7": {
            "command": "npx",
            "args": ["-y", "@upstash/context7-mcp"],
            "timeout": 120,
            "connect_timeout": 60,
        },
    }


def generate_context7_mcp_yaml() -> str:
    """Generate the Context7 MCP server configuration as a YAML string.

    Returns a ready-to-paste YAML block for ~/.hermes/config.yaml.

    Returns:
        str: YAML-formatted Context7 MCP server configuration.
    """
    return """\
# Context7 MCP Server — real-time, version-specific code documentation
# https://github.com/upstash/context7
mcp_servers:
  context7:
    command: "npx"
    args: ["-y", "@upstash/context7-mcp"]
    timeout: 120
    connect_timeout: 60
"""


# ─── Cache Configuration ────────────────────────────────────────────────────────

DEFAULT_CACHE_TTL_SECONDS = 86400  # 24 hours
MAX_CACHE_SIZE_MB = 100
MAX_DOC_SIZE_CHARS = 100_000  # Cap returned documentation size


def _get_docs_cache_dir() -> Path:
    """Return the documentation cache directory under HERMES_HOME."""
    cache_dir = get_hermes_home() / "cache" / "docs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _cache_key(library: str, version: Optional[str] = None, query: Optional[str] = None) -> str:
    """Generate a cache key from library, version, and optional query."""
    parts = [library.lower().strip()]
    if version:
        parts.append(version.lower().strip())
    if query:
        # Hash query to keep key length manageable
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        parts.append(query_hash)
    return "_".join(parts)


def _cache_path(key: str) -> Path:
    """Return the cache file path for a given key."""
    return _get_docs_cache_dir() / f"{key}.json"


def _read_cache(key: str) -> Optional[Dict[str, Any]]:
    """Read cached documentation if available and not expired."""
    cache_file = _cache_path(key)
    if not cache_file.exists():
        return None
    
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        timestamp = data.get("timestamp", 0)
        ttl = data.get("ttl", DEFAULT_CACHE_TTL_SECONDS)
        
        if time.time() - timestamp > ttl:
            # Cache expired
            cache_file.unlink(missing_ok=True)
            return None
        
        return data.get("content")
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read cache %s: %s", cache_file, e)
        return None


def _write_cache(key: str, content: Dict[str, Any], ttl: int = DEFAULT_CACHE_TTL_SECONDS) -> None:
    """Write documentation to cache with timestamp."""
    cache_file = _cache_path(key)
    
    try:
        data = {
            "timestamp": time.time(),
            "ttl": ttl,
            "content": content,
        }
        cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        
        # Prune cache if too large
        _prune_cache_if_needed()
    except OSError as e:
        logger.warning("Failed to write cache %s: %s", cache_file, e)


def _prune_cache_if_needed() -> None:
    """Remove oldest cache files if total size exceeds limit."""
    cache_dir = _get_docs_cache_dir()
    
    try:
        total_size = sum(f.stat().st_size for f in cache_dir.glob("*.json"))
        max_size = MAX_CACHE_SIZE_MB * 1024 * 1024
        
        if total_size <= max_size:
            return
        
        # Sort by modification time, oldest first
        files = sorted(cache_dir.glob("*.json"), key=lambda f: f.stat().st_mtime)
        
        for f in files:
            if total_size <= max_size * 0.8:  # Stop at 80% of limit
                break
            total_size -= f.stat().st_size
            f.unlink(missing_ok=True)
            logger.debug("Pruned cache file: %s", f)
    except OSError as e:
        logger.warning("Cache pruning failed: %s", e)


# ─── llms.txt Parser ─────────────────────────────────────────────────────────────

def _find_llms_txt(project_root: Optional[str] = None) -> Optional[Path]:
    """Find llms.txt file in project root or current directory."""
    search_paths = []
    
    if project_root:
        search_paths.append(Path(project_root) / "llms.txt")
    
    # Also search current working directory
    cwd = Path.cwd()
    search_paths.extend([
        cwd / "llms.txt",
        cwd / ".llms.txt",
        cwd.parent / "llms.txt",
    ])
    
    for path in search_paths:
        if path.exists() and path.is_file():
            return path
    
    return None


def _parse_llms_txt(path: Path | str) -> Dict[str, Any]:
    """Parse llms.txt file and extract structured documentation.
    
    llms.txt format (https://llmstxt.org/):
    - Uses Markdown-style format
    - First line is title: # LibraryName
    - Optional description after title
    - Sections marked with ##
    - URLs/links for external references
    
    Returns dict with title, description, sections, and links.
    """
    path = Path(path)
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        return {"error": f"Failed to read llms.txt: {e}"}
    
    result = {
        "source": "llms.txt",
        "path": str(path),
        "title": "",
        "description": "",
        "sections": [],
        "links": [],
        "raw_content": content[:MAX_DOC_SIZE_CHARS],
    }
    
    lines = content.splitlines()
    
    # Parse title (first # line)
    for line in lines:
        if line.startswith("# "):
            result["title"] = line[2:].strip()
            break
    
    # Parse sections (## headings)
    current_section: Dict[str, Any] = {}
    section_content: List[str] = []
    
    for line in lines:
        if line.startswith("## "):
            # Save previous section
            if current_section and section_content:
                current_section["content"] = "\n".join(section_content).strip()
                result["sections"].append(current_section)
            
            # Start new section
            current_section = {"title": line[3:].strip()}
            section_content = []
        elif current_section:
            section_content.append(line)
        
        # Extract links
        link_matches = re.findall(r'https?://[^\s<>"\)]+', line)
        for link in link_matches:
            if link not in result["links"]:
                result["links"].append(link)
    
    # Save last section
    if current_section and section_content:
        current_section["content"] = "\n".join(section_content).strip()
        result["sections"].append(current_section)
    
    # Extract description (text after title, before first section)
    if result["sections"]:
        first_section_idx = content.find("## ")
        if first_section_idx > 0:
            title_end = content.find("\n", content.find("# "))
            desc_text = content[title_end:first_section_idx].strip()
            if desc_text:
                result["description"] = desc_text
    
    return result


def _search_in_llms_content(parsed: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Search for query terms in parsed llms.txt content."""
    if "error" in parsed:
        return parsed
    
    query_lower = query.lower()
    matches = []
    
    # Search in sections
    for section in parsed.get("sections", []):
        title = section.get("title", "").lower()
        content = section.get("content", "").lower()
        
        if query_lower in title or query_lower in content:
            matches.append({
                "section": section.get("title", ""),
                "relevance": "high" if query_lower in title else "medium",
                "snippet": _extract_snippet(section.get("content", ""), query_lower),
            })
    
    # Search in raw content if no section matches
    if not matches:
        raw = parsed.get("raw_content", "").lower()
        if query_lower in raw:
            matches.append({
                "section": "general",
                "relevance": "low",
                "snippet": _extract_snippet(parsed.get("raw_content", ""), query_lower),
            })
    
    result = {
        "source": "llms.txt",
        "path": parsed.get("path", ""),
        "title": parsed.get("title", ""),
        "query": query,
        "matches": matches,
        "total_sections": len(parsed.get("sections", [])),
    }
    
    return result


def _extract_snippet(content: str, query: str, max_len: int = 500) -> str:
    """Extract a relevant snippet around the query match."""
    content_lower = content.lower()
    idx = content_lower.find(query)
    
    if idx == -1:
        return content[:max_len].strip() + "..." if len(content) > max_len else content.strip()
    
    # Extract context around the match
    start = max(0, idx - 100)
    end = min(len(content), idx + len(query) + 200)
    
    snippet = content[start:end].strip()
    if start > 0:
        snippet = "... " + snippet
    if end < len(content):
        snippet = snippet + " ..."
    
    return snippet


# ─── Context7 API Client ────────────────────────────────────────────────────────

CONTEXT7_API_BASE = "https://context7.com/api"
CONTEXT7_SEARCH_URL = f"{CONTEXT7_API_BASE}/search"
CONTEXT7_FETCH_URL = f"{CONTEXT7_API_BASE}/fetch"

# Alternative endpoint from GitHub repo
CONTEXT7_GITHUB_API = "https://api.context7.com"


def _get_context7_client() -> httpx.Client:
    """Get httpx client for Context7 API."""
    return httpx.Client(
        timeout=30.0,
        headers={
            "Accept": "application/json",
            "User-Agent": "Hermes-Agent/1.0",
        },
    )


def _search_context7(library: str, version: Optional[str] = None) -> Dict[str, Any]:
    """Search Context7 API for library documentation."""
    client = _get_context7_client()
    
    params = {
        "q": library,
    }
    if version:
        params["version"] = version
    
    try:
        # Try primary endpoint
        response = client.get(CONTEXT7_SEARCH_URL, params=params)
        
        if response.status_code == 404:
            # Try alternative endpoint
            response = client.get(f"{CONTEXT7_GITHUB_API}/search", params=params)
        
        response.raise_for_status()
        data = response.json()
        
        return {
            "source": "context7",
            "library": library,
            "version": version,
            "results": data.get("results", data.get("libraries", [])),
            "success": True,
        }
    except httpx.HTTPError as e:
        logger.warning("Context7 search failed for %s: %s", library, e)
        return {
            "source": "context7",
            "library": library,
            "error": f"Context7 API error: {type(e).__name__}",
            "success": False,
        }
    finally:
        client.close()


def _fetch_context7_docs(library_id: str, query: Optional[str] = None) -> Dict[str, Any]:
    """Fetch full documentation from Context7 for a library ID."""
    client = _get_context7_client()
    
    params = {"id": library_id}
    if query:
        params["q"] = query
    
    try:
        # Try primary endpoint
        response = client.get(CONTEXT7_FETCH_URL, params=params)
        
        if response.status_code == 404:
            # Try alternative endpoint structure
            response = client.get(f"{CONTEXT7_GITHUB_API}/libraries/{library_id}", params=params)
        
        response.raise_for_status()
        data = response.json()
        
        # Extract documentation content
        docs_content = data.get("content", data.get("documentation", data.get("docs", "")))
        
        if isinstance(docs_content, list):
            # Multiple sections/pages
            docs_content = "\n\n".join(
                item.get("content", item.get("text", str(item)))
                for item in docs_content
            )
        
        # Cap content size
        if len(docs_content) > MAX_DOC_SIZE_CHARS:
            docs_content = docs_content[:MAX_DOC_SIZE_CHARS] + "\n... [truncated]"
        
        return {
            "source": "context7",
            "library_id": library_id,
            "query": query,
            "content": docs_content,
            "title": data.get("title", data.get("name", library_id)),
            "version": data.get("version", ""),
            "url": data.get("url", ""),
            "success": True,
        }
    except httpx.HTTPError as e:
        logger.warning("Context7 fetch failed for %s: %s", library_id, e)
        return {
            "source": "context7",
            "library_id": library_id,
            "error": f"Context7 API error: {type(e).__name__}",
            "success": False,
        }
    finally:
        client.close()


# ─── Main Tool Handler ──────────────────────────────────────────────────────────

def fetch_docs_handler(args: Dict[str, Any], **kwargs) -> str:
    """Main handler for fetch_docs tool.
    
    Args:
        query: Search query (optional)
        library: Library name, e.g., "react", "vue", "next.js"
        version: Version number (optional)
        project_root: Path to project root for llms.txt lookup (optional)
        source: Force source: "llms_txt", "context7", or "auto" (default)
    
    Returns JSON with documentation content.
    """
    query = args.get("query", "").strip()
    library = args.get("library", "").strip().lower()
    version = args.get("version", "").strip()
    project_root = args.get("project_root", "")
    source = args.get("source", "auto").strip().lower()
    
    if not library and not query:
        return tool_error(
            "Either 'library' or 'query' parameter is required.",
            usage="fetch_docs(library='react', query='hooks', version='18')"
        )
    
    # Determine effective library from query if not specified
    if not library and query:
        # Try to detect library from query (e.g., "react hooks" -> "react")
        common_libs = [
            "react", "vue", "angular", "svelte", "nextjs", "next.js",
            "python", "django", "flask", "fastapi", "numpy", "pandas",
            "typescript", "javascript", "nodejs", "express",
            "tailwindcss", "bootstrap", "sass",
            "rust", "go", "golang",
        ]
        query_lower = query.lower()
        for lib in common_libs:
            if lib in query_lower:
                library = lib
                break
    
    results: Dict[str, Any] = {
        "query": query,
        "library": library,
        "version": version,
        "sources_checked": [],
        "documentation": None,
    }
    
    # Step 1: Check llms.txt if source allows it
    if source in ("auto", "llms_txt"):
        llms_path = _find_llms_txt(project_root)
        if llms_path:
            results["sources_checked"].append("llms.txt")
            
            parsed = _parse_llms_txt(llms_path)
            
            if query:
                search_result = _search_in_llms_content(parsed, query)
                if search_result.get("matches"):
                    results["documentation"] = search_result
                    results["source_used"] = "llms.txt"
                    return tool_result(results)
            else:
                # No query - return full parsed content
                results["documentation"] = parsed
                results["source_used"] = "llms.txt"
                return tool_result(results)
    
    # Step 2: Check Context7 API if source allows it
    if source in ("auto", "context7") and library:
        results["sources_checked"].append("context7")
        
        # Check cache first
        cache_key = _cache_key(library, version, query)
        cached = _read_cache(cache_key)
        
        if cached:
            results["documentation"] = cached
            results["source_used"] = "context7 (cached)"
            return tool_result(results)
        
        # Search Context7 for library
        search_result = _search_context7(library, version)
        
        if search_result.get("success") and search_result.get("results"):
            # Get first matching library ID
            results_list = search_result.get("results", [])
            if results_list:
                first_result = results_list[0]
                lib_id = first_result.get("id", first_result.get("library_id", ""))
                
                if lib_id:
                    # Fetch full documentation
                    fetch_result = _fetch_context7_docs(lib_id, query)
                    
                    if fetch_result.get("success"):
                        # Cache the result
                        _write_cache(cache_key, fetch_result)
                        
                        results["documentation"] = fetch_result
                        results["source_used"] = "context7"
                        return tool_result(results)
                    else:
                        results["context7_error"] = fetch_result.get("error")
        else:
            results["context7_error"] = search_result.get("error")
    
    # No documentation found
    if results["sources_checked"]:
        results["error"] = f"No documentation found for '{library or query}' in checked sources."
    else:
        results["error"] = "No documentation sources available. Check llms.txt or Context7 API."
    
    results["success"] = False
    return tool_result(results)


# ─── Tool Schema ────────────────────────────────────────────────────────────────

FETCH_DOCS_SCHEMA = {
    "name": "fetch_docs",
    "description": (
        "Fetch documentation for libraries and frameworks from llms.txt or Context7 API. "
        "Use this tool when you need up-to-date documentation for programming libraries.\n\n"
        "Sources (checked in order):\n"
        "1. llms.txt: Project-local documentation file (https://llmstxt.org/)\n"
        "2. Context7 API: External documentation service (https://context7.com)\n\n"
        "Features:\n"
        "- Caching: Popular docs cached for 24 hours\n"
        "- Search: Query-based search within docs\n"
        "- Version-specific: Fetch docs for specific library versions\n\n"
        "Examples:\n"
        "- fetch_docs(library='react') - Get React documentation\n"
        "- fetch_docs(library='vue', version='3') - Vue 3 docs\n"
        "- fetch_docs(library='nextjs', query='routing') - Search Next.js routing docs\n"
        "- fetch_docs(query='python asyncio') - Detect library and search\n\n"
        "Tip: For project-specific docs, ensure llms.txt exists in project root."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query to find specific documentation sections. "
                    "Optional - if omitted, returns general library docs."
                ),
            },
            "library": {
                "type": "string",
                "description": (
                    "Library or framework name, e.g., 'react', 'vue', 'next.js', 'python'. "
                    "Required unless query contains a recognizable library name."
                ),
            },
            "version": {
                "type": "string",
                "description": (
                    "Library version number, e.g., '18', '3.4', '2.0'. "
                    "Optional - defaults to latest/stable."
                ),
            },
            "project_root": {
                "type": "string",
                "description": (
                    "Path to project root for llms.txt lookup. "
                    "Optional - defaults to current working directory."
                ),
            },
            "source": {
                "type": "string",
                "enum": ["auto", "llms_txt", "context7"],
                "description": (
                    "Documentation source to use. "
                    "'auto' checks llms.txt first, then Context7. "
                    "'llms_txt' only uses local llms.txt file. "
                    "'context7' only uses Context7 API."
                ),
            },
        },
        "required": [],  # Either library or query required, enforced in handler
    },
}


# ─── Availability Check ──────────────────────────────────────────────────────────

def check_docs_fetcher_requirements() -> bool:
    """Check if documentation fetcher is available.
    
    Always available - llms.txt is optional, Context7 has no auth requirement.
    """
    return True


# ─── Registry Registration ───────────────────────────────────────────────────────

registry.register(
    name="fetch_docs",
    toolset="documentation",
    schema=FETCH_DOCS_SCHEMA,
    handler=fetch_docs_handler,
    check_fn=check_docs_fetcher_requirements,
    requires_env=[],  # No required environment variables
    is_async=False,
    description="Fetch library documentation from llms.txt and Context7 API",
    emoji="📚",
)