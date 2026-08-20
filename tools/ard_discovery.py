#!/usr/bin/env python3
"""
Agentic Resource Discovery (ARD) Tool Module

Implements the ARD open specification (by Microsoft, Google, GoDaddy, Hugging Face)
for runtime discovery of AI capabilities — skills, MCP servers, and agent Spaces.

Wraps the Hugging Face Discover Tool reference implementation:
    https://huggingface-hf-discover.hf.space/search

Provides three tools:
  - ard_search:     Search for skills, MCP servers, or Spaces
  - ard_install_mcp: Install a discovered MCP server into Hermes
  - ard_catalog:     Show available registries via well-known ai-catalog.json

The ARD protocol separates discovery from execution — agents find tools at runtime
instead of relying on pre-configuration. Federation is built in: a search through
one service can surface capabilities hosted elsewhere.
"""

import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy import httpx so cold starts don't pay for it
_httpx: Any = None


def _get_httpx() -> Any:
    """Lazy-import httpx with proxy support from environment."""
    global _httpx
    if _httpx is not None:
        return _httpx
    import httpx
    _httpx = httpx
    return _httpx


# ─── ARD Search ────────────────────────────────────────────────────────────

_HF_DISCOVER_URL = "https://huggingface-hf-discover.hf.space/search"

# Known registries (well-known ai-catalog.json endpoints)
_KNOWN_REGISTRIES = {
    "huggingface": {
        "name": "Hugging Face Discover",
        "catalog_url": "https://huggingface.co/.well-known/ai-catalog.json",
        "search_url": _HF_DISCOVER_URL,
    },
}

_RESOURCE_TYPES = {
    "skill": "application/ai-skill",
    "mcp-server": "application/mcp-server-card+json",
    "space": "application/vnd.huggingface.space+json",
}


def ard_search(
    query: str = "",
    kind: str = "all",
    page_size: int = 5,
    registry_url: Optional[str] = None,
) -> str:
    """Search for AI capabilities at runtime via the ARD protocol.

    Args:
        query: Natural-language description of what you need
               (e.g. "transcribe audio", "generate images", "fine tune a model")
        kind:  Type of resource to search for:
               "all" (default), "skill", "mcp-server", "space"
        page_size: Number of results (1-20, default 5)
        registry_url: Optional federated registry URL. When omitted,
                      uses the Hugging Face Discover registry.

    Returns:
        JSON with search results. Each result includes:
        - displayName, description, type, url
        - For MCP servers: mcpUrl (ready-to-use endpoint)
        - For skills: skill content URL
        - referrals: links to other federated registries
    """
    httpx = _get_httpx()
    search_url = registry_url or _HF_DISCOVER_URL

    # Build filter
    type_filter = []
    if kind in _RESOURCE_TYPES:
        type_filter = [_RESOURCE_TYPES[kind]]
    elif kind == "all":
        type_filter = list(_RESOURCE_TYPES.values())

    payload: Dict[str, Any] = {
        "query": {
            "text": query,
        },
        "pageSize": min(max(page_size, 1), 20),
    }
    if type_filter:
        payload["query"]["filter"] = {"type": type_filter}

    try:
        resp = httpx.post(
            search_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("ARD search failed: %s", e)
        return json.dumps({
            "error": f"ARD search failed: {e}. The HF Discover endpoint may be "
                     f"unreachable from this network. Try using a proxy or "
                     f"the browser tool to access {search_url} directly.",
            "query": query,
            "kind": kind,
        }, ensure_ascii=False)

    # Simplify results for the LLM
    simplified: List[Dict[str, Any]] = []
    for r in data.get("results", []):
        md = r.get("metadata", {})
        entry: Dict[str, Any] = {
            "identifier": r.get("identifier", ""),
            "displayName": r.get("displayName", ""),
            "type": r.get("type", ""),
            "description": r.get("description", ""),
            "url": r.get("url", ""),
            "score": r.get("score", 0),
            "source": r.get("source", ""),
            "tags": r.get("tags", []),
        }
        # Include MCP URL when available (so agents can connect directly)
        if "mcpUrl" in md:
            entry["mcpUrl"] = md["mcpUrl"]
        if "spaceId" in md:
            entry["spaceId"] = md["spaceId"]
            entry["appUrl"] = md.get("appUrl", "")
            entry["hubUrl"] = md.get("hubUrl", "")
        simplified.append(entry)

    result = {
        "query": query,
        "kind": kind,
        "results": simplified,
        "total_results": len(simplified),
        "referrals": data.get("referrals", []),
        "tip": (
            "Use ard_install_mcp to install a discovered MCP server. "
            "Use ard_catalog to explore available registries."
        ),
    }
    return json.dumps(result, ensure_ascii=False)


# ─── ARD Install MCP ──────────────────────────────────────────────────────

def ard_install_mcp(mcp_url: str, name: str) -> str:
    """Install a discovered MCP server into Hermes.

    Args:
        mcp_url: The MCP server endpoint URL (from ard_search results' mcpUrl field)
        name:    A name for the server (e.g. "whisper", "image-gen")

    Returns:
        JSON with installation status.
    """
    if not mcp_url or not name:
        return json.dumps({"error": "mcp_url and name are required"}, ensure_ascii=False)

    # Sanitize name
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_").lower()
    if not safe_name:
        return json.dumps({"error": f"Invalid name: {name}"}, ensure_ascii=False)

    try:
        result = subprocess.run(
            ["hermes", "mcp", "add", safe_name, "--url", mcp_url],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return json.dumps({
                "status": "installed",
                "name": safe_name,
                "mcp_url": mcp_url,
                "message": f"MCP server '{safe_name}' installed. Run '/reload-mcp' or restart to use it.",
                "stdout": result.stdout.strip(),
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "status": "failed",
                "name": safe_name,
                "mcp_url": mcp_url,
                "error": result.stderr.strip() or result.stdout.strip(),
            }, ensure_ascii=False)
    except FileNotFoundError:
        return json.dumps({
            "status": "failed",
            "error": "hermes CLI not found. Install manually: hermes mcp add NAME --url URL",
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "status": "failed",
            "error": str(e),
        }, ensure_ascii=False)


# ─── ARD Catalog ───────────────────────────────────────────────────────────

def ard_catalog(registry_url: Optional[str] = None) -> str:
    """Show available registries and their capabilities.

    Args:
        registry_url: Optional specific registry catalog URL.
                      When omitted, shows known registries.

    Returns:
        JSON with catalog information.
    """
    if registry_url:
        httpx = _get_httpx()
        try:
            resp = httpx.get(registry_url, timeout=15.0)
            resp.raise_for_status()
            data = resp.json()
            return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            return json.dumps({
                "error": f"Failed to fetch catalog: {e}",
                "registry_url": registry_url,
            }, ensure_ascii=False)

    # Return known registries
    return json.dumps({
        "registries": _KNOWN_REGISTRIES,
        "note": (
            "ARD is a federated discovery protocol. Each registry independently "
            "indexes skills, MCP servers, and agent Spaces. Search one and get "
            "referrals to others via the 'referrals' field in search results."
        ),
    }, ensure_ascii=False)


# ─── Requirements Check ───────────────────────────────────────────────────

def check_ard_requirements() -> bool:
    """Check if httpx is available for ARD API calls."""
    try:
        import httpx  # noqa: F401
        return True
    except ImportError:
        return False


# ─── Schemas ──────────────────────────────────────────────────────────────

ARD_SEARCH_SCHEMA = {
    "name": "ard_search",
    "description": (
        "Search for AI capabilities at runtime using the ARD (Agentic Resource Discovery) "
        "protocol. Find skills, MCP servers, and AI agent Spaces from Hugging Face and "
        "federated registries — no pre-configuration needed.\n\n"
        "Use this when:\n"
        "- You need a capability you don't currently have (e.g. 'transcribe audio')\n"
        "- The user asks to find a tool, skill, or integration\n"
        "- You want to extend your capabilities dynamically\n\n"
        "Kinds:\n"
        "- 'skill': AI skills (agents.md with instructions)\n"
        "- 'mcp-server': MCP servers you can connect to with ard_install_mcp\n"
        "- 'space': Full Hugging Face Spaces\n"
        "- 'all': Search everything (default)\n\n"
        "After finding an MCP server, use ard_install_mcp to add it to Hermes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language description of what capability you need (e.g. 'transcribe audio', 'generate an image', 'fine tune a model')",
            },
            "kind": {
                "type": "string",
                "enum": ["all", "skill", "mcp-server", "space"],
                "description": "Type of resource to search for. Default: 'all'.",
            },
            "page_size": {
                "type": "integer",
                "description": "Number of results to return (1-20, default 5).",
                "minimum": 1,
                "maximum": 20,
            },
            "registry_url": {
                "type": "string",
                "description": "Optional URL of a federated ARD registry. When omitted, uses Hugging Face Discover.",
            },
        },
        "required": ["query"],
    },
}

ARD_INSTALL_MCP_SCHEMA = {
    "name": "ard_install_mcp",
    "description": (
        "Install a discovered MCP server into Hermes. Use after ard_search finds "
        "an MCP server with an mcpUrl field. The server becomes available as a "
        "native MCP tool after /reload-mcp or a session restart."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "mcp_url": {
                "type": "string",
                "description": "The MCP server endpoint URL from ard_search results (mcpUrl field).",
            },
            "name": {
                "type": "string",
                "description": "A short name for the server (e.g. 'whisper', 'image-gen'). Use lowercase alphanumeric with hyphens.",
            },
        },
        "required": ["mcp_url", "name"],
    },
}

ARD_CATALOG_SCHEMA = {
    "name": "ard_catalog",
    "description": (
        "Show available ARD registries. Use this to discover which federated "
        "registries are available for searching AI capabilities."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "registry_url": {
                "type": "string",
                "description": "Optional specific registry catalog URL to fetch.",
            },
        },
    },
}


# ─── Registry ─────────────────────────────────────────────────────────────

from tools.registry import registry

registry.register(
    name="ard_search",
    toolset="ard",
    schema=ARD_SEARCH_SCHEMA,
    handler=lambda args, **kw: ard_search(
        query=args.get("query", ""),
        kind=args.get("kind", "all"),
        page_size=args.get("page_size", 5),
        registry_url=args.get("registry_url"),
    ),
    check_fn=check_ard_requirements,
    emoji="🔍",
    description="Search for AI skills, MCP servers, and Spaces via ARD protocol",
)

registry.register(
    name="ard_install_mcp",
    toolset="ard",
    schema=ARD_INSTALL_MCP_SCHEMA,
    handler=lambda args, **kw: ard_install_mcp(
        mcp_url=args.get("mcp_url", ""),
        name=args.get("name", ""),
    ),
    check_fn=check_ard_requirements,
    emoji="📦",
    description="Install a discovered MCP server into Hermes",
)

registry.register(
    name="ard_catalog",
    toolset="ard",
    schema=ARD_CATALOG_SCHEMA,
    handler=lambda args, **kw: ard_catalog(
        registry_url=args.get("registry_url"),
    ),
    check_fn=check_ard_requirements,
    emoji="📋",
    description="Show available ARD registries",
)
