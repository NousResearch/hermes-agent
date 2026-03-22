#!/usr/bin/env python3
"""
Standalone Web Tools Module

This module provides generic web tools that work with multiple backend providers.
Backend is selected during ``hermes tools`` setup (web.backend in config.yaml).

Available tools:
- web_search_tool: Search the web for information
- web_extract_tool: Extract content from specific web pages
- web_crawl_tool: Crawl websites with specific instructions (Firecrawl only)

Page Storage tools (persistent storage for scraped pages):
- web_page_search_tool: Full-text search over saved pages
- web_page_list_tool: List all saved pages with optional filtering
- web_page_get_tool: Retrieve a specific saved page by URL or ID
- web_page_delete_tool: Delete a page from storage
- web_page_stats_tool: Get storage statistics

Backend compatibility:
- Firecrawl: https://docs.firecrawl.dev/introduction (search, extract, crawl)
- Parallel: https://docs.parallel.ai (search, extract)

LLM Processing:
- Uses OpenRouter API with Gemini 3 Flash Preview for intelligent content extraction
- Extracts key excerpts and creates markdown summaries to reduce token usage

Page Storage:
- SQLite database stored at ~/.hermes/data/web_pages.db
- Full-text search using SQLite FTS5
- Automatic storage of pages extracted via web_extract_tool and web_crawl_tool
- Set WEB_PAGES_DB_PATH env var for custom storage location

Debug Mode:
- Set WEB_TOOLS_DEBUG=true to enable detailed logging
- Creates web_tools_debug_UUID.json in ./logs directory
- Captures all tool calls, results, and compression metrics

Usage:
    from web_tools import web_search_tool, web_extract_tool, web_crawl_tool
    from web_tools import web_page_search_tool, web_page_list_tool
    
    # Search the web
    results = web_search_tool("Python machine learning libraries", limit=3)
    
    # Extract content from URLs  
    content = web_extract_tool(["https://example.com"], format="markdown")
    
    # Crawl a website
    crawl_data = web_crawl_tool("example.com", "Find contact information")
    
    # Search saved pages
    saved = web_page_search_tool("machine learning")
    
    # List saved pages
    pages = web_page_list_tool(limit=10)
"""

# ─── Page Storage System ─────────────────────────────────────────────────────
"""
Persistent storage for scraped/extracted web pages with search capabilities.

Features:
- SQLite-based storage for efficient full-text search
- Automatic page storage from web_extract and web_crawl operations
- Search over saved page content
- List and manage saved pages
"""

import json
import sqlite3
import hashlib
import threading
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

# Storage directory - use ~/.hermes/data for persistence
_STORAGE_DIR = Path.home() / ".hermes" / "data"
_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
_PAGES_DB = _STORAGE_DIR / "web_pages.db"

# Thread-local storage for database connections
_local = threading.local()


def _get_db_path() -> Path:
    """Get the database path, checking for custom location via env var."""
    custom_path = os.getenv("WEB_PAGES_DB_PATH", "").strip()
    if custom_path:
        return Path(custom_path)
    return _PAGES_DB


@contextmanager
def _get_db_connection():
    """Get a thread-local SQLite connection with proper configuration."""
    if not hasattr(_local, 'connection') or _local.connection is None:
        db_path = _get_db_path()
        _local.connection = sqlite3.connect(str(db_path), check_same_thread=False)
        _local.connection.row_factory = sqlite3.Row
        # Enable FTS5 for full-text search
        _local.connection.execute("PRAGMA journal_mode=WAL")
        _local.connection.execute("PRAGMA synchronous=NORMAL")
        _init_db(_local.connection)
    try:
        yield _local.connection
    except Exception:
        if _local.connection:
            _local.connection.rollback()
        raise


def _init_db(conn: sqlite3.Connection) -> None:
    """Initialize the database schema with FTS support."""
    # Main pages table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL UNIQUE,
            url_hash TEXT NOT NULL,
            title TEXT,
            content TEXT,
            raw_content TEXT,
            metadata TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            source_tool TEXT,
            content_length INTEGER DEFAULT 0
        )
    """)
    
    # Full-text search virtual table
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
            url,
            title,
            content,
            content='pages',
            content_rowid='id'
        )
    """)
    
    # Triggers to keep FTS in sync
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS pages_ai AFTER INSERT ON pages BEGIN
            INSERT INTO pages_fts(rowid, url, title, content)
            VALUES (new.id, new.url, new.title, new.content);
        END
    """)
    
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS pages_ad AFTER DELETE ON pages BEGIN
            INSERT INTO pages_fts(pages_fts, rowid, url, title, content)
            VALUES('delete', old.id, old.url, old.title, old.content);
        END
    """)
    
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS pages_au AFTER UPDATE ON pages BEGIN
            INSERT INTO pages_fts(pages_fts, rowid, url, title, content)
            VALUES('delete', old.id, old.url, old.title, old.content);
            INSERT INTO pages_fts(rowid, url, title, content)
            VALUES (new.id, new.url, new.title, new.content);
        END
    """)
    
    conn.commit()


def _hash_url(url: str) -> str:
    """Create a hash of the URL for quick lookups."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def store_page(
    url: str,
    title: str = "",
    content: str = "",
    raw_content: str = "",
    metadata: dict = None,
    source_tool: str = "unknown"
) -> dict:
    """
    Store a web page in persistent storage.
    
    Args:
        url: The page URL (unique identifier)
        title: Page title
        content: Processed/summarized content
        raw_content: Original content
        metadata: Additional metadata dict
        source_tool: Tool that extracted this page (web_extract, web_crawl)
    
    Returns:
        dict with success status and page id
    """
    if not url:
        return {"success": False, "error": "URL is required"}
    
    try:
        now = datetime.utcnow().isoformat()
        url_hash = _hash_url(url)
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        content_len = len(content) if content else 0
        
        with _get_db_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO pages (url, url_hash, title, content, raw_content, metadata, 
                                   created_at, updated_at, source_tool, content_length)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    title = excluded.title,
                    content = excluded.content,
                    raw_content = excluded.raw_content,
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at,
                    source_tool = excluded.source_tool,
                    content_length = excluded.content_length
            """, (url, url_hash, title, content, raw_content, metadata_json,
                  now, now, source_tool, content_len))
            
            conn.commit()
            page_id = cursor.lastrowid
            
            logger.info("Stored page: %s (id=%d, %d chars)", url, page_id, content_len)
            
            return {
                "success": True,
                "page_id": page_id,
                "url": url,
                "content_length": content_len,
                "stored_at": now
            }
            
    except Exception as e:
        logger.error("Failed to store page %s: %s", url, e)
        return {"success": False, "error": str(e)}


def store_pages_from_results(results: list, source_tool: str = "unknown") -> list:
    """
    Store multiple pages from web_extract or web_crawl results.
    
    Args:
        results: List of result dicts with url, title, content, etc.
        source_tool: Tool that extracted these pages
    
    Returns:
        List of storage results
    """
    stored = []
    for result in results:
        if result.get("error") or result.get("blocked_by_policy"):
            continue  # Skip failed extractions
        
        store_result = store_page(
            url=result.get("url", ""),
            title=result.get("title", ""),
            content=result.get("content", ""),
            raw_content=result.get("raw_content", ""),
            metadata=result.get("metadata"),
            source_tool=source_tool
        )
        stored.append(store_result)
    
    return stored


def search_pages(query: str, limit: int = 10, include_content: bool = True) -> str:
    """
    Search over saved web pages using full-text search.
    
    Args:
        query: Search query (supports SQLite FTS5 syntax)
        limit: Maximum number of results
        include_content: Whether to include full content in results
    
    Returns:
        JSON string with search results
    """
    if not query or not query.strip():
        return json.dumps({"error": "Search query is required", "results": []})
    
    try:
        # Escape special FTS characters and build query
        search_query = query.strip().replace('"', '""')
        
        with _get_db_connection() as conn:
            # Use FTS5 for full-text search with ranking
            if include_content:
                columns = "p.id, p.url, p.title, p.content, p.created_at, p.updated_at, p.source_tool, p.content_length"
            else:
                columns = "p.id, p.url, p.title, p.created_at, p.updated_at, p.source_tool, p.content_length"
            
            cursor = conn.execute(f"""
                SELECT {columns}, pages_fts.rank
                FROM pages p
                JOIN pages_fts ON p.id = pages_fts.rowid
                WHERE pages_fts MATCH ?
                ORDER BY pages_fts.rank
                LIMIT ?
            """, (search_query, limit))
            
            results = []
            for row in cursor.fetchall():
                result = {
                    "id": row["id"],
                    "url": row["url"],
                    "title": row["title"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "source_tool": row["source_tool"],
                    "content_length": row["content_length"],
                    "relevance_score": -row["rank"] if row["rank"] else 0
                }
                if include_content:
                    result["content"] = row["content"]
                results.append(result)
            
            return json.dumps({
                "success": True,
                "query": query,
                "results_count": len(results),
                "results": results
            }, indent=2, ensure_ascii=False)
            
    except Exception as e:
        logger.error("Search failed: %s", e)
        return json.dumps({"error": f"Search failed: {str(e)}", "results": []})


def list_pages(
    limit: int = 50,
    offset: int = 0,
    source_tool: str = None,
    min_length: int = None,
    sort_by: str = "updated_at"
) -> str:
    """
    List saved web pages with optional filtering.
    
    Args:
        limit: Maximum number of results (default 50)
        offset: Offset for pagination
        source_tool: Filter by source tool (web_extract, web_crawl)
        min_length: Minimum content length filter
        sort_by: Sort field (updated_at, created_at, content_length, title)
    
    Returns:
        JSON string with page list
    """
    try:
        valid_sort_fields = {"updated_at", "created_at", "content_length", "title", "url"}
        if sort_by not in valid_sort_fields:
            sort_by = "updated_at"
        
        with _get_db_connection() as conn:
            # Build query with filters
            sql = """
                SELECT id, url, title, created_at, updated_at, source_tool, content_length
                FROM pages
            """
            params = []
            conditions = []
            
            if source_tool:
                conditions.append("source_tool = ?")
                params.append(source_tool)
            
            if min_length is not None:
                conditions.append("content_length >= ?")
                params.append(min_length)
            
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            
            sql += f" ORDER BY {sort_by} DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor = conn.execute(sql, params)
            
            # Get total count for pagination
            count_sql = "SELECT COUNT(*) as total FROM pages"
            if conditions:
                count_sql += " WHERE " + " AND ".join(conditions[:-2] if not (limit or offset) else conditions)
            count_params = params[:-2] if params else []
            total = conn.execute(count_sql, count_params).fetchone()["total"]
            
            pages = []
            for row in cursor.fetchall():
                pages.append({
                    "id": row["id"],
                    "url": row["url"],
                    "title": row["title"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "source_tool": row["source_tool"],
                    "content_length": row["content_length"]
                })
            
            return json.dumps({
                "success": True,
                "total": total,
                "limit": limit,
                "offset": offset,
                "pages_count": len(pages),
                "pages": pages
            }, indent=2, ensure_ascii=False)
            
    except Exception as e:
        logger.error("List pages failed: %s", e)
        return json.dumps({"error": f"Failed to list pages: {str(e)}", "pages": []})


def get_page(url: str = None, page_id: int = None) -> str:
    """
    Retrieve a specific page by URL or ID.
    
    Args:
        url: Page URL
        page_id: Page ID
    
    Returns:
        JSON string with page data
    """
    if not url and not page_id:
        return json.dumps({"error": "Either url or page_id is required"})
    
    try:
        with _get_db_connection() as conn:
            if page_id:
                cursor = conn.execute("""
                    SELECT id, url, title, content, raw_content, metadata, 
                           created_at, updated_at, source_tool, content_length
                    FROM pages WHERE id = ?
                """, (page_id,))
            else:
                cursor = conn.execute("""
                    SELECT id, url, title, content, raw_content, metadata,
                           created_at, updated_at, source_tool, content_length
                    FROM pages WHERE url = ?
                """, (url,))
            
            row = cursor.fetchone()
            if not row:
                return json.dumps({"error": "Page not found", "success": False})
            
            return json.dumps({
                "success": True,
                "page": {
                    "id": row["id"],
                    "url": row["url"],
                    "title": row["title"],
                    "content": row["content"],
                    "raw_content": row["raw_content"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "source_tool": row["source_tool"],
                    "content_length": row["content_length"]
                }
            }, indent=2, ensure_ascii=False)
            
    except Exception as e:
        logger.error("Get page failed: %s", e)
        return json.dumps({"error": f"Failed to get page: {str(e)}"})


def delete_page(url: str = None, page_id: int = None) -> str:
    """
    Delete a page from storage.
    
    Args:
        url: Page URL
        page_id: Page ID
    
    Returns:
        JSON string with result
    """
    if not url and not page_id:
        return json.dumps({"error": "Either url or page_id is required"})
    
    try:
        with _get_db_connection() as conn:
            if page_id:
                cursor = conn.execute("DELETE FROM pages WHERE id = ?", (page_id,))
            else:
                cursor = conn.execute("DELETE FROM pages WHERE url = ?", (url,))
            
            conn.commit()
            
            if cursor.rowcount == 0:
                return json.dumps({"error": "Page not found", "success": False})
            
            return json.dumps({
                "success": True,
                "deleted": cursor.rowcount,
                "url": url,
                "page_id": page_id
            })
            
    except Exception as e:
        logger.error("Delete page failed: %s", e)
        return json.dumps({"error": f"Failed to delete page: {str(e)}"})


def get_storage_stats() -> dict:
    """Get statistics about the page storage."""
    try:
        with _get_db_connection() as conn:
            stats = conn.execute("""
                SELECT 
                    COUNT(*) as total_pages,
                    SUM(content_length) as total_bytes,
                    AVG(content_length) as avg_content_length,
                    MAX(updated_at) as last_updated
                FROM pages
            """).fetchone()
            
            by_tool = conn.execute("""
                SELECT source_tool, COUNT(*) as count
                FROM pages
                GROUP BY source_tool
            """).fetchall()
            
            return {
                "success": True,
                "total_pages": stats["total_pages"] or 0,
                "total_bytes": stats["total_bytes"] or 0,
                "total_size_mb": round((stats["total_bytes"] or 0) / 1024 / 1024, 2),
                "avg_content_length": round(stats["avg_content_length"] or 0, 0),
                "last_updated": stats["last_updated"],
                "by_source_tool": {row["source_tool"]: row["count"] for row in by_tool},
                "storage_path": str(_get_db_path())
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


# Public tool functions (wrappers for registry)
def web_page_search_tool(query: str, limit: int = 10) -> str:
    """Search over saved web pages using full-text search."""
    return search_pages(query, limit=limit, include_content=True)


def web_page_list_tool(limit: int = 50, offset: int = 0, source: str = None) -> str:
    """List saved web pages with optional filtering."""
    return list_pages(limit=limit, offset=offset, source_tool=source)


def web_page_get_tool(url: str = None, page_id: int = None) -> str:
    """Retrieve a specific saved page by URL or ID."""
    return get_page(url=url, page_id=page_id)


def web_page_delete_tool(url: str = None, page_id: int = None) -> str:
    """Delete a page from storage."""
    return delete_page(url=url, page_id=page_id)


def web_page_stats_tool() -> str:
    """Get statistics about saved pages."""
    return json.dumps(get_storage_stats(), indent=2)


import logging
import os
import re
import asyncio
from typing import List, Dict, Any, Optional
import httpx
from firecrawl import Firecrawl
from agent.auxiliary_client import async_call_llm
from tools.debug_helpers import DebugSession
from tools.website_policy import check_website_access

logger = logging.getLogger(__name__)


# ─── Backend Selection ────────────────────────────────────────────────────────

def _has_env(name: str) -> bool:
    val = os.getenv(name)
    return bool(val and val.strip())

def _load_web_config() -> dict:
    """Load the ``web:`` section from ~/.hermes/config.yaml."""
    try:
        from hermes_cli.config import load_config
        return load_config().get("web", {})
    except (ImportError, Exception):
        return {}

def _get_backend() -> str:
    """Determine which web backend to use.

    Reads ``web.backend`` from config.yaml (set by ``hermes tools``).
    Falls back to whichever API key is present for users who configured
    keys manually without running setup.
    """
    configured = _load_web_config().get("backend", "").lower().strip()
    if configured in ("parallel", "firecrawl", "tavily"):
        return configured

    # Fallback for manual / legacy config — use whichever key is present.
    has_firecrawl = _has_env("FIRECRAWL_API_KEY") or _has_env("FIRECRAWL_API_URL")
    has_parallel = _has_env("PARALLEL_API_KEY")
    has_tavily = _has_env("TAVILY_API_KEY")

    if has_tavily and not has_firecrawl and not has_parallel:
        return "tavily"
    if has_parallel and not has_firecrawl:
        return "parallel"

    # Default to firecrawl (backward compat, or when both are set)
    return "firecrawl"

# ─── Firecrawl Client ────────────────────────────────────────────────────────

_firecrawl_client = None

def _get_firecrawl_client():
    """Get or create the Firecrawl client (lazy initialization).

    Uses the cloud API by default (requires FIRECRAWL_API_KEY).
    Set FIRECRAWL_API_URL to point at a self-hosted instance instead —
    in that case the API key is optional (set USE_DB_AUTHENTICATION=false
    on your Firecrawl server to disable auth entirely).
    """
    global _firecrawl_client
    if _firecrawl_client is None:
        api_key = os.getenv("FIRECRAWL_API_KEY")
        api_url = os.getenv("FIRECRAWL_API_URL")
        if not api_key and not api_url:
            logger.error("Firecrawl client initialization failed: missing configuration.")
            raise ValueError(
                "Firecrawl client not configured. "
                "Set FIRECRAWL_API_KEY (cloud) or FIRECRAWL_API_URL (self-hosted). "
                "This tool requires Firecrawl to be available."
            )
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if api_url:
            kwargs["api_url"] = api_url
        _firecrawl_client = Firecrawl(**kwargs)
    return _firecrawl_client

# ─── Parallel Client ─────────────────────────────────────────────────────────

_parallel_client = None
_async_parallel_client = None

def _get_parallel_client():
    """Get or create the Parallel sync client (lazy initialization).

    Requires PARALLEL_API_KEY environment variable.
    """
    from parallel import Parallel
    global _parallel_client
    if _parallel_client is None:
        api_key = os.getenv("PARALLEL_API_KEY")
        if not api_key:
            raise ValueError(
                "PARALLEL_API_KEY environment variable not set. "
                "Get your API key at https://parallel.ai"
            )
        _parallel_client = Parallel(api_key=api_key)
    return _parallel_client


def _get_async_parallel_client():
    """Get or create the Parallel async client (lazy initialization).

    Requires PARALLEL_API_KEY environment variable.
    """
    from parallel import AsyncParallel
    global _async_parallel_client
    if _async_parallel_client is None:
        api_key = os.getenv("PARALLEL_API_KEY")
        if not api_key:
            raise ValueError(
                "PARALLEL_API_KEY environment variable not set. "
                "Get your API key at https://parallel.ai"
            )
        _async_parallel_client = AsyncParallel(api_key=api_key)
    return _async_parallel_client

# ─── Tavily Client ───────────────────────────────────────────────────────────

_TAVILY_BASE_URL = "https://api.tavily.com"


def _tavily_request(endpoint: str, payload: dict) -> dict:
    """Send a POST request to the Tavily API.

    Auth is provided via ``api_key`` in the JSON body (no header-based auth).
    Raises ``ValueError`` if ``TAVILY_API_KEY`` is not set.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY environment variable not set. "
            "Get your API key at https://app.tavily.com/home"
        )
    payload["api_key"] = api_key
    url = f"{_TAVILY_BASE_URL}/{endpoint.lstrip('/')}"
    logger.info("Tavily %s request to %s", endpoint, url)
    response = httpx.post(url, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def _normalize_tavily_search_results(response: dict) -> dict:
    """Normalize Tavily /search response to the standard web search format.

    Tavily returns ``{results: [{title, url, content, score, ...}]}``.
    We map to ``{success, data: {web: [{title, url, description, position}]}}``.
    """
    web_results = []
    for i, result in enumerate(response.get("results", [])):
        web_results.append({
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "description": result.get("content", ""),
            "position": i + 1,
        })
    return {"success": True, "data": {"web": web_results}}


def _normalize_tavily_documents(response: dict, fallback_url: str = "") -> List[Dict[str, Any]]:
    """Normalize Tavily /extract or /crawl response to the standard document format.

    Maps results to ``{url, title, content, raw_content, metadata}`` and
    includes any ``failed_results`` / ``failed_urls`` as error entries.
    """
    documents: List[Dict[str, Any]] = []
    for result in response.get("results", []):
        url = result.get("url", fallback_url)
        raw = result.get("raw_content", "") or result.get("content", "")
        documents.append({
            "url": url,
            "title": result.get("title", ""),
            "content": raw,
            "raw_content": raw,
            "metadata": {"sourceURL": url, "title": result.get("title", "")},
        })
    # Handle failed results
    for fail in response.get("failed_results", []):
        documents.append({
            "url": fail.get("url", fallback_url),
            "title": "",
            "content": "",
            "raw_content": "",
            "error": fail.get("error", "extraction failed"),
            "metadata": {"sourceURL": fail.get("url", fallback_url)},
        })
    for fail_url in response.get("failed_urls", []):
        url_str = fail_url if isinstance(fail_url, str) else str(fail_url)
        documents.append({
            "url": url_str,
            "title": "",
            "content": "",
            "raw_content": "",
            "error": "extraction failed",
            "metadata": {"sourceURL": url_str},
        })
    return documents


DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION = 5000

# Allow per-task override via env var
DEFAULT_SUMMARIZER_MODEL = os.getenv("AUXILIARY_WEB_EXTRACT_MODEL", "").strip() or None

_debug = DebugSession("web_tools", env_var="WEB_TOOLS_DEBUG")


async def process_content_with_llm(
    content: str, 
    url: str = "", 
    title: str = "",
    model: str = DEFAULT_SUMMARIZER_MODEL,
    min_length: int = DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION
) -> Optional[str]:
    """
    Process web content using LLM to create intelligent summaries with key excerpts.
    
    This function uses Gemini 3 Flash Preview (or specified model) via OpenRouter API 
    to intelligently extract key information and create markdown summaries,
    significantly reducing token usage while preserving all important information.
    
    For very large content (>500k chars), uses chunked processing with synthesis.
    For extremely large content (>2M chars), refuses to process entirely.
    
    Args:
        content (str): The raw content to process
        url (str): The source URL (for context, optional)
        title (str): The page title (for context, optional)
        model (str): The model to use for processing (default: google/gemini-3-flash-preview)
        min_length (int): Minimum content length to trigger processing (default: 5000)
        
    Returns:
        Optional[str]: Processed markdown content, or None if content too short or processing fails
    """
    # Size thresholds
    MAX_CONTENT_SIZE = 2_000_000  # 2M chars - refuse entirely above this
    CHUNK_THRESHOLD = 500_000     # 500k chars - use chunked processing above this
    CHUNK_SIZE = 100_000          # 100k chars per chunk
    MAX_OUTPUT_SIZE = 5000        # Hard cap on final output size
    
    try:
        content_len = len(content)
        
        # Refuse if content is absurdly large
        if content_len > MAX_CONTENT_SIZE:
            size_mb = content_len / 1_000_000
            logger.warning("Content too large (%.1fMB > 2MB limit). Refusing to process.", size_mb)
            return f"[Content too large to process: {size_mb:.1f}MB. Try using web_crawl with specific extraction instructions, or search for a more focused source.]"
        
        # Skip processing if content is too short
        if content_len < min_length:
            logger.debug("Content too short (%d < %d chars), skipping LLM processing", content_len, min_length)
            return None
        
        # Create context information
        context_info = []
        if title:
            context_info.append(f"Title: {title}")
        if url:
            context_info.append(f"Source: {url}")
        context_str = "\n".join(context_info) + "\n\n" if context_info else ""
        
        # Check if we need chunked processing
        if content_len > CHUNK_THRESHOLD:
            logger.info("Content large (%d chars). Using chunked processing...", content_len)
            return await _process_large_content_chunked(
                content, context_str, model, CHUNK_SIZE, MAX_OUTPUT_SIZE
            )
        
        # Standard single-pass processing for normal content
        logger.info("Processing content with LLM (%d characters)", content_len)
        
        processed_content = await _call_summarizer_llm(content, context_str, model)
        
        if processed_content:
            # Enforce output cap
            if len(processed_content) > MAX_OUTPUT_SIZE:
                processed_content = processed_content[:MAX_OUTPUT_SIZE] + "\n\n[... summary truncated for context management ...]"
            
            # Log compression metrics
            processed_length = len(processed_content)
            compression_ratio = processed_length / content_len if content_len > 0 else 1.0
            logger.info("Content processed: %d -> %d chars (%.1f%%)", content_len, processed_length, compression_ratio * 100)
        
        return processed_content
        
    except Exception as e:
        logger.debug("Error processing content with LLM: %s", e)
        return f"[Failed to process content: {str(e)[:100]}. Content size: {len(content):,} chars]"


async def _call_summarizer_llm(
    content: str, 
    context_str: str, 
    model: str, 
    max_tokens: int = 20000,
    is_chunk: bool = False,
    chunk_info: str = ""
) -> Optional[str]:
    """
    Make a single LLM call to summarize content.
    
    Args:
        content: The content to summarize
        context_str: Context information (title, URL)
        model: Model to use
        max_tokens: Maximum output tokens
        is_chunk: Whether this is a chunk of a larger document
        chunk_info: Information about chunk position (e.g., "Chunk 2/5")
        
    Returns:
        Summarized content or None on failure
    """
    if is_chunk:
        # Chunk-specific prompt - aware that this is partial content
        system_prompt = """You are an expert content analyst processing a SECTION of a larger document. Your job is to extract and summarize the key information from THIS SECTION ONLY.

Important guidelines for chunk processing:
1. Do NOT write introductions or conclusions - this is a partial document
2. Focus on extracting ALL key facts, figures, data points, and insights from this section
3. Preserve important quotes, code snippets, and specific details verbatim
4. Use bullet points and structured formatting for easy synthesis later
5. Note any references to other sections (e.g., "as mentioned earlier", "see below") without trying to resolve them

Your output will be combined with summaries of other sections, so focus on thorough extraction rather than narrative flow."""

        user_prompt = f"""Extract key information from this SECTION of a larger document:

{context_str}{chunk_info}

SECTION CONTENT:
{content}

Extract all important information from this section in a structured format. Focus on facts, data, insights, and key details. Do not add introductions or conclusions."""

    else:
        # Standard full-document prompt
        system_prompt = """You are an expert content analyst. Your job is to process web content and create a comprehensive yet concise summary that preserves all important information while dramatically reducing bulk.

Create a well-structured markdown summary that includes:
1. Key excerpts (quotes, code snippets, important facts) in their original format
2. Comprehensive summary of all other important information
3. Proper markdown formatting with headers, bullets, and emphasis

Your goal is to preserve ALL important information while reducing length. Never lose key facts, figures, insights, or actionable information. Make it scannable and well-organized."""

        user_prompt = f"""Please process this web content and create a comprehensive markdown summary:

{context_str}CONTENT TO PROCESS:
{content}

Create a markdown summary that captures all key information in a well-organized, scannable format. Include important quotes and code snippets in their original formatting. Focus on actionable information, specific details, and unique insights."""

    # Call the LLM with retry logic
    max_retries = 6
    retry_delay = 2
    last_error = None

    for attempt in range(max_retries):
        try:
            call_kwargs = {
                "task": "web_extract",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": max_tokens,
            }
            if model:
                call_kwargs["model"] = model
            response = await async_call_llm(**call_kwargs)
            return response.choices[0].message.content.strip()
        except RuntimeError:
            logger.warning("No auxiliary model available for web content processing")
            return None
        except Exception as api_error:
            last_error = api_error
            if attempt < max_retries - 1:
                logger.warning("LLM API call failed (attempt %d/%d): %s", attempt + 1, max_retries, str(api_error)[:100])
                logger.warning("Retrying in %ds...", retry_delay)
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
            else:
                raise last_error
    
    return None


async def _process_large_content_chunked(
    content: str, 
    context_str: str, 
    model: str, 
    chunk_size: int,
    max_output_size: int
) -> Optional[str]:
    """
    Process large content by chunking, summarizing each chunk in parallel,
    then synthesizing the summaries.
    
    Args:
        content: The large content to process
        context_str: Context information
        model: Model to use
        chunk_size: Size of each chunk in characters
        max_output_size: Maximum final output size
        
    Returns:
        Synthesized summary or None on failure
    """
    # Split content into chunks
    chunks = []
    for i in range(0, len(content), chunk_size):
        chunk = content[i:i + chunk_size]
        chunks.append(chunk)
    
    logger.info("Split into %d chunks of ~%d chars each", len(chunks), chunk_size)
    
    # Summarize each chunk in parallel
    async def summarize_chunk(chunk_idx: int, chunk_content: str) -> tuple[int, Optional[str]]:
        """Summarize a single chunk."""
        try:
            chunk_info = f"[Processing chunk {chunk_idx + 1} of {len(chunks)}]"
            summary = await _call_summarizer_llm(
                chunk_content, 
                context_str, 
                model, 
                max_tokens=10000,
                is_chunk=True,
                chunk_info=chunk_info
            )
            if summary:
                logger.info("Chunk %d/%d summarized: %d -> %d chars", chunk_idx + 1, len(chunks), len(chunk_content), len(summary))
            return chunk_idx, summary
        except Exception as e:
            logger.warning("Chunk %d/%d failed: %s", chunk_idx + 1, len(chunks), str(e)[:50])
            return chunk_idx, None
    
    # Run all chunk summarizations in parallel
    tasks = [summarize_chunk(i, chunk) for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks)
    
    # Collect successful summaries in order
    summaries = []
    for chunk_idx, summary in sorted(results, key=lambda x: x[0]):
        if summary:
            summaries.append(f"## Section {chunk_idx + 1}\n{summary}")
    
    if not summaries:
        logger.debug("All chunk summarizations failed")
        return "[Failed to process large content: all chunk summarizations failed]"
    
    logger.info("Got %d/%d chunk summaries", len(summaries), len(chunks))
    
    # If only one chunk succeeded, just return it (with cap)
    if len(summaries) == 1:
        result = summaries[0]
        if len(result) > max_output_size:
            result = result[:max_output_size] + "\n\n[... truncated ...]"
        return result
    
    # Synthesize the summaries into a final summary
    logger.info("Synthesizing %d summaries...", len(summaries))
    
    combined_summaries = "\n\n---\n\n".join(summaries)
    
    synthesis_prompt = f"""You have been given summaries of different sections of a large document. 
Synthesize these into ONE cohesive, comprehensive summary that:
1. Removes redundancy between sections
2. Preserves all key facts, figures, and actionable information
3. Is well-organized with clear structure
4. Is under {max_output_size} characters

{context_str}SECTION SUMMARIES:
{combined_summaries}

Create a single, unified markdown summary."""

    try:
        call_kwargs = {
            "task": "web_extract",
            "messages": [
                {"role": "system", "content": "You synthesize multiple summaries into one cohesive, comprehensive summary. Be thorough but concise."},
                {"role": "user", "content": synthesis_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 20000,
        }
        if model:
            call_kwargs["model"] = model
        response = await async_call_llm(**call_kwargs)
        final_summary = response.choices[0].message.content.strip()
        
        # Enforce hard cap
        if len(final_summary) > max_output_size:
            final_summary = final_summary[:max_output_size] + "\n\n[... summary truncated for context management ...]"
        
        original_len = len(content)
        final_len = len(final_summary)
        compression = final_len / original_len if original_len > 0 else 1.0
        
        logger.info("Synthesis complete: %d -> %d chars (%.2f%%)", original_len, final_len, compression * 100)
        return final_summary
        
    except Exception as e:
        logger.warning("Synthesis failed: %s", str(e)[:100])
        # Fall back to concatenated summaries with truncation
        fallback = "\n\n".join(summaries)
        if len(fallback) > max_output_size:
            fallback = fallback[:max_output_size] + "\n\n[... truncated due to synthesis failure ...]"
        return fallback


def clean_base64_images(text: str) -> str:
    """
    Remove base64 encoded images from text to reduce token count and clutter.
    
    This function finds and removes base64 encoded images in various formats:
    - (data:image/png;base64,...)
    - (data:image/jpeg;base64,...)
    - (data:image/svg+xml;base64,...)
    - data:image/[type];base64,... (without parentheses)
    
    Args:
        text: The text content to clean
        
    Returns:
        Cleaned text with base64 images replaced with placeholders
    """
    # Pattern to match base64 encoded images wrapped in parentheses
    # Matches: (data:image/[type];base64,[base64-string])
    base64_with_parens_pattern = r'\(data:image/[^;]+;base64,[A-Za-z0-9+/=]+\)'
    
    # Pattern to match base64 encoded images without parentheses
    # Matches: data:image/[type];base64,[base64-string]
    base64_pattern = r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+'
    
    # Replace parentheses-wrapped images first
    cleaned_text = re.sub(base64_with_parens_pattern, '[BASE64_IMAGE_REMOVED]', text)
    
    # Then replace any remaining non-parentheses images
    cleaned_text = re.sub(base64_pattern, '[BASE64_IMAGE_REMOVED]', cleaned_text)
    
    return cleaned_text


# ─── Parallel Search & Extract Helpers ────────────────────────────────────────

def _parallel_search(query: str, limit: int = 5) -> dict:
    """Search using the Parallel SDK and return results as a dict."""
    from tools.interrupt import is_interrupted
    if is_interrupted():
        return {"error": "Interrupted", "success": False}

    mode = os.getenv("PARALLEL_SEARCH_MODE", "agentic").lower().strip()
    if mode not in ("fast", "one-shot", "agentic"):
        mode = "agentic"

    logger.info("Parallel search: '%s' (mode=%s, limit=%d)", query, mode, limit)
    response = _get_parallel_client().beta.search(
        search_queries=[query],
        objective=query,
        mode=mode,
        max_results=min(limit, 20),
    )

    web_results = []
    for i, result in enumerate(response.results or []):
        excerpts = result.excerpts or []
        web_results.append({
            "url": result.url or "",
            "title": result.title or "",
            "description": " ".join(excerpts) if excerpts else "",
            "position": i + 1,
        })

    return {"success": True, "data": {"web": web_results}}


async def _parallel_extract(urls: List[str]) -> List[Dict[str, Any]]:
    """Extract content from URLs using the Parallel async SDK.

    Returns a list of result dicts matching the structure expected by the
    LLM post-processing pipeline (url, title, content, metadata).
    """
    from tools.interrupt import is_interrupted
    if is_interrupted():
        return [{"url": u, "error": "Interrupted", "title": ""} for u in urls]

    logger.info("Parallel extract: %d URL(s)", len(urls))
    response = await _get_async_parallel_client().beta.extract(
        urls=urls,
        full_content=True,
    )

    results = []
    for result in response.results or []:
        content = result.full_content or ""
        if not content:
            content = "\n\n".join(result.excerpts or [])
        url = result.url or ""
        title = result.title or ""
        results.append({
            "url": url,
            "title": title,
            "content": content,
            "raw_content": content,
            "metadata": {"sourceURL": url, "title": title},
        })

    for error in response.errors or []:
        results.append({
            "url": error.url or "",
            "title": "",
            "content": "",
            "error": error.content or error.error_type or "extraction failed",
            "metadata": {"sourceURL": error.url or ""},
        })

    return results


def web_search_tool(query: str, limit: int = 5) -> str:
    """
    Search the web for information using available search API backend.

    This function provides a generic interface for web search that can work
    with multiple backends (Parallel or Firecrawl).

    Note: This function returns search result metadata only (URLs, titles, descriptions).
    Use web_extract_tool to get full content from specific URLs.
    
    Args:
        query (str): The search query to look up
        limit (int): Maximum number of results to return (default: 5)
    
    Returns:
        str: JSON string containing search results with the following structure:
             {
                 "success": bool,
                 "data": {
                     "web": [
                         {
                             "title": str,
                             "url": str,
                             "description": str,
                             "position": int
                         },
                         ...
                     ]
                 }
             }
    
    Raises:
        Exception: If search fails or API key is not set
    """
    debug_call_data = {
        "parameters": {
            "query": query,
            "limit": limit
        },
        "error": None,
        "results_count": 0,
        "original_response_size": 0,
        "final_response_size": 0
    }
    
    try:
        from tools.interrupt import is_interrupted
        if is_interrupted():
            return json.dumps({"error": "Interrupted", "success": False})

        # Dispatch to the configured backend
        backend = _get_backend()
        if backend == "parallel":
            response_data = _parallel_search(query, limit)
            debug_call_data["results_count"] = len(response_data.get("data", {}).get("web", []))
            result_json = json.dumps(response_data, indent=2, ensure_ascii=False)
            debug_call_data["final_response_size"] = len(result_json)
            _debug.log_call("web_search_tool", debug_call_data)
            _debug.save()
            return result_json

        if backend == "tavily":
            logger.info("Tavily search: '%s' (limit: %d)", query, limit)
            raw = _tavily_request("search", {
                "query": query,
                "max_results": min(limit, 20),
                "include_raw_content": False,
                "include_images": False,
            })
            response_data = _normalize_tavily_search_results(raw)
            debug_call_data["results_count"] = len(response_data.get("data", {}).get("web", []))
            result_json = json.dumps(response_data, indent=2, ensure_ascii=False)
            debug_call_data["final_response_size"] = len(result_json)
            _debug.log_call("web_search_tool", debug_call_data)
            _debug.save()
            return result_json

        logger.info("Searching the web for: '%s' (limit: %d)", query, limit)

        response = _get_firecrawl_client().search(
            query=query,
            limit=limit
        )

        # The response is a SearchData object with web, news, and images attributes
        # When not scraping, the results are directly in these attributes
        web_results = []

        # Check if response has web attribute (SearchData object)
        if hasattr(response, 'web'):
            # Response is a SearchData object with web attribute
            if response.web:
                # Convert each SearchResultWeb object to dict
                for result in response.web:
                    if hasattr(result, 'model_dump'):
                        # Pydantic model - use model_dump
                        web_results.append(result.model_dump())
                    elif hasattr(result, '__dict__'):
                        # Regular object - use __dict__
                        web_results.append(result.__dict__)
                    elif isinstance(result, dict):
                        # Already a dict
                        web_results.append(result)
        elif hasattr(response, 'model_dump'):
            # Response has model_dump method - use it to get dict
            response_dict = response.model_dump()
            if 'web' in response_dict and response_dict['web']:
                web_results = response_dict['web']
        elif isinstance(response, dict):
            # Response is already a dictionary
            if 'web' in response and response['web']:
                web_results = response['web']
        
        results_count = len(web_results)
        logger.info("Found %d search results", results_count)
        
        # Build response with just search metadata (URLs, titles, descriptions)
        response_data = {
            "success": True,
            "data": {
                "web": web_results
            }
        }
        
        # Capture debug information
        debug_call_data["results_count"] = results_count
        
        # Convert to JSON
        result_json = json.dumps(response_data, indent=2, ensure_ascii=False)
        
        debug_call_data["final_response_size"] = len(result_json)
        
        # Log debug information
        _debug.log_call("web_search_tool", debug_call_data)
        _debug.save()
        
        return result_json
        
    except Exception as e:
        error_msg = f"Error searching web: {str(e)}"
        logger.debug("%s", error_msg)
        
        debug_call_data["error"] = error_msg
        _debug.log_call("web_search_tool", debug_call_data)
        _debug.save()
        
        return json.dumps({"error": error_msg}, ensure_ascii=False)


async def web_extract_tool(
    urls: List[str], 
    format: str = None, 
    use_llm_processing: bool = True,
    model: str = DEFAULT_SUMMARIZER_MODEL,
    min_length: int = DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION
) -> str:
    """
    Extract content from specific web pages using available extraction API backend.
    
    This function provides a generic interface for web content extraction that
    can work with multiple backends. Currently uses Firecrawl.
    
    Args:
        urls (List[str]): List of URLs to extract content from
        format (str): Desired output format ("markdown" or "html", optional)
        use_llm_processing (bool): Whether to process content with LLM for summarization (default: True)
        model (str): The model to use for LLM processing (default: google/gemini-3-flash-preview)
        min_length (int): Minimum content length to trigger LLM processing (default: 5000)
    
    Returns:
        str: JSON string containing extracted content. If LLM processing is enabled and successful,
             the 'content' field will contain the processed markdown summary instead of raw content.
    
    Raises:
        Exception: If extraction fails or API key is not set
    """
    debug_call_data = {
        "parameters": {
            "urls": urls,
            "format": format,
            "use_llm_processing": use_llm_processing,
            "model": model,
            "min_length": min_length
        },
        "error": None,
        "pages_extracted": 0,
        "pages_processed_with_llm": 0,
        "original_response_size": 0,
        "final_response_size": 0,
        "compression_metrics": [],
        "processing_applied": []
    }
    
    try:
        logger.info("Extracting content from %d URL(s)", len(urls))

        # Dispatch to the configured backend
        backend = _get_backend()

        if backend == "parallel":
            results = await _parallel_extract(urls)
        elif backend == "tavily":
            logger.info("Tavily extract: %d URL(s)", len(urls))
            raw = _tavily_request("extract", {
                "urls": urls,
                "include_images": False,
            })
            results = _normalize_tavily_documents(raw, fallback_url=urls[0] if urls else "")
        else:
            # ── Firecrawl extraction ──
            # Determine requested formats for Firecrawl v2
            formats: List[str] = []
            if format == "markdown":
                formats = ["markdown"]
            elif format == "html":
                formats = ["html"]
            else:
                # Default: request markdown for LLM-readiness and include html as backup
                formats = ["markdown", "html"]

            # Always use individual scraping for simplicity and reliability
            # Batch scraping adds complexity without much benefit for small numbers of URLs
            results: List[Dict[str, Any]] = []

            from tools.interrupt import is_interrupted as _is_interrupted
            for url in urls:
                if _is_interrupted():
                    results.append({"url": url, "error": "Interrupted", "title": ""})
                    continue

                # Website policy check — block before fetching
                blocked = check_website_access(url)
                if blocked:
                    logger.info("Blocked web_extract for %s by rule %s", blocked["host"], blocked["rule"])
                    results.append({
                        "url": url, "title": "", "content": "",
                        "error": blocked["message"],
                        "blocked_by_policy": {"host": blocked["host"], "rule": blocked["rule"], "source": blocked["source"]},
                    })
                    continue

                try:
                    logger.info("Scraping: %s", url)
                    scrape_result = _get_firecrawl_client().scrape(
                        url=url,
                        formats=formats
                    )

                    # Process the result - properly handle object serialization
                    metadata = {}
                    title = ""
                    content_markdown = None
                    content_html = None

                    # Extract data from the scrape result
                    if hasattr(scrape_result, 'model_dump'):
                        # Pydantic model - use model_dump to get dict
                        result_dict = scrape_result.model_dump()
                        content_markdown = result_dict.get('markdown')
                        content_html = result_dict.get('html')
                        metadata = result_dict.get('metadata', {})
                    elif hasattr(scrape_result, '__dict__'):
                        # Regular object with attributes
                        content_markdown = getattr(scrape_result, 'markdown', None)
                        content_html = getattr(scrape_result, 'html', None)

                        # Handle metadata - convert to dict if it's an object
                        metadata_obj = getattr(scrape_result, 'metadata', {})
                        if hasattr(metadata_obj, 'model_dump'):
                            metadata = metadata_obj.model_dump()
                        elif hasattr(metadata_obj, '__dict__'):
                            metadata = metadata_obj.__dict__
                        elif isinstance(metadata_obj, dict):
                            metadata = metadata_obj
                        else:
                            metadata = {}
                    elif isinstance(scrape_result, dict):
                        # Already a dictionary
                        content_markdown = scrape_result.get('markdown')
                        content_html = scrape_result.get('html')
                        metadata = scrape_result.get('metadata', {})

                    # Ensure metadata is a dict (not an object)
                    if not isinstance(metadata, dict):
                        if hasattr(metadata, 'model_dump'):
                            metadata = metadata.model_dump()
                        elif hasattr(metadata, '__dict__'):
                            metadata = metadata.__dict__
                        else:
                            metadata = {}

                    # Get title from metadata
                    title = metadata.get("title", "")

                    # Re-check final URL after redirect
                    final_url = metadata.get("sourceURL", url)
                    final_blocked = check_website_access(final_url)
                    if final_blocked:
                        logger.info("Blocked redirected web_extract for %s by rule %s", final_blocked["host"], final_blocked["rule"])
                        results.append({
                            "url": final_url, "title": title, "content": "", "raw_content": "",
                            "error": final_blocked["message"],
                            "blocked_by_policy": {"host": final_blocked["host"], "rule": final_blocked["rule"], "source": final_blocked["source"]},
                        })
                        continue

                    # Choose content based on requested format
                    chosen_content = content_markdown if (format == "markdown" or (format is None and content_markdown)) else content_html or content_markdown or ""

                    results.append({
                        "url": final_url,
                        "title": title,
                        "content": chosen_content,
                        "raw_content": chosen_content,
                        "metadata": metadata  # Now guaranteed to be a dict
                    })

                except Exception as scrape_err:
                    logger.debug("Scrape failed for %s: %s", url, scrape_err)
                    results.append({
                        "url": url,
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "error": str(scrape_err)
                    })

        response = {"results": results}
        
        pages_extracted = len(response.get('results', []))
        logger.info("Extracted content from %d pages", pages_extracted)
        
        debug_call_data["pages_extracted"] = pages_extracted
        debug_call_data["original_response_size"] = len(json.dumps(response))
        
        # Process each result with LLM if enabled
        if use_llm_processing:
            logger.info("Processing extracted content with LLM (parallel)...")
            debug_call_data["processing_applied"].append("llm_processing")
            
            # Prepare tasks for parallel processing
            async def process_single_result(result):
                """Process a single result with LLM and return updated result with metrics."""
                url = result.get('url', 'Unknown URL')
                title = result.get('title', '')
                raw_content = result.get('raw_content', '') or result.get('content', '')
                
                if not raw_content:
                    return result, None, "no_content"
                
                original_size = len(raw_content)
                
                # Process content with LLM
                processed = await process_content_with_llm(
                    raw_content, url, title, model, min_length
                )
                
                if processed:
                    processed_size = len(processed)
                    compression_ratio = processed_size / original_size if original_size > 0 else 1.0
                    
                    # Update result with processed content
                    result['content'] = processed
                    result['raw_content'] = raw_content
                    
                    metrics = {
                        "url": url,
                        "original_size": original_size,
                        "processed_size": processed_size,
                        "compression_ratio": compression_ratio,
                        "model_used": model
                    }
                    return result, metrics, "processed"
                else:
                    metrics = {
                        "url": url,
                        "original_size": original_size,
                        "processed_size": original_size,
                        "compression_ratio": 1.0,
                        "model_used": None,
                        "reason": "content_too_short"
                    }
                    return result, metrics, "too_short"
            
            # Run all LLM processing in parallel
            results_list = response.get('results', [])
            tasks = [process_single_result(result) for result in results_list]
            processed_results = await asyncio.gather(*tasks)
            
            # Collect metrics and print results
            for result, metrics, status in processed_results:
                url = result.get('url', 'Unknown URL')
                if status == "processed":
                    debug_call_data["compression_metrics"].append(metrics)
                    debug_call_data["pages_processed_with_llm"] += 1
                    logger.info("%s (processed)", url)
                elif status == "too_short":
                    debug_call_data["compression_metrics"].append(metrics)
                    logger.info("%s (no processing - content too short)", url)
                else:
                    logger.warning("%s (no content to process)", url)
        else:
            # Print summary of extracted pages for debugging (original behavior)
            for result in response.get('results', []):
                url = result.get('url', 'Unknown URL')
                content_length = len(result.get('raw_content', ''))
                logger.info("%s (%d characters)", url, content_length)
        
        # Trim output to minimal fields per entry: title, content, error
        trimmed_results = [
            {
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "error": r.get("error"),
                **({  "blocked_by_policy": r["blocked_by_policy"]} if "blocked_by_policy" in r else {}),
            }
            for r in response.get("results", [])
        ]
        trimmed_response = {"results": trimmed_results}

        if trimmed_response.get("results") == []:
            result_json = json.dumps({"error": "Content was inaccessible or not found"}, ensure_ascii=False)

            cleaned_result = clean_base64_images(result_json)
        
        else:
            result_json = json.dumps(trimmed_response, indent=2, ensure_ascii=False)
            
            cleaned_result = clean_base64_images(result_json)
        
        debug_call_data["final_response_size"] = len(cleaned_result)
        debug_call_data["processing_applied"].append("base64_image_removal")
        
        # Log debug information
        _debug.log_call("web_extract_tool", debug_call_data)
        _debug.save()
        
        # Store extracted pages in persistent storage
        try:
            store_pages_from_results(trimmed_results, source_tool="web_extract")
        except Exception as store_err:
            logger.debug("Failed to store pages: %s", store_err)
        
        return cleaned_result
            
    except Exception as e:
        error_msg = f"Error extracting content: {str(e)}"
        logger.debug("%s", error_msg)
        
        debug_call_data["error"] = error_msg
        _debug.log_call("web_extract_tool", debug_call_data)
        _debug.save()
        
        return json.dumps({"error": error_msg}, ensure_ascii=False)


async def web_crawl_tool(
    url: str, 
    instructions: str = None, 
    depth: str = "basic", 
    use_llm_processing: bool = True,
    model: str = DEFAULT_SUMMARIZER_MODEL,
    min_length: int = DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION
) -> str:
    """
    Crawl a website with specific instructions using available crawling API backend.
    
    This function provides a generic interface for web crawling that can work
    with multiple backends. Currently uses Firecrawl.
    
    Args:
        url (str): The base URL to crawl (can include or exclude https://)
        instructions (str): Instructions for what to crawl/extract using LLM intelligence (optional)
        depth (str): Depth of extraction ("basic" or "advanced", default: "basic")
        use_llm_processing (bool): Whether to process content with LLM for summarization (default: True)
        model (str): The model to use for LLM processing (default: google/gemini-3-flash-preview)
        min_length (int): Minimum content length to trigger LLM processing (default: 5000)
    
    Returns:
        str: JSON string containing crawled content. If LLM processing is enabled and successful,
             the 'content' field will contain the processed markdown summary instead of raw content.
             Each page is processed individually.
    
    Raises:
        Exception: If crawling fails or API key is not set
    """
    debug_call_data = {
        "parameters": {
            "url": url,
            "instructions": instructions,
            "depth": depth,
            "use_llm_processing": use_llm_processing,
            "model": model,
            "min_length": min_length
        },
        "error": None,
        "pages_crawled": 0,
        "pages_processed_with_llm": 0,
        "original_response_size": 0,
        "final_response_size": 0,
        "compression_metrics": [],
        "processing_applied": []
    }
    
    try:
        backend = _get_backend()

        # Tavily supports crawl via its /crawl endpoint
        if backend == "tavily":
            # Ensure URL has protocol
            if not url.startswith(('http://', 'https://')):
                url = f'https://{url}'

            # Website policy check
            blocked = check_website_access(url)
            if blocked:
                logger.info("Blocked web_crawl for %s by rule %s", blocked["host"], blocked["rule"])
                return json.dumps({"results": [{"url": url, "title": "", "content": "", "error": blocked["message"],
                    "blocked_by_policy": {"host": blocked["host"], "rule": blocked["rule"], "source": blocked["source"]}}]}, ensure_ascii=False)

            from tools.interrupt import is_interrupted as _is_int
            if _is_int():
                return json.dumps({"error": "Interrupted", "success": False})

            logger.info("Tavily crawl: %s", url)
            payload: Dict[str, Any] = {
                "url": url,
                "limit": 20,
                "extract_depth": depth,
            }
            if instructions:
                payload["instructions"] = instructions
            raw = _tavily_request("crawl", payload)
            results = _normalize_tavily_documents(raw, fallback_url=url)

            response = {"results": results}
            # Fall through to the shared LLM processing and trimming below
            # (skip the Firecrawl-specific crawl logic)
            pages_crawled = len(response.get('results', []))
            logger.info("Crawled %d pages", pages_crawled)
            debug_call_data["pages_crawled"] = pages_crawled
            debug_call_data["original_response_size"] = len(json.dumps(response))

            # Process each result with LLM if enabled
            if use_llm_processing:
                logger.info("Processing crawled content with LLM (parallel)...")
                debug_call_data["processing_applied"].append("llm_processing")

                async def _process_tavily_crawl(result):
                    page_url = result.get('url', 'Unknown URL')
                    title = result.get('title', '')
                    content = result.get('content', '')
                    if not content:
                        return result, None, "no_content"
                    original_size = len(content)
                    processed = await process_content_with_llm(content, page_url, title, model, min_length)
                    if processed:
                        result['raw_content'] = content
                        result['content'] = processed
                        metrics = {"url": page_url, "original_size": original_size, "processed_size": len(processed),
                                   "compression_ratio": len(processed) / original_size if original_size else 1.0, "model_used": model}
                        return result, metrics, "processed"
                    metrics = {"url": page_url, "original_size": original_size, "processed_size": original_size,
                               "compression_ratio": 1.0, "model_used": None, "reason": "content_too_short"}
                    return result, metrics, "too_short"

                tasks = [_process_tavily_crawl(r) for r in response.get('results', [])]
                processed_results = await asyncio.gather(*tasks)
                for result, metrics, status in processed_results:
                    if status == "processed":
                        debug_call_data["compression_metrics"].append(metrics)
                        debug_call_data["pages_processed_with_llm"] += 1

            trimmed_results = [{"url": r.get("url", ""), "title": r.get("title", ""), "content": r.get("content", ""), "error": r.get("error"),
                **({  "blocked_by_policy": r["blocked_by_policy"]} if "blocked_by_policy" in r else {})} for r in response.get("results", [])]
            result_json = json.dumps({"results": trimmed_results}, indent=2, ensure_ascii=False)
            cleaned_result = clean_base64_images(result_json)
            debug_call_data["final_response_size"] = len(cleaned_result)
            _debug.log_call("web_crawl_tool", debug_call_data)
            _debug.save()
            
            # Store crawled pages in persistent storage
            try:
                store_pages_from_results(trimmed_results, source_tool="web_crawl")
            except Exception as store_err:
                logger.debug("Failed to store crawled pages: %s", store_err)
            
            return cleaned_result

        # web_crawl requires Firecrawl — Parallel has no crawl API
        if not (os.getenv("FIRECRAWL_API_KEY") or os.getenv("FIRECRAWL_API_URL")):
            return json.dumps({
                "error": "web_crawl requires Firecrawl. Set FIRECRAWL_API_KEY, "
                         "or use web_search + web_extract instead.",
                "success": False,
            }, ensure_ascii=False)

        # Ensure URL has protocol
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'
            logger.info("Added https:// prefix to URL: %s", url)
        
        instructions_text = f" with instructions: '{instructions}'" if instructions else ""
        logger.info("Crawling %s%s", url, instructions_text)
        
        # Website policy check — block before crawling
        blocked = check_website_access(url)
        if blocked:
            logger.info("Blocked web_crawl for %s by rule %s", blocked["host"], blocked["rule"])
            return json.dumps({"results": [{"url": url, "title": "", "content": "", "error": blocked["message"],
                "blocked_by_policy": {"host": blocked["host"], "rule": blocked["rule"], "source": blocked["source"]}}]}, ensure_ascii=False)

        # Use Firecrawl's v2 crawl functionality
        # Docs: https://docs.firecrawl.dev/features/crawl
        # The crawl() method automatically waits for completion and returns all data
        
        # Build crawl parameters - keep it simple
        crawl_params = {
            "limit": 20,  # Limit number of pages to crawl
            "scrape_options": {
                "formats": ["markdown"]  # Just markdown for simplicity
            }
        }
        
        # Note: The 'prompt' parameter is not documented for crawl
        # Instructions are typically used with the Extract endpoint, not Crawl
        if instructions:
            logger.info("Instructions parameter ignored (not supported in crawl API)")
        
        from tools.interrupt import is_interrupted as _is_int
        if _is_int():
            return json.dumps({"error": "Interrupted", "success": False})

        try:
            crawl_result = _get_firecrawl_client().crawl(
                url=url,
                **crawl_params
            )
        except Exception as e:
            logger.debug("Crawl API call failed: %s", e)
            raise

        pages: List[Dict[str, Any]] = []
        
        # Process crawl results - the crawl method returns a CrawlJob object with data attribute
        data_list = []
        
        # The crawl_result is a CrawlJob object with a 'data' attribute containing list of Document objects
        if hasattr(crawl_result, 'data'):
            data_list = crawl_result.data if crawl_result.data else []
            logger.info("Status: %s", getattr(crawl_result, 'status', 'unknown'))
            logger.info("Retrieved %d pages", len(data_list))
            
            # Debug: Check other attributes if no data
            if not data_list:
                logger.debug("CrawlJob attributes: %s", [attr for attr in dir(crawl_result) if not attr.startswith('_')])
                logger.debug("Status: %s", getattr(crawl_result, 'status', 'N/A'))
                logger.debug("Total: %s", getattr(crawl_result, 'total', 'N/A'))
                logger.debug("Completed: %s", getattr(crawl_result, 'completed', 'N/A'))
                
        elif isinstance(crawl_result, dict) and 'data' in crawl_result:
            data_list = crawl_result.get("data", [])
        else:
            logger.warning("Unexpected crawl result type")
            logger.debug("Result type: %s", type(crawl_result))
            if hasattr(crawl_result, '__dict__'):
                logger.debug("Result attributes: %s", list(crawl_result.__dict__.keys()))
        
        for item in data_list:
            # Process each crawled page - properly handle object serialization
            page_url = "Unknown URL"
            title = ""
            content_markdown = None
            content_html = None
            metadata = {}
            
            # Extract data from the item
            if hasattr(item, 'model_dump'):
                # Pydantic model - use model_dump to get dict
                item_dict = item.model_dump()
                content_markdown = item_dict.get('markdown')
                content_html = item_dict.get('html')
                metadata = item_dict.get('metadata', {})
            elif hasattr(item, '__dict__'):
                # Regular object with attributes
                content_markdown = getattr(item, 'markdown', None)
                content_html = getattr(item, 'html', None)
                
                # Handle metadata - convert to dict if it's an object
                metadata_obj = getattr(item, 'metadata', {})
                if hasattr(metadata_obj, 'model_dump'):
                    metadata = metadata_obj.model_dump()
                elif hasattr(metadata_obj, '__dict__'):
                    metadata = metadata_obj.__dict__
                elif isinstance(metadata_obj, dict):
                    metadata = metadata_obj
                else:
                    metadata = {}
            elif isinstance(item, dict):
                # Already a dictionary
                content_markdown = item.get('markdown')
                content_html = item.get('html')
                metadata = item.get('metadata', {})
            
            # Ensure metadata is a dict (not an object)
            if not isinstance(metadata, dict):
                if hasattr(metadata, 'model_dump'):
                    metadata = metadata.model_dump()
                elif hasattr(metadata, '__dict__'):
                    metadata = metadata.__dict__
                else:
                    metadata = {}
            
            # Extract URL and title from metadata
            page_url = metadata.get("sourceURL", metadata.get("url", "Unknown URL"))
            title = metadata.get("title", "")
            
            # Re-check crawled page URL against policy
            page_blocked = check_website_access(page_url)
            if page_blocked:
                logger.info("Blocked crawled page %s by rule %s", page_blocked["host"], page_blocked["rule"])
                pages.append({
                    "url": page_url, "title": title, "content": "", "raw_content": "",
                    "error": page_blocked["message"],
                    "blocked_by_policy": {"host": page_blocked["host"], "rule": page_blocked["rule"], "source": page_blocked["source"]},
                })
                continue

            # Choose content (prefer markdown)
            content = content_markdown or content_html or ""
            
            pages.append({
                "url": page_url,
                "title": title,
                "content": content,
                "raw_content": content,
                "metadata": metadata  # Now guaranteed to be a dict
            })

        response = {"results": pages}
        
        pages_crawled = len(response.get('results', []))
        logger.info("Crawled %d pages", pages_crawled)
        
        debug_call_data["pages_crawled"] = pages_crawled
        debug_call_data["original_response_size"] = len(json.dumps(response))
        
        # Process each result with LLM if enabled
        if use_llm_processing:
            logger.info("Processing crawled content with LLM (parallel)...")
            debug_call_data["processing_applied"].append("llm_processing")
            
            # Prepare tasks for parallel processing
            async def process_single_crawl_result(result):
                """Process a single crawl result with LLM and return updated result with metrics."""
                page_url = result.get('url', 'Unknown URL')
                title = result.get('title', '')
                content = result.get('content', '')
                
                if not content:
                    return result, None, "no_content"
                
                original_size = len(content)
                
                # Process content with LLM
                processed = await process_content_with_llm(
                    content, page_url, title, model, min_length
                )
                
                if processed:
                    processed_size = len(processed)
                    compression_ratio = processed_size / original_size if original_size > 0 else 1.0
                    
                    # Update result with processed content
                    result['raw_content'] = content
                    result['content'] = processed
                    
                    metrics = {
                        "url": page_url,
                        "original_size": original_size,
                        "processed_size": processed_size,
                        "compression_ratio": compression_ratio,
                        "model_used": model
                    }
                    return result, metrics, "processed"
                else:
                    metrics = {
                        "url": page_url,
                        "original_size": original_size,
                        "processed_size": original_size,
                        "compression_ratio": 1.0,
                        "model_used": None,
                        "reason": "content_too_short"
                    }
                    return result, metrics, "too_short"
            
            # Run all LLM processing in parallel
            results_list = response.get('results', [])
            tasks = [process_single_crawl_result(result) for result in results_list]
            processed_results = await asyncio.gather(*tasks)
            
            # Collect metrics and print results
            for result, metrics, status in processed_results:
                page_url = result.get('url', 'Unknown URL')
                if status == "processed":
                    debug_call_data["compression_metrics"].append(metrics)
                    debug_call_data["pages_processed_with_llm"] += 1
                    logger.info("%s (processed)", page_url)
                elif status == "too_short":
                    debug_call_data["compression_metrics"].append(metrics)
                    logger.info("%s (no processing - content too short)", page_url)
                else:
                    logger.warning("%s (no content to process)", page_url)
        else:
            # Print summary of crawled pages for debugging (original behavior)
            for result in response.get('results', []):
                page_url = result.get('url', 'Unknown URL')
                content_length = len(result.get('content', ''))
                logger.info("%s (%d characters)", page_url, content_length)
        
        # Trim output to minimal fields per entry: title, content, error
        trimmed_results = [
            {
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "error": r.get("error"),
                **({  "blocked_by_policy": r["blocked_by_policy"]} if "blocked_by_policy" in r else {}),
            }
            for r in response.get("results", [])
        ]
        trimmed_response = {"results": trimmed_results}
        
        result_json = json.dumps(trimmed_response, indent=2, ensure_ascii=False)
        # Clean base64 images from crawled content
        cleaned_result = clean_base64_images(result_json)
        
        debug_call_data["final_response_size"] = len(cleaned_result)
        debug_call_data["processing_applied"].append("base64_image_removal")
        
        # Log debug information
        _debug.log_call("web_crawl_tool", debug_call_data)
        _debug.save()
        
        # Store crawled pages in persistent storage
        try:
            store_pages_from_results(trimmed_results, source_tool="web_crawl")
        except Exception as store_err:
            logger.debug("Failed to store crawled pages: %s", store_err)
        
        return cleaned_result
        
    except Exception as e:
        error_msg = f"Error crawling website: {str(e)}"
        logger.debug("%s", error_msg)
        
        debug_call_data["error"] = error_msg
        _debug.log_call("web_crawl_tool", debug_call_data)
        _debug.save()
        
        return json.dumps({"error": error_msg}, ensure_ascii=False)


# Convenience function to check if API key is available
def check_firecrawl_api_key() -> bool:
    """
    Check if the Firecrawl API key is available in environment variables.

    Returns:
        bool: True if API key is set, False otherwise
    """
    return bool(os.getenv("FIRECRAWL_API_KEY"))


def check_web_api_key() -> bool:
    """Check if any web backend API key is available (Parallel, Firecrawl, or Tavily)."""
    return bool(
        os.getenv("PARALLEL_API_KEY")
        or os.getenv("FIRECRAWL_API_KEY")
        or os.getenv("FIRECRAWL_API_URL")
        or os.getenv("TAVILY_API_KEY")
    )


def check_auxiliary_model() -> bool:
    """Check if an auxiliary text model is available for LLM content processing."""
    try:
        from agent.auxiliary_client import resolve_provider_client
        for p in ("openrouter", "nous", "custom", "codex"):
            client, _ = resolve_provider_client(p)
            if client is not None:
                return True
        return False
    except Exception:
        return False


def get_debug_session_info() -> Dict[str, Any]:
    """Get information about the current debug session."""
    return _debug.get_session_info()


if __name__ == "__main__":
    """
    Simple test/demo when run directly
    """
    print("🌐 Standalone Web Tools Module")
    print("=" * 40)
    
    # Check if API keys are available
    web_available = check_web_api_key()
    nous_available = check_auxiliary_model()

    if web_available:
        backend = _get_backend()
        print(f"✅ Web backend: {backend}")
        if backend == "parallel":
            print("   Using Parallel API (https://parallel.ai)")
        elif backend == "tavily":
            print("   Using Tavily API (https://tavily.com)")
        else:
            print("   Using Firecrawl API (https://firecrawl.dev)")
    else:
        print("❌ No web search backend configured")
        print("Set PARALLEL_API_KEY, TAVILY_API_KEY, or FIRECRAWL_API_KEY")

    if not nous_available:
        print("❌ No auxiliary model available for LLM content processing")
        print("Set OPENROUTER_API_KEY, configure Nous Portal, or set OPENAI_BASE_URL + OPENAI_API_KEY")
        print("⚠️  Without an auxiliary model, LLM content processing will be disabled")
    else:
        print(f"✅ Auxiliary model available: {DEFAULT_SUMMARIZER_MODEL}")

    if not web_available:
        exit(1)

    print("🛠️  Web tools ready for use!")
    
    if nous_available:
        print(f"🧠 LLM content processing available with {DEFAULT_SUMMARIZER_MODEL}")
        print(f"   Default min length for processing: {DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION} chars")
    
    # Show debug mode status
    if _debug.active:
        print(f"🐛 Debug mode ENABLED - Session ID: {_debug.session_id}")
        print(f"   Debug logs will be saved to: {_debug.log_dir}/web_tools_debug_{_debug.session_id}.json")
    else:
        print("🐛 Debug mode disabled (set WEB_TOOLS_DEBUG=true to enable)")
    
    print("\nBasic usage:")
    print("  from web_tools import web_search_tool, web_extract_tool, web_crawl_tool")
    print("  import asyncio")
    print("")
    print("  # Search (synchronous)")
    print("  results = web_search_tool('Python tutorials')")
    print("")
    print("  # Extract and crawl (asynchronous)")
    print("  async def main():")
    print("      content = await web_extract_tool(['https://example.com'])")
    print("      crawl_data = await web_crawl_tool('example.com', 'Find docs')")
    print("  asyncio.run(main())")
    
    if nous_available:
        print("\nLLM-enhanced usage:")
        print("  # Content automatically processed for pages >5000 chars (default)")
        print("  content = await web_extract_tool(['https://python.org/about/'])")
        print("")
        print("  # Customize processing parameters")
        print("  crawl_data = await web_crawl_tool(")
        print("      'docs.python.org',")
        print("      'Find key concepts',")
        print("      model='google/gemini-3-flash-preview',")
        print("      min_length=3000")
        print("  )")
        print("")
        print("  # Disable LLM processing")
        print("  raw_content = await web_extract_tool(['https://example.com'], use_llm_processing=False)")
    
    print("\nDebug mode:")
    print("  # Enable debug logging")
    print("  export WEB_TOOLS_DEBUG=true")
    print("  # Debug logs capture:")
    print("  # - All tool calls with parameters")
    print("  # - Original API responses")
    print("  # - LLM compression metrics")
    print("  # - Final processed results")
    print("  # Logs saved to: ./logs/web_tools_debug_UUID.json")
    
    print(f"\n📝 Run 'python test_web_tools_llm.py' to test LLM processing capabilities")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry

WEB_SEARCH_SCHEMA = {
    "name": "web_search",
    "description": "Search the web for information on any topic. Returns up to 5 relevant results with titles, URLs, and descriptions.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up on the web"
            }
        },
        "required": ["query"]
    }
}

WEB_EXTRACT_SCHEMA = {
    "name": "web_extract",
    "description": "Extract content from web page URLs. Returns page content in markdown format. Also works with PDF URLs (arxiv papers, documents, etc.) — pass the PDF link directly and it converts to markdown text. Pages under 5000 chars return full markdown; larger pages are LLM-summarized and capped at ~5000 chars per page. Pages over 2M chars are refused. If a URL fails or times out, use the browser tool to access it instead.",
    "parameters": {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of URLs to extract content from (max 5 URLs per call)",
                "maxItems": 5
            }
        },
        "required": ["urls"]
    }
}

registry.register(
    name="web_search",
    toolset="web",
    schema=WEB_SEARCH_SCHEMA,
    handler=lambda args, **kw: web_search_tool(args.get("query", ""), limit=5),
    check_fn=check_web_api_key,
    requires_env=["PARALLEL_API_KEY", "FIRECRAWL_API_KEY", "TAVILY_API_KEY"],
    emoji="🔍",
)
registry.register(
    name="web_extract",
    toolset="web",
    schema=WEB_EXTRACT_SCHEMA,
    handler=lambda args, **kw: web_extract_tool(
        args.get("urls", [])[:5] if isinstance(args.get("urls"), list) else [], "markdown"),
    check_fn=check_web_api_key,
    requires_env=["PARALLEL_API_KEY", "FIRECRAWL_API_KEY", "TAVILY_API_KEY"],
    is_async=True,
    emoji="📄",
)

# ─── Page Storage Tools Registry ────────────────────────────────────────────

WEB_PAGE_SEARCH_SCHEMA = {
    "name": "web_page_search",
    "description": "Search over previously extracted/saved web pages using full-text search. Use this to find information in pages you've previously extracted with web_extract or web_crawl.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to find in saved pages"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 10)",
                "default": 10
            }
        },
        "required": ["query"]
    }
}

WEB_PAGE_LIST_SCHEMA = {
    "name": "web_page_list",
    "description": "List all saved web pages. Shows pages previously extracted with web_extract or web_crawl.",
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of pages to list (default: 50)",
                "default": 50
            },
            "offset": {
                "type": "integer",
                "description": "Offset for pagination (default: 0)",
                "default": 0
            },
            "source": {
                "type": "string",
                "description": "Filter by source tool: 'web_extract' or 'web_crawl'"
            }
        }
    }
}

WEB_PAGE_GET_SCHEMA = {
    "name": "web_page_get",
    "description": "Retrieve a specific saved page by URL or page ID. Returns full content and metadata.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL of the page to retrieve"
            },
            "page_id": {
                "type": "integer",
                "description": "ID of the page to retrieve"
            }
        }
    }
}

WEB_PAGE_DELETE_SCHEMA = {
    "name": "web_page_delete",
    "description": "Delete a saved page from storage by URL or page ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL of the page to delete"
            },
            "page_id": {
                "type": "integer",
                "description": "ID of the page to delete"
            }
        }
    }
}

WEB_PAGE_STATS_SCHEMA = {
    "name": "web_page_stats",
    "description": "Get statistics about saved web pages (count, size, storage location).",
    "parameters": {
        "type": "object",
        "properties": {}
    }
}

# Register page storage tools
registry.register(
    name="web_page_search",
    toolset="web",
    schema=WEB_PAGE_SEARCH_SCHEMA,
    handler=lambda args, **kw: web_page_search_tool(
        args.get("query", ""), limit=args.get("limit", 10)),
    emoji="🔎",
)

registry.register(
    name="web_page_list",
    toolset="web",
    schema=WEB_PAGE_LIST_SCHEMA,
    handler=lambda args, **kw: web_page_list_tool(
        limit=args.get("limit", 50),
        offset=args.get("offset", 0),
        source=args.get("source")),
    emoji="📋",
)

registry.register(
    name="web_page_get",
    toolset="web",
    schema=WEB_PAGE_GET_SCHEMA,
    handler=lambda args, **kw: web_page_get_tool(
        url=args.get("url"), page_id=args.get("page_id")),
    emoji="📖",
)

registry.register(
    name="web_page_delete",
    toolset="web",
    schema=WEB_PAGE_DELETE_SCHEMA,
    handler=lambda args, **kw: web_page_delete_tool(
        url=args.get("url"), page_id=args.get("page_id")),
    emoji="🗑️",
)

registry.register(
    name="web_page_stats",
    toolset="web",
    schema=WEB_PAGE_STATS_SCHEMA,
    handler=lambda args, **kw: web_page_stats_tool(),
    emoji="📊",
)
