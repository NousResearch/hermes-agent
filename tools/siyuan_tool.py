"""SiYuan Note tool for managing a personal knowledge base via REST API.

Registers seven LLM-callable tools:
- ``siyuan_search``  -- full-text or SQL search across notes
- ``siyuan_read``    -- read block content, children, path, or attributes
- ``siyuan_write``   -- create notebooks, documents, or blocks
- ``siyuan_update``  -- update block content, rename docs, set attributes
- ``siyuan_delete``  -- delete blocks, documents, or notebooks
- ``siyuan_list``    -- list notebooks and documents
- ``siyuan_export``  -- export a document as Markdown

Authentication uses an API token via ``SIYUAN_TOKEN`` env var (Settings > About in SiYuan).
The SiYuan instance URL is read from ``SIYUAN_URL`` (default: http://127.0.0.1:6806).

Notebook-level access control: set ``SIYUAN_ALLOWED_NOTEBOOKS`` to a comma-separated
list of notebook IDs. When set, all operations are restricted to those notebooks only.
When unset, all notebooks are accessible.

Alternative (MCP): if you prefer zero custom code, install @porkll/siyuan-mcp and
add it to ~/.hermes/config.yaml under mcp_servers. See the Hermes MCP docs.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

import re

logger = logging.getLogger(__name__)

# Regex for valid SiYuan block/notebook IDs (e.g. "20210808180117-6v0mkxr")
_SIYUAN_ID_RE = re.compile(r"^[0-9]{14}-[a-z0-9]{7}$")


def _validate_id(block_id: str) -> Optional[str]:
    """Return an error message if the ID format is invalid, or None if OK."""
    if not block_id:
        return "ID cannot be empty"
    if not _SIYUAN_ID_RE.match(block_id):
        return f"Invalid SiYuan ID format: '{block_id}'. Expected format: YYYYMMDDHHmmss-xxxxxxx"
    return None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SIYUAN_URL: str = ""
_SIYUAN_TOKEN: str = ""


def _get_config():
    """Return (siyuan_url, siyuan_token) from env vars at call time."""
    return (
        (_SIYUAN_URL or os.getenv("SIYUAN_URL", "http://127.0.0.1:6806")).rstrip("/"),
        _SIYUAN_TOKEN or os.getenv("SIYUAN_TOKEN", ""),
    )


def _get_allowed_notebooks() -> Optional[List[str]]:
    """Return list of allowed notebook IDs, or None if unrestricted."""
    raw = os.getenv("SIYUAN_ALLOWED_NOTEBOOKS", "").strip()
    if not raw:
        return None
    return [nb.strip() for nb in raw.split(",") if nb.strip()]


# ---------------------------------------------------------------------------
# Async HTTP helper (all SiYuan API calls are POST + JSON)
# ---------------------------------------------------------------------------

async def _async_siyuan_request(endpoint: str, payload: dict = None) -> dict:
    """Send a POST request to SiYuan kernel API and return the data field."""
    import aiohttp

    url, token = _get_config()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Token {token}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{url}{endpoint}",
                headers=headers,
                json=payload or {},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
                result = await resp.json()
    except aiohttp.ClientConnectorError:
        raise ConnectionError(
            f"Cannot connect to SiYuan at {url}. "
            "Make sure SiYuan is running and the URL is correct."
        )

    if result.get("code") != 0:
        raise RuntimeError(f"SiYuan API error: {result.get('msg', 'Unknown error')}")
    return result.get("data", {})


# ---------------------------------------------------------------------------
# Notebook access control
# ---------------------------------------------------------------------------

async def _async_get_block_notebook(block_id: str) -> str:
    """Resolve which notebook a block belongs to via SQL."""
    # Validate ID format to prevent SQL injection
    err = _validate_id(block_id)
    if err:
        return ""
    data = await _async_siyuan_request("/api/query/sql", {
        "stmt": f"SELECT box FROM blocks WHERE id = '{block_id}' LIMIT 1"
    })
    rows = data if isinstance(data, list) else []
    if not rows:
        return ""
    return rows[0].get("box", "")


def _check_notebook_access(notebook_id: str) -> Optional[str]:
    """Return an error message if the notebook is not in the allowlist, or None if OK."""
    allowed = _get_allowed_notebooks()
    if allowed is None:
        return None  # unrestricted
    if notebook_id not in allowed:
        return (
            f"Access denied: notebook '{notebook_id}' is not in the allowed list. "
            f"Allowed notebooks: {', '.join(allowed)}"
        )
    return None


async def _async_check_block_access(block_id: str) -> Optional[str]:
    """Check if a block's parent notebook is in the allowlist."""
    allowed = _get_allowed_notebooks()
    if allowed is None:
        return None
    notebook_id = await _async_get_block_notebook(block_id)
    if not notebook_id:
        return f"Could not determine notebook for block '{block_id}'"
    return _check_notebook_access(notebook_id)


# ---------------------------------------------------------------------------
# Async core functions
# ---------------------------------------------------------------------------

def _filter_by_allowed_notebooks(results: List[dict], box_key: str = "box") -> List[dict]:
    """Remove results from notebooks not in the allowlist."""
    allowed = _get_allowed_notebooks()
    if allowed is None:
        return results
    return [r for r in results if r.get(box_key, "") in allowed]


async def _async_search(
    query: str,
    method: str = "fulltext",
    limit: int = 20,
) -> Dict[str, Any]:
    """Search SiYuan content via full-text or SQL."""
    if method == "sql":
        # Safety: only allow SELECT statements
        stmt = query.strip()
        if not stmt.upper().startswith("SELECT"):
            return {"error": "Only SELECT SQL statements are allowed for security."}
        # Block dangerous SQL keywords
        upper = stmt.upper()
        for forbidden in ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE"):
            if forbidden in upper:
                return {"error": f"SQL statement contains forbidden keyword: {forbidden}"}
        if not stmt.rstrip().endswith(f"LIMIT {limit}") and "LIMIT" not in upper:
            stmt = f"{stmt.rstrip().rstrip(';')} LIMIT {limit}"
        data = await _async_siyuan_request("/api/query/sql", {"stmt": stmt})
        rows = data if isinstance(data, list) else []
        # Filter by allowed notebooks
        rows = _filter_by_allowed_notebooks(rows, "box")
        # Truncate content fields
        for row in rows:
            if "content" in row and len(row["content"]) > 500:
                row["content"] = row["content"][:500] + "...[truncated]"
        return {"count": len(rows), "results": rows[:limit]}
    else:
        # Full-text search
        data = await _async_siyuan_request("/api/search/fullTextSearchBlock", {
            "query": query,
            "page": 0,
        })
        blocks = data.get("blocks", []) if isinstance(data, dict) else []
        # Filter by allowed notebooks
        blocks = _filter_by_allowed_notebooks(blocks, "box")
        results = []
        for b in blocks[:limit]:
            content = b.get("content", "")
            if len(content) > 500:
                content = content[:500] + "...[truncated]"
            results.append({
                "id": b.get("id", ""),
                "rootID": b.get("rootID", ""),
                "box": b.get("box", ""),
                "path": b.get("hPath", b.get("path", "")),
                "content": content,
                "type": b.get("type", ""),
            })
        return {"count": len(results), "results": results}


async def _async_read(
    block_id: str,
    action: str = "content",
) -> Dict[str, Any]:
    """Read a block's content, children, path, or attributes."""
    # Check notebook access
    err = await _async_check_block_access(block_id)
    if err:
        return {"error": err}

    if action == "content":
        data = await _async_siyuan_request("/api/block/getBlockKramdown", {"id": block_id})
        kramdown = data.get("kramdown", "")
        if len(kramdown) > 2000:
            kramdown = kramdown[:2000] + "\n...[truncated]"
        return {"id": block_id, "kramdown": kramdown}

    elif action == "children":
        data = await _async_siyuan_request("/api/block/getChildBlocks", {"id": block_id})
        children = data if isinstance(data, list) else []
        results = []
        for c in children[:50]:
            content = c.get("content", c.get("markdown", ""))
            if len(content) > 200:
                content = content[:200] + "..."
            results.append({
                "id": c.get("id", ""),
                "type": c.get("type", ""),
                "subType": c.get("subType", ""),
                "content": content,
            })
        return {"id": block_id, "count": len(results), "children": results}

    elif action == "path":
        data = await _async_siyuan_request("/api/filetree/getHPathByID", {"id": block_id})
        return {"id": block_id, "path": data if isinstance(data, str) else str(data)}

    elif action == "attrs":
        data = await _async_siyuan_request("/api/attr/getBlockAttrs", {"id": block_id})
        return {"id": block_id, "attrs": data if isinstance(data, dict) else {}}

    return {"error": f"Unknown read action: {action}"}


async def _async_write(
    action: str,
    content: str = "",
    notebook: str = "",
    path: str = "",
    parent_id: str = "",
    previous_id: str = "",
    name: str = "",
) -> Dict[str, Any]:
    """Create new content in SiYuan."""
    if action == "create_notebook":
        if not name:
            return {"error": "Missing required parameter: name"}
        # Block creation when allowlist is active (new notebook would be outside it)
        allowed = _get_allowed_notebooks()
        if allowed is not None:
            return {
                "error": "Cannot create notebooks when SIYUAN_ALLOWED_NOTEBOOKS is set. "
                "New notebooks would be outside the allowlist. "
                "Ask the user to add the new notebook ID to the allowlist after creation."
            }
        data = await _async_siyuan_request("/api/notebook/createNotebook", {"name": name})
        nb = data.get("notebook", {})
        return {"success": True, "notebook_id": nb.get("id", ""), "name": name}

    elif action == "create_doc":
        if not notebook:
            return {"error": "Missing required parameter: notebook"}
        if not path:
            return {"error": "Missing required parameter: path"}
        err = _check_notebook_access(notebook)
        if err:
            return {"error": err}
        data = await _async_siyuan_request("/api/filetree/createDocWithMd", {
            "notebook": notebook,
            "path": path,
            "markdown": content,
        })
        return {"success": True, "doc_id": data if isinstance(data, str) else str(data)}

    elif action == "append_block":
        if not parent_id:
            return {"error": "Missing required parameter: parent_id"}
        if not content:
            return {"error": "Missing required parameter: content"}
        err = await _async_check_block_access(parent_id)
        if err:
            return {"error": err}
        data = await _async_siyuan_request("/api/block/appendBlock", {
            "parentID": parent_id,
            "data": content,
            "dataType": "markdown",
        })
        return {"success": True, "data": data}

    elif action == "prepend_block":
        if not parent_id:
            return {"error": "Missing required parameter: parent_id"}
        if not content:
            return {"error": "Missing required parameter: content"}
        err = await _async_check_block_access(parent_id)
        if err:
            return {"error": err}
        data = await _async_siyuan_request("/api/block/prependBlock", {
            "parentID": parent_id,
            "data": content,
            "dataType": "markdown",
        })
        return {"success": True, "data": data}

    elif action == "insert_block":
        if not previous_id:
            return {"error": "Missing required parameter: previous_id"}
        if not content:
            return {"error": "Missing required parameter: content"}
        err = await _async_check_block_access(previous_id)
        if err:
            return {"error": err}
        data = await _async_siyuan_request("/api/block/insertBlock", {
            "previousID": previous_id,
            "data": content,
            "dataType": "markdown",
        })
        return {"success": True, "data": data}

    return {"error": f"Unknown write action: {action}"}


async def _async_update(
    action: str,
    block_id: str,
    content: str = "",
    title: str = "",
    attrs: dict = None,
) -> Dict[str, Any]:
    """Update existing content in SiYuan."""
    if action == "update_block":
        if not block_id:
            return {"error": "Missing required parameter: id"}
        if not content:
            return {"error": "Missing required parameter: content"}
        err = await _async_check_block_access(block_id)
        if err:
            return {"error": err}
        data = await _async_siyuan_request("/api/block/updateBlock", {
            "id": block_id,
            "data": content,
            "dataType": "markdown",
        })
        return {"success": True, "data": data}

    elif action == "rename_doc":
        if not block_id:
            return {"error": "Missing required parameter: id"}
        if not title:
            return {"error": "Missing required parameter: title"}
        err = await _async_check_block_access(block_id)
        if err:
            return {"error": err}
        await _async_siyuan_request("/api/filetree/renameDocByID", {
            "id": block_id,
            "title": title,
        })
        return {"success": True, "id": block_id, "new_title": title}

    elif action == "set_attrs":
        if not block_id:
            return {"error": "Missing required parameter: id"}
        if not attrs:
            return {"error": "Missing required parameter: attrs"}
        err = await _async_check_block_access(block_id)
        if err:
            return {"error": err}
        # Auto-prefix custom attributes
        prefixed = {}
        for k, v in attrs.items():
            key = k if k.startswith("custom-") else f"custom-{k}"
            prefixed[key] = str(v)
        await _async_siyuan_request("/api/attr/setBlockAttrs", {
            "id": block_id,
            "attrs": prefixed,
        })
        return {"success": True, "id": block_id, "attrs": prefixed}

    return {"error": f"Unknown update action: {action}"}


async def _async_delete(
    target: str,
    target_id: str,
) -> Dict[str, Any]:
    """Delete a block, document, or notebook in SiYuan."""
    if target == "block":
        err = await _async_check_block_access(target_id)
        if err:
            return {"error": err}
        await _async_siyuan_request("/api/block/deleteBlock", {"id": target_id})
        return {"success": True, "deleted": "block", "id": target_id}

    elif target == "document":
        err = await _async_check_block_access(target_id)
        if err:
            return {"error": err}
        await _async_siyuan_request("/api/filetree/removeDocByID", {"id": target_id})
        return {"success": True, "deleted": "document", "id": target_id}

    elif target == "notebook":
        err_msg = _check_notebook_access(target_id)
        if err_msg:
            return {"error": err_msg}
        await _async_siyuan_request("/api/notebook/removeNotebook", {"notebook": target_id})
        return {"success": True, "deleted": "notebook", "id": target_id}

    return {"error": f"Unknown delete target: {target}"}


async def _async_list_items(
    action: str = "notebooks",
    notebook: str = "",
    path: str = "/",
) -> Dict[str, Any]:
    """List notebooks or documents."""
    if action == "notebooks":
        data = await _async_siyuan_request("/api/notebook/lsNotebooks", {})
        notebooks = data.get("notebooks", []) if isinstance(data, dict) else []
        allowed = _get_allowed_notebooks()
        results = []
        for nb in notebooks:
            nb_id = nb.get("id", "")
            if allowed and nb_id not in allowed:
                continue
            results.append({
                "id": nb_id,
                "name": nb.get("name", ""),
                "closed": nb.get("closed", False),
            })
        return {"count": len(results), "notebooks": results}

    elif action == "documents":
        if not notebook:
            return {"error": "Missing required parameter: notebook"}
        err = _check_notebook_access(notebook)
        if err:
            return {"error": err}
        data = await _async_siyuan_request("/api/filetree/listDocsByPath", {
            "notebook": notebook,
            "path": path,
        })
        files = data.get("files", []) if isinstance(data, dict) else []
        results = []
        for f in files[:50]:
            results.append({
                "id": f.get("id", ""),
                "name": f.get("name", ""),
                "subFileCount": f.get("subFileCount", 0),
            })
        return {"count": len(results), "documents": results}

    return {"error": f"Unknown list action: {action}"}


async def _async_export(doc_id: str) -> Dict[str, Any]:
    """Export a document as Markdown."""
    err = await _async_check_block_access(doc_id)
    if err:
        return {"error": err}
    data = await _async_siyuan_request("/api/export/exportMdContent", {"id": doc_id})
    content = data.get("content", "") if isinstance(data, dict) else ""
    if len(content) > 5000:
        content = content[:5000] + "\n\n...[truncated, document too large]"
    return {"id": doc_id, "markdown": content}


# ---------------------------------------------------------------------------
# Sync bridge
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Run an async coroutine from a sync handler."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=30)
    else:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Handlers  (signature: (args, **kw) -> str)
# ---------------------------------------------------------------------------

def _handle_search(args: dict, **kw) -> str:
    """Handler for siyuan_search tool."""
    query = args.get("query", "")
    if not query:
        return json.dumps({"error": "Missing required parameter: query"})
    method = args.get("method", "fulltext")
    limit = args.get("limit", 20)
    try:
        result = _run_async(_async_search(query=query, method=method, limit=limit))
        return json.dumps({"result": result}, ensure_ascii=False)
    except Exception as e:
        logger.error("siyuan_search error: %s", e)
        return json.dumps({"error": f"Search failed: {e}"})


def _handle_read(args: dict, **kw) -> str:
    """Handler for siyuan_read tool."""
    block_id = args.get("id", "")
    if not block_id:
        return json.dumps({"error": "Missing required parameter: id"})
    id_err = _validate_id(block_id)
    if id_err:
        return json.dumps({"error": id_err})
    action = args.get("action", "content")
    try:
        result = _run_async(_async_read(block_id=block_id, action=action))
        return json.dumps({"result": result}, ensure_ascii=False)
    except Exception as e:
        logger.error("siyuan_read error: %s", e)
        return json.dumps({"error": f"Read failed: {e}"})


def _handle_write(args: dict, **kw) -> str:
    """Handler for siyuan_write tool."""
    action = args.get("action", "")
    if not action:
        return json.dumps({"error": "Missing required parameter: action"})
    try:
        result = _run_async(_async_write(
            action=action,
            content=args.get("content", ""),
            notebook=args.get("notebook", ""),
            path=args.get("path", ""),
            parent_id=args.get("parent_id", ""),
            previous_id=args.get("previous_id", ""),
            name=args.get("name", ""),
        ))
        return json.dumps({"result": result}, ensure_ascii=False)
    except Exception as e:
        logger.error("siyuan_write error: %s", e)
        return json.dumps({"error": f"Write failed: {e}"})


def _handle_update(args: dict, **kw) -> str:
    """Handler for siyuan_update tool."""
    action = args.get("action", "")
    block_id = args.get("id", "")
    if not action:
        return json.dumps({"error": "Missing required parameter: action"})
    if not block_id:
        return json.dumps({"error": "Missing required parameter: id"})
    id_err = _validate_id(block_id)
    if id_err:
        return json.dumps({"error": id_err})
    try:
        result = _run_async(_async_update(
            action=action,
            block_id=block_id,
            content=args.get("content", ""),
            title=args.get("title", ""),
            attrs=args.get("attrs"),
        ))
        return json.dumps({"result": result}, ensure_ascii=False)
    except Exception as e:
        logger.error("siyuan_update error: %s", e)
        return json.dumps({"error": f"Update failed: {e}"})


def _handle_delete(args: dict, **kw) -> str:
    """Handler for siyuan_delete tool."""
    target = args.get("target", "")
    target_id = args.get("id", "")
    if not target:
        return json.dumps({"error": "Missing required parameter: target"})
    if not target_id:
        return json.dumps({"error": "Missing required parameter: id"})
    id_err = _validate_id(target_id)
    if id_err:
        return json.dumps({"error": id_err})
    try:
        result = _run_async(_async_delete(target=target, target_id=target_id))
        return json.dumps({"result": result}, ensure_ascii=False)
    except Exception as e:
        logger.error("siyuan_delete error: %s", e)
        return json.dumps({"error": f"Delete failed: {e}"})


def _handle_list(args: dict, **kw) -> str:
    """Handler for siyuan_list tool."""
    action = args.get("action", "notebooks")
    try:
        result = _run_async(_async_list_items(
            action=action,
            notebook=args.get("notebook", ""),
            path=args.get("path", "/"),
        ))
        return json.dumps({"result": result}, ensure_ascii=False)
    except Exception as e:
        logger.error("siyuan_list error: %s", e)
        return json.dumps({"error": f"List failed: {e}"})


def _handle_export(args: dict, **kw) -> str:
    """Handler for siyuan_export tool."""
    doc_id = args.get("id", "")
    if not doc_id:
        return json.dumps({"error": "Missing required parameter: id"})
    id_err = _validate_id(doc_id)
    if id_err:
        return json.dumps({"error": id_err})
    try:
        result = _run_async(_async_export(doc_id=doc_id))
        return json.dumps({"result": result}, ensure_ascii=False)
    except Exception as e:
        logger.error("siyuan_export error: %s", e)
        return json.dumps({"error": f"Export failed: {e}"})


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def _check_siyuan_available() -> bool:
    """Tool is only available when SIYUAN_TOKEN is set."""
    return bool(os.getenv("SIYUAN_TOKEN"))


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

SIYUAN_SEARCH_SCHEMA = {
    "name": "siyuan_search",
    "description": (
        "Search content in SiYuan Note. Supports full-text search or SQL queries "
        "against the block database. Use SQL for precise queries "
        "(e.g. SELECT * FROM blocks WHERE content LIKE '%keyword%' AND type='p')."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query. For fulltext: natural language keywords. "
                    "For sql: a SELECT statement against the blocks table "
                    "(columns: id, parent_id, root_id, box, path, content, type, subtype, created, updated)."
                ),
            },
            "method": {
                "type": "string",
                "enum": ["fulltext", "sql"],
                "description": "Search method. 'fulltext' for keyword search, 'sql' for SQL queries. Default: fulltext.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return. Default: 20.",
            },
        },
        "required": ["query"],
    },
}

SIYUAN_READ_SCHEMA = {
    "name": "siyuan_read",
    "description": (
        "Read content from a SiYuan block. Can read the block's Kramdown/Markdown content, "
        "list its child blocks, get its human-readable path, or get its custom attributes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "The block or document ID to read.",
            },
            "action": {
                "type": "string",
                "enum": ["content", "children", "path", "attrs"],
                "description": (
                    "What to read. 'content': block Kramdown text, 'children': list child blocks, "
                    "'path': human-readable document path, 'attrs': custom attributes. Default: content."
                ),
            },
        },
        "required": ["id"],
    },
}

SIYUAN_WRITE_SCHEMA = {
    "name": "siyuan_write",
    "description": (
        "Create new content in SiYuan: notebooks, documents, or blocks. "
        "Use 'create_doc' to create a full document from Markdown, or "
        "'append_block'/'prepend_block'/'insert_block' to add blocks to existing documents."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create_notebook", "create_doc", "append_block", "prepend_block", "insert_block"],
                "description": "The write operation to perform.",
            },
            "content": {
                "type": "string",
                "description": "Markdown content for the new document or block.",
            },
            "notebook": {
                "type": "string",
                "description": "Notebook ID (required for create_doc).",
            },
            "path": {
                "type": "string",
                "description": "Document path within the notebook (required for create_doc, e.g. '/Meeting Notes').",
            },
            "parent_id": {
                "type": "string",
                "description": "Parent block ID (required for append_block/prepend_block).",
            },
            "previous_id": {
                "type": "string",
                "description": "Previous sibling block ID (required for insert_block).",
            },
            "name": {
                "type": "string",
                "description": "Notebook name (required for create_notebook).",
            },
        },
        "required": ["action"],
    },
}

SIYUAN_UPDATE_SCHEMA = {
    "name": "siyuan_update",
    "description": (
        "Update existing content in SiYuan. Can update a block's Markdown content, "
        "rename a document, or set custom attributes on a block."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["update_block", "rename_doc", "set_attrs"],
                "description": "The update operation to perform.",
            },
            "id": {
                "type": "string",
                "description": "The block or document ID to update.",
            },
            "content": {
                "type": "string",
                "description": "New Markdown content (for update_block).",
            },
            "title": {
                "type": "string",
                "description": "New document title (for rename_doc).",
            },
            "attrs": {
                "type": "object",
                "description": (
                    "Key-value attributes to set (for set_attrs). "
                    "Keys are auto-prefixed with 'custom-' if not already."
                ),
            },
        },
        "required": ["action", "id"],
    },
}

SIYUAN_DELETE_SCHEMA = {
    "name": "siyuan_delete",
    "description": (
        "Delete content from SiYuan. Can delete a block, a document, or an entire notebook. "
        "This action is irreversible."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "enum": ["block", "document", "notebook"],
                "description": "Type of item to delete.",
            },
            "id": {
                "type": "string",
                "description": "The ID of the block, document, or notebook to delete.",
            },
        },
        "required": ["target", "id"],
    },
}

SIYUAN_LIST_SCHEMA = {
    "name": "siyuan_list",
    "description": (
        "List notebooks or documents in SiYuan to navigate the workspace structure. "
        "Use 'notebooks' to see all available notebooks, or 'documents' to list docs in a notebook."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["notebooks", "documents"],
                "description": "What to list. Default: notebooks.",
            },
            "notebook": {
                "type": "string",
                "description": "Notebook ID (required when action is 'documents').",
            },
            "path": {
                "type": "string",
                "description": "Path within the notebook to list (default: '/').",
            },
        },
        "required": [],
    },
}

SIYUAN_EXPORT_SCHEMA = {
    "name": "siyuan_export",
    "description": (
        "Export a SiYuan document as clean Markdown. Useful for extracting "
        "a full document's content in a portable format."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "The document ID to export.",
            },
        },
        "required": ["id"],
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

from tools.registry import registry

registry.register(
    name="siyuan_search",
    toolset="siyuan",
    schema=SIYUAN_SEARCH_SCHEMA,
    handler=_handle_search,
    check_fn=_check_siyuan_available,
    requires_env=["SIYUAN_TOKEN"],
    emoji="📝",
)

registry.register(
    name="siyuan_read",
    toolset="siyuan",
    schema=SIYUAN_READ_SCHEMA,
    handler=_handle_read,
    check_fn=_check_siyuan_available,
    requires_env=["SIYUAN_TOKEN"],
    emoji="📝",
)

registry.register(
    name="siyuan_write",
    toolset="siyuan",
    schema=SIYUAN_WRITE_SCHEMA,
    handler=_handle_write,
    check_fn=_check_siyuan_available,
    requires_env=["SIYUAN_TOKEN"],
    emoji="📝",
)

registry.register(
    name="siyuan_update",
    toolset="siyuan",
    schema=SIYUAN_UPDATE_SCHEMA,
    handler=_handle_update,
    check_fn=_check_siyuan_available,
    requires_env=["SIYUAN_TOKEN"],
    emoji="📝",
)

registry.register(
    name="siyuan_delete",
    toolset="siyuan",
    schema=SIYUAN_DELETE_SCHEMA,
    handler=_handle_delete,
    check_fn=_check_siyuan_available,
    requires_env=["SIYUAN_TOKEN"],
    emoji="📝",
)

registry.register(
    name="siyuan_list",
    toolset="siyuan",
    schema=SIYUAN_LIST_SCHEMA,
    handler=_handle_list,
    check_fn=_check_siyuan_available,
    requires_env=["SIYUAN_TOKEN"],
    emoji="📝",
)

registry.register(
    name="siyuan_export",
    toolset="siyuan",
    schema=SIYUAN_EXPORT_SCHEMA,
    handler=_handle_export,
    check_fn=_check_siyuan_available,
    requires_env=["SIYUAN_TOKEN"],
    emoji="📝",
)
