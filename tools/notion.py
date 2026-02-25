"""
Notion integration tool for Hermes Agent.

Allows the agent to read/write Notion pages, databases, and blocks.

Setup:
    1. Go to https://www.notion.so/my-integrations
    2. Create a new integration, copy the "Internal Integration Secret"
    3. Add NOTION_API_KEY to ~/.hermes/.env
    4. Share your Notion pages/databases with the integration

Usage:
    hermes --toolsets notion -q "List my Notion databases"
    hermes --toolsets notion -q "Create a new page called 'Meeting Notes' in database <id>"
"""

import json
import os
from typing import Any

import requests

NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"


def _get_headers() -> dict:
    api_key = os.environ.get("NOTION_API_KEY", "")
    if not api_key:
        raise ValueError(
            "NOTION_API_KEY not set. Add it to ~/.hermes/.env\n"
            "Get one at: https://www.notion.so/my-integrations"
        )
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION,
    }


def _req(method: str, path: str, **kwargs) -> dict:
    """Make a Notion API request and return parsed JSON."""
    url = f"{NOTION_API_BASE}{path}"
    resp = requests.request(method, url, headers=_get_headers(), timeout=30, **kwargs)
    try:
        data = resp.json()
    except Exception:
        data = {"error": resp.text}
    if not resp.ok:
        error_msg = data.get("message", resp.text)
        raise RuntimeError(f"Notion API error {resp.status_code}: {error_msg}")
    return data


# ---------------------------------------------------------------------------
# Rich-text helpers
# ---------------------------------------------------------------------------

def _plain_text(rich_text: list) -> str:
    """Extract plain text from a Notion rich_text array."""
    return "".join(rt.get("plain_text", "") for rt in rich_text)


def _make_rich_text(text: str) -> list:
    return [{"type": "text", "text": {"content": text}}]


def _summarize_block(block: dict) -> str:
    """Convert a block object to a human-readable one-liner."""
    btype = block.get("type", "unknown")
    bdata = block.get(btype, {})
    rich = bdata.get("rich_text", [])
    text = _plain_text(rich) if rich else ""
    url = bdata.get("url", "")
    checked = bdata.get("checked", None)

    if btype == "paragraph":
        return text or "(empty paragraph)"
    elif btype in ("heading_1", "heading_2", "heading_3"):
        level = btype[-1]
        return f"{'#' * int(level)} {text}"
    elif btype == "bulleted_list_item":
        return f"• {text}"
    elif btype == "numbered_list_item":
        return f"1. {text}"
    elif btype == "to_do":
        mark = "[x]" if checked else "[ ]"
        return f"{mark} {text}"
    elif btype == "toggle":
        return f"▶ {text}"
    elif btype == "code":
        lang = bdata.get("language", "")
        return f"```{lang}\n{text}\n```"
    elif btype == "quote":
        return f"> {text}"
    elif btype == "divider":
        return "---"
    elif btype == "image":
        img_url = bdata.get("external", {}).get("url") or bdata.get("file", {}).get("url", "")
        caption = _plain_text(bdata.get("caption", []))
        return f"[image: {caption or img_url}]"
    elif btype == "bookmark":
        caption = _plain_text(bdata.get("caption", []))
        return f"[bookmark: {caption or url}]({url})"
    elif btype == "child_page":
        return f"📄 Sub-page: {bdata.get('title', '(untitled)')}"
    elif btype == "child_database":
        return f"🗄 Sub-database: {bdata.get('title', '(untitled)')}"
    elif btype == "callout":
        icon = bdata.get("icon", {}).get("emoji", "💡")
        return f"{icon} {text}"
    else:
        return f"[{btype}] {text or '(no text)'}"


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------

def notion_search(query: str, filter_type: str = "all") -> str:
    """
    Search across all Notion pages and databases accessible to the integration.

    Args:
        query: Text to search for. Pass an empty string to list everything.
        filter_type: Filter results. One of: "all", "page", "database".

    Returns:
        A formatted list of matching pages/databases with their IDs and URLs.

    Examples:
        notion_search("meeting notes")
        notion_search("", filter_type="database")
    """
    payload: dict[str, Any] = {"query": query, "page_size": 20}
    if filter_type in ("page", "database"):
        payload["filter"] = {"value": filter_type, "property": "object"}

    data = _req("POST", "/search", json=payload)
    results = data.get("results", [])
    if not results:
        return f"No results found for '{query}'."

    lines = [f"Found {len(results)} result(s) for '{query}':\n"]
    for item in results:
        obj_type = item.get("object", "unknown")
        item_id = item.get("id", "")
        url = item.get("url", "")
        # Extract title
        if obj_type == "page":
            props = item.get("properties", {})
            title_prop = props.get("title") or props.get("Name") or {}
            title_parts = title_prop.get("title", [])
            title = _plain_text(title_parts) if title_parts else "(untitled)"
        else:  # database
            title_arr = item.get("title", [])
            title = _plain_text(title_arr) if title_arr else "(untitled)"

        icon = "📄" if obj_type == "page" else "🗄"
        lines.append(f"{icon} [{obj_type}] {title}")
        lines.append(f"   ID:  {item_id}")
        lines.append(f"   URL: {url}")
        lines.append("")

    return "\n".join(lines)


def notion_get_page(page_id: str) -> str:
    """
    Retrieve a Notion page's metadata and full content (all blocks).

    Args:
        page_id: The Notion page ID (e.g. "abc123de-..." or the ID from notion_search).

    Returns:
        The page title, properties, and all block content as readable text.
    """
    # Fetch page metadata
    page = _req("GET", f"/pages/{page_id}")
    props = page.get("properties", {})
    url = page.get("url", "")

    # Extract title
    title_prop = props.get("title") or props.get("Name") or {}
    title = _plain_text(title_prop.get("title", [])) or "(untitled)"

    lines = [f"# {title}", f"URL: {url}", ""]

    # Fetch all blocks (with pagination)
    has_more = True
    cursor = None
    block_lines = []
    while has_more:
        params: dict = {"page_size": 100}
        if cursor:
            params["start_cursor"] = cursor
        blocks_data = _req("GET", f"/blocks/{page_id}/children", params=params)
        for block in blocks_data.get("results", []):
            block_lines.append(_summarize_block(block))
        has_more = blocks_data.get("has_more", False)
        cursor = blocks_data.get("next_cursor")

    if block_lines:
        lines += block_lines
    else:
        lines.append("(No content)")

    return "\n".join(lines)


def notion_create_page(
    parent_id: str,
    title: str,
    content: str = "",
    parent_type: str = "database",
) -> str:
    """
    Create a new page in a Notion database or as a child of another page.

    Args:
        parent_id: ID of the parent database or page.
        title: Title for the new page.
        content: Optional page body text (creates a single paragraph block).
        parent_type: "database" (default) or "page".

    Returns:
        Confirmation with the new page's ID and URL.

    Examples:
        notion_create_page("db-id-here", "My New Task", "Details go here")
        notion_create_page("page-id-here", "Sub-page", parent_type="page")
    """
    if parent_type == "database":
        parent = {"database_id": parent_id}
        properties = {
            "Name": {"title": _make_rich_text(title)},
        }
    else:
        parent = {"page_id": parent_id}
        properties = {
            "title": {"title": _make_rich_text(title)},
        }

    payload: dict[str, Any] = {"parent": parent, "properties": properties}

    if content:
        payload["children"] = [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": _make_rich_text(content)},
            }
        ]

    new_page = _req("POST", "/pages", json=payload)
    page_id = new_page.get("id", "")
    page_url = new_page.get("url", "")
    return f"✅ Page created successfully!\nTitle: {title}\nID: {page_id}\nURL: {page_url}"


def notion_append_blocks(page_id: str, text: str, block_type: str = "paragraph") -> str:
    """
    Append text blocks to an existing Notion page.

    Args:
        page_id: The Notion page ID.
        text: The text content to append. For multiple blocks, separate with '\\n\\n'.
        block_type: Block type for all new blocks. Options:
                    "paragraph" (default), "heading_1", "heading_2", "heading_3",
                    "bulleted_list_item", "numbered_list_item", "to_do", "quote", "callout".

    Returns:
        Confirmation message.

    Examples:
        notion_append_blocks("page-id", "New paragraph text")
        notion_append_blocks("page-id", "# My Heading", block_type="heading_2")
        notion_append_blocks("page-id", "Task 1\\n\\nTask 2", block_type="bulleted_list_item")
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    children = []
    for para in paragraphs:
        block: dict[str, Any] = {
            "object": "block",
            "type": block_type,
            block_type: {"rich_text": _make_rich_text(para)},
        }
        # to_do blocks need a checked field
        if block_type == "to_do":
            block[block_type]["checked"] = False
        children.append(block)

    _req("PATCH", f"/blocks/{page_id}/children", json={"children": children})
    return f"✅ Appended {len(children)} block(s) to page {page_id}."


def notion_update_page(page_id: str, properties: str) -> str:
    """
    Update properties of an existing Notion page (e.g. status, tags, dates).

    Args:
        page_id: The Notion page ID.
        properties: A JSON string mapping property names to Notion property values.
                    Example for a checkbox: '{"Done": {"checkbox": true}}'
                    Example for a select: '{"Status": {"select": {"name": "In Progress"}}}'
                    Example for a title: '{"Name": {"title": [{"text": {"content": "New Title"}}]}}'

    Returns:
        Confirmation with updated properties.

    Examples:
        notion_update_page("page-id", '{"Done": {"checkbox": true}}')
        notion_update_page("page-id", '{"Priority": {"select": {"name": "High"}}}')
    """
    try:
        props = json.loads(properties)
    except json.JSONDecodeError as e:
        return f"❌ Invalid JSON for properties: {e}\nProvide valid JSON, e.g.: '{{\"Done\": {{\"checkbox\": true}}}}'"

    _req("PATCH", f"/pages/{page_id}", json={"properties": props})
    prop_names = ", ".join(props.keys())
    return f"✅ Updated page {page_id}. Properties changed: {prop_names}"


def notion_query_database(
    database_id: str,
    filter_json: str = "",
    sorts_json: str = "",
    page_size: int = 20,
) -> str:
    """
    Query a Notion database with optional filters and sorts.

    Args:
        database_id: The Notion database ID.
        filter_json: Optional JSON filter object (Notion filter syntax).
                     Example: '{"property": "Status", "select": {"equals": "Done"}}'
        sorts_json: Optional JSON array of sort objects.
                    Example: '[{"property": "Created", "direction": "descending"}]'
        page_size: Number of results to return (1-100, default 20).

    Returns:
        A formatted list of database entries with their IDs and key properties.

    Examples:
        notion_query_database("db-id")
        notion_query_database("db-id", filter_json='{"property": "Done", "checkbox": {"equals": false}}')
    """
    payload: dict[str, Any] = {"page_size": min(max(page_size, 1), 100)}
    if filter_json:
        try:
            payload["filter"] = json.loads(filter_json)
        except json.JSONDecodeError as e:
            return f"❌ Invalid filter JSON: {e}"
    if sorts_json:
        try:
            payload["sorts"] = json.loads(sorts_json)
        except json.JSONDecodeError as e:
            return f"❌ Invalid sorts JSON: {e}"

    data = _req("POST", f"/databases/{database_id}/query", json=payload)
    results = data.get("results", [])
    if not results:
        return "No entries found in database."

    lines = [f"Found {len(results)} entries:\n"]
    for item in results:
        item_id = item.get("id", "")
        url = item.get("url", "")
        props = item.get("properties", {})

        # Find title property
        title = "(untitled)"
        for _pname, pval in props.items():
            ptype = pval.get("type")
            if ptype == "title":
                title = _plain_text(pval.get("title", [])) or "(untitled)"
                break

        lines.append(f"• {title}")
        lines.append(f"  ID:  {item_id}")
        lines.append(f"  URL: {url}")

        # Show a few other properties
        shown = 0
        for pname, pval in props.items():
            ptype = pval.get("type")
            if ptype == "title":
                continue
            if shown >= 4:
                break
            if ptype == "checkbox":
                lines.append(f"  {pname}: {'✅' if pval.get('checkbox') else '☐'}")
            elif ptype == "select":
                sel = pval.get("select")
                lines.append(f"  {pname}: {sel['name'] if sel else '—'}")
            elif ptype == "multi_select":
                names = [s["name"] for s in pval.get("multi_select", [])]
                lines.append(f"  {pname}: {', '.join(names) or '—'}")
            elif ptype == "rich_text":
                lines.append(f"  {pname}: {_plain_text(pval.get('rich_text', [])) or '—'}")
            elif ptype == "number":
                lines.append(f"  {pname}: {pval.get('number', '—')}")
            elif ptype == "date":
                date = pval.get("date")
                lines.append(f"  {pname}: {date['start'] if date else '—'}")
            elif ptype == "people":
                people = [p.get("name", "?") for p in pval.get("people", [])]
                lines.append(f"  {pname}: {', '.join(people) or '—'}")
            elif ptype == "url":
                lines.append(f"  {pname}: {pval.get('url', '—')}")
            else:
                continue
            shown += 1
        lines.append("")

    if data.get("has_more"):
        lines.append("(More results available — increase page_size or use filters to narrow down)")

    return "\n".join(lines)


def notion_delete_block(block_id: str) -> str:
    """
    Archive (soft-delete) a Notion block or page.

    Args:
        block_id: The ID of the block or page to archive.

    Returns:
        Confirmation message.
    """
    _req("DELETE", f"/blocks/{block_id}")
    return f"✅ Block/page {block_id} has been archived."


# ---------------------------------------------------------------------------
# Tool registry entry
# ---------------------------------------------------------------------------

NOTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "notion_search",
            "description": (
                "Search Notion pages and databases accessible to the integration. "
                "Use this to find pages by name or list all available databases."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text. Pass empty string to list everything.",
                    },
                    "filter_type": {
                        "type": "string",
                        "enum": ["all", "page", "database"],
                        "description": "Filter to only pages, only databases, or all (default).",
                        "default": "all",
                    },
                },
                "required": ["query"],
            },
        },
        "impl": notion_search,
    },
    {
        "type": "function",
        "function": {
            "name": "notion_get_page",
            "description": (
                "Retrieve the full content of a Notion page including all its blocks. "
                "Returns the title, metadata, and all text content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "The Notion page ID (from notion_search results).",
                    },
                },
                "required": ["page_id"],
            },
        },
        "impl": notion_get_page,
    },
    {
        "type": "function",
        "function": {
            "name": "notion_create_page",
            "description": (
                "Create a new Notion page inside a database or as a child of another page."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "parent_id": {
                        "type": "string",
                        "description": "ID of the parent database or page.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Title for the new page.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Optional initial content (plain text paragraph).",
                        "default": "",
                    },
                    "parent_type": {
                        "type": "string",
                        "enum": ["database", "page"],
                        "description": "Whether parent_id refers to a database (default) or page.",
                        "default": "database",
                    },
                },
                "required": ["parent_id", "title"],
            },
        },
        "impl": notion_create_page,
    },
    {
        "type": "function",
        "function": {
            "name": "notion_append_blocks",
            "description": (
                "Append text content to an existing Notion page. "
                "Supports multiple block types: paragraphs, headings, lists, todos, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "The Notion page ID to append to.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to append. Separate multiple blocks with blank lines.",
                    },
                    "block_type": {
                        "type": "string",
                        "enum": [
                            "paragraph",
                            "heading_1",
                            "heading_2",
                            "heading_3",
                            "bulleted_list_item",
                            "numbered_list_item",
                            "to_do",
                            "quote",
                            "callout",
                        ],
                        "description": "Block type for the appended content.",
                        "default": "paragraph",
                    },
                },
                "required": ["page_id", "text"],
            },
        },
        "impl": notion_append_blocks,
    },
    {
        "type": "function",
        "function": {
            "name": "notion_update_page",
            "description": (
                "Update Notion page properties such as status, checkbox, tags, dates, etc. "
                "Useful for marking tasks done, changing priority, updating labels."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "The Notion page ID to update.",
                    },
                    "properties": {
                        "type": "string",
                        "description": (
                            "JSON string of Notion property values to update. "
                            'Example: \'{"Done": {"checkbox": true}}\' '
                            'or \'{"Status": {"select": {"name": "Done"}}}\''
                        ),
                    },
                },
                "required": ["page_id", "properties"],
            },
        },
        "impl": notion_update_page,
    },
    {
        "type": "function",
        "function": {
            "name": "notion_query_database",
            "description": (
                "Query a Notion database to list or filter its entries. "
                "Supports Notion filter and sort syntax."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "database_id": {
                        "type": "string",
                        "description": "The Notion database ID.",
                    },
                    "filter_json": {
                        "type": "string",
                        "description": "Optional Notion filter JSON string.",
                        "default": "",
                    },
                    "sorts_json": {
                        "type": "string",
                        "description": "Optional Notion sorts JSON array string.",
                        "default": "",
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "Number of results to return (1-100, default 20).",
                        "default": 20,
                    },
                },
                "required": ["database_id"],
            },
        },
        "impl": notion_query_database,
    },
    {
        "type": "function",
        "function": {
            "name": "notion_delete_block",
            "description": (
                "Archive (soft-delete) a Notion block or page. "
                "Archived items can be restored from Notion's trash."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "block_id": {
                        "type": "string",
                        "description": "The block or page ID to archive.",
                    },
                },
                "required": ["block_id"],
            },
        },
        "impl": notion_delete_block,
    },
]
