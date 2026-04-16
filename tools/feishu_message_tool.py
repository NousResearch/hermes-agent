"""Feishu Message Tool -- read/search Feishu messages via Feishu Open API.

Provides the ability to read and search Feishu messages from group chats and
direct messages. Supports filtering by message type, time range, and sender.

Usage:
    from tools.feishu_message_tool import feishu_message_tool, FEISHU_MESSAGE_SCHEMA
    
    # Search messages in a group
    result = feishu_message_tool({
        "action": "search",
        "chat_id": "oc_xxx",
        "query": "",
        "page_size": 50
    })
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

FEISHU_MESSAGE_SCHEMA = {
    "name": "feishu_message",
    "description": (
        "Read and search Feishu message history from group chats or direct messages.\n\n"
        "Supports:\n"
        "- Searching messages by keyword or time range\n"
        "- Filtering by message type (text, interactive/card, image, etc.)\n"
        "- Getting messages from a specific chat/conversation\n"
        "- Extracting card content from interactive card messages\n\n"
        "Use this when user asks to 'read group chat messages', 'search Feishu history', "
        "'get chat records', or similar requests.\n\n"
        "Note: chat_id can be found in the conversation metadata."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "list"],
                "description": "'search' (default) searches messages, 'list' returns recent messages from a chat"
            },
            "chat_id": {
                "type": "string",
                "description": "Feishu chat ID (e.g. 'oc_xxx'). For group chats starts with 'oc_', for DMs starts with 'ou_'."
            },
            "query": {
                "type": "string",
                "description": "Search keyword. Empty query returns all messages (subject to other filters)."
            },
            "message_type": {
                "type": "string",
                "description": "Filter by message type: 'text', 'interactive' (card), 'image', 'file', 'media'."
            },
            "start_time": {
                "type": "string",
                "description": "Start time in ISO 8601 format (e.g. '2026-04-14T00:00:00+08:00')."
            },
            "end_time": {
                "type": "string",
                "description": "End time in ISO 8601 format (e.g. '2026-04-14T23:59:59+08:00')."
            },
            "page_size": {
                "type": "integer",
                "description": "Number of messages to return per page (default 50, max 50)."
            },
            "page_token": {
                "type": "string",
                "description": "Pagination token from previous response for fetching next page."
            },
            "thread_id": {
                "type": "string",
                "description": "Thread/topic ID to get messages from (starts with 'omt_')."
            }
        },
        "required": []
    }
}


# ---------------------------------------------------------------------------
# Card Parser (subset of feishu_card_parser for extracting card content)
# ---------------------------------------------------------------------------

def _parse_card_content(content_str: str) -> Dict[str, Any]:
    """
    Parse Feishu card content from message.
    Returns dict with title, body_text, actions.
    """
    import json as _json
    
    # content_str might be wrapped in <card>...</card>
    match = re.search(r'<card[^>]*>(.*?)</card>', content_str, re.DOTALL)
    if match:
        content_str = match.group(1)
    
    # Unescape HTML entities
    content_str = content_str.replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"')
    
    try:
        card = _json.loads(content_str)
    except _json.JSONDecodeError:
        return {"error": "Invalid JSON", "raw": content_str[:200]}
    
    result = {
        "title": "",
        "body_text": "",
        "actions": [],
        "raw_keys": list(card.keys()) if isinstance(card, dict) else []
    }
    
    # Extract title
    header = card.get("header") or card.get("card_header") or {}
    if isinstance(header, dict):
        title = header.get("title") or header.get("property", {}).get("title") or {}
        if isinstance(title, dict):
            result["title"] = title.get("content") or title.get("text", "")
        elif isinstance(title, str):
            result["title"] = title
    
    # Extract text content by walking all nodes
    texts = []
    
    def walk(node):
        if isinstance(node, dict):
            tag = node.get("tag", "").lower()
            content = node.get("content") or node.get("text", "")
            
            if content and tag in (
                "text", "plain_text", "lark_md", "markdown", 
                "a", "div", "note"
            ):
                texts.append(content)
            
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)
    
    walk(card.get("body") or card.get("elements") or card)
    result["body_text"] = "\n".join(texts[:20])  # Limit to 20 lines
    
    # Extract action labels
    def walk_actions(node):
        if isinstance(node, dict):
            tag = node.get("tag", "").lower()
            if tag in ("button", "select_static", "overflow", "date_picker"):
                label = node.get("text") or node.get("name") or node.get("value", "")
                if label:
                    result["actions"].append(label)
            for v in node.values():
                walk_actions(v)
        elif isinstance(node, list):
            for item in node:
                walk_actions(item)
    
    walk_actions(card)
    
    return result


# ---------------------------------------------------------------------------
# Feishu API helpers
# ---------------------------------------------------------------------------

def _get_feishu_credentials() -> tuple:
    """Get Feishu app_id and app_secret from various sources."""
    import os
    
    # Try environment variables first
    app_id = os.environ.get("FEISHU_APP_ID")
    app_secret = os.environ.get("FEISHU_APP_SECRET")
    if app_id and app_secret:
        return app_id, app_secret
    
    # Try Hermes config.yaml
    try:
        config_path = os.path.expanduser("~/.hermes/config.yaml")
        if os.path.exists(config_path):
            with open(config_path) as f:
                content = f.read()
            id_match = re.search(r'feishu_app_id:\s*[\'"]?([\w-]+)', content, re.IGNORECASE)
            secret_match = re.search(r'feishu_app_secret:\s*[\'"]?([\w-]+)', content, re.IGNORECASE)
            if id_match and secret_match:
                return id_match.group(1), secret_match.group(1)
    except Exception:
        pass
    
    # Try OpenClaw config.json (main account)
    try:
        openclaw_config = os.path.expanduser("~/.openclaw/openclaw.json")
        if os.path.exists(openclaw_config):
            with open(openclaw_config) as f:
                config = json.loads(f.read())
            # OpenClaw stores feishu at channels.feishu.accounts.main
            feishu = config.get("channels", {}).get("feishu", {})
            accounts = feishu.get("accounts", {})
            main = accounts.get("main", {})
            app_id = main.get("appId")
            app_secret = main.get("appSecret")
            if app_id and app_secret:
                return app_id, app_secret
    except Exception:
        pass
    
    return None, None


def _get_feishu_token() -> Optional[str]:
    """Get Feishu tenant access token using app_id and app_secret."""
    import urllib.request
    
    app_id, app_secret = _get_feishu_credentials()
    if not app_id or not app_secret:
        return None
    
    # Get tenant access token via OAuth2
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    data = json.dumps({"app_id": app_id, "app_secret": app_secret}).encode()
    
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            if result.get("code") == 0:
                return result.get("tenant_access_token")
    except Exception as e:
        logger.warning(f"Failed to get Feishu token: {e}")
    
    return None


def _call_feishu_api(method: str, endpoint: str, token: str, data: Dict = None) -> Dict:
    """Make a call to Feishu Open API."""
    import urllib.request
    import urllib.parse
    
    base_url = "https://open.feishu.cn/open-apis"
    url = f"{base_url}{endpoint}"
    
    body = json.dumps(data).encode() if data else None
    
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            if result.get("code") != 0:
                logger.warning(f"Feishu API error: {result}")
            return result
    except Exception as e:
        logger.error(f"Feishu API call failed: {e}")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_search(args: Dict) -> str:
    """Search messages in a chat."""
    chat_id = args.get("chat_id")
    query = args.get("query", "")
    message_type = args.get("message_type")
    start_time = args.get("start_time")
    end_time = args.get("end_time")
    page_size = min(args.get("page_size", 50), 50)
    page_token = args.get("page_token")
    
    if not chat_id:
        return json.dumps({"error": "chat_id is required"})
    
    token = _get_feishu_token()
    if not token:
        return json.dumps({"error": "Feishu access token not found. Set FEISHU_ACCESS_TOKEN env var or feishu.access_token in config.yaml"})
    
    # Build request
    data = {
        "container": {"type": "chat", "chat_id": chat_id},
        "page_size": page_size,
    }
    
    if query:
        data["query"] = query
    
    if start_time:
        data["start_time"] = start_time
    
    if end_time:
        data["end_time"] = end_time
    
    if page_token:
        data["page_token"] = page_token
    
    # Filter by message type
    msg_type_map = {
        "text": "text",
        "interactive": "interactive", 
        "card": "interactive",
        "image": "image",
        "file": "file",
        "media": "media"
    }
    if message_type:
        mt = msg_type_map.get(message_type.lower())
        if mt:
            data["message_type"] = mt
    
    result = _call_feishu_api("POST", "/im/v1/messages/search", token, data)
    
    if "error" in result:
        return json.dumps(result)
    
    # Parse response
    items = result.get("data", {}).get("items", [])
    
    parsed_messages = []
    for item in items:
        msg_type = item.get("msg_type", "")
        content_str = item.get("body", {}).get("content", "")
        
        parsed = {
            "message_id": item.get("message_id"),
            "msg_type": msg_type,
            "create_time": item.get("create_time"),
            "sender": item.get("sender", {}).get("sender_type", ""),
            "sender_id": item.get("sender", {}).get("id", ""),
        }
        
        # Parse content based on type
        if msg_type == "text":
            try:
                content_obj = json.loads(content_str)
                parsed["text"] = content_obj.get("text", content_str)
            except:
                parsed["text"] = content_str
        elif msg_type == "interactive":
            card_info = _parse_card_content(content_str)
            parsed["card"] = card_info
        else:
            # For other types, just store raw content preview
            parsed["content"] = content_str[:500] if len(content_str) > 500 else content_str
        
        parsed_messages.append(parsed)
    
    has_more = result.get("data", {}).get("has_more", False)
    next_page_token = result.get("data", {}).get("next_page_token", "")
    
    return json.dumps({
        "messages": parsed_messages,
        "total": len(parsed_messages),
        "has_more": has_more,
        "next_page_token": next_page_token
    }, ensure_ascii=False)


def _handle_list(args: Dict) -> str:
    """Get recent messages from a chat."""
    chat_id = args.get("chat_id")
    thread_id = args.get("thread_id")
    page_size = min(args.get("page_size", 50), 50)
    page_token = args.get("page_token")
    start_time = args.get("start_time")
    end_time = args.get("end_time")
    
    if not chat_id:
        return json.dumps({"error": "chat_id is required"})
    
    token = _get_feishu_token()
    if not token:
        return json.dumps({"error": "Feishu access token not found"})
    
    # Build endpoint - use thread messages if thread_id provided
    # Note: container_id_type is required (chat or thread)
    if thread_id:
        endpoint = f"/im/v1/messages?container_id_type=thread&container_id={thread_id}&page_size={page_size}"
        if page_token:
            endpoint += f"&page_token={page_token}"
    else:
        endpoint = f"/im/v1/messages?container_id_type=chat&container_id={chat_id}&page_size={page_size}"
        if page_token:
            endpoint += f"&page_token={page_token}"
    
    result = _call_feishu_api("GET", endpoint, token)
    
    if "error" in result:
        return json.dumps(result)
    
    items = result.get("data", {}).get("items", [])
    
    parsed_messages = []
    for item in items:
        msg_type = item.get("msg_type", "")
        content_str = item.get("body", {}).get("content", "")
        
        parsed = {
            "message_id": item.get("message_id"),
            "msg_type": msg_type,
            "create_time": item.get("create_time"),
            "sender": item.get("sender", {}).get("sender_type", ""),
            "sender_id": item.get("sender", {}).get("id", ""),
            "update_time": item.get("update_time", ""),
        }
        
        if msg_type == "text":
            try:
                content_obj = json.loads(content_str)
                parsed["text"] = content_obj.get("text", content_str)
            except:
                parsed["text"] = content_str
        elif msg_type == "interactive":
            card_info = _parse_card_content(content_str)
            parsed["card"] = card_info
        else:
            parsed["content"] = content_str[:500] if len(content_str) > 500 else content_str
        
        parsed_messages.append(parsed)
    
    has_more = result.get("data", {}).get("has_more", False)
    next_page_token = result.get("data", {}).get("next_page_token", "")
    
    return json.dumps({
        "messages": parsed_messages,
        "total": len(parsed_messages),
        "has_more": has_more,
        "next_page_token": next_page_token
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def feishu_message_tool(args: Dict, **kw) -> str:
    """
    Handle feishu_message tool calls.
    
    Args:
        args: Dict with action, chat_id, query, etc.
        **kw: Additional keyword arguments (ignored)
    
    Returns:
        JSON string with results
    """
    action = args.get("action", "search")
    
    try:
        if action == "search":
            return _handle_search(args)
        elif action == "list":
            return _handle_list(args)
        else:
            return json.dumps({"error": f"Unknown action: {action}"})
    except Exception as e:
        logger.exception("feishu_message_tool failed")
        return json.dumps({"error": str(e)})


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="feishu_message",
    toolset="messaging",
    schema=FEISHU_MESSAGE_SCHEMA,
    handler=feishu_message_tool,
    emoji="💬",
)
