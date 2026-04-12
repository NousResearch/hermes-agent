#!/usr/bin/env python3
"""
OpenViking Tool - Agent-native context database integration.

Provides semantic search, memory storage, and content retrieval
against a local or remote OpenViking server.

Environment:
    OPENVIKING_BASE_URL  (default: http://127.0.0.1:1933)
    OPENVIKING_API_KEY   (optional, defaults to ov.conf root_api_key)
    OPENVIKING_ACCOUNT   (default: default)
    OPENVIKING_USER      (default: hermes)
"""

import json
import logging
import os
import subprocess
from typing import Any, Dict, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://127.0.0.1:1933"
_DEFAULT_ACCOUNT = "default"
_DEFAULT_USER = "hermes"


def _get_ov_conf_api_key() -> Optional[str]:
    """Try to read root_api_key from ~/.openviking/ov.conf"""
    try:
        conf_path = os.path.expanduser("~/.openviking/ov.conf")
        if os.path.exists(conf_path):
            with open(conf_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("server", {}).get("root_api_key")
    except Exception:
        pass
    return None


def _base_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("OPENVIKING_BASE_URL", _DEFAULT_BASE_URL)
    api_key = env.get("OPENVIKING_API_KEY") or _get_ov_conf_api_key()
    if api_key:
        env["OPENVIKING_API_KEY"] = api_key
    return env


def _run_ov(*args: str) -> Dict[str, Any]:
    """Run the `ov` CLI and return parsed JSON or text output."""
    env = _base_env()
    account = env.get("OPENVIKING_ACCOUNT", _DEFAULT_ACCOUNT)
    user = env.get("OPENVIKING_USER", _DEFAULT_USER)

    cmd = ["/root/.venvs/openviking/bin/ov"]
    cmd.extend(args)
    cmd.extend(["--account", account, "--user", user, "--output", "json"])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        if result.returncode != 0:
            return {
                "success": False,
                "error": result.stderr.strip() or f"ov exited with code {result.returncode}",
                "stdout": result.stdout,
            }
        stdout = result.stdout.strip()
        # ov --output json prefixes the command line; skip it and parse the JSON body.
        if stdout:
            lines = stdout.splitlines()
            json_text = stdout
            if lines and lines[0].startswith("cmd:"):
                json_text = "\n".join(lines[1:]).strip()
            if json_text:
                try:
                    parsed = json.loads(json_text)
                    return {"success": True, "data": parsed}
                except json.JSONDecodeError:
                    return {"success": True, "data": json_text}
            return {"success": True, "data": None}
        return {"success": True, "data": None}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "OpenViking CLI timed out after 30s"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _check_openviking_available() -> bool:
    """Check whether the OpenViking server responds to health checks."""
    import urllib.request
    base_url = os.getenv("OPENVIKING_BASE_URL", _DEFAULT_BASE_URL)
    api_key = os.getenv("OPENVIKING_API_KEY") or _get_ov_conf_api_key()
    try:
        req = urllib.request.Request(f"{base_url}/api/v1/debug/health", method="GET")
        if api_key:
            req.add_header("X-API-Key", api_key)
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def openviking_search(query: str, n: int = 5, task_id: str = None) -> str:
    """Semantic search in OpenViking memory and resources."""
    if not query:
        return json.dumps({"success": False, "error": "query is required"})
    result = _run_ov("search", query, "-n", str(n))
    return json.dumps(result, ensure_ascii=False, default=str)


def openviking_read(uri: str, level: str = "read", task_id: str = None) -> str:
    """Read content from an OpenViking URI.

    level: "read" (raw text), "overview" (L1 summary), or "abstract" (L0 summary).
    """
    if not uri:
        return json.dumps({"success": False, "error": "uri is required"})
    cmd = {
        "read": "read",
        "overview": "overview",
        "abstract": "abstract",
    }.get(level, "read")
    result = _run_ov(cmd, uri)
    return json.dumps(result, ensure_ascii=False, default=str)


def openviking_add_memory(content: str, task_id: str = None) -> str:
    """Add a memory entry to OpenViking.

    Content can be plain text (stored as a user message) or a JSON object/array
    of {role, content} messages.
    """
    if not content:
        return json.dumps({"success": False, "error": "content is required"})
    result = _run_ov("add-memory", content)
    return json.dumps(result, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="openviking_search",
    toolset="openviking",
    schema={
        "name": "openviking_search",
        "description": "Semantic search in the OpenViking context database for past memories, resources, and sessions.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query.",
                },
                "n": {
                    "type": "integer",
                    "description": "Number of top results to return (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    handler=lambda args, **kw: openviking_search(
        query=args.get("query", ""),
        n=args.get("n", 5),
        task_id=kw.get("task_id"),
    ),
    check_fn=_check_openviking_available,
    requires_env=[],
    description="Semantic search in OpenViking memory and resources.",
    emoji="🛡️",
)

registry.register(
    name="openviking_read",
    toolset="openviking",
    schema={
        "name": "openviking_read",
        "description": "Read raw text, overview, or abstract content from an OpenViking URI.",
        "parameters": {
            "type": "object",
            "properties": {
                "uri": {
                    "type": "string",
                    "description": "OpenViking URI to read, e.g. viking://user/hermes/memories/.abstract.md",
                },
                "level": {
                    "type": "string",
                    "enum": ["read", "overview", "abstract"],
                    "description": "Content level: read=raw text, overview=L1 summary, abstract=L0 summary.",
                    "default": "read",
                },
            },
            "required": ["uri"],
        },
    },
    handler=lambda args, **kw: openviking_read(
        uri=args.get("uri", ""),
        level=args.get("level", "read"),
        task_id=kw.get("task_id"),
    ),
    check_fn=_check_openviking_available,
    requires_env=[],
    description="Read content from an OpenViking URI.",
    emoji="🛡️",
)

registry.register(
    name="openviking_add_memory",
    toolset="openviking",
    schema={
        "name": "openviking_add_memory",
        "description": "Add a memory entry to OpenViking. Plain text is stored as a user message; JSON objects/arrays are stored as structured messages.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Memory content to store. Can be plain text or JSON message(s).",
                },
            },
            "required": ["content"],
        },
    },
    handler=lambda args, **kw: openviking_add_memory(
        content=args.get("content", ""),
        task_id=kw.get("task_id"),
    ),
    check_fn=_check_openviking_available,
    requires_env=[],
    description="Add a memory entry to OpenViking.",
    emoji="🛡️",
)
