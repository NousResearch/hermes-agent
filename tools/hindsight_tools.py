"""Hindsight tools for persistent long-term memory.

Registers three tools for structured memory operations:

  hindsight_retain   — store information to long-term memory
  hindsight_recall   — search memories with multi-strategy retrieval
  hindsight_reflect  — synthesize a reasoned answer from stored memories

Hindsight extracts structured facts, resolves entities, builds a
knowledge graph, and uses cross-encoder reranking for retrieval.

Configuration via environment variables:
  HINDSIGHT_API_URL   — API endpoint (default: https://api.hindsight.vectorize.io)
  HINDSIGHT_API_KEY   — API key (required for Hindsight Cloud)
  HINDSIGHT_BANK_ID   — memory bank identifier (required)
  HINDSIGHT_BUDGET    — recall budget: low/mid/high (default: mid)
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

# ── Configuration ──

_DEFAULT_API_URL = "https://api.hindsight.vectorize.io"

_client = None
_bank_id: str | None = None
_budget: str = "mid"
_created_banks: set[str] = set()


def _get_client():
    """Lazily initialize the Hindsight client."""
    global _client, _bank_id, _budget

    if _client is not None:
        return _client

    try:
        from hindsight_client import Hindsight
    except ImportError:
        logger.debug("hindsight-client not installed, Hindsight tools unavailable")
        return None

    api_url = os.environ.get("HINDSIGHT_API_URL")
    api_key = os.environ.get("HINDSIGHT_API_KEY")
    _bank_id = os.environ.get("HINDSIGHT_BANK_ID")
    _budget = os.environ.get("HINDSIGHT_BUDGET", "mid")

    if not api_url and not api_key:
        logger.debug("No HINDSIGHT_API_URL or HINDSIGHT_API_KEY set, skipping")
        return None

    url = api_url or _DEFAULT_API_URL
    kwargs = {"base_url": url, "timeout": 30.0}
    if api_key:
        kwargs["api_key"] = api_key

    try:
        _client = Hindsight(**kwargs)
        logger.info("Hindsight client initialized (url: %s, bank: %s)", url, _bank_id)
        return _client
    except Exception as e:
        logger.warning("Failed to initialize Hindsight client: %s", e)
        return None


def _ensure_bank(bank_id: str) -> None:
    """Create the memory bank if not already created this session."""
    if bank_id in _created_banks:
        return
    client = _get_client()
    if not client:
        return
    try:
        client.create_bank(bank_id=bank_id, name=bank_id)
        _created_banks.add(bank_id)
    except Exception:
        _created_banks.add(bank_id)  # Likely already exists


def _resolve_bank_id() -> str | None:
    """Get the current bank ID from env."""
    return _bank_id or os.environ.get("HINDSIGHT_BANK_ID")


# ── Availability check ──

def _check_hindsight_available() -> bool:
    """Tool is available when hindsight-client is installed and configured."""
    if _get_client() is None:
        return False
    return _resolve_bank_id() is not None


# ── hindsight_retain ──

_RETAIN_SCHEMA = {
    "name": "hindsight_retain",
    "description": (
        "Store information to long-term memory for later retrieval. "
        "Use this to save important facts, user preferences, decisions, "
        "or any information that should be remembered across sessions. "
        "Hindsight automatically extracts structured facts, resolves entities, "
        "and indexes the memory for retrieval."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The information to store in memory.",
            },
        },
        "required": ["content"],
    },
}


def _handle_hindsight_retain(args: dict, **kw) -> str:
    content = args.get("content", "")
    if not content:
        return json.dumps({"error": "Missing required parameter: content"})
    client = _get_client()
    bank_id = _resolve_bank_id()
    if not client or not bank_id:
        return json.dumps({"error": "Hindsight is not configured."})
    try:
        _ensure_bank(bank_id)
        client.retain(bank_id=bank_id, content=content)
        return json.dumps({"result": "Memory stored successfully."})
    except Exception as e:
        logger.error("Hindsight retain failed: %s", e)
        return json.dumps({"error": f"Failed to store memory: {e}"})


# ── hindsight_recall ──

_RECALL_SCHEMA = {
    "name": "hindsight_recall",
    "description": (
        "Search long-term memory for relevant information. "
        "Returns matching memories ranked by relevance using semantic search, "
        "keyword matching, entity graph traversal, and cross-encoder reranking. "
        "Use this to find previously stored facts, preferences, or context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in memory.",
            },
        },
        "required": ["query"],
    },
}


def _handle_hindsight_recall(args: dict, **kw) -> str:
    query = args.get("query", "")
    if not query:
        return json.dumps({"error": "Missing required parameter: query"})
    client = _get_client()
    bank_id = _resolve_bank_id()
    if not client or not bank_id:
        return json.dumps({"error": "Hindsight is not configured."})
    try:
        response = client.recall(bank_id=bank_id, query=query, budget=_budget)
        if not response.results:
            return json.dumps({"result": "No relevant memories found."})
        lines = [f"{i}. {r.text}" for i, r in enumerate(response.results, 1)]
        return json.dumps({"result": "\n".join(lines)})
    except Exception as e:
        logger.error("Hindsight recall failed: %s", e)
        return json.dumps({"error": f"Failed to search memory: {e}"})


# ── hindsight_reflect ──

_REFLECT_SCHEMA = {
    "name": "hindsight_reflect",
    "description": (
        "Synthesize a thoughtful answer from long-term memories. "
        "Unlike recall, this reasons across all stored memories to produce "
        "a coherent response. Use this when you need a summary, recommendation, "
        "or reasoned answer rather than raw facts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question to reflect on using stored memories.",
            },
        },
        "required": ["query"],
    },
}


def _handle_hindsight_reflect(args: dict, **kw) -> str:
    query = args.get("query", "")
    if not query:
        return json.dumps({"error": "Missing required parameter: query"})
    client = _get_client()
    bank_id = _resolve_bank_id()
    if not client or not bank_id:
        return json.dumps({"error": "Hindsight is not configured."})
    try:
        response = client.reflect(bank_id=bank_id, query=query, budget=_budget)
        return json.dumps({"result": response.text or "No relevant memories found."})
    except Exception as e:
        logger.error("Hindsight reflect failed: %s", e)
        return json.dumps({"error": f"Failed to reflect on memory: {e}"})


# ── Registration ──

from tools.registry import registry

registry.register(
    name="hindsight_retain",
    toolset="hindsight",
    schema=_RETAIN_SCHEMA,
    handler=_handle_hindsight_retain,
    check_fn=_check_hindsight_available,
    emoji="⚡",
)

registry.register(
    name="hindsight_recall",
    toolset="hindsight",
    schema=_RECALL_SCHEMA,
    handler=_handle_hindsight_recall,
    check_fn=_check_hindsight_available,
    emoji="⚡",
)

registry.register(
    name="hindsight_reflect",
    toolset="hindsight",
    schema=_REFLECT_SCHEMA,
    handler=_handle_hindsight_reflect,
    check_fn=_check_hindsight_available,
    emoji="⚡",
)
