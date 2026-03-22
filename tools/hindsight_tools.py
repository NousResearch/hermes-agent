"""Hindsight tools for persistent long-term memory.

Registers three tools for structured memory operations:

  hindsight_retain   — store information to long-term memory
  hindsight_recall   — search memories with multi-strategy retrieval
  hindsight_reflect  — synthesize a reasoned answer from stored memories

Hindsight extracts structured facts, resolves entities, builds a
knowledge graph, and uses cross-encoder reranking for retrieval.

Configuration is read from ~/.hindsight/config.json (written by
'hermes hindsight setup'), with environment variable fallbacks:
  HINDSIGHT_API_KEY   — API key (required for Hindsight Cloud)
  HINDSIGHT_BANK_ID   — memory bank identifier (required)
  HINDSIGHT_BUDGET    — recall budget: low/mid/high (default: mid)
  HINDSIGHT_API_URL   — API endpoint (default: https://api.hindsight.vectorize.io)
"""

import json
import logging
import queue
import threading

logger = logging.getLogger(__name__)


def _run_in_thread(fn, timeout: float = 30.0):
    """Run a zero-argument callable in a fresh daemon thread with no inherited event loop.

    The hindsight-client SDK uses aiohttp whose ``_run_async`` helper calls
    ``asyncio.get_event_loop()``.  If called from inside a running asyncio event
    loop (as Hermes does), the getter can return the main loop and
    ``loop.run_until_complete()`` fails.  Worse, an ``aiohttp.ClientSession``
    is bound to the loop that was active when the ``Hindsight`` client was
    created; reusing that singleton from worker threads with different loops
    causes intermittent "Timeout context manager should be used inside a task"
    errors.

    The solution: each call receives a zero-argument lambda that also creates
    a fresh Hindsight client (see ``_fresh_client``).  ``asyncio.set_event_loop(None)``
    is called at the start of the thread so aiohttp always sees a clean slate
    and creates its ClientSession bound to the correct new event loop.
    """
    result_q: queue.Queue = queue.Queue(maxsize=1)

    def _run() -> None:
        import asyncio as _asyncio
        _asyncio.set_event_loop(None)   # no inherited loop
        try:
            result_q.put(("ok", fn()))
        except Exception as exc:
            result_q.put(("err", exc))

    t = threading.Thread(target=_run, daemon=True, name="hindsight-tool-call")
    t.start()
    kind, value = result_q.get(timeout=timeout)
    if kind == "err":
        raise value
    return value


def _fresh_client():
    """Create a new Hindsight client (not the cached singleton).

    Called inside worker threads so aiohttp binds its ClientSession to the
    thread-local event loop, not the main asyncio loop.
    """
    cfg = _get_config()
    if cfg is None or not cfg.enabled:
        raise RuntimeError("Hindsight is not configured.")
    if cfg.mode == "local":
        try:
            from hindsight import HindsightEmbedded
        except ImportError:
            raise RuntimeError("hindsight-all is not installed.")
        return HindsightEmbedded(
            profile=cfg.local_profile,
            llm_provider=cfg.llm_provider,
            llm_api_key=cfg.llm_api_key or "",
            llm_model=cfg.llm_model,
            llm_base_url=cfg.llm_base_url,
        )
    if not cfg.api_key:
        raise RuntimeError("Hindsight is not configured.")
    try:
        from hindsight_client import Hindsight
    except ImportError:
        raise RuntimeError("hindsight-client is not installed.")
    return Hindsight(api_key=cfg.api_key, base_url=cfg.base_url, timeout=30.0)

# ── Lazy state ──

_config = None

# Session manager injected by AIAgent when Hindsight is active
_session_manager = None


def set_session_context(session_manager) -> None:
    """Register the active HindsightSessionManager.

    Called by AIAgent._activate_hindsight() so the availability check
    can confirm Hindsight is fully initialised for this session.
    """
    global _session_manager
    _session_manager = session_manager


def clear_session_context() -> None:
    """Clear session context (for testing or shutdown)."""
    global _session_manager
    _session_manager = None


def _get_config():
    """Lazily load HindsightClientConfig (reads ~/.hindsight/config.json)."""
    global _config
    if _config is not None:
        return _config
    try:
        from hindsight_integration.client import HindsightClientConfig
        _config = HindsightClientConfig.from_global_config()
    except Exception as e:
        logger.debug("Failed to load HindsightClientConfig: %s", e)
        _config = None
    return _config


def _resolve_bank_id() -> str | None:
    cfg = _get_config()
    return cfg.bank_id if cfg else None


def _resolve_budget() -> str:
    cfg = _get_config()
    return cfg.budget if cfg else "mid"


# ── Availability check ──

def _check_hindsight_available() -> bool:
    """Tool is available when Hindsight is active (session manager set) or configured."""
    if _session_manager is not None:
        return True
    cfg = _get_config()
    if cfg is None:
        return False
    if cfg.mode == "local":
        return bool(cfg.llm_api_key and cfg.bank_id)
    return bool(cfg.api_key and cfg.bank_id)


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
            "context": {
                "type": "string",
                "description": (
                    "Short label describing the source or situation "
                    "(e.g. 'user preference', 'project decision', 'meeting note'). "
                    "Providing context significantly improves memory quality and retrieval."
                ),
            },
        },
        "required": ["content"],
    },
}


def _handle_hindsight_retain(args: dict, **kw) -> str:
    content = args.get("content", "")
    if not content:
        return json.dumps({"error": "Missing required parameter: content"})
    context = args.get("context") or None
    bank_id = _resolve_bank_id()
    if not bank_id:
        return json.dumps({"error": "Hindsight is not configured."})
    try:
        _run_in_thread(lambda: _fresh_client().retain(bank_id=bank_id, content=content, context=context))
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
    bank_id = _resolve_bank_id()
    if not bank_id:
        return json.dumps({"error": "Hindsight is not configured."})
    try:
        budget = _resolve_budget()
        response = _run_in_thread(
            lambda: _fresh_client().recall(bank_id=bank_id, query=query, budget=budget)
        )
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
    bank_id = _resolve_bank_id()
    if not bank_id:
        return json.dumps({"error": "Hindsight is not configured."})
    try:
        budget = _resolve_budget()
        response = _run_in_thread(
            lambda: _fresh_client().reflect(bank_id=bank_id, query=query, budget=budget)
        )
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
