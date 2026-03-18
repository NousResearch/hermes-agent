"""
Cognitive Memory Tool - Semantic recall and intelligent storage.

Provides the agent with semantic memory capabilities:
  - recall: Find relevant memories using natural language queries
  - store: Intelligently store new information with auto-classification
  - forget: Explicitly forget memories by scope or content
  - status: Get memory system health and statistics

Unlike the basic memory tool (MEMORY.md), cognitive memory uses
vector embeddings for semantic search and automatically handles
categorization, importance scoring, and contradiction detection.
"""

import json
import logging
from typing import Any, Dict, Optional

from cognitive_memory.encoding import encode
from cognitive_memory.extraction import ForgettingManager, extract_facts
from cognitive_memory.recall import RecallEngine
from cognitive_memory.store import CognitiveStore

logger = logging.getLogger(__name__)


def cognitive_memory_tool(
    action: str,
    query: str = None,
    content: str = None,
    scope: str = "/",
    importance: Optional[float] = None,
    categories: Optional[list] = None,
    limit: int = 5,
    engine: Optional[RecallEngine] = None,
    store: Optional[CognitiveStore] = None,
    forgetting: Optional[ForgettingManager] = None,
) -> str:
    """
    Single entry point for cognitive memory operations.

    Returns JSON string with results.
    """
    if engine is None or store is None:
        return json.dumps({
            "success": False,
            "error": "Cognitive memory is not available. Check config or API key.",
        }, ensure_ascii=False)

    if action == "recall":
        result = _handle_recall(engine, query, scope, categories, limit)
    elif action == "store":
        result = _handle_store(engine, store, content, scope, importance, categories)
    elif action == "forget":
        result = _handle_forget(store, query, scope)
    elif action == "status":
        result = _handle_status(store)
    else:
        return json.dumps({
            "success": False,
            "error": f"Unknown action '{action}'. Use: recall, store, forget, status",
        }, ensure_ascii=False)

    # Trigger forgetting cycle after any successful action (if due)
    if forgetting:
        try:
            forgetting.maybe_run_cycle(exempt_scopes=["/user"])
        except Exception as e:
            logger.debug("Forgetting cycle failed (non-fatal): %s", e)

    return result


def _handle_recall(
    engine: RecallEngine,
    query: Optional[str],
    scope: str,
    categories: Optional[list],
    limit: int,
) -> str:
    if not query:
        return json.dumps({
            "success": False,
            "error": "Query is required for recall action.",
        }, ensure_ascii=False)

    try:
        results = engine.recall(
            query=query,
            limit=limit,
            scope=scope if scope != "/" else None,
            categories=categories,
        )
    except Exception as e:
        logger.warning("Recall failed: %s", e)
        return json.dumps({
            "success": False,
            "error": f"Recall failed (embedding API may be down): {e}",
        }, ensure_ascii=False)

    memories = []
    for sm in results:
        memories.append({
            "id": sm.memory.id,
            "content": sm.memory.content,
            "scope": sm.memory.scope,
            "categories": sm.memory.categories,
            "importance": round(sm.memory.importance, 3),
            "score": round(sm.score, 3),
            "similarity": round(sm.similarity, 3),
            "match_reasons": sm.match_reasons,
        })

    return json.dumps({
        "success": True,
        "action": "recall",
        "query": query,
        "count": len(memories),
        "memories": memories,
    }, ensure_ascii=False)


def _handle_store(
    engine: RecallEngine,
    store: CognitiveStore,
    content: Optional[str],
    scope: str,
    importance: Optional[float],
    categories: Optional[list],
) -> str:
    if not content:
        return json.dumps({
            "success": False,
            "error": "Content is required for store action.",
        }, ensure_ascii=False)

    # Step 1: Embed content and find similar existing memories BEFORE encoding
    embedding = None
    candidates = []
    try:
        embedding = engine._embedder.embed_text(content)
        candidates = store.search_similar(embedding, threshold=0.5, limit=5)
    except Exception as e:
        logger.warning("Embedding failed during store (will store without): %s", e)

    # Step 2: Encode with candidates so contradiction detection actually works
    encoding = encode(content, candidates=candidates if candidates else None)

    # Step 3: Dedup — skip if near-identical memory exists AND no contradiction detected
    if not encoding.contradictions:
        for sm in candidates:
            if sm.similarity >= 0.95:
                return json.dumps({
                    "success": True,
                    "action": "store",
                    "memory_id": sm.memory.id,
                    "duplicate": True,
                    "existing_content": sm.memory.content[:100],
                    "similarity": round(sm.similarity, 3),
                }, ensure_ascii=False)
    final_categories = categories or encoding.categories
    final_importance = importance if importance is not None else encoding.importance

    # Step 4: Store the memory
    memory_id = store.add_memory(
        content=content,
        embedding=embedding,
        scope=scope,
        importance=final_importance,
        categories=final_categories,
    )

    # Step 5: Supersede contradicted memories
    superseded = []
    for c in encoding.contradictions:
        if c.existing_memory:
            store.soft_delete(c.existing_memory.id)
            superseded.append(c.existing_memory.id)

    response = {
        "success": True,
        "action": "store",
        "memory_id": memory_id,
        "categories": final_categories,
        "importance": round(final_importance, 3),
    }

    if encoding.contradictions:
        response["contradictions"] = [
            {
                "existing_id": c.existing_memory.id if c.existing_memory else None,
                "existing_content": c.existing_memory.content[:100] if c.existing_memory else None,
                "confidence": round(c.confidence, 3),
                "reason": c.reason,
                "superseded": c.existing_memory.id in superseded if c.existing_memory else False,
            }
            for c in encoding.contradictions
        ]

    # Related memories (exclude self and superseded)
    if candidates:
        superseded_set = set(superseded)
        related = [sm for sm in candidates if sm.memory.id not in superseded_set][:3]
        if related:
            response["related"] = [
                {
                    "id": sm.memory.id,
                    "content": sm.memory.content[:100],
                    "similarity": round(sm.similarity, 3),
                }
                for sm in related
            ]

    return json.dumps(response, ensure_ascii=False)


def _handle_forget(
    store: CognitiveStore,
    query: Optional[str],
    scope: str,
) -> str:
    if scope and scope != "/":
        count = store.soft_delete_by_scope(scope)
        return json.dumps({
            "success": True,
            "action": "forget",
            "forgotten_count": count,
            "scope": scope,
        }, ensure_ascii=False)

    if query:
        # Find memories matching query text and forget them
        active = store.get_all_active()
        forgotten = 0
        for mem in active:
            if query.lower() in mem.content.lower():
                store.soft_delete(mem.id)
                forgotten += 1

        return json.dumps({
            "success": True,
            "action": "forget",
            "forgotten_count": forgotten,
            "query": query,
        }, ensure_ascii=False)

    return json.dumps({
        "success": False,
        "error": "Provide either a query or a non-root scope to forget memories.",
    }, ensure_ascii=False)


def _handle_status(store: CognitiveStore) -> str:
    active = store.count(include_forgotten=False)
    total = store.count(include_forgotten=True)
    forgotten = total - active

    return json.dumps({
        "success": True,
        "action": "status",
        "active_memories": active,
        "forgotten_memories": forgotten,
        "total_memories": total,
    }, ensure_ascii=False)


def check_cognitive_memory_requirements() -> bool:
    """Cognitive memory requires litellm for embeddings."""
    try:
        import litellm
        return True
    except ImportError:
        return False


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

COGNITIVE_MEMORY_SCHEMA = {
    "name": "cognitive_recall",
    "description": (
        "Semantic memory system with intelligent recall and storage. "
        "Uses vector embeddings to find relevant memories by meaning, not just keywords.\n\n"
        "ACTIONS:\n"
        "- recall: Search memories by meaning. Use natural language queries to find relevant "
        "information from past conversations and stored knowledge.\n"
        "- store: Save important information with automatic categorization and importance scoring. "
        "Detects contradictions with existing memories.\n"
        "- forget: Remove memories by scope prefix or content match.\n"
        "- status: Check memory system health (active/forgotten counts).\n\n"
        "USE CASES:\n"
        "- Before answering questions, recall relevant context from past sessions\n"
        "- Store user preferences, project facts, environment details\n"
        "- Update outdated information (store will flag contradictions)\n"
        "- Forget obsolete project-specific memories when switching contexts\n\n"
        "This complements the basic memory tool (MEMORY.md/USER.md) by providing "
        "unlimited, searchable, auto-organized semantic memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["recall", "store", "forget", "status"],
                "description": "The action to perform.",
            },
            "query": {
                "type": "string",
                "description": (
                    "For 'recall': natural language search query. "
                    "For 'forget': text to match against memory content."
                ),
            },
            "content": {
                "type": "string",
                "description": "Content to store (required for 'store' action).",
            },
            "scope": {
                "type": "string",
                "description": (
                    "Hierarchical scope prefix (e.g., '/user', '/projects/myapp'). "
                    "Defaults to '/'. For 'forget', deletes all memories under this scope."
                ),
            },
            "importance": {
                "type": "number",
                "description": "Override auto-estimated importance (0.0-1.0). Usually auto-detected.",
            },
            "categories": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Override auto-detected categories. Options: "
                    "fact, preference, procedure, observation, convention, "
                    "environment, correction, skill."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Max results for recall (default 5).",
            },
        },
        "required": ["action"],
    },
}


# --- Registry ---
from tools.registry import registry

registry.register(
    name="cognitive_recall",
    toolset="memory",
    schema=COGNITIVE_MEMORY_SCHEMA,
    handler=lambda args, **kw: cognitive_memory_tool(
        action=args.get("action", ""),
        query=args.get("query"),
        content=args.get("content"),
        scope=args.get("scope", "/"),
        importance=args.get("importance"),
        categories=args.get("categories"),
        limit=args.get("limit", 5),
        engine=kw.get("engine"),
        store=kw.get("cognitive_store"),
        forgetting=kw.get("forgetting"),
    ),
    check_fn=check_cognitive_memory_requirements,
)
