"""knowledge_search / knowledge_sync / knowledge_health — Hermes tools (Step 5).

These are the ONLY seam between Hermes and retrieval. They call
KnowledgeService, which hides the backend entirely.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)


def _service():
    from packages.knowledge import get_knowledge_service

    return get_knowledge_service()


def _knowledge_enabled() -> bool:
    try:
        from packages.knowledge.config import KnowledgeConfig

        return bool(KnowledgeConfig.load().enabled)
    except Exception:
        return False


# ---------------------------------------------------------------- search
def knowledge_search(query: str, limit: int = 5, workspace: Optional[str] = None,
                     filters: Optional[Dict[str, Any]] = None,
                     mode: str = "search", task_id: Optional[str] = None) -> str:
    if not (query or "").strip():
        return tool_error("query is required")
    try:
        svc = _service()
        if mode == "similar":
            res = svc.find_similar(query, limit=limit, workspace=workspace)
        elif mode == "retrieve":
            res = svc.retrieve(query, limit=limit, workspace=workspace, filters=filters)
        else:
            res = svc.search(query, limit=limit, workspace=workspace, filters=filters)
        payload = res.to_dict()
        payload["success"] = not res.error or bool(res.chunks)
        if not res.chunks:
            payload["hint"] = (
                "No indexed knowledge matched. Run knowledge_sync to index a "
                "vault/repo, or answer from general reasoning and say so."
            )
        return json.dumps(payload, ensure_ascii=False)
    except Exception as exc:
        logger.exception("knowledge_search failed")
        return tool_error(f"knowledge_search failed: {exc}")


# ------------------------------------------------------------------ sync
def knowledge_sync(path: str = "", source: str = "markdown",
                   workspace: Optional[str] = None, include_code: bool = False,
                   task_id: Optional[str] = None) -> str:
    try:
        from packages.knowledge.sync import DocumentSynchronizer

        svc = _service()
        syncer = DocumentSynchronizer(svc)
        if not path:
            return json.dumps({"success": True, "sources": syncer.sync_configured()},
                              ensure_ascii=False)
        if source == "obsidian":
            rep = syncer.sync_obsidian(path, workspace)
        elif source == "git":
            rep = syncer.sync_git_repo(path, workspace, include_code)
        elif source == "mkdocs":
            rep = syncer.sync_mkdocs(path, workspace)
        elif source == "conversations":
            rep = syncer.sync_conversations(workspace=workspace)
        else:
            rep = syncer.sync_path(path, source, workspace, include_code)
        return json.dumps({"success": True, "report": rep.to_dict()}, ensure_ascii=False)
    except Exception as exc:
        logger.exception("knowledge_sync failed")
        return tool_error(f"knowledge_sync failed: {exc}")


# ---------------------------------------------------------------- health
def knowledge_health(task_id: Optional[str] = None) -> str:
    try:
        return json.dumps({"success": True, **_service().health()}, ensure_ascii=False)
    except Exception as exc:
        return tool_error(f"knowledge_health failed: {exc}")


registry.register(
    name="knowledge_search",
    toolset="knowledge",
    emoji="📚",
    schema={
        "name": "knowledge_search",
        "description": (
            "Semantic search over the user's indexed long-term knowledge: Obsidian "
            "notes, project docs, READMEs, architecture decisions (ADRs), meeting "
            "notes, code snippets, PDFs and past conversations. Call this BEFORE "
            "answering any question that depends on the user's own material rather "
            "than general knowledge, then cite the returned sources as [n]. Returns "
            "answer, sources, chunks, confidence, provider and elapsedTime. The "
            "retrieval backend is pluggable and invisible to you."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language query."},
                "limit": {"type": "integer", "default": 5,
                          "description": "Top K chunks to return (1-25)."},
                "workspace": {"type": "string",
                              "description": "Knowledge workspace/collection to search."},
                "filters": {
                    "type": "object",
                    "description": "Optional filters: source (obsidian|git|mkdocs|"
                                   "markdown|pdf|conversation), path_prefix, score_threshold.",
                },
                "mode": {"type": "string", "enum": ["search", "retrieve", "similar"],
                         "default": "search",
                         "description": "search=chunks only; retrieve=chunks+answer; "
                                        "similar=find documents similar to the given document id."},
            },
            "required": ["query"],
        },
    },
    handler=lambda args, **kw: knowledge_search(
        query=args.get("query", ""),
        limit=int(args.get("limit", 5) or 5),
        workspace=args.get("workspace"),
        filters=args.get("filters"),
        mode=args.get("mode", "search"),
        task_id=kw.get("task_id"),
    ),
    check_fn=_knowledge_enabled,
)

registry.register(
    name="knowledge_sync",
    toolset="knowledge",
    emoji="🔄",
    schema={
        "name": "knowledge_sync",
        "description": (
            "Index or re-index knowledge sources into the retrieval backend. "
            "Incremental: only new, changed and deleted files are pushed. With no "
            "path, syncs every source configured under knowledge.sync_sources."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to index. Empty = configured sources."},
                "source": {"type": "string",
                           "enum": ["obsidian", "git", "mkdocs", "markdown", "pdf", "conversations"],
                           "default": "markdown"},
                "workspace": {"type": "string"},
                "include_code": {"type": "boolean", "default": False,
                                 "description": "Also index source-code files."},
            },
        },
    },
    handler=lambda args, **kw: knowledge_sync(
        path=args.get("path", ""),
        source=args.get("source", "markdown"),
        workspace=args.get("workspace"),
        include_code=bool(args.get("include_code", False)),
        task_id=kw.get("task_id"),
    ),
    check_fn=_knowledge_enabled,
)

registry.register(
    name="knowledge_health",
    toolset="knowledge",
    emoji="🩺",
    schema={
        "name": "knowledge_health",
        "description": "Health, cache and provider status of the knowledge subsystem.",
        "parameters": {"type": "object", "properties": {}},
    },
    handler=lambda args, **kw: knowledge_health(task_id=kw.get("task_id")),
    check_fn=_knowledge_enabled,
)
