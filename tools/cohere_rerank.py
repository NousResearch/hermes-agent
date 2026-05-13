"""Cohere Rerank tool — relevance-score a list of documents against a query.

Wraps ``cohere.ClientV2.rerank`` so the agent can ask Cohere to score and
sort an arbitrary list of candidate passages by how well they match a
query. This is the canonical "stage 2" reranker that goes after a coarse
retrieval pass (BM25, embedding search, web search) and before feeding
the top-k passages into the LLM context.

Gated on ``COHERE_API_KEY`` via the registry's ``check_fn``; stays hidden
from the model when the user hasn't configured Cohere.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)


def check_cohere_requirements() -> bool:
    """Return True when a Cohere API key is configured (env or .env)."""
    return bool(os.getenv("COHERE_API_KEY") or os.getenv("CO_API_KEY"))


def _resolve_api_key() -> str:
    return (os.getenv("COHERE_API_KEY") or os.getenv("CO_API_KEY") or "").strip()


COHERE_RERANK_SCHEMA: Dict[str, Any] = {
    "name": "cohere_rerank",
    "description": (
        "Rerank a list of candidate documents by their relevance to a "
        "query, using Cohere's rerank API. Returns the top_n results "
        "sorted by relevance score with their original indices preserved. "
        "Use this after a coarse retrieval step (web search, BM25, vector "
        "search) to pick the best passages before feeding them into the "
        "model's context. Do NOT use as a general search tool — it scores "
        "documents the caller provides; it does not retrieve new ones. "
        "Requires the Cohere API key."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user's query or information need.",
            },
            "documents": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Candidate passages to score. Each entry is treated as "
                    "a single document. Limit: 1000 documents per call; "
                    "shorter passages (200-1000 tokens) work best."
                ),
                "minItems": 1,
                "maxItems": 1000,
            },
            "top_n": {
                "type": "integer",
                "description": (
                    "Number of top results to return. Defaults to 5. "
                    "Set to the number of passages you want to feed into "
                    "the next prompt."
                ),
                "default": 5,
                "minimum": 1,
            },
            "model": {
                "type": "string",
                "description": (
                    "Cohere rerank model name. Defaults to 'rerank-v3.5' "
                    "(latest, multilingual). Older: 'rerank-english-v3.0', "
                    "'rerank-multilingual-v3.0'."
                ),
                "default": "rerank-v3.5",
            },
        },
        "required": ["query", "documents"],
    },
}


def cohere_rerank(
    query: str,
    documents: List[str],
    top_n: int = 5,
    model: str = "rerank-v3.5",
    task_id: str | None = None,
) -> str:
    """Call ``cohere.ClientV2.rerank`` and return the ranked results as JSON."""
    if not isinstance(query, str) or not query.strip():
        return tool_error("`query` must be a non-empty string.")
    if not isinstance(documents, list) or not documents:
        return tool_error("`documents` must be a non-empty list of strings.")
    if not all(isinstance(d, str) and d for d in documents):
        return tool_error("Each entry in `documents` must be a non-empty string.")
    if not isinstance(top_n, int) or top_n < 1:
        top_n = 5
    top_n = min(top_n, len(documents))

    api_key = _resolve_api_key()
    if not api_key:
        return tool_error(
            "COHERE_API_KEY is not set. Configure it via `hermes setup` or "
            "export it in your environment to enable cohere_rerank."
        )

    try:
        from agent.cohere_adapter import build_cohere_client
        client = build_cohere_client(api_key)
    except ImportError as exc:
        return tool_error(
            f"The 'cohere' package is not installed: {exc}. "
            "Install it with: pip install cohere>=5.13"
        )
    except Exception as exc:
        logger.exception("Failed to build Cohere client")
        return tool_error(f"Failed to build Cohere client: {exc}")

    try:
        response = client.rerank(
            query=query,
            documents=documents,
            model=model,
            top_n=top_n,
        )
    except Exception as exc:
        logger.exception("Cohere rerank call failed")
        return tool_error(f"Cohere rerank call failed: {exc}")

    # Normalize the SDK's results into a plain list of dicts. Each result
    # carries an index (into the input documents list) and a relevance
    # score in [0, 1]. We also echo back the matched document text so the
    # caller doesn't have to re-correlate by index.
    raw_results = getattr(response, "results", None)
    if raw_results is None and isinstance(response, dict):
        raw_results = response.get("results")
    if not isinstance(raw_results, list):
        return tool_error("Cohere returned no results — check the model name.")

    results: List[Dict[str, Any]] = []
    for item in raw_results:
        idx = (
            getattr(item, "index", None)
            if not isinstance(item, dict)
            else item.get("index")
        )
        score = (
            getattr(item, "relevance_score", None)
            if not isinstance(item, dict)
            else item.get("relevance_score")
        )
        if idx is None or score is None:
            continue
        try:
            idx_int = int(idx)
            score_f = float(score)
        except (TypeError, ValueError):
            continue
        if not (0 <= idx_int < len(documents)):
            continue
        results.append(
            {
                "index": idx_int,
                "relevance_score": score_f,
                "document": documents[idx_int],
            }
        )

    return json.dumps(
        {
            "success": True,
            "model": model,
            "query": query,
            "top_n": top_n,
            "results": results,
        }
    )


def _handle_cohere_rerank(args: Dict[str, Any], **kw: Any) -> str:
    return cohere_rerank(
        query=args.get("query", ""),
        documents=args.get("documents") or [],
        top_n=int(args.get("top_n") or 5),
        model=(args.get("model") or "rerank-v3.5").strip() or "rerank-v3.5",
        task_id=kw.get("task_id"),
    )


registry.register(
    name="cohere_rerank",
    toolset="cohere",
    schema=COHERE_RERANK_SCHEMA,
    handler=_handle_cohere_rerank,
    check_fn=check_cohere_requirements,
    requires_env=["COHERE_API_KEY"],
    is_async=False,
    emoji="📊",
)
