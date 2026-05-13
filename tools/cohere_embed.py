"""Cohere Embed tool — generate text embeddings via the Cohere v2 API.

Wraps ``cohere.ClientV2.embed`` as a first-class agent tool. The agent can
call this inside a reasoning loop to vectorize a batch of texts for
downstream similarity search, clustering, or hybrid retrieval. The tool is
gated on ``COHERE_API_KEY`` via the registry's ``check_fn`` so it stays
hidden from the model unless the user has configured Cohere.

Why this lives as a built-in tool rather than an auxiliary surface:
Hermes does not currently have a generic embedding plumbing layer. The
agent-facing tool keeps the integration self-contained and lets the model
opt in only when embedding is genuinely useful for the task at hand.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)


# Cohere documents the following input_type values for embed-v4.0 and
# embed-english-v3.0. Rejecting unknown values keeps malformed agent
# calls from leaking into the Cohere API and confusing debugging.
_VALID_INPUT_TYPES = {
    "search_document",
    "search_query",
    "classification",
    "clustering",
    "image",
}


def check_cohere_requirements() -> bool:
    """Return True when a Cohere API key is configured.

    Checks ``COHERE_API_KEY`` first, falling back to ``CO_API_KEY`` for
    parity with the official SDK's lookup order.
    """
    return bool(os.getenv("COHERE_API_KEY") or os.getenv("CO_API_KEY"))


def _resolve_api_key() -> str:
    return (os.getenv("COHERE_API_KEY") or os.getenv("CO_API_KEY") or "").strip()


COHERE_EMBED_SCHEMA: Dict[str, Any] = {
    "name": "cohere_embed",
    "description": (
        "Generate dense vector embeddings for one or more text passages "
        "using Cohere's embed API. Returns a list of float vectors and the "
        "embedding dimension. Use this when the user asks you to compare, "
        "cluster, or semantically search a batch of texts (for downstream "
        "cosine similarity, vector store ingestion, or de-duplication). "
        "Do NOT call this for ordinary question answering — use the model's "
        "context window instead. Requires the Cohere API key."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "texts": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of text passages to embed. Each entry should be a "
                    "self-contained chunk (paragraph, document section, or "
                    "search query). Limit: 96 texts per call."
                ),
                "minItems": 1,
                "maxItems": 96,
            },
            "input_type": {
                "type": "string",
                "description": (
                    "Embedding optimisation hint. Use 'search_document' for "
                    "passages you'll later search over, 'search_query' for "
                    "the user's query against those passages, 'classification' "
                    "for labelling tasks, or 'clustering' for grouping tasks."
                ),
                "enum": sorted(_VALID_INPUT_TYPES),
            },
            "model": {
                "type": "string",
                "description": (
                    "Cohere embed model name. Defaults to 'embed-v4.0' "
                    "(latest, multilingual, 1536-dim). Older alternatives: "
                    "'embed-english-v3.0', 'embed-multilingual-v3.0'."
                ),
                "default": "embed-v4.0",
            },
        },
        "required": ["texts", "input_type"],
    },
}


def cohere_embed(
    texts: List[str],
    input_type: str,
    model: str = "embed-v4.0",
    task_id: str | None = None,
) -> str:
    """Call ``cohere.ClientV2.embed`` and return the embeddings as JSON."""
    if not isinstance(texts, list) or not texts:
        return tool_error("`texts` must be a non-empty list of strings.")
    if not all(isinstance(t, str) and t for t in texts):
        return tool_error("Each entry in `texts` must be a non-empty string.")
    if input_type not in _VALID_INPUT_TYPES:
        return tool_error(
            f"`input_type` must be one of: {sorted(_VALID_INPUT_TYPES)}; "
            f"got {input_type!r}."
        )

    api_key = _resolve_api_key()
    if not api_key:
        return tool_error(
            "COHERE_API_KEY is not set. Configure it via `hermes setup` or "
            "export it in your environment to enable cohere_embed."
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
        response = client.embed(
            texts=texts,
            model=model,
            input_type=input_type,
            embedding_types=["float"],
        )
    except Exception as exc:
        logger.exception("Cohere embed call failed")
        return tool_error(f"Cohere embed call failed: {exc}")

    # The SDK returns a typed object whose .embeddings attribute is a
    # container with a .float_ list (renamed from .float because float
    # is a Python builtin). Accept dict shapes too for testability.
    embeddings: List[List[float]] = []
    embeddings_attr = getattr(response, "embeddings", None)
    if embeddings_attr is None and isinstance(response, dict):
        embeddings_attr = response.get("embeddings")

    if embeddings_attr is not None:
        # New SDK: response.embeddings.float_  (or .float)
        float_list = (
            getattr(embeddings_attr, "float_", None)
            or getattr(embeddings_attr, "float", None)
        )
        if float_list is None and isinstance(embeddings_attr, dict):
            float_list = embeddings_attr.get("float_") or embeddings_attr.get("float")
        if isinstance(float_list, list):
            embeddings = [list(vec) for vec in float_list]
        elif isinstance(embeddings_attr, list):
            # Legacy SDK shape: response.embeddings is a list of lists.
            embeddings = [list(vec) for vec in embeddings_attr]

    if not embeddings:
        return tool_error(
            "Cohere returned no embeddings — check the model name and "
            "input_type and try again."
        )

    dim = len(embeddings[0]) if embeddings else 0
    return json.dumps(
        {
            "success": True,
            "model": model,
            "input_type": input_type,
            "count": len(embeddings),
            "dim": dim,
            "embeddings": embeddings,
        }
    )


def _handle_cohere_embed(args: Dict[str, Any], **kw: Any) -> str:
    return cohere_embed(
        texts=args.get("texts") or [],
        input_type=args.get("input_type", ""),
        model=(args.get("model") or "embed-v4.0").strip() or "embed-v4.0",
        task_id=kw.get("task_id"),
    )


registry.register(
    name="cohere_embed",
    toolset="cohere",
    schema=COHERE_EMBED_SCHEMA,
    handler=_handle_cohere_embed,
    check_fn=check_cohere_requirements,
    requires_env=["COHERE_API_KEY"],
    is_async=False,
    emoji="🧬",
)
