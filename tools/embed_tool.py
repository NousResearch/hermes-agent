#!/usr/bin/env python3
"""Text embedding tool — Venice-backed.

Exposes Venice's OpenAI-compatible ``/embeddings`` endpoint as an
agent tool. Useful for RAG pre-processing, similarity search, and
memory indexing tasks the LLM can hand off.

Authentication: ``VENICE_API_KEY`` (same key as the rest of the
Venice multi-modal stack). Base URL is overridable via
``VENICE_BASE_URL`` so this works on either Venice direct or the
HermesOS managed-Venice proxy.

The tool returns the raw embedding vectors plus the model and token
usage. For very large inputs (>20 strings or >4KB combined), callers
should batch — Venice caps input at 2048 array entries / 8192 tokens.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

from tools.registry import registry

logger = logging.getLogger(__name__)


DEFAULT_VENICE_BASE_URL = "https://api.venice.ai/api/v1"
DEFAULT_EMBED_MODEL = "text-embedding-bge-m3"


def _resolve_credentials() -> tuple[str, str]:
    api_key = os.environ.get("VENICE_API_KEY", "").strip()
    base_url = (
        os.environ.get("VENICE_BASE_URL", "").strip()
        or DEFAULT_VENICE_BASE_URL
    ).rstrip("/")
    return api_key, base_url


def check_text_embed_requirements() -> bool:
    api_key, _ = _resolve_credentials()
    return bool(api_key)


def text_embed_tool(
    input: Union[str, List[str]],
    model: Optional[str] = None,
    encoding_format: str = "float",
    dimensions: Optional[int] = None,
) -> str:
    """Embed text via Venice ``/embeddings`` (OpenAI-compatible).

    ``input`` accepts either a single string or a list of strings.
    Returns a JSON blob with the embedding vectors and usage stats.
    """
    api_key, base_url = _resolve_credentials()
    if not api_key:
        return json.dumps(
            {
                "success": False,
                "error": "VENICE_API_KEY not set. Configure Venice in `hermes model` first.",
                "error_type": "missing_api_key",
            }
        )

    if not isinstance(input, (str, list)):
        return json.dumps(
            {"success": False, "error": "input must be a string or list of strings", "error_type": "bad_input"}
        )
    if isinstance(input, list):
        if not input:
            return json.dumps(
                {"success": False, "error": "input list cannot be empty", "error_type": "bad_input"}
            )
        if len(input) > 2048:
            return json.dumps(
                {"success": False, "error": "Venice caps embedding input at 2048 entries", "error_type": "too_large"}
            )
        for entry in input:
            if not isinstance(entry, str):
                return json.dumps(
                    {"success": False, "error": "all input entries must be strings", "error_type": "bad_input"}
                )

    chosen_model = (model or DEFAULT_EMBED_MODEL).strip() or DEFAULT_EMBED_MODEL
    enc = (encoding_format or "float").strip().lower()
    if enc not in {"float", "base64"}:
        enc = "float"

    payload: Dict[str, Any] = {
        "model": chosen_model,
        "input": input,
        "encoding_format": enc,
    }
    if isinstance(dimensions, int) and dimensions > 0:
        payload["dimensions"] = dimensions

    try:
        import requests

        response = requests.post(
            f"{base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "hermes-agent/embeddings-venice",
            },
            json=payload,
            timeout=60,
        )
    except Exception as exc:
        return json.dumps(
            {"success": False, "error": f"Venice request failed: {exc}", "error_type": "connection_error"}
        )

    if response.status_code != 200:
        try:
            detail = response.json().get("error", {}).get("message") or response.text[:300]
        except Exception:
            detail = response.text[:300]
        return json.dumps(
            {
                "success": False,
                "error": f"Venice embeddings failed ({response.status_code}): {detail}",
                "error_type": "api_error",
            }
        )

    try:
        result = response.json()
    except Exception as exc:
        return json.dumps(
            {"success": False, "error": f"Invalid JSON: {exc}", "error_type": "invalid_response"}
        )

    data = result.get("data") or []
    vectors = [entry.get("embedding") for entry in data if isinstance(entry, dict)]
    usage = result.get("usage") or {}

    return json.dumps(
        {
            "success": True,
            "model": result.get("model") or chosen_model,
            "provider": "venice",
            "embeddings": vectors,
            "count": len(vectors),
            "dimensions": len(vectors[0]) if vectors and isinstance(vectors[0], list) else None,
            "prompt_tokens": usage.get("prompt_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
    )


TEXT_EMBED_SCHEMA = {
    "name": "text_embed",
    "description": (
        "Generate vector embeddings for text. Useful for RAG, similarity "
        "search, and memory indexing. Accepts a single string or a list "
        "of up to 2048 strings (each up to 8192 tokens). Returns the "
        "embedding vectors plus the model name and token usage."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "input": {
                "description": "Text to embed. Either a single string or a list of strings.",
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}, "maxItems": 2048, "minItems": 1},
                ],
            },
            "model": {
                "type": "string",
                "description": "Venice embedding model (defaults to text-embedding-bge-m3).",
            },
            "encoding_format": {
                "type": "string",
                "enum": ["float", "base64"],
                "description": "Vector encoding. 'float' returns numeric arrays; 'base64' returns compact strings.",
                "default": "float",
            },
            "dimensions": {
                "type": "integer",
                "description": "Optional output dimension count (only honored by models that support reshaping).",
            },
        },
        "required": ["input"],
    },
}


registry.register(
    name="text_embed",
    toolset="memory",
    schema=TEXT_EMBED_SCHEMA,
    handler=lambda **kw: text_embed_tool(**kw),
    check_fn=check_text_embed_requirements,
    requires_env=["VENICE_API_KEY"],
    is_async=False,
    emoji="📐",
)
