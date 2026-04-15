"""
Model Discovery Tool -- Lumenfall Provider

Lists available Lumenfall models with optional filtering by capability
(e.g. text-to-image, text-to-video, image-to-video, image-edit).

Available tool:
  lumenfall_list_models -- Discover available models and their capabilities

Requires:
  LUMENFALL_API_KEY environment variable (get one at https://lumenfall.ai)
"""

import json
import logging
from typing import Optional

from tools.lumenfall_client import (
    check_lumenfall_available,
    list_models,
    LumenfallError,
)

logger = logging.getLogger(__name__)


def lumenfall_list_models_tool(capability: Optional[str] = None) -> str:
    """List available Lumenfall models, optionally filtered by capability.

    Args:
        capability: If provided, only return models whose ``modes`` list
                    contains this value (e.g. "text-to-image", "text-to-video").

    Returns:
        JSON string with {"success": bool, "models": [...], "total": N}
    """
    try:
        models = list_models()

        # Filter by capability if requested
        if capability:
            models = [m for m in models if capability in m.get("modes", [])]

        # Normalize output: ensure each model has id, name, modes
        output_models = []
        for m in models:
            output_models.append({
                "id": m.get("id", ""),
                "name": m.get("name", m.get("id", "")),
                "modes": m.get("modes", []),
            })

        return json.dumps({
            "success": True,
            "models": output_models,
            "total": len(output_models),
        }, indent=2, ensure_ascii=False)

    except LumenfallError as e:
        logger.error("Lumenfall list_models error: %s", e)
        return json.dumps({
            "success": False,
            "models": [],
            "error": str(e),
        }, indent=2)

    except Exception as e:
        logger.error("Unexpected error listing models: %s", e, exc_info=True)
        return json.dumps({
            "success": False,
            "models": [],
            "error": str(e),
        }, indent=2)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry  # noqa: E402

LUMENFALL_LIST_MODELS_SCHEMA = {
    "name": "lumenfall_list_models",
    "description": (
        "Discover available Lumenfall models and their capabilities. "
        "Returns a list of models with their IDs and supported modes "
        "(e.g. text-to-image, text-to-video, image-to-video, image-edit). "
        "Optionally filter by a specific capability."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "capability": {
                "type": "string",
                "description": (
                    "Filter models by capability. Examples: "
                    "text-to-image, text-to-video, image-to-video, image-edit. "
                    "Leave empty to list all models."
                ),
            },
        },
        "required": [],
    },
}


def _handle_lumenfall_list_models(args, **kw):
    return lumenfall_list_models_tool(
        capability=args.get("capability"),
    )


registry.register(
    name="lumenfall_list_models",
    toolset="lumenfall",
    schema=LUMENFALL_LIST_MODELS_SCHEMA,
    handler=_handle_lumenfall_list_models,
    check_fn=check_lumenfall_available,
    requires_env=[],
    is_async=False,
    emoji="\U0001f4cb",
)
