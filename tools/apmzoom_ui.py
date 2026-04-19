"""
Synthetic UI tools — let the LLM summon rich frontend components via normal
OpenAI function-calling.

Each `ui_*` tool doesn't make an HTTP call; its handler pushes a structured
payload onto the SSE stream as an ``event: hermes.ui.prompt`` event.  The
frontend (agent-claw) routes the payload's ``component`` field to the right
React bubble (Uploader, PriceConfirm, ProductEditor, …).  After the user
interacts, the frontend sends a follow-up chat message containing an
``[ui-result]`` marker with the structured result, which the LLM reads
on the next agent turn to continue the workflow.

This makes "summon UI" a first-class agent primitive:
  - LLM already knows function-calling — no new syntax
  - Type-safe via tool schemas (frontend gets validated props)
  - Works across all workflows — no per-flow state machine needed

The handler returns a lightweight "awaiting_user_input" JSON to the LLM so
the agent loop pauses naturally: LLM sees the placeholder, issues its final
text chunk ("请在下方上传图片…"), and ends the turn.  The next user message
from the frontend carries the real result.
"""

import json
import logging
import uuid
from typing import Any, Callable, Dict, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-request UI event emitter
# ---------------------------------------------------------------------------
#
# api_server.py sets this before the agent loop runs (same lifecycle as the
# merchant_id / active_skill module globals in tools/apmzoom.py).  The emitter
# pushes a payload onto the streaming response's queue so the SSE writer
# emits `event: hermes.ui.prompt`.
#
# Module-global instead of ContextVar for the same reason as elsewhere —
# hermes' ThreadPoolExecutor tool dispatch drops ContextVar state.
_ui_event_emitter: Optional[Callable[[Dict[str, Any]], None]] = None


def set_ui_emitter(emitter: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    """Install the SSE push function for this request.

    Pass None to clear (no-stream requests don't get UI events — they fall
    back to the LLM's awaiting_user_input placeholder, which is enough for
    headless clients).
    """
    global _ui_event_emitter
    _ui_event_emitter = emitter


# ---------------------------------------------------------------------------
# UI component catalogue
# ---------------------------------------------------------------------------
#
# Each entry describes a UI component the frontend knows how to render plus
# the props the LLM should pass when invoking it.  Keep props tight — every
# extra field is more the LLM has to get right.
#
# To add a new component: add an entry here + add a corresponding React
# component on the frontend + document it in the relevant workflow SKILL.md.

UI_COMPONENTS: Dict[str, Dict[str, Any]] = {
    "ui_uploader": {
        "display": "Prompt user to upload files via the chat UI",
        "description": (
            "Ask the merchant to upload one or more files. The frontend "
            "renders an upload widget inline in the chat. Use this when you "
            "need image_url(s) before running vision_analyze."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "Short message shown above the uploader (e.g. '请上传商品主图')",
                },
                "max_files": {
                    "type": "integer",
                    "description": "Max number of files (1 for single, up to 20 for batch)",
                    "minimum": 1,
                    "maximum": 20,
                },
                "accept": {
                    "type": "string",
                    "description": "MIME filter, default 'image/*'",
                },
            },
            "required": ["instruction"],
        },
    },
    "ui_price_confirm": {
        "display": "Confirm a price change with the merchant",
        "description": (
            "Show a diff-style confirm dialog for a price change. Use before "
            "calling gds_m_editgoodsprice. User clicks Confirm or Cancel."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goods_id": {"type": "integer", "description": "Target goods id"},
                "goods_name": {"type": "string", "description": "Display name"},
                "old_price": {"type": "number", "description": "Current sale_price"},
                "new_price": {"type": "number", "description": "Proposed new sale_price"},
            },
            "required": ["goods_id", "goods_name", "old_price", "new_price"],
        },
    },
    "ui_product_editor": {
        "display": "Render the full product editor for a single-item upload",
        "description": (
            "Shows vision_analyze results prefilled and lets the merchant "
            "edit + confirm. Use in upload-image-and-publish-product after "
            "vision + goodsclasslist. On submit, frontend returns the full "
            "addgoods payload; you then call gds_m_addgoods with it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "preview_url": {"type": "string"},
                "vision_fields": {
                    "type": "object",
                    "description": "Category, color, material, suggested_price, selling_points…",
                },
                "recommended_class_id": {"type": "integer"},
                "class_options": {
                    "type": "array",
                    "description": "Full [{id, name}] list from goodsclasslist so user can override",
                },
            },
            "required": ["preview_url", "vision_fields"],
        },
    },
    "ui_batch_editor": {
        "display": "Render a batch product editor for multi-item upload",
        "description": (
            "Shows N items with vision results, lets the merchant edit each "
            "+ check the ones to publish. Use in batch-upload-and-publish "
            "after concurrent vision + goodsclasslist. On submit, frontend "
            "returns an array of addgoods payloads (only for checked items)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": (
                        "One entry per uploaded image: "
                        "{preview_url, vision_fields, recommended_class_id, "
                        "class_options, llm_note?}"
                    ),
                },
            },
            "required": ["items"],
        },
    },
}


def _make_ui_handler(component_name: str):
    """Closure — when the LLM calls the synthetic tool, emit an SSE event
    and return a lightweight placeholder so the agent loop ends the turn
    gracefully."""
    def _handler(args: Dict[str, Any], **kwargs) -> str:
        # Accept both {"params": {...}} and bare dict (matches apmzoom.py handler)
        if isinstance(args, dict) and "params" in args and isinstance(args["params"], dict):
            props = args["params"]
        elif isinstance(args, dict):
            props = args
        else:
            return json.dumps({
                "error": "bad_params",
                "message": "arguments must be an object",
            }, ensure_ascii=False)

        correlation_id = f"ui-{uuid.uuid4().hex[:12]}"
        component = component_name.removeprefix("ui_")
        payload = {
            "component": component,
            "correlation_id": correlation_id,
            "props": props,
        }

        if _ui_event_emitter is not None:
            try:
                _ui_event_emitter(payload)
                logger.info(
                    "[apmzoom_ui] emitted %s correlation_id=%s",
                    component_name, correlation_id,
                )
            except Exception as e:
                logger.warning("[apmzoom_ui] emitter failed for %s: %s",
                               component_name, e)
        else:
            logger.info(
                "[apmzoom_ui] %s called but no emitter set (non-streaming request)",
                component_name,
            )

        # Placeholder response that the LLM will see.  Contains enough hints
        # for the model to reason about the next step after the user responds.
        return json.dumps({
            "status": "awaiting_user_input",
            "component": component,
            "correlation_id": correlation_id,
            "hint": (
                f"UI component '{component}' is now shown to the user. "
                "End this turn with a short prompt. On next turn, the user "
                "message will contain '[ui-result]' with the structured "
                f"result — use correlation_id={correlation_id} to match."
            ),
        }, ensure_ascii=False)
    return _handler


def _register_ui_tools() -> int:
    """Register all ui_* synthetic tools.  Returns count registered."""
    count = 0
    for name, spec in UI_COMPONENTS.items():
        schema = {
            "name": name,
            "description": spec["description"][:500],
            "parameters": {
                "type": "object",
                "properties": {
                    "params": {
                        **spec["parameters"],
                        "description": spec["display"],
                    },
                },
                "required": ["params"],
            },
        }
        handler = _make_ui_handler(name)
        # Separate toolset so callers can enable/disable UI tools independently.
        try:
            registry.register(
                name=name,
                toolset="apmzoom_ui",
                schema=schema,
                handler=handler,
            )
            count += 1
        except Exception as e:
            logger.warning("[apmzoom_ui] failed to register %s: %s", name, e)
    logger.info("[apmzoom_ui] registered %d UI components", count)
    return count


# Register at import time — matches tools/apmzoom.py pattern so AIAgent sees
# the synthetic tools before its tool-enumeration pass.
_register_ui_tools()
