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

# Route ui emitter logs into the same apmzoom.log file that tools/apmzoom.py
# targets, so `tail -f ~/.hermes/logs/apmzoom.log` shows both HTTP skill
# invocations and synthetic ui_* events in one stream.  Reuses the file
# handler if the apmzoom logger already installed one at import time.
def _attach_to_apmzoom_log():
    from pathlib import Path
    import os
    try:
        home = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
        log_path = home / "logs" / "apmzoom.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if any(getattr(h, "_apmzoom_file", False) for h in logger.handlers):
            return
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh._apmzoom_file = True
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    except Exception:
        pass


_attach_to_apmzoom_log()


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
        "display": "MUST invoke: opens the image upload widget",
        "description": (
            "🚨 THIS IS A TOOL YOU MUST INVOKE. Saying '请上传图片' / "
            "'Please upload' is NOT enough — the uploader only renders "
            "when you emit a tool_call. EMIT THIS TOOL CALL as the next "
            "action, do not end the turn with text. "
            "**ONLY call this when the task truly needs visual input** — "
            "e.g. 新品上架 / 发图 / 传照片 / 以图搜款 / 识图改款 / 商品主图. "
            "DO NOT call for: price changes, stock queries, text-only "
            "searches, category lookups, or any operation on an existing "
            "商品 (those each have dedicated apm_* tools). Typical "
            "upstream of vision_analyze + ui_product_editor."
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
        "display": "MUST invoke: shows price-change confirmation dialog",
        "description": (
            "🚨 THIS IS A TOOL YOU MUST INVOKE. Saying '请确认改价' / "
            "'Please confirm' is NOT enough — the dialog only renders when "
            "you emit a tool_call. EMIT THIS TOOL CALL as the next action, "
            "do not end the turn with text. Shows diff (old vs new price); "
            "merchant clicks Confirm or Cancel. Use before gds_m_editgoodsprice."
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
        "display": "MUST invoke: opens the product editor for merchant to confirm",
        "description": (
            "🚨 THIS IS A TOOL YOU MUST INVOKE. Saying '现在打开商品编辑器' / "
            "'Now I'll open the editor' is NOT enough — the UI only renders "
            "when you emit a tool_call to this function. If you plan to open "
            "the editor, EMIT THIS TOOL CALL as the next action in the same "
            "turn; do not end the turn with text. "
            "Pre-fills vision_analyze results; merchant confirms or edits. "
            "On submit the frontend returns the full addgoods payload. "
            "Use after vision + goodsclasslist in upload-image-and-publish-product. "
            "Props carry EVERY required addgoods field so the merchant isn't "
            "stuck on a form missing make_address_id / goods_detail / "
            "stock_count."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "preview_url": {
                    "type": "string",
                    "description": "CDN URL of the main product image",
                },
                "vision_fields": {
                    "type": "object",
                    "description": (
                        "Vision extraction: {category, color, material, "
                        "size_range[], suggested_price, selling_points[]}. "
                        "null when vision refused."
                    ),
                },
                "recommended_goods_name": {
                    "type": "string",
                    "description": "LLM-suggested goods_name (can be vision's category + color)",
                },
                "recommended_class_id": {
                    "type": "integer",
                    "description": (
                        "Leaf class_id the LLM picked based on vision.category "
                        "(e.g. 12 for 毛衣).  The bridge will auto-compute the "
                        "'1-6-12' cascade format on addgoods — LLM doesn't have "
                        "to build it."
                    ),
                },
                "class_tree": {
                    "type": "array",
                    "description": (
                        "DO NOT PASS THIS FIELD.  Bridge auto-injects the "
                        "full goodsclasslist tree server-side so the LLM "
                        "doesn't waste 60-90s generating ~8KB of JSON "
                        "tokens.  Listed here only to document the prop "
                        "the frontend receives."
                    ),
                },
                "make_address_options": {
                    "type": "array",
                    "description": (
                        "DO NOT PASS THIS FIELD.  Bridge auto-injects the "
                        "3 origin options (韩国/中国/其他).  Listed only to "
                        "document the prop the frontend receives."
                    ),
                },
                "recommended_make_address_id": {
                    "type": "integer",
                    "description": "Default origin (usually 1 韩国 for apmzoom merchants)",
                },
                "suggested_stock": {
                    "type": "integer",
                    "description": "Default stock_count to prefill (e.g. 10).  Merchant can override.",
                },
                "suggested_detail": {
                    "type": "string",
                    "description": (
                        "Draft goods_detail (简介) — build from vision's selling_points joined "
                        "with newlines.  goods_detail is REQUIRED for addgoods."
                    ),
                },
            },
            "required": [
                "preview_url",
                "vision_fields",
            ],
        },
    },
    "ui_batch_editor": {
        "display": "MUST invoke: opens the batch product editor",
        "description": (
            "🚨 THIS IS A TOOL YOU MUST INVOKE. Saying '打开批量编辑器' / "
            "'Opening batch editor' is NOT enough — the batch editor only "
            "renders when you emit a tool_call. EMIT THIS TOOL CALL as the "
            "next action, do not end the turn with text. "
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


def _inject_ui_props(component: str, props: Dict[str, Any]) -> None:
    """Bridge-side enrichment for ui_* tool props.

    Reason: some required props (class_tree, make_address_options) carry
    data volumes (~8KB) that DeepSeek takes 60-90 SECONDS to emit as
    tool_call args — that's pure token generation time for data the
    bridge already has.  Workflow now tells LLM NOT to pass these fields;
    bridge fetches + injects them server-side before pushing the
    hermes.ui.prompt event, so the frontend receives them unchanged and
    the LLM's turn ends in 5-10s instead of 90s.
    """
    # product_editor needs goodsclasslist tree + make_address options
    if component in ("product_editor", "batch_editor"):
        if "class_tree" not in props or not props["class_tree"]:
            try:
                from tools.apmzoom import _execute_skill
                raw = _execute_skill(
                    "gds_m_goodsclasslist", {}, _internal=True,
                )
                tree = json.loads(raw).get("result") or []
                props["class_tree"] = tree
                logger.info(
                    "[apmzoom_ui] %s: auto-injected class_tree (%d top-level nodes)",
                    component, len(tree),
                )
            except Exception as e:
                logger.debug("[apmzoom_ui] class_tree injection failed: %s", e)
                props["class_tree"] = []

        if "make_address_options" not in props or not props["make_address_options"]:
            # Hardcoded; apmzoom has 3 origins, they never change
            props["make_address_options"] = [
                {"id": 1, "name": "韩国"},
                {"id": 2, "name": "中国"},
                {"id": 3, "name": "其他"},
            ]
        props.setdefault("recommended_make_address_id", 1)


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

        # Bridge auto-injects heavy static props (class_tree ~8KB,
        # make_address_options) that LLM would otherwise spend 60-90s
        # generating as tool_call args.  Silent enrichment — LLM doesn't
        # know the field was added, frontend receives the full props.
        _inject_ui_props(component, props)
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
