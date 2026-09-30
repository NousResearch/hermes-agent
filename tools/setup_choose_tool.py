import json
from typing import Callable, Optional

from tools.registry import registry, tool_error

KINDS = ("question", "accent", "theme", "layout", "connectors", "plugins")
INTENT_KINDS = ("connectors", "plugins")
MAX_OPTIONS = 12
_NO_ANSWER = ("The card got no answer: it timed out, the turn was interrupted, or no Hermes desktop "
              "window answered.")


def _normalize_options(options) -> tuple:
    if options is None:
        return None, None
    if not isinstance(options, list) or not options:
        return None, "options must be a non-empty array of {id, label, detail?}, or omitted."
    if len(options) > MAX_OPTIONS:
        return None, f"options has {len(options)} entries; the limit is {MAX_OPTIONS}."
    normalized, seen = [], set()
    for index, item in enumerate(options):
        if not isinstance(item, dict):
            return None, f"options[{index}] must be an object with id and label."
        option_id, label, detail = item.get("id"), item.get("label"), item.get("detail")
        if not isinstance(option_id, str) or not option_id.strip():
            return None, f"options[{index}].id must be non-empty text."
        if not isinstance(label, str) or not label.strip():
            return None, f"options[{index}].label must be non-empty text."
        if detail is not None and not isinstance(detail, str):
            return None, f"options[{index}].detail must be text."
        if option_id.strip() in seen:
            return None, f"options[{index}].id {option_id.strip()!r} repeats an earlier id."
        seen.add(option_id.strip())
        entry = {"id": option_id.strip(), "label": label.strip()}
        if detail and detail.strip():
            entry["detail"] = detail.strip()
        normalized.append(entry)
    return normalized, None


def _result(reply: Optional[dict]) -> str:
    if reply is None:
        return json.dumps({"outcome": "no_answer", "picked": None, "notice": _NO_ANSWER})
    if reply.get("picked") is None:
        return json.dumps({"outcome": "cancelled", "picked": None})
    result = {"outcome": "submitted", "picked": reply["picked"]}
    if reply.get("intent"):
        result["intent"] = reply["intent"]
    return json.dumps(result, ensure_ascii=False)


def setup_choose_tool(kind: str = "", question: str = "", options=None, multi_select=None, intent=None,
                      callback: Optional[Callable] = None) -> str:
    if callback is None:
        return tool_error("setup_choose is only available in the Hermes desktop app.")
    if kind not in KINDS:
        return tool_error(f"kind must be one of: {', '.join(KINDS)}.")
    text = str(question or "").strip()
    if not text:
        return tool_error("question must be non-empty text.")
    normalized, error = _normalize_options(options)
    if error:
        return tool_error(error)
    if intent and kind not in INTENT_KINDS:
        return tool_error(f"intent applies only to kind {' or '.join(INTENT_KINDS)}.")
    payload = {"kind": kind, "question": text, "options": normalized,
               "multi_select": bool(multi_select) and (normalized is not None or kind != "question"),
               "intent": bool(intent)}
    try:
        return _result(callback(payload))
    except Exception as exc:
        return tool_error(f"Failed to get user input: {exc}")


SETUP_CHOOSE_SCHEMA = {
    "name": "setup_choose",
    "description": (
        "Ask the user one thing in the setup chat through a card: a question, or a "
        "picker for accent, theme, layout, connectors or plugins. The card shows "
        "`question` itself, so your message text must not repeat it. Omit `options` "
        "for accent, theme, layout, connectors or plugins to show the app's own list. "
        "kind='question' without options asks for free text; with options the user may "
        "still type an answer. multi_select lets the user pick several rows. intent "
        "(connectors and plugins only) adds a now / later / save choice per row. "
        "Result: {outcome, picked, intent?}. outcome is submitted, cancelled or "
        "no_answer (with a notice saying why). picked is the chosen option id (or the "
        "typed text) as a string, or a list of ids with multi_select; intent maps each "
        "picked id to now, later or save."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": list(KINDS),
                "description": "question, or the picker to show.",
            },
            "question": {
                "type": "string",
                "description": "The card's heading; do not repeat it in your message.",
            },
            "options": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_OPTIONS,
                "description": "Rows to offer; omit for the app's own list (or free text for kind='question').",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "label": {"type": "string"},
                        "detail": {"type": "string"},
                    },
                    "required": ["id", "label"],
                },
            },
            "multi_select": {"type": "boolean", "description": "Let the user pick several rows."},
            "intent": {
                "type": "boolean",
                "description": "connectors / plugins: ask now, later or save per picked row.",
            },
        },
        "required": ["kind", "question"],
    },
}


registry.register(
    name="setup_choose", toolset="setup", schema=SETUP_CHOOSE_SCHEMA,
    handler=lambda args, **kw: setup_choose_tool(
        kind=args.get("kind", ""), question=args.get("question", ""), callback=kw.get("callback"),
        **{k: args.get(k) for k in ("options", "multi_select", "intent")}),
    emoji="🎛")
