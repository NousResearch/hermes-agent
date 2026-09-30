import json
from typing import Callable, Optional

from tools.registry import registry, tool_error

KINDS = ("question", "accent", "theme", "layout", "connectors", "plugins", "tour", "fork")
MAX_OPTIONS = 12
# The skill branches on the tour and fork ids, so their rows are filled here, whatever the model sent.
_TOUR_ROWS = [
    {"id": "basics", "label": "Quick tour"},
    {"id": "tour", "label": "Show me everything"},
    {"id": "none", "label": "Skip, let's build something"},
]
# The fork when the /initiate-setup turn recorded none (host facts unknown).
_FORK = {"options": [
    {"id": "mind", "label": "I have something in mind"},
    {"id": "automate", "label": "Automate something I already do"},
    {"id": "machine", "label": "Help me set up this computer"},
    {"id": "figure", "label": "Let's figure it out together"},
    {"id": "skip", "label": "Skip this for now"},
]}
_NO_ANSWER = ("The card got no answer: it timed out, the turn was interrupted, or no Hermes desktop "
              "window answered.")


def _normalize_options(options) -> tuple:
    if options is None or options == []:
        return None, None
    if not isinstance(options, list):
        return None, "options must be an array of {id, label, detail?}; [] for the app's list or free text."
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


def _result(reply: Optional[dict], options: Optional[list]) -> str:
    if reply is None:
        return json.dumps({"outcome": "no_answer", "picked": None, "notice": _NO_ANSWER})
    picked = reply.get("picked")
    if picked is None:
        return json.dumps({"outcome": "cancelled", "picked": None})
    result = {"outcome": "submitted", "picked": picked}
    # The settled card and the model read the pick by name; rows the backend filled are not in the call's args.
    labels = {option["id"]: option["label"] for option in options or ()}
    if isinstance(picked, list) and any(value in labels for value in picked):
        result["label"] = [labels.get(value, value) for value in picked]
    elif isinstance(picked, str) and picked in labels:
        result["label"] = labels[picked]
    return json.dumps(result, ensure_ascii=False)


# App-owned parts of a card, filled here from the recorded facts so the model can neither drop nor edit them.
_APP_FILLED: dict[str, Callable[[dict], dict]] = {
    "tour": lambda cards: {"options": _TOUR_ROWS, "multi_select": False},
    "fork": lambda cards: {"options": (cards.get("fork") or _FORK)["options"], "multi_select": False},
    "connectors": lambda cards: {"preselected": (cards.get("preselected") or {}).get("connectors") or []},
    "plugins": lambda cards: {"preselected": (cards.get("preselected") or {}).get("plugins") or []},
}


def setup_choose_tool(kind: str = "", question: str = "", options=None, multi_select=None,
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
    payload = {"kind": kind, "question": text, "options": normalized,
               "multi_select": bool(multi_select) and (normalized is not None or kind != "question")}
    from hermes_cli.setup_profile import read_cards
    filler = _APP_FILLED.get(kind)
    try:
        cards = read_cards() if filler else {}
        payload.update(filler(cards) if filler else {})
        reply = callback(payload)
        fork = cards.get("fork") if kind == "fork" else None
        # "Something else" on a machine-first fork opens the rest of the fork in the same call.
        if fork and fork.get("fallback_options") and (reply or {}).get("picked") == "something_else":
            payload = {**payload, "question": fork["fallback_question"], "options": fork["fallback_options"]}
            reply = callback(payload)
        return _result(reply, payload["options"])
    except Exception as exc:
        return tool_error(f"Failed to get user input: {exc}")


SETUP_CHOOSE_SCHEMA = {
    "name": "setup_choose",
    "description": (
        "Ask the user one thing in the setup chat through a card: a question, or a "
        "picker for accent, theme, layout, connectors or plugins, or the app's own "
        "tour offer or fork. The card shows `question` itself, so your message text "
        "must not repeat it. Always send `options`: [] shows the app's own list for "
        "accent, theme, layout, connectors and plugins, and free text for kind='question'. "
        "tour and fork always show the app's rows; send [] for them. "
        "With options the user may still type an answer. multi_select lets the user "
        "pick several rows. Result: {outcome, picked, label?}. outcome is submitted, "
        "cancelled or no_answer (with a notice saying why). picked is the chosen option "
        "id (or the typed text) as a string, or a list of ids with multi_select; label "
        "names a picked row."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": list(KINDS),
                "description": "question, a picker, or the app's tour offer or fork.",
            },
            "question": {
                "type": "string",
                "description": "The card's heading; do not repeat it in your message.",
            },
            "options": {
                "type": "array",
                "minItems": 0,
                "maxItems": MAX_OPTIONS,
                "description": "Rows to offer; [] for the app's own list (free text for kind='question').",
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
        },
        "required": ["kind", "question", "options"],
    },
}


registry.register(
    name="setup_choose", toolset="setup", schema=SETUP_CHOOSE_SCHEMA,
    handler=lambda args, **kw: setup_choose_tool(
        kind=args.get("kind", ""), question=args.get("question", ""), callback=kw.get("callback"),
        **{k: args.get(k) for k in ("options", "multi_select")}),
    emoji="🎛")
