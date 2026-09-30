import json
from typing import Callable, Optional

from tools.registry import registry, tool_error

KINDS = ("question", "accent", "theme", "layout", "connectors", "plugins", "tour", "fork", "machine_use")
MAX_OPTIONS = 12
# The skill branches on the ids of these rows, so they are filled here and any rows the model sent are dropped.
_TOUR_ROWS = [
    {"id": "basics", "label": "Quick tour"},
    {"id": "tour", "label": "Show me everything"},
    {"id": "none", "label": "Skip, let's build something"},
]
_MACHINE_USE_ROWS = [
    {"id": "work", "label": "Work"},
    {"id": "gaming", "label": "Gaming"},
    {"id": "school", "label": "School"},
    {"id": "creative", "label": "Creative"},
    {"id": "mix", "label": "A bit of everything"},
]
# The fork when the /initiate-setup turn recorded none (host facts unknown).
_FORK = {"question": "Know what you'd like it to make?", "options": [
    {"id": "mind", "label": "I have something in mind"},
    {"id": "automate", "label": "Automate something I already do"},
    {"id": "machine", "label": "Help me set up this computer"},
    {"id": "figure", "label": "Let's figure it out together"},
    {"id": "skip", "label": "Skip this for now"},
]}
_NO_ANSWER = ("The card got no answer: it timed out, the turn was interrupted, or no Hermes desktop "
              "window answered.")
# From the fork on, each skipped card steps down this ladder, so setup ends in a handoff or a stop.
_SKIP_LADDER = (
    "No question: send a kind='question' card with three or four first tasks built from the scan and the apps they "
    "picked.",
    "Send a kind='question' card with ONE concrete first task built from the scan and the apps they picked; when "
    "they pick it, hand off with start_chat.",
    "Stop setup: say only \"It's all yours, and this chat stays here if you want a hand.\" No card and no handoff.",
)
_MACHINE_USE_SKIPPED = "Hand off with the machine-setup plan and leave their use out."
_RESEND = ("If this text does not answer the card, answer it in a sentence or two, then send this card again in "
           "the same turn: {card}")


def _card(kind: str, question: str, multi_select: bool = False, options: Optional[list] = None) -> str:
    return json.dumps({"kind": kind, "question": question, "options": options or [], "multi_select": multi_select},
                      ensure_ascii=False, separators=(",", ":"))


# The card that follows each of these in the flow, so the model cannot drop a beat.
_THEN: dict[str, Callable[[dict, object], str]] = {
    "connectors": lambda cards, picked: "Send card plugins: " + _card("plugins", "Want any of these?", True),
    "plugins": lambda cards, picked: "Send card layout: " + _card("layout", "Which layout?"),
    "layout": lambda cards, picked: "Send card tour: " + _card("tour", "Want a look around first?"),
    "tour": lambda cards, picked: (
        ("After the gui_tour call, send" if picked in ("basics", "tour") else "No tour. Send")
        + " card fork in this same turn: " + _card("fork", (cards.get("fork") or _FORK)["question"])),
}


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


def _result(reply: Optional[dict], options: Optional[list]) -> dict:
    if reply is None:
        return {"outcome": "no_answer", "picked": None, "notice": _NO_ANSWER}
    picked = reply.get("picked")
    if picked is None:
        return {"outcome": "cancelled", "picked": None}
    result = {"outcome": "submitted", "picked": picked}
    # The settled card and the model read the pick by name; rows the backend filled are not in the call's args.
    labels = {option["id"]: option["label"] for option in options or ()}
    if isinstance(picked, list) and any(value in labels for value in picked):
        result["label"] = [labels.get(value, value) for value in picked]
    elif isinstance(picked, str) and picked in labels:
        result["label"] = labels[picked]
    return result


def _follow_up(kind: str, card: str, result: dict, rows: Optional[list], cards: dict) -> tuple[dict, dict]:
    """``next`` (and, from the fork, ``handoff``) for this answer, and the conversation's updated card state."""
    state, extra = dict(cards), {}
    outcome, picked = result["outcome"], result["picked"]
    if outcome == "no_answer":
        return extra, state
    if kind == "fork":
        plan = "machine" if picked == "machine" else "build"
        handoff = cards.get("handoff")
        if handoff and state.get("plan_sent") != plan:
            extra["handoff"] = {"message": handoff["message"], "plan": handoff[plan]}
            state["plan_sent"] = plan
        state["fork_seen"] = True
    step = state.get("step", -1)
    skipped = outcome == "cancelled" or (kind == "fork" and picked == "skip")
    if skipped and kind == "machine_use":
        extra["next"] = _MACHINE_USE_SKIPPED
    elif skipped and (kind == "fork" or (kind == "question" and state.get("fork_seen"))):
        # A skipped card of several first tasks steps to one task, and a skipped single task to the stop.
        shape = 0 if kind == "fork" or not rows else 2 if len(rows) == 1 else 1
        state["step"] = min(max(step + 1, shape), len(_SKIP_LADDER) - 1)
        extra["next"] = _SKIP_LADDER[state["step"]]
    elif kind == "fork" and picked == "figure":
        state["step"] = max(step, 0)
        extra["next"] = _SKIP_LADDER[state["step"]]
    elif outcome == "submitted" and isinstance(picked, str) and rows and picked not in {row["id"] for row in rows}:
        extra["next"] = _RESEND.format(card=card)
    elif kind in _THEN:
        extra["next"] = _THEN[kind](cards, picked)
    return extra, state


# App-owned parts of a card, filled here from the recorded facts so the model can neither drop nor edit them.
_APP_FILLED: dict[str, Callable[[dict], dict]] = {
    "tour": lambda cards: {"options": _TOUR_ROWS, "multi_select": False},
    "machine_use": lambda cards: {"options": _MACHINE_USE_ROWS, "multi_select": False},
    "fork": lambda cards: {"options": (cards.get("fork") or _FORK)["options"], "multi_select": False},
    "connectors": lambda cards: {"preselected": (cards.get("preselected") or {}).get("connectors") or [],
                                 "multi_select": True},
    "plugins": lambda cards: {"preselected": (cards.get("preselected") or {}).get("plugins") or [],
                              "multi_select": True},
}
_APP_ROWS = frozenset({"tour", "machine_use", "fork"})


def setup_choose_tool(kind: str = "", question: str = "", options=None, multi_select=None,
                      callback: Optional[Callable] = None, session_id: Optional[str] = None) -> str:
    if callback is None:
        return tool_error("setup_choose is only available in the Hermes desktop app.")
    if kind not in KINDS:
        return tool_error(f"kind must be one of: {', '.join(KINDS)}.")
    text = str(question or "").strip()
    if not text:
        return tool_error("question must be non-empty text.")
    normalized, error = (None, None) if kind in _APP_ROWS else _normalize_options(options)
    if error:
        return tool_error(error)
    payload = {"kind": kind, "question": text, "options": normalized,
               "multi_select": bool(multi_select) and (normalized is not None or kind != "question")}
    card = _card(kind, text, payload["multi_select"], normalized)
    from hermes_cli.setup_profile import read_cards, record_cards
    try:
        cards = read_cards(session_id) if session_id else {}
        if kind == "fork" and "fork" not in cards:
            from agent.initiate_setup_prompt import collect_setup_cards
            cards = {**cards, **collect_setup_cards()}
        payload.update(_APP_FILLED[kind](cards) if kind in _APP_FILLED else {})
        reply = callback(payload)
        fork = cards.get("fork") if kind == "fork" else None
        # "Something else" on a machine-first fork opens the rest of the fork in the same call.
        if fork and fork.get("fallback_options") and (reply or {}).get("picked") == "something_else":
            payload = {**payload, "question": fork["fallback_question"], "options": fork["fallback_options"]}
            reply = callback(payload)
        result = _result(reply, payload["options"])
        extra, state = _follow_up(kind, card, result, payload["options"], cards)
        if session_id and state != cards:
            record_cards(session_id, state)
        return json.dumps({**result, **extra}, ensure_ascii=False)
    except Exception as exc:
        return tool_error(f"Failed to get user input: {exc}")


SETUP_CHOOSE_SCHEMA = {
    "name": "setup_choose",
    "description": (
        "Ask the user one thing in the setup chat through a card: a question, a "
        "picker for accent, theme, layout, connectors or plugins, or the app's own "
        "tour offer, fork or machine_use rows. The card shows `question` itself, so "
        "your message text must not repeat it. Always send `options`: [] shows the "
        "app's own list for the pickers, tour, fork and machine_use, and free text "
        "for kind='question'. With options the user may still type an answer. "
        "multi_select lets the user pick several rows. Result: {outcome, picked, "
        "label?, next?, handoff?}. outcome is submitted, cancelled or no_answer (with "
        "a notice saying why). picked is the chosen option id (or the typed text) as "
        "a string, or a list of ids with multi_select; label names a picked row. "
        "Do what `next` says. `handoff` holds the start_chat message's parts and plan."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": list(KINDS),
                "description": "question, a picker, or the app's tour, fork or machine_use rows.",
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
