import json
import logging
from typing import Callable, Optional

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

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
# Why an app-owned list has nothing to offer; the card is not shown and the call returns at once.
_NO_CONNECTORS = ("Connecting apps needs a Nous account (free) and none is signed in here, so no card was shown; "
                  "it can be set up later.")
_NO_PLUGINS = "No plugins on the list run on this computer, so no card was shown."
# The task chat's first-task skill offers the options for a vague ask, so setup hands off at once.
_FIGURE = "Hand off now with start_chat; the ask is \"Let's figure out a first task together.\""
_MACHINE_USE_SKIPPED = "Hand off with the machine-setup plan and leave their use out."
_RESEND = ("If this text does not answer the card, answer it in a sentence or two, then send this card again in "
           "the same turn: {card}")
# Composer text that names no row of a card with fixed rows: their words, not a pick.
_TYPED = ("They typed this instead of picking. Reply to it in a sentence or two, then send this card again in the "
          "same turn so they can pick: {card}")
# On a first-task card typed words can be the task itself.
_TYPED_TASK = ("They typed this instead of picking. If it names a task, that is their first task: hand off with it. "
               "Otherwise reply to it in a sentence or two, then send this card again in the same turn: {card}")


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
        + " card fork in this same turn: " + _card("fork", (cards.get("fork") or {}).get("question") or _fork_question())),
}


def _fork_question() -> str:
    # The fork question when the /initiate-setup turn recorded none (host facts unknown). Imported late: tool
    # discovery imports this module, and the facts module loads the host probes.
    from agent.initiate_setup_facts import FORK_QUESTION

    return FORK_QUESTION


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
    # The desktop sends `said` only after matching the text against the card's rows by id and label.
    said = reply.get("said")
    if isinstance(said, str) and said.strip():
        return {"outcome": "typed", "picked": None, "said": said.strip()}
    picked = reply.get("picked")
    if picked is None:
        return {"outcome": "cancelled", "picked": None}
    result = {"outcome": "submitted", "picked": picked}
    # The settled card and the model read the pick by the name the user saw. Rows the backend filled are named
    # here; the app's own lists (accent, layout, connectors...) are named by the card that answered.
    labels = {option["id"]: option["label"] for option in options or ()}
    named = reply.get("label")
    if isinstance(picked, list) and any(value in labels for value in picked):
        result["label"] = [labels.get(value, value) for value in picked]
    elif isinstance(picked, str) and picked in labels:
        result["label"] = labels[picked]
    elif isinstance(picked, str) and isinstance(named, str) and named.strip():
        result["label"] = named.strip()
    elif (isinstance(picked, list) and isinstance(named, list) and len(named) == len(picked)
          and all(isinstance(value, str) for value in named)):
        result["label"] = named
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
    if outcome == "typed":
        task_card = kind == "fork" or (kind == "question" and state.get("fork_seen"))
        return {**extra, "next": (_TYPED_TASK if task_card else _TYPED).format(card=card)}, state
    step = state.get("step", -1)
    skipped = outcome == "cancelled"
    if skipped and kind == "machine_use":
        extra["next"] = _MACHINE_USE_SKIPPED
    elif skipped and (kind == "fork" or (kind == "question" and state.get("fork_seen"))):
        # A skipped card of several first tasks steps to one task, and a skipped single task to the stop.
        shape = 0 if kind == "fork" or not rows else 2 if len(rows) == 1 else 1
        state["step"] = min(max(step + 1, shape), len(_SKIP_LADDER) - 1)
        extra["next"] = _SKIP_LADDER[state["step"]]
    elif kind == "fork" and picked == "figure":
        extra["next"] = _FIGURE
    elif outcome == "submitted" and isinstance(picked, str) and rows and picked not in {row["id"] for row in rows}:
        extra["next"] = _RESEND.format(card=card)
    elif kind in _THEN:
        extra["next"] = _THEN[kind](cards, picked)
    return extra, state


# The picks start_chat hands the task chat from the setup profile, keyed by card kind.
_REMEMBERED = frozenset({"connectors", "plugins", "layout"})


def _remember(kind: str, question: str, result: dict, state: dict) -> dict:
    """Also the connector and plugin ids (the task chat connects and installs them first) and the fork pick (a
    plugin task's plugins join those installs)."""
    from agent.initiate_setup_prompt import NAME_QUESTION

    if result["outcome"] != "submitted":
        return state
    picked = result["picked"]
    if kind == "fork":
        return {**state, "fork_pick": picked}
    key = "name" if kind == "question" and question == NAME_QUESTION else kind if kind in _REMEMBERED else None
    if key is None:
        return state
    state = {**state, "picks": {**state.get("picks", {}), key: result.get("label") or picked}}
    if kind in ("connectors", "plugins") and isinstance(picked, list):
        state["pick_ids"] = {**state.get("pick_ids", {}), kind: picked}
    return state


def _fork_rows(cards: dict) -> list:
    # Built when the card is shown: the apps and plugins cards before it have been answered by then.
    from agent.initiate_setup_facts import fork_card

    return fork_card(cards)["options"]


# App-owned parts of a card, filled here from the recorded facts so the model can neither drop nor edit them.
_APP_FILLED: dict[str, Callable[[dict], dict]] = {
    "tour": lambda cards: {"options": _TOUR_ROWS, "multi_select": False},
    "machine_use": lambda cards: {"options": _MACHINE_USE_ROWS, "multi_select": False},
    "fork": lambda cards: {"options": _fork_rows(cards), "multi_select": False},
    "connectors": lambda cards: {"preselected": (cards.get("preselected") or {}).get("connectors") or [],
                                 "multi_select": True},
    "plugins": lambda cards: {"preselected": (cards.get("preselected") or {}).get("plugins") or [],
                              "multi_select": True},
}
_APP_ROWS = frozenset({"tour", "machine_use", "fork"})


def _name_rows() -> Optional[list]:
    # The account's full name stays on this computer: it is added here, after the model's call was saved, so the
    # model sees it only in the result when the user picks it.
    from agent.initiate_setup_facts import suggested_name

    name = suggested_name()
    return [{"id": "suggested", "label": name}] if name else None


def _connectors_closed() -> Optional[str]:
    from tools.connectors.gateway.config import connectors_available

    # A user with no Nous identity is open too: the first connect creates the guest.
    return None if connectors_available() else _NO_CONNECTORS


def _plugins_closed() -> Optional[str]:
    from hermes_cli.plugin_catalog_presence import onboarding_entries

    return None if onboarding_entries() else _NO_PLUGINS


# App-owned lists that can come up empty for this session: the reason, checked before the card is shown.
_CLOSED: dict[str, Callable[[], Optional[str]]] = {"connectors": _connectors_closed, "plugins": _plugins_closed}


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
    from agent.initiate_setup_prompt import NAME_QUESTION
    from hermes_cli.setup_profile import read_cards, record_cards
    try:
        cards = read_cards(session_id) if session_id else {}
        if kind == "fork" and "fork" not in cards:
            # Compression gave the conversation a new id, so its /initiate-setup turn recorded nothing here.
            from agent.initiate_setup_facts import facts, setup_cards
            cards = {**cards, **setup_cards(facts())}
        closed = _CLOSED[kind]() if kind in _CLOSED and normalized is None else None
        if closed:
            return json.dumps({"outcome": "no_answer", "picked": None, "notice": closed,
                               "next": _THEN[kind](cards, None)}, ensure_ascii=False)
        payload.update(_APP_FILLED[kind](cards) if kind in _APP_FILLED else {})
        if kind == "question" and text == NAME_QUESTION:
            payload.update(options=_name_rows(), multi_select=False)
        reply = callback(payload)
        result = _result(reply, payload["options"])
        if kind == "question" and text == NAME_QUESTION and result["outcome"] == "typed":
            # The composer answers the name card like its free-text field: the words are the name, not a missed row.
            result = {"outcome": "submitted", "picked": result["said"]}
        extra, state = _follow_up(kind, card, result, payload["options"], cards)
        state = _remember(kind, text, result, state)
        if session_id and state != cards:
            record_cards(session_id, state)
        return json.dumps({**result, **extra}, ensure_ascii=False)
    except Exception as exc:
        logger.exception("setup_choose failed")
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
        "label?, said?, next?, handoff?}. outcome is submitted, typed, cancelled or "
        "no_answer (with a notice saying why; an app list with nothing to offer "
        "returns it at once, with no card). typed means the user wrote words that "
        "match no row: `said` holds them and nothing was picked. picked is the "
        "chosen option id (or the free-text answer) as a string, or a list of ids "
        "with multi_select; label is the name the user saw "
        "for each pick, so say the label, never the id. "
        "Do what `next` says. `handoff` holds the handoff message's parts and plan."
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
