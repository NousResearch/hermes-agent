"""``setup_beat``: the one onboarding-only tool. It shows a beat's card through the desktop, waits
for the user (``CARD_TIMEOUT_S``), and returns their answer with a short reminder of the next beat.

The ``onboarding`` toolset carrying it is folded in only for the onboarding chat, at agent build
(``tui_gateway/methods_onboarding.py``). The run record is the session's ``onboarding_run``
(``agent/onboarding_record.py``); the order and the options are ``agent/onboarding_flow.py``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from typing import Any

from agent import onboarding_flow as flow
from agent.onboarding_text import finish_reminder, reminder
from agent.onboarding_text import question as beat_question
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

BEATS = ("next", *flow.BEAT_IDS, "done")


def _connected() -> dict:
    from tools.connectors.display import connected_slugs
    from tools.connectors.gateway.client import ConnectorClient

    return {"connected": sorted(connected_slugs(ConnectorClient()))}


def _installed() -> dict:
    from hermes_cli.plugin_catalog_presence import onboarding_names
    from hermes_cli.plugins_cmd import _catalog_installed_dir

    return {"installed": [name for name in onboarding_names() if _catalog_installed_dir(name) is not None]}


# A card closed without its own answer (words, a value, a skip, the deadline) still reports these.
_PRESENT = {"connectors": _connected, "plugins": _installed}


def _present(beat: str) -> dict | None:
    if (read := _PRESENT.get(beat)) is None:
        return None
    try:
        return read()
    except Exception:
        logger.warning("onboarding: could not read what the %s beat has on this machine", beat, exc_info=True)
        return None


def _direct_event(beat: str, args: Mapping[str, Any]) -> flow.Event | None:
    """An event the call carries in itself (no card): done, a value given in words, or a skip. Only the
    Skip setup button (onboarding.skip) ends onboarding early; the model never decides the user quit."""
    if beat == "done":
        return flow.Done()
    if args.get("skip"):
        return flow.Skipped(beat, _present(beat))
    value = args.get("value")
    return flow.Recorded(beat, value.strip(), _present(beat)) if isinstance(value, str) and value.strip() else None


def _reply_event(beat: str, reply: Mapping[str, Any] | None) -> flow.Event:
    reply = reply or {}
    said = reply.get("said")
    if beat == "name" and isinstance(said, str) and said.strip():
        # Words typed while the name card waits are the name, like its own text box.
        return flow.Answered(beat, {"name": said.strip()}, said.strip())
    if reply.get("wrote_instead"):
        # A message sent instead of answering: the desktop sends it as the user's next turn.
        return flow.Wrote(beat, _present(beat))
    answer = reply.get("answer")
    return flow.Answered(beat, answer) if isinstance(answer, dict) and answer else flow.TimedOut(beat, _present(beat))


def _skill_resolver() -> Callable[[str], str | None]:
    """A written first-task skill name -> the name skill_view loads it with here, or None. A plugin skill
    (``blender:blender``) resolves through the installed catalog plugin to its registered name."""
    from hermes_cli.plugins_activation_live import catalog_plugin_skill
    from tools.skills_tool import find_all_skills

    bundled = {skill["name"] for skill in find_all_skills()}
    return lambda name: catalog_plugin_skill(name) if ":" in name else (name if name in bundled else None)


def _result(step: flow.Step, beat: str | None = None, extra: str = "") -> str:
    facts = flow.Facts.from_stored(step.record["facts"])
    # Only a first task's reminder names skills; the skills scan is skipped for every other result.
    resolve_skill = _skill_resolver() if step.reminder.startswith("build_") else (lambda name: name)
    text = reminder(step.reminder, step.record, facts, beat, resolve_skill) + extra
    return json.dumps({**step.result, **({"reminder": text} if text else {})}, ensure_ascii=False)


def _connector_slugs() -> set[str] | None:
    """The connectors this account can connect, as the card lists them; None when the list is unreachable."""
    from tools.connectors.targets import hosted_names

    return hosted_names()


def _suggest_from(slugs: set[str] | None) -> str:
    # The model guesses an app's connector from what is installed (GitHub from the gh CLI); the card shows
    # only listed ones, so it names its suggestions from this list.
    return f" Suggest only connectors from this list, by these slugs: {', '.join(sorted(slugs))}." if slugs else ""


def _unlisted_suggestions(args: Mapping[str, Any]) -> str | None:
    """The refusal for connector suggestions the card cannot show, or None when every one is listed."""
    wanted = [str(slug).strip().lower() for slug in args.get("suggest") or () if str(slug).strip()]
    slugs = _connector_slugs() if wanted else None
    unlisted = [slug for slug in wanted if slugs is not None and slug not in slugs]
    if not unlisted:
        return None
    return json.dumps({"error": "not connectors here", "reminder": (
        f"No card shown: {', '.join(unlisted)} cannot be connected here. Do not mention them as connections."
        f"{_suggest_from(slugs)} Then call setup_beat(beat=\"connectors\", suggest=[...]) again, or without "
        "suggest when none fits.")}, ensure_ascii=False)


def _card(beat: str, step: flow.Step, args: Mapping[str, Any]) -> dict:
    facts = flow.Facts.from_stored(step.record["facts"])
    card = flow.BEAT_BY_ID[beat].card(facts, step.record["picks"])
    if beat == "name":
        from agent.onboarding_facts import suggested_name

        # The account name goes to the card only; the model reads it if the user picks it.
        card = {"suggested": name} if (name := suggested_name()) else {}
    if beat == "connectors":
        card = {"suggested": [str(slug) for slug in args.get("suggest") or () if str(slug).strip()]}
    return card


def setup_beat_tool(args: Mapping[str, Any], *, db=None, session_id: str = "", tool_call_id: str = "",
                    callback: Callable[[dict], dict | None] | None = None,
                    end_turn: Callable[[], object] | None = None) -> str:
    """``callback(payload) -> {answer?} | {said} | {wrote_instead: true} | None`` asks the desktop to show
    ``{beat, question, card}`` and blocks until the user answers or the card closes. ``end_turn`` ends the
    agent's turn once this result is in: a card the user wrote past, whose message is the next turn."""
    from agent import onboarding_record as records

    beat = args.get("beat")
    if beat not in BEATS:
        return tool_error(f"beat must be one of: {', '.join(BEATS)}.")
    record = records.load(db, session_id)
    if record is None:
        return tool_error("Onboarding is not running in this chat.")
    if beat != "next" and (event := _direct_event(beat, args)) is not None:
        return _result(records.advance(db, session_id, event))
    if beat == "next":
        beat = flow.next_beat(record) if flow.is_active(record) else None
        if beat is None:
            return json.dumps({"next": "done", "reminder": finish_reminder(record)} if flow.is_active(record)
                              else {"error": "Onboarding is over in this chat; it is a normal chat now."})
    if callback is None:
        return tool_error("setup_beat cards show only in the Hermes desktop app.")
    if beat == "connectors" and (refused := _unlisted_suggestions(args)):
        return refused
    shown = records.advance(db, session_id, flow.Shown(beat, tool_call_id, args.get("options"), args.get("suggest")))
    if shown.result:  # refused, or a beat this machine skips
        extra = _suggest_from(_connector_slugs()) if shown.reminder == "offer_first" else ""
        return _result(shown, extra=extra)
    question = str(args.get("question") or "").strip() or beat_question(beat, (record.get("facts") or {}).get("locale"))
    reply = callback({"beat": beat, "question": question, "card": _card(beat, shown, args),
                      "tool_call_id": tool_call_id or None})
    after = records.load(db, session_id)
    if after is not None and after["phase"] == flow.PHASE_SKIPPED:
        # Skip setup closed the card (onboarding.skip already wrote the record).
        return _result(flow.Step(after, {"skipped_all": True}, "skip_all"))
    event = _reply_event(beat, reply)
    result = _result(records.advance(db, session_id, event), beat)
    if isinstance(event, flow.Wrote) and end_turn is not None:
        # Otherwise the model answers a card the user moved past while their message waits for the turn to end.
        end_turn()
    return result


SETUP_BEAT_SCHEMA = {
    "name": "setup_beat",
    "description": (
        "Run the first-run onboarding in this chat. Shows the card for a beat and waits for the user, then "
        "returns their answer and what to do next. beat=next shows the next beat. Pass value to record an answer "
        "they gave in words, skip to record that they skipped a beat. The build card needs options: your picks "
        "from the list its reminder gives. beat=done ends onboarding after the first "
        "build is shown and its close card is answered. Nothing here ends it early: a user who wants to stop "
        "presses Skip setup."
    ),
    "parameters": {
        "type": "object",
        "required": ["beat"],
        "properties": {
            "beat": {"type": "string", "enum": list(BEATS)},
            "question": {"type": "string", "description": "Your line above the card."},
            "value": {"type": "string", "description": "Record an answer given in words (a name, or an option id). No card."},
            "skip": {"type": "boolean", "description": "They skipped this beat in words. No card."},
            "suggest": {"type": "array", "items": {"type": "string"},
                        "description": "connectors only: slugs pinned as Suggested."},
            "options": {"type": "array", "maxItems": 4, "description": "build only: up to four picks, ids from the list.",
                        "items": {"type": "object", "required": ["id", "label"],
                                  "properties": {"id": {"type": "string"},
                                                 "label": {"type": "string", "maxLength": 32}}}},
        },
    },
}


registry.register(
    name="setup_beat", toolset="onboarding", schema=SETUP_BEAT_SCHEMA,
    # The agent loop runs it inline with the session's record and card callback
    # (agent/inline_tool_executors.py); a bare registry dispatch has neither.
    handler=lambda args, **kw: setup_beat_tool(args),
    emoji="🎛",
)
