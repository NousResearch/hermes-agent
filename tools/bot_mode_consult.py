"""``message_agent(mode="consult")``: ask a teammate profile and get its answer back in the same tool call.

Each (asking conversation, teammate) pair owns one scoped child session instead of a post into the teammate's
forever Bot Chat. The child is a ``delegate_task`` child on a profile route: it runs as the teammate (its
config.yaml, primary model, credentials, SOUL.md and terminal policy, bound by ``delegated_profile_scope``, the
binding a multiplexed gateway applies to a routed turn), inside THIS process, so one gateway or ``hermes serve``
serves every consult and no per-profile process starts. Its row has ``parent_session_id`` = the asking session
and carries ``_bot_consult`` = the teammate's profile id; it is seeded with the brief only. A later consult of the
same teammate from the same conversation (compression rotations included) continues that child
(``delegate_tool_continue``); another conversation gets its own. The child is a plain subagent session, so it
gets no ``message_agent`` and cannot consult further.

Opt-in per asking profile: ``bot_mode.consult_roster`` lists the teammates that may be consulted. Empty (the
default) leaves ``message_agent``'s schema byte-identical and refuses the mode.
"""

from __future__ import annotations

import json
from typing import Any, Optional

CONSULT_MARKER = "_bot_consult"
CONSULT_MODE = "consult"
# Every delegate_task(action='continue') refusal for a thread that is mid-run says it is "still" running/stopping
# or "already being continued": the teammate is busy on this pair's thread, so the asker hears that, not a new thread.
_BUSY = ("still running", "still stopping", "already being continued")


def consult_roster() -> list[str]:
    """``bot_mode.consult_roster`` of the profile in scope, as sorted unique names (empty = mode off)."""
    from tools.bot_relay import _bot_mode_cfg

    raw = _bot_mode_cfg("consult_roster", loader="load_config_readonly")
    if not isinstance(raw, list):
        return []
    return sorted({name.strip().lstrip("@") for name in raw if isinstance(name, str) and name.strip()})


def consult_mode_schema(roster: list[str]) -> dict:
    """The ``mode`` property ``message_agent`` gains while a consult roster is configured."""
    return {
        "type": "string",
        "enum": ["message", CONSULT_MODE],
        "description": (
            "'message' (default): the Bot Chat DM described above. 'consult': ask ONE teammate from "
            f"[{', '.join(roster)}] and get its answer back IN THIS tool call (no completion wait). Each "
            "teammate keeps one private thread per conversation of yours, so a follow-up consult continues "
            "it; send the full question the first time."
        ),
    }


def _ancestors(db: Any, session_id: str) -> frozenset[str]:
    return frozenset(db.get_compression_lineage(session_id) or [session_id])


def _sticky_subagent(db: Any, asker_sid: str, profile: str) -> Optional[str]:
    """The subagent_id of this conversation's consult child for ``profile``, newest first, or None."""
    from hermes_state_sessions import _parse_model_config
    from tools.delegate_tool_sessions import SUBAGENT_ID_KEY, _lineage_reader

    get, owns = _lineage_reader(db, _ancestors(db, asker_sid))
    for sid in db.session_ids_with_model_config_key(CONSULT_MARKER):
        cfg = _parse_model_config((get(sid) or {}).get("model_config"))
        if cfg.get(CONSULT_MARKER) == profile and cfg.get(SUBAGENT_ID_KEY) and owns(sid):
            return str(cfg[SUBAGENT_ID_KEY])
    return None


def _mark_child(db: Any, asker_sid: str, subagent_id: str, profile: str) -> Optional[str]:
    """Stamp the new child's session with the teammate it consults; returns that session id."""
    from tools.delegate_tool_sessions import stored_child_session

    child_sid = stored_child_session(db, subagent_id, _ancestors(db, asker_sid))
    if child_sid:
        db.patch_session_model_config(child_sid, {CONSULT_MARKER: profile})
    return child_sid


def _answer(raw: str, label: str, *, resumed: bool, child_sid: Optional[str]) -> tuple[Optional[str], str]:
    """``(subagent_id, tool result JSON)`` from a joined delegate_task result."""
    out = json.loads(raw)
    results = out.get("results") if isinstance(out, dict) else None
    if not results:
        return None, json.dumps({"error": out.get("error") or f"Consulting {label} did not run.", "to": label})
    entry = results[0]
    sid = entry.get("subagent_id")
    payload: dict[str, Any] = {"status": "answered" if entry.get("status") == "completed" else entry.get("status"),
                               "to": label, "subagent_id": sid, "resumed": resumed}
    if entry.get("status") == "completed":
        payload["answer"] = entry.get("summary") or ""
    else:
        payload["error"] = entry.get("error") or f"{label} did not answer."
    if child_sid:
        payload["session_id"] = child_sid
    return sid, json.dumps(payload, ensure_ascii=False)


def _brief_context(signature: str) -> str:
    return (f"{signature.strip()} is consulting you from one of their conversations. This is a private thread "
            "between the two of you: they read only your final response, returned to them as the answer, and may "
            "follow up here later. Answer the question directly and completely.")


def consult_teammate(target: str, body: str, sender: Any, signature: str) -> str:
    """Run one consult of ``target`` for ``sender`` (``tools.bot_mode_dm.Sender``) and return its answer JSON."""
    from tools.bot_mode_dm import _err, _resolve_local_name
    from tools.bot_mode_probe import _handle
    from tools.delegate_tool import delegate_task
    from tools.delegate_tool_continue import _transcript_store, continue_subagent

    roster = consult_roster()
    names = list(sender.roster_homes)
    raw = str(target or "").strip().lstrip("@")
    if "," in raw or not raw:
        return _err("mode='consult' asks exactly one teammate per call.", roster=roster)
    profile = _resolve_local_name(raw, names, sender.root)
    allowed = {_resolve_local_name(name, names, sender.root) for name in roster} - {None}
    if profile is None or profile == sender.me or profile not in allowed:
        return _err(f"'{raw}' is not on this profile's consult roster (bot_mode.consult_roster). "
                    "Consult one of the listed teammates, or use mode='message'.", roster=roster)
    agent = sender.agent
    asker_sid = str(getattr(agent, "session_id", "") or "")
    label = f"@{_handle(profile, names)}"
    with _transcript_store(agent) as db:
        if db is None or not asker_sid:
            return _err("mode='consult' needs a saved conversation to attach the teammate's thread to.")
        sid = _sticky_subagent(db, asker_sid, profile)
    if sid:
        raw_result = continue_subagent(sid, body, agent, background=False, joined=True)
        error = json.loads(raw_result).get("error")
        if not error or any(busy in error for busy in _BUSY):
            with _transcript_store(agent) as db:
                return _answer(raw_result, label, resumed=True,
                               child_sid=_sticky_child_sid(db, asker_sid, sid))[1]
        # Not retained here (a restart dropped its in-process route): a fresh thread, never a refusal.
    raw_result = delegate_task(tasks=[{"goal": body, "context": _brief_context(signature)}], background=False,
                               parent_agent=agent, credentials_cfg={"profile": profile, "label": label})
    sid, payload = _answer(raw_result, label, resumed=False, child_sid=None)
    if not sid:
        return payload
    with _transcript_store(agent) as db:
        child_sid = _mark_child(db, asker_sid, sid, profile)
    return json.dumps({**json.loads(payload), **({"session_id": child_sid} if child_sid else {})}, ensure_ascii=False)


def _sticky_child_sid(db: Any, asker_sid: str, subagent_id: str) -> Optional[str]:
    from tools.delegate_tool_sessions import stored_child_session

    return stored_child_session(db, subagent_id, _ancestors(db, asker_sid))
