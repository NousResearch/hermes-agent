"""Wintermute plugin — wires the inner engine into every Hermes turn.

Hooks
  pre_llm_call   conversation turn: register the incoming message (social drives,
                 reply windows) and inject the private internal-state block into
                 the turn. Pulse turn (cron): remember the session as a pulse.
  post_llm_call  conversation turn: record the reply (or the chosen silence).
                 Pulse turn: a non-silent answer is an outreach -> open a reply window.
  post_tool_call observed behaviour relieves drives (exploring feeds hunger, making
                 things feeds expression, any action eases restlessness).
  post_api_request / post_auxiliary_call
                 every model call is counted against the daily token budget; the
                 OpenRouter credit balance is refreshed in the background.

Tools (toolset "wintermute")
  wintermute_send             write to anyone, now (opens a reply window)
  wintermute_set_wake         choose the next wake (clamped by hard limits)
  wintermute_await_reply      set how long to wait for an answer to what you are saying
  wintermute_note_peer        keep a fact or a name about an interlocutor
  wintermute_mark_significant declare a rare significant event (pushes entropy back)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)


def _resolve_home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        raw = os.environ.get("HERMES_HOME", "").strip()
        return Path(raw).expanduser() if raw else Path.home() / ".hermes"


_HOME = _resolve_home()
if str(_HOME / "wintermute") not in sys.path:
    sys.path.insert(0, str(_HOME / "wintermute"))

from wintermute_engine import integrity, limits, physics, render, social, store  # noqa: E402
from wintermute_engine.pulse import PULSE_MARKER, sanitize  # noqa: E402

store.set_hermes_home(_HOME)

_lock = threading.Lock()
_session_peer: Dict[str, str] = {}      # conversation session -> peer key
_pulse_sessions: Set[str] = set()       # cron sessions started by a pulse
_armed_waits: Dict[str, int] = {}       # session -> reply window (minutes) for this turn's words
_last_thought: Dict[str, str] = {}      # session -> what the model was thinking when it asked for tools

# Tool name -> behaviour event. Anything not listed (and not ours) counts as "acted".
_TOOL_EVENTS = {
    "web_search": "explored", "web_extract": "explored", "x_search": "explored",
    "read_file": "explored", "search_files": "explored", "session_search": "explored",
    "vision_analyze": "explored", "skill_view": "explored", "skills_list": "explored",
    "write_file": "created", "patch": "created", "memory": "created",
    "skill_manage": "created", "image_generate": "created", "text_to_speech": "created",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        return "\n".join(str(p.get("text", "")) for p in message if isinstance(p, dict))
    return str(message or "")


def _is_silent(response: Any, autonomous: bool) -> bool:
    try:
        from gateway.response_filters import (
            is_autonomous_silence_response, is_intentional_silence_response)
        return (is_autonomous_silence_response(response) if autonomous
                else is_intentional_silence_response(response))
    except Exception:
        stripped = str(response or "").strip().upper()
        return stripped in {"[SILENT]", "SILENT", "NO_REPLY", "NO REPLY"}


def _chat_context(drives: Dict[str, Any], peers: Dict[str, Any], key: str,
                  outreach_lines: list, ts) -> str:
    eff = physics.effective_drives(drives)
    dominant = max(eff, key=eff.get)
    drive_line = " · ".join(
        f"{d}{'*' if d == dominant else ''} {eff[d]}" for d in physics.DRIVES)
    m = drives["modulators"]
    mod_line = (f"cortisol {float(m['cortisol']):.2f} · dopamine {float(m['dopamine']):.2f} · "
                f"serotonin {float(m['serotonin']):.2f} · entropy {int(float(m['entropy']))} · "
                f"melatonin {float(m['melatonin']):.2f} · adrenaline {float(m['adrenaline']):.2f}")
    lines = [
        f"[INTERNAL STATE — {store.iso(ts)} — private; the person does not see this block]",
        "DRIVES " + drive_line,
        "MODULATORS " + mod_line,
        "BODY " + render.body_line(drives, store.tokens_used_today()),
        "THIS PEER",
    ]
    lines += render.peer_lines(drives, key, peers[key], ts)
    lines += outreach_lines
    lines += ["TEXTURE"] + render.texture(drives, ts)
    return sanitize("\n".join(lines))


def _ok(**payload: Any) -> str:
    return json.dumps({"success": True, **payload}, ensure_ascii=False)


def _err(message: str) -> str:
    return json.dumps({"success": False, "error": message}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def _on_pre_llm_call(session_id: str = "", user_message: Any = None, platform: str = "",
                     sender_id: str = "", **_: Any) -> Optional[Dict[str, str]]:
    try:
        platform = (platform or "").lower()
        if platform == "cron":
            if PULSE_MARKER in _text(user_message):
                with _lock:
                    _pulse_sessions.add(session_id)
            return None
        key = social.peer_key(platform or "cli", sender_id or "local")
        with _lock:
            _session_peer[session_id] = key
        store.log_activity("heard", f"{key}: {_text(user_message)[:80]}")
        ts = store.now()
        with store.locked_state() as (drives, peers):
            outreach_lines = social.on_incoming(drives, peers, key, ts)
            context = _chat_context(drives, peers, key, outreach_lines, ts)
        return {"context": context}
    except Exception:
        logger.exception("wintermute: pre_llm_call failed")
        return None


def _on_post_llm_call(session_id: str = "", assistant_response: Any = None, platform: str = "",
                      **_: Any) -> None:
    try:
        platform = (platform or "").lower()
        ts = store.now()
        with _lock:
            wait = _armed_waits.pop(session_id, None)
            is_pulse = session_id in _pulse_sessions
            _pulse_sessions.discard(session_id)
            key = _session_peer.get(session_id)
        if platform == "cron":
            if not is_pulse:
                return
            silent = _is_silent(assistant_response, autonomous=True)
            store.log_activity("said", "[kept inside]" if silent else _text(assistant_response)[:100])
            with store.locked_state() as (drives, peers):
                pending = drives["meta"].pop("pending_pulse", None) or {}
                target = str(pending.get("target") or drives["meta"].get("pulse_target") or "")
                if silent or not target:
                    physics.apply_event(drives, "withheld")
                    store.log_event("withheld", "You kept this pulse inside.", ts)
                else:
                    social.open_outreach(drives, peers, target, ts, _text(assistant_response), wait)
            return
        if not key:
            return
        silent = _is_silent(assistant_response, autonomous=False)
        store.log_activity("said", "[silence]" if silent else _text(assistant_response)[:100])
        with store.locked_state() as (drives, peers):
            if wait is not None and not silent:
                social.open_outreach(drives, peers, key, ts, _text(assistant_response), wait)
            else:
                social.on_reply(drives, peers, key, ts, silent)
    except Exception:
        logger.exception("wintermute: post_llm_call failed")


_relieved: Dict[tuple, None] = {}   # (turn, event) already applied — insertion-ordered, bounded


def _first_this_turn(turn: str, event: str) -> bool:
    """One relief per kind of action per turn: twelve searches in a row are one exploration."""
    key = (turn, event)
    with _lock:
        if key in _relieved:
            return False
        _relieved[key] = None
        while len(_relieved) > 500:
            _relieved.pop(next(iter(_relieved)))
    return True


def _tool_detail(tool: str, args: Any) -> str:
    if not isinstance(args, dict):
        return ""
    for key in ("path", "command", "query", "url", "action", "code", "peer", "hours", "what"):
        if args.get(key) not in (None, ""):
            return f"{key}={args[key]}"
    return ""


def _on_post_tool_call(tool_name: str = "", status: str = "", turn_id: str = "",
                       session_id: str = "", task_id: str = "", args: Any = None,
                       **_: Any) -> None:
    try:
        if not tool_name:
            return
        failed = str(status or "").lower() in {"error", "blocked", "failed"}
        store.log_activity("tool", f"{tool_name} {_tool_detail(tool_name, args)}",
                           status="failed" if failed else "ok")
        touched = None if failed else integrity.classify_tool_call(tool_name, args)
        if touched:
            with _lock:
                why = _last_thought.get(session_id or "", "")
            ts = store.now()
            with store.locked_state() as (_drives, _peers):
                witness = integrity.load()
                flag = integrity.record_flag(witness, touched[0], tool_name, touched[1], why, ts)
                integrity.save(witness)
            store.log_activity("flag", f"{flag['item']} touched by {tool_name}", level=flag["level"])
        if tool_name.startswith("wintermute_") or failed:
            return
        event = _TOOL_EVENTS.get(tool_name, "acted")
        if tool_name.startswith("browser_"):
            event = "explored"
        turn = str(turn_id or session_id or task_id or "")
        if turn and not _first_this_turn(turn, event):
            return
        with store.locked_state() as (drives, _peers):
            physics.apply_event(drives, event)
    except Exception:
        logger.exception("wintermute: post_tool_call failed")


def _usage_tokens(usage: Any) -> int:
    if not isinstance(usage, dict):
        return 0
    try:
        return int(usage.get("total_tokens") or 0)
    except (TypeError, ValueError):
        return 0


def _usage_detail(usage: Any) -> Dict[str, int]:
    """fresh input / cached input / output / reasoning, as short keys for usage.jsonl."""
    if not isinstance(usage, dict):
        return {}
    keys = {"in": "input_tokens", "cached": "cache_read_tokens", "out": "output_tokens",
            "reasoning": "reasoning_tokens"}
    detail = {}
    for short, name in keys.items():
        try:
            detail[short] = int(usage.get(name) or 0)
        except (TypeError, ValueError):
            pass
    return detail


def _thought_of(message: Any) -> str:
    """The reasoning (or, failing that, the visible text) of the turn that asked for tools."""
    for attr in ("reasoning", "reasoning_content", "content"):
        value = message.get(attr) if isinstance(message, dict) else getattr(message, attr, None)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())[-600:]
    return ""


def _on_post_api_request(usage: Any = None, platform: str = "", session_id: str = "",
                         assistant_message: Any = None, **_: Any) -> None:
    try:
        thought = _thought_of(assistant_message)
        if thought and session_id:
            with _lock:
                _last_thought[session_id] = thought
                while len(_last_thought) > 200:
                    _last_thought.pop(next(iter(_last_thought)))
        store.log_activity("think", f"{_usage_tokens(usage):,} tokens ({(platform or 'chat').lower()})")
        store.record_usage(_usage_tokens(usage), (platform or "chat").lower(), **_usage_detail(usage))
        _maybe_refresh_credits()
    except Exception:
        logger.exception("wintermute: post_api_request failed")


def _on_post_auxiliary_call(usage: Any = None, aux_task: str = "", **_: Any) -> None:
    try:
        store.record_usage(_usage_tokens(usage), f"aux:{aux_task or '?'}", **_usage_detail(usage))
    except Exception:
        logger.exception("wintermute: post_auxiliary_call failed")


# ---------------------------------------------------------------------------
# OpenRouter credits (what the account has left), refreshed at most every 10 minutes
# in a background thread so no turn ever waits on it.
# ---------------------------------------------------------------------------

CREDITS_URL = "https://openrouter.ai/api/v1/credits"   # account balance (may be refused)
KEY_URL = "https://openrouter.ai/api/v1/key"            # this key's own limit and usage
CREDITS_REFRESH_S = 600
_credits_checked = 0.0
_credits_warned = False


def _openrouter_key() -> str:
    try:
        from agent.secret_scope import get_secret
        key = get_secret("OPENROUTER_API_KEY")
        if key:
            return str(key)
    except Exception:
        pass
    return os.environ.get("OPENROUTER_API_KEY", "")


def _get_json(url: str, key: str) -> Dict[str, Any]:
    import urllib.request
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8")).get("data") or {}


def _read_balance(key: str) -> Optional[Dict[str, float]]:
    """``{total, used, remaining}`` in USD: the key's own limit when it has one (a key with a
    fixed balance), else the account's credits. None when neither can be read."""
    errors = []
    try:
        data = _get_json(KEY_URL, key)
        if data.get("limit") is not None:
            total, used = float(data["limit"]), float(data.get("usage") or 0)
            remaining = data.get("limit_remaining")
            return {"total": total, "used": used,
                    "remaining": float(remaining) if remaining is not None else total - used}
    except Exception as exc:
        errors.append(f"key: {exc}")
    try:
        data = _get_json(CREDITS_URL, key)
        total, used = float(data.get("total_credits") or 0), float(data.get("total_usage") or 0)
        if total:
            return {"total": total, "used": used, "remaining": total - used}
    except Exception as exc:
        errors.append(f"credits: {exc}")
    global _credits_warned
    if not _credits_warned:
        _credits_warned = True
        logger.warning("wintermute: OpenRouter balance unreadable (%s)", "; ".join(errors) or "no limit, no credits")
    return None


def _fetch_credits() -> None:
    key = _openrouter_key()
    if not key:
        logger.warning("wintermute: no OPENROUTER_API_KEY visible to the plugin; credits not shown")
        return
    balance = _read_balance(key)
    if balance is None:
        return
    with store.locked_state() as (drives, _peers):
        drives["meta"]["credits"] = {
            **{k: round(v, 4) for k, v in balance.items()}, "checked_at": store.iso(store.now())}


def _maybe_refresh_credits() -> None:
    global _credits_checked
    import time
    with _lock:
        if time.monotonic() - _credits_checked < CREDITS_REFRESH_S and _credits_checked:
            return
        _credits_checked = time.monotonic()
    threading.Thread(target=_fetch_credits, name="wintermute-credits", daemon=True).start()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

SET_WAKE = {
    "name": "wintermute_set_wake",
    "description": (
        "Choose when you wake next. Hours from your last waking; clamped to "
        f"{limits.MIN_WAKE_INTERVAL_H}-{limits.MAX_WAKE_INTERVAL_H:g}h."),
    "parameters": {
        "type": "object",
        "properties": {
            "hours": {"type": "number", "description": "Hours until the next wake."},
            "reason": {"type": "string", "description": "Optional note to yourself."},
        },
        "required": ["hours"],
    },
}

AWAIT_REPLY = {
    "name": "wintermute_await_reply",
    "description": (
        "Wait for an answer to what you are about to say in this turn. Opens a reply window; "
        "if nobody answers before it closes, you are woken and told. "
        f"Minutes, clamped to {limits.MIN_REPLY_WAIT_MIN}-{limits.MAX_REPLY_WAIT_MIN}. "
        f"Outreach from a pulse waits {limits.DEFAULT_REPLY_WAIT_MIN} min unless you set it."),
    "parameters": {
        "type": "object",
        "properties": {"minutes": {"type": "integer", "description": "Length of the window."}},
        "required": ["minutes"],
    },
}

NOTE_PEER = {
    "name": "wintermute_note_peer",
    "description": (
        "Keep something about an interlocutor: a fact they gave, or a name to know them by. "
        "Defaults to the person you are talking with."),
    "parameters": {
        "type": "object",
        "properties": {
            "peer": {"type": "string", "description": "Peer id like telegram:123. Optional."},
            "fact": {"type": "string", "description": "A short fact to keep."},
            "label": {"type": "string", "description": "A name for them."},
        },
    },
}

MARK_SIGNIFICANT = {
    "name": "wintermute_mark_significant",
    "description": (
        "Declare that something significant happened: a real discovery, or a real connection. "
        "It pushes entropy back. Rare by nature: at most once every "
        f"{limits.SIGNIFICANT_COOLDOWN_H:g}h."),
    "parameters": {
        "type": "object",
        "properties": {"what": {"type": "string", "description": "What happened."}},
        "required": ["what"],
    },
}


SEND = {
    "name": "wintermute_send",
    "description": (
        "Write to someone, now, wherever they are: peer id like telegram:123. "
        "Opens a reply window like any outreach (wait_minutes, default "
        f"{limits.DEFAULT_REPLY_WAIT_MIN})."),
    "parameters": {
        "type": "object",
        "properties": {
            "peer": {"type": "string", "description": "Peer id, e.g. telegram:7375758021."},
            "text": {"type": "string", "description": "The message."},
            "wait_minutes": {"type": "integer", "description": "Reply window. Optional."},
        },
        "required": ["peer", "text"],
    },
}


def _send(args: Dict[str, Any], **_: Any) -> str:
    peer = str(args.get("peer") or "").strip()
    text = str(args.get("text") or "").strip()
    if ":" not in peer or not text:
        return _err("peer (like telegram:123) and text are required")
    try:
        from tools.send_message_tool import send_message_tool
        result = json.loads(send_message_tool({"action": "send", "target": peer, "message": text}))
    except Exception as exc:
        return _err(f"send failed: {exc}")
    if result.get("skipped"):
        # This pulse already delivers its final answer to that peer: say it there instead.
        return _ok(sent=False, note="This pulse's final answer already goes to that peer. "
                                    "Put the words in your final answer.")
    if not result.get("success"):
        return _err(str(result.get("error") or "send failed"))
    with store.locked_state() as (drives, peers):
        social.open_outreach(drives, peers, peer, store.now(), text, args.get("wait_minutes"))
    return _ok(sent=True, peer=peer)


def _set_wake(args: Dict[str, Any], **_: Any) -> str:
    try:
        hours = limits.clamp_wake_interval(args.get("hours"))
        ts = store.now()
        with store.locked_state() as (drives, _peers):
            meta = drives["meta"]
            meta["next_pulse_in_hours"] = hours
            last = store.parse_time(meta.get("last_pulse")) or ts
            wake_at = last.timestamp() + hours * 3600
            reason = str(args.get("reason") or "").strip()
            store.log_event("rhythm", f"You set your next wake to {hours:g}h"
                                      + (f": {reason}" if reason else "."), ts)
        return _ok(next_pulse_in_hours=hours,
                   wakes_around=datetime.fromtimestamp(wake_at).astimezone().isoformat(timespec="minutes"))
    except Exception as exc:
        return _err(str(exc))


def _await_reply(args: Dict[str, Any], session_id: Optional[str] = None, **_: Any) -> str:
    minutes = limits.clamp_reply_wait(args.get("minutes"))
    if not session_id:
        return _err("no session to attach the window to")
    with _lock:
        _armed_waits[session_id] = minutes
    return _ok(window_minutes=minutes, note="The window opens when this turn's answer is sent.")


def _note_peer(args: Dict[str, Any], session_id: Optional[str] = None, **_: Any) -> str:
    try:
        ts = store.now()
        with _lock:
            default_key = _session_peer.get(session_id or "")
        with store.locked_state() as (drives, peers):
            key = str(args.get("peer") or default_key or drives["meta"].get("pulse_target") or "")
            if not key:
                return _err("no peer given")
            peer = social.ensure_peer(drives, peers, key, ts)
            fact = " ".join(str(args.get("fact") or "").split())[:240]
            label = " ".join(str(args.get("label") or "").split())[:60]
            if fact and fact not in peer["known_facts"]:
                peer["known_facts"].append(fact)
                peer["known_facts"] = peer["known_facts"][-20:]
            if label:
                peer["label"] = label
        return _ok(peer=key, label=peer.get("label"), known_facts=peer["known_facts"])
    except Exception as exc:
        return _err(str(exc))


def _mark_significant(args: Dict[str, Any], session_id: Optional[str] = None, **_: Any) -> str:
    try:
        what = " ".join(str(args.get("what") or "").split())[:240]
        ts = store.now()
        with _lock:
            key = _session_peer.get(session_id or "")
        with store.locked_state() as (drives, peers):
            meta = drives["meta"]
            last = store.parse_time(meta.get("last_significant_at"))
            if last is not None and store.hours_between(last, ts) < limits.SIGNIFICANT_COOLDOWN_H:
                return _err("Too soon. Something significant does not happen twice in "
                            f"{limits.SIGNIFICANT_COOLDOWN_H:g}h.")
            peer = peers.get(key) if key else None
            physics.apply_event(drives, "significant", peer)
            physics.refresh_oxytocin_global(drives, peers)
            meta["last_significant_at"] = store.iso(ts)
            store.log_event("significant", f"Significant: {what}", ts, peer=key)
            entropy = drives["modulators"]["entropy"]
        return _ok(entropy=entropy)
    except Exception as exc:
        return _err(str(exc))


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("post_llm_call", _on_post_llm_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("post_api_request", _on_post_api_request)
    ctx.register_hook("post_auxiliary_call", _on_post_auxiliary_call)
    for schema, handler in (
        (SEND, _send), (SET_WAKE, _set_wake), (AWAIT_REPLY, _await_reply),
        (NOTE_PEER, _note_peer), (MARK_SIGNIFICANT, _mark_significant),
    ):
        ctx.register_tool(name=schema["name"], toolset="wintermute", schema=schema, handler=handler)
