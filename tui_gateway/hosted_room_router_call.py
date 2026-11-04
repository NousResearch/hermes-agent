"""The router call behind ``routing: auto`` rooms (``gateway/hosted_room_router.py``).

One tool-less ``auxiliary.group_router`` request per unaddressed message, made off the policy lock on its own
thread so a Send RPC never waits on a model. Its answer is recorded as ``room.routed`` only when the planner,
re-run under the policy lock after the call, still wants exactly that message routed: a Stop, a newer message
in the thread, or another worker's answer landing meanwhile discards it.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from agent.memory_provider import spawn_context_thread
from gateway import hosted_room_router as router
from gateway import hosted_rooms

logger = logging.getLogger(__name__)

_ROUTER_TASK = "group_router"
_ROUTER_MAX_TOKENS = 200


def _member_roster(room: Mapping[str, Any], homes: Mapping[str, Path]) -> list[dict[str, str]]:
    from hermes_cli.profiles import read_profile_meta

    roster = []
    for member in room.get("members") or ():
        home = homes.get(str(member.get("profile") or ""))
        local = (member.get("target") or {}).get("kind", "local") == "local"
        meta = read_profile_meta(home) if local and home is not None else {}
        roster.append({"handle": str(member["handle"]), "name": str(member.get("display_name") or ""),
                       "description": str(meta.get("description") or "")})
    return roster


def _recent_messages(db_path: Path, room: Mapping[str, Any], before_seq: int) -> list[dict[str, str]]:
    """The last few user/member messages anywhere in the room before the routed one."""
    page = hosted_rooms.read_events(db_path, room_id=room["room_id"], since_seq=max(0, before_seq - 64), limit=64)
    names = {str(m["member_id"]): str(m.get("display_name") or m["handle"]) for m in room.get("members") or ()}
    rows = [event for event in page.get("events") or () if int(event["seq"]) < before_seq
            and event["kind"] in {"message.user", "message.member"}]
    return [{"author": "user" if event["kind"] == "message.user"
             else names.get(str(event["payload"].get("member_id")), "member"),
             "text": str(event["payload"].get("text") or "")} for event in rows[-router.ROUTER_RECENT_MESSAGES:]]


def ask_router(*, root: Path, db_path: Path, room: Mapping[str, Any], discussion: Mapping[str, Any],
               context: Mapping[str, str], homes: Mapping[str, Path]) -> tuple[str, ...] | None:
    """Member ids the router picked (``()`` = everyone), or None when the call failed or answered invalidly.
    Runs under the room owner's profile scope: its ``auxiliary.group_router`` config and credentials."""
    from agent.auxiliary_client import call_llm, extract_content_or_reasoning
    from gateway.run import _profile_runtime_scope

    payload = discussion.get("payload") or {}
    prompt = router.router_prompt(
        room_name=str(room.get("name") or ""), purpose=context.get("purpose", ""), notes=context.get("notes", ""),
        roster=_member_roster(room, homes), recent=_recent_messages(db_path, room, int(discussion["seq"])),
        message=str(payload.get("text") or ""),
        attachments=[str(item.get("name") or "file") for item in payload.get("attachments") or ()])
    handles = {str(m["handle"]).casefold(): str(m["member_id"]) for m in room.get("members") or ()}
    try:
        with _profile_runtime_scope(root):
            response = call_llm(
                task=_ROUTER_TASK, temperature=0, max_tokens=_ROUTER_MAX_TOKENS,
                messages=[{"role": "system", "content": router.ROUTER_SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}])
        text = extract_content_or_reasoning(response)
    except Exception as exc:  # health: allow BLE001 -- any router failure falls back to everyone
        logger.warning("hosted room %s: router call failed, everyone answers: %s", room["room_id"], exc)
        return None
    picked = router.parse_router_answer(text, handles)
    if picked is None:
        # Only the answer's size: its text may quote the user's message.
        logger.warning("hosted room %s: router answer was not a valid choice (%d chars), everyone answers",
                       room["room_id"], len(str(text or "")))
    return picked


class RouterCalls:
    """At most one in-flight router call per (room, message); ``record`` runs once it answers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inflight: set[tuple[str, str]] = set()

    def start(self, key: tuple[str, str], call: Callable[[], tuple[str, ...] | None],
              record: Callable[[tuple[str, ...] | None], None]) -> bool:
        with self._lock:
            if key in self._inflight:
                return False
            self._inflight.add(key)

        def run() -> None:
            try:
                record(call())
            except Exception:  # health: allow BLE001 -- a lost answer is re-asked on the next preparation
                logger.warning("hosted room %s: recording the router answer failed", key[0], exc_info=True)
            finally:
                with self._lock:
                    self._inflight.discard(key)
        # The owner's profile scope is bound explicitly inside ask_router (_profile_runtime_scope).
        spawn_context_thread(run, name=f"hosted-room-router-{key[0][:16]}").start()
        return True

    def inflight(self, key: tuple[str, str]) -> bool:
        with self._lock:
            return key in self._inflight
