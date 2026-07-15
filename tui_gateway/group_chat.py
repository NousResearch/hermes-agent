"""Persistent local group-chat orchestration for TUI gateway surfaces."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable


class GroupChatStore:
    DEFAULT_MAX_MENTION_DEPTH = 4
    MAX_MENTION_DEPTH = 10
    DEFAULT_TRIGGER_TOKENS = 48_000
    DEFAULT_MAX_HISTORY_TOKENS = 32_000
    DEFAULT_TAIL_MESSAGE_COUNT = 12

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._db:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS group_rooms (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at REAL NOT NULL,
                    workspace TEXT,
                    max_mention_depth INTEGER NOT NULL DEFAULT 4,
                    trigger_tokens INTEGER NOT NULL DEFAULT 48000,
                    max_history_tokens INTEGER NOT NULL DEFAULT 32000,
                    tail_message_count INTEGER NOT NULL DEFAULT 12,
                    summary TEXT NOT NULL DEFAULT '',
                    summary_through_seq INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS group_members (
                    room_id TEXT NOT NULL, profile TEXT NOT NULL, name TEXT NOT NULL,
                    runtime_session_id TEXT, stored_session_id TEXT, ordinal INTEGER NOT NULL,
                    PRIMARY KEY (room_id, profile)
                );
                CREATE TABLE IF NOT EXISTS group_timeline (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, room_id TEXT NOT NULL,
                    event_type TEXT NOT NULL, member_profile TEXT,
                    payload_json TEXT NOT NULL, created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS group_projection_cursors (
                    room_id TEXT NOT NULL, profile TEXT NOT NULL, seq INTEGER NOT NULL,
                    PRIMARY KEY (room_id, profile)
                );
                CREATE TABLE IF NOT EXISTS group_mention_dispatches (
                    room_id TEXT NOT NULL, idempotency_key TEXT NOT NULL,
                    PRIMARY KEY (room_id, idempotency_key)
                );
                """
            )
            columns = {
                row["name"]
                for row in self._db.execute("PRAGMA table_info(group_members)")
            }
            if "runtime_session_id" not in columns:
                self._db.execute("ALTER TABLE group_members ADD COLUMN runtime_session_id TEXT")
                if "session_id" in columns:
                    self._db.execute(
                        "UPDATE group_members SET runtime_session_id=session_id"
                    )
            if "stored_session_id" not in columns:
                self._db.execute("ALTER TABLE group_members ADD COLUMN stored_session_id TEXT")
            room_columns = {
                row["name"] for row in self._db.execute("PRAGMA table_info(group_rooms)")
            }
            if "workspace" not in room_columns:
                self._db.execute("ALTER TABLE group_rooms ADD COLUMN workspace TEXT")
            room_columns = {
                row["name"] for row in self._db.execute("PRAGMA table_info(group_rooms)")
            }
            for name, definition in (
                ("max_mention_depth", "INTEGER NOT NULL DEFAULT 4"),
                ("trigger_tokens", "INTEGER NOT NULL DEFAULT 48000"),
                ("max_history_tokens", "INTEGER NOT NULL DEFAULT 32000"),
                ("tail_message_count", "INTEGER NOT NULL DEFAULT 12"),
                ("summary", "TEXT NOT NULL DEFAULT ''"),
                ("summary_through_seq", "INTEGER NOT NULL DEFAULT 0"),
            ):
                if name not in room_columns:
                    self._db.execute(f"ALTER TABLE group_rooms ADD COLUMN {name} {definition}")

    def close(self) -> None:
        self._db.close()

    def create_room(
        self,
        name: str,
        members: list[dict],
        context: dict | None = None,
        workspace: str | None = None,
        max_mention_depth: int = DEFAULT_MAX_MENTION_DEPTH,
    ) -> dict:
        room_id = uuid.uuid4().hex
        now = time.time()
        context = context or {}
        with self._lock, self._db:
            self._db.execute(
                """INSERT INTO group_rooms(
                       id,name,created_at,workspace,max_mention_depth,trigger_tokens,
                       max_history_tokens,tail_message_count
                   ) VALUES(?,?,?,?,?,?,?,?)""",
                (
                    room_id,
                    name,
                    now,
                    workspace,
                    max(0, min(self.MAX_MENTION_DEPTH, int(max_mention_depth))),
                    int(context.get("trigger_tokens", self.DEFAULT_TRIGGER_TOKENS)),
                    int(context.get("max_history_tokens", self.DEFAULT_MAX_HISTORY_TOKENS)),
                    int(context.get("tail_message_count", self.DEFAULT_TAIL_MESSAGE_COUNT)),
                ),
            )
            for ordinal, member in enumerate(members):
                self._db.execute(
                    """INSERT INTO group_members(
                           room_id,profile,name,runtime_session_id,stored_session_id,ordinal
                       ) VALUES(?,?,?,?,?,?)""",
                    (room_id, member["profile"], member.get("name") or member["profile"],
                     member.get("runtime_session_id") or member.get("session_id"),
                     member.get("stored_session_id"), ordinal),
                )
        room = self.get_room(room_id)
        assert room is not None
        return room

    def get_room(self, room_id: str) -> dict | None:
        with self._lock:
            room = self._db.execute("SELECT * FROM group_rooms WHERE id=?", (room_id,)).fetchone()
            if room is None:
                return None
            members = self._db.execute(
                """SELECT profile,name,runtime_session_id,stored_session_id
                   FROM group_members WHERE room_id=? ORDER BY ordinal""",
                (room_id,),
            ).fetchall()
        member_list = [
            {k: row[k] for k in ("profile", "name", "runtime_session_id", "stored_session_id") if row[k] is not None}
            for row in members
        ]
        return {
            "id": room["id"],
            "name": room["name"],
            "created_at": room["created_at"],
            "workspace": room["workspace"],
            "max_mention_depth": room["max_mention_depth"],
            "trigger_tokens": room["trigger_tokens"],
            "max_history_tokens": room["max_history_tokens"],
            "tail_message_count": room["tail_message_count"],
            "summary": room["summary"],
            "summary_through_seq": room["summary_through_seq"],
            "members": member_list,
            "profiles": [member["profile"] for member in member_list],
            "messages": self.messages(room_id),
        }

    def list_rooms(self) -> list[dict]:
        with self._lock:
            ids = [r[0] for r in self._db.execute("SELECT id FROM group_rooms ORDER BY created_at DESC")]
        return [room for room_id in ids if (room := self.get_room(room_id))]

    def save_summary(self, room_id: str, summary: str, through_seq: int) -> None:
        with self._lock, self._db:
            cur = self._db.execute(
                "UPDATE group_rooms SET summary=?, summary_through_seq=? WHERE id=?",
                (summary, int(through_seq), room_id),
            )
            if cur.rowcount == 0:
                raise KeyError(room_id)

    def set_member_session(
        self, room_id: str, profile: str, runtime_session_id: str,
        stored_session_id: str | None = None,
    ) -> None:
        with self._lock, self._db:
            if stored_session_id is None:
                self._db.execute(
                    "UPDATE group_members SET runtime_session_id=? WHERE room_id=? AND profile=?",
                    (runtime_session_id, room_id, profile),
                )
            else:
                self._db.execute(
                    """UPDATE group_members
                       SET runtime_session_id=?, stored_session_id=?
                       WHERE room_id=? AND profile=?""",
                    (runtime_session_id, stored_session_id, room_id, profile),
                )

    def clear_member_runtime_session(self, room_id: str, profile: str) -> None:
        with self._lock, self._db:
            self._db.execute(
                "UPDATE group_members SET runtime_session_id=NULL WHERE room_id=? AND profile=?",
                (room_id, profile),
            )

    def delete_room(self, room_id: str) -> bool:
        with self._lock, self._db:
            exists = self._db.execute(
                "SELECT 1 FROM group_rooms WHERE id=?", (room_id,)
            ).fetchone()
            if exists is None:
                return False
            self._db.execute("DELETE FROM group_projection_cursors WHERE room_id=?", (room_id,))
            self._db.execute("DELETE FROM group_mention_dispatches WHERE room_id=?", (room_id,))
            self._db.execute("DELETE FROM group_timeline WHERE room_id=?", (room_id,))
            self._db.execute("DELETE FROM group_members WHERE room_id=?", (room_id,))
            self._db.execute("DELETE FROM group_rooms WHERE id=?", (room_id,))
        return True

    def append_event(self, room_id: str, event_type: str, payload: dict, member_profile: str | None = None) -> dict:
        now = time.time()
        with self._lock, self._db:
            exists = self._db.execute(
                "SELECT 1 FROM group_rooms WHERE id=?", (room_id,)
            ).fetchone()
            if exists is None:
                raise KeyError(room_id)
            cur = self._db.execute(
                "INSERT INTO group_timeline(room_id,event_type,member_profile,payload_json,created_at) VALUES(?,?,?,?,?)",
                (room_id, event_type, member_profile, json.dumps(payload, ensure_ascii=False), now),
            )
        return {"seq": cur.lastrowid, "room_id": room_id, "type": event_type,
                "member_profile": member_profile, "payload": payload, "created_at": now}

    def timeline(self, room_id: str, after: int = 0) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM group_timeline WHERE room_id=? AND seq>? ORDER BY seq", (room_id, after)
            ).fetchall()
        return [{"seq": r["seq"], "room_id": r["room_id"], "type": r["event_type"],
                 "member_profile": r["member_profile"], "payload": json.loads(r["payload_json"]),
                 "created_at": r["created_at"]} for r in rows]

    def messages(self, room_id: str) -> list[dict]:
        messages: list[dict] = []
        active_by_profile: dict[str, dict] = {}
        for event in self.timeline(room_id):
            payload = event["payload"]
            profile = event["member_profile"]
            event_type = event["type"]
            if event_type == "user.message":
                messages.append({
                    "id": f"group-{event['seq']}",
                    "seq": event["seq"],
                    "role": "user",
                    "content": str(payload.get("text") or ""),
                    "status": "complete",
                    "created_at": event["created_at"],
                })
                continue
            if not profile or event_type not in {
                "message.start", "message.delta", "message.complete", "error", "agent.error",
                "approval.request", "tool.start", "tool.progress", "tool.complete",
            }:
                continue
            if event_type == "message.start" or profile not in active_by_profile:
                message_id = str(payload.get("message_id") or f"group-{event['seq']}")
                message = {
                    "id": message_id,
                    "seq": event["seq"],
                    "role": "assistant",
                    "profile": profile,
                    "content": "",
                    "status": "streaming",
                    "created_at": event["created_at"],
                    "tools": [],
                }
                messages.append(message)
                active_by_profile[profile] = message
            elif payload.get("message_id"):
                message_id = str(payload["message_id"])
                existing = next((item for item in messages if item["id"] == message_id), None)
                if existing is not None:
                    active_by_profile[profile] = existing
            message = active_by_profile[profile]
            if event_type.startswith("tool."):
                tool_id = str(payload.get("tool_id") or "")
                tool = next((item for item in message["tools"] if item.get("tool_id") == tool_id), None)
                if tool is None:
                    tool = {key: payload[key] for key in ("tool_id", "name", "context") if key in payload}
                    tool["status"] = "running"
                    message["tools"].append(tool)
                if event_type == "tool.progress":
                    tool.update({key: value for key, value in payload.items() if key != "message_id"})
                elif event_type == "tool.complete":
                    tool.update({key: value for key, value in payload.items() if key not in {"message_id", "context"}})
                    tool["status"] = "complete"
                continue
            if event_type == "message.delta":
                message["content"] += str(payload.get("text") or "")
            elif event_type == "message.complete":
                message["content"] = str(payload.get("text") or message["content"])
                message["status"] = "complete"
                active_by_profile.pop(profile, None)
            elif event_type in {"error", "agent.error"}:
                message["content"] = str(payload.get("message") or message["content"])
                message["status"] = "error"
                active_by_profile.pop(profile, None)
            elif event_type == "approval.request":
                message["status"] = "approval"
                message["runtime_session_id"] = str(payload.get("session_id") or "")
                message["approval"] = {
                    "command": str(payload.get("command") or ""),
                    "description": str(payload.get("description") or ""),
                    "choices": payload.get("choices"),
                    "allow_permanent": payload.get("allow_permanent"),
                    "smart_denied": bool(payload.get("smart_denied")),
                }
        return messages

    def message_page(
        self, room_id: str, before_seq: int | None = None, limit: int = 50,
    ) -> dict:
        """Return completed canonical messages without splitting event/tool groups."""
        limit = max(1, min(int(limit), 200))
        events = self.timeline(room_id)
        messages = self.messages(room_id)
        boundaries: dict[str, tuple[int, int]] = {}
        active: dict[str, tuple[str, int]] = {}
        for event in events:
            payload = event["payload"]
            if event["type"] == "user.message":
                boundaries[f"group-{event['seq']}"] = (event["seq"], event["seq"])
                continue
            profile = event["member_profile"]
            if not profile:
                continue
            message_id = str(payload.get("message_id") or "")
            if event["type"] == "message.start":
                message_id = message_id or f"group-{event['seq']}"
                active[profile] = (message_id, event["seq"])
            elif message_id and profile not in active:
                active[profile] = (message_id, event["seq"])
            elif not message_id and profile in active:
                message_id = active[profile][0]
            if message_id and profile in active:
                boundaries[message_id] = (active[profile][1], event["seq"])
                if event["type"] in {"message.complete", "error", "agent.error"}:
                    active.pop(profile, None)
        canonical = sorted(
            [
                (boundaries[item["id"]][0], boundaries[item["id"]][1], item)
                for item in messages
                if item.get("status") in {"complete", "error"} and item["id"] in boundaries
            ],
            key=lambda entry: (entry[1], entry[0], entry[2]["id"]),
        )
        for index, (_start, _end, item) in enumerate(canonical, start=1):
            item["seq"] = index
        cutoff = int(before_seq) if before_seq is not None else None
        eligible = [entry for index, entry in enumerate(canonical, start=1) if cutoff is None or index < cutoff]
        selected = eligible[-limit:]
        first_index = canonical.index(selected[0]) + 1 if selected else None
        return {
            "messages": [entry[2] for entry in selected],
            "before_seq": first_index,
            "has_more": len(eligible) > len(selected),
        }

    def projection_cursor(self, room_id: str, profile: str) -> int:
        with self._lock:
            row = self._db.execute(
                "SELECT seq FROM group_projection_cursors WHERE room_id=? AND profile=?",
                (room_id, profile),
            ).fetchone()
        return int(row["seq"]) if row else 0

    def set_projection_cursor(self, room_id: str, profile: str, seq: int) -> None:
        with self._lock, self._db:
            self._db.execute(
                """INSERT INTO group_projection_cursors(room_id,profile,seq)
                   VALUES(?,?,?)
                   ON CONFLICT(room_id,profile) DO UPDATE SET seq=excluded.seq""",
                (room_id, profile, int(seq)),
            )

    def claim_mention_dispatch(self, room_id: str, idempotency_key: str) -> bool:
        with self._lock, self._db:
            if self._db.execute("SELECT 1 FROM group_rooms WHERE id=?", (room_id,)).fetchone() is None:
                return False
            cursor = self._db.execute(
                "INSERT OR IGNORE INTO group_mention_dispatches(room_id,idempotency_key) VALUES(?,?)",
                (room_id, idempotency_key),
            )
        return cursor.rowcount == 1


def route_mentions(text: str, members: list[dict]) -> list[dict]:
    """Route explicit @name/@profile mentions; no mention broadcasts to all."""
    has_mention = "@" in text
    explicit_all = bool(re.search(r"(?<!\w)@all(?![\w-])", text, re.IGNORECASE))
    selected = []
    for member in members:
        aliases = {str(member.get("profile", "")), str(member.get("name", ""))}
        matched = any(
            alias and re.search(rf"(?<!\w)@{re.escape(alias)}(?![\w-])", text, re.IGNORECASE)
            for alias in aliases
        )
        if explicit_all or matched:
            selected.append(member)
    return selected if selected or has_mention else list(members)


class GroupChatCoordinator:
    """Runs independent profile sessions and projects their events to one timeline."""

    def __init__(self, store: GroupChatStore, create_session: Callable[[dict], dict[str, str]],
                 submit: Callable[[str, str, Callable[[str, dict], None]], Any],
                 interrupt: Callable[[str], Any], approve: Callable[[str, str, bool], Any],
                 recover_session: Callable[[str, str, str | None, str | None], str | None] | None = None,
                 summarizer: Callable[[str, list[dict]], str] | None = None,
                 token_counter: Callable[[str], int] | None = None):
        self.store, self.create_session, self.submit = store, create_session, submit
        self.interrupt, self.approve = interrupt, approve
        self.recover_session = recover_session
        self.summarizer = summarizer or self._structured_summary
        if token_counter is None:
            from agent.model_metadata import estimate_tokens_rough
            token_counter = estimate_tokens_rough
        self.token_counter = token_counter
        self._pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="group-chat")
        self._turn_locks: dict[tuple[str, str], threading.Lock] = {}
        self._turn_locks_lock = threading.Lock()
        self._session_locks: dict[tuple[str, str], threading.Lock] = {}
        self._compression_locks: dict[str, threading.Lock] = {}

    def _ensure_member_session(self, room_id: str, profile: str) -> str | None:
        key = (room_id, profile)
        with self._turn_locks_lock:
            lock = self._session_locks.setdefault(key, threading.Lock())
        with lock:
            room = self.store.get_room(room_id)
            member = next((item for item in (room or {}).get("members", []) if item["profile"] == profile), None)
            if member is None:
                return None
            if member.get("runtime_session_id"):
                return str(member["runtime_session_id"])
            created = self.create_session({
                "profile": profile,
                "room_id": room_id,
                "workspace": (room or {}).get("workspace"),
            })
            session_id = created["runtime_session_id"]
            self.store.set_member_session(room_id, profile, session_id, created["stored_session_id"])
            return session_id

    @staticmethod
    def _structured_summary(previous: str, messages: list[dict]) -> str:
        lines = ["## 房间上下文摘要"]
        if previous:
            lines.extend(["### 先前摘要", previous])
        lines.append("### 已完成消息")
        for message in messages:
            speaker = message.get("profile") or message.get("role") or "unknown"
            lines.append(f"- {speaker}: {message.get('content', '')}")
        return "\n".join(lines)

    def compress_room(self, room_id: str) -> bool:
        with self._turn_locks_lock:
            lock = self._compression_locks.setdefault(room_id, threading.Lock())
        with lock:
            return self._compress_room_serialized(room_id)

    def _compress_room_serialized(self, room_id: str) -> bool:
        room = self.store.get_room(room_id)
        if room is None:
            raise KeyError(room_id)
        completed = [
            message for message in self.store.messages(room_id)
            if message.get("status") in {"complete", "error"}
        ]
        current = [
            message for message in completed
            if self._message_end_seq(room_id, message["id"]) > room["summary_through_seq"]
        ]
        def message_tokens(items: list[dict]) -> int:
            return sum(self.token_counter(str(item.get("content") or "")) for item in items)

        if message_tokens(current) < room["trigger_tokens"]:
            return False
        tail_count = max(0, int(room["tail_message_count"]))
        keep_from = max(0, len(current) - tail_count)
        while keep_from < len(current) and message_tokens(current[keep_from:]) > int(room["max_history_tokens"]):
            keep_from += 1
        old = current[:keep_from]
        if not old:
            return False
        through = self._message_end_seq(room_id, old[-1]["id"])
        summary = self.summarizer(room["summary"], old)
        self.store.save_summary(room_id, summary, through)
        return True

    def _message_end_seq(self, room_id: str, message_id: str) -> int:
        if message_id.startswith("group-"):
            return int(message_id.removeprefix("group-"))
        result = 0
        for event in self.store.timeline(room_id):
            if str(event["payload"].get("message_id") or "") == message_id:
                result = event["seq"]
        return result

    def send(self, room_id: str, text: str) -> dict:
        room = self.store.get_room(room_id)
        if room is None:
            raise KeyError(room_id)
        targets = route_mentions(text, room["members"])
        user_event = self.store.append_event(room_id, "user.message", {"text": text})
        for member in targets:
            session_id = self._ensure_member_session(room_id, member["profile"])
            if not session_id:
                continue
            self._pool.submit(self._run_member, room_id, member["profile"], session_id, text, 0)
        return {"event": user_event, "targets": [m["profile"] for m in targets]}

    def _incremental_prompt(self, room_id: str, profile: str, text: str, boundary: int | None = None) -> str:
        room = self.store.get_room(room_id)
        summary = str((room or {}).get("summary") or "")
        summary_through = int((room or {}).get("summary_through_seq") or 0)
        cursor = max(self.store.projection_cursor(room_id, profile), summary_through)
        events = self.store.timeline(room_id, cursor)
        if boundary is not None:
            events = [event for event in events if event["seq"] <= boundary]
        peer_lines = [
            f"@{event['member_profile']}: {event['payload'].get('text', '')}"
            for event in events
            if event["type"] == "message.complete"
            and event["member_profile"]
            and event["member_profile"] != profile
            and event["payload"].get("text")
        ]
        sections = []
        if summary:
            sections.append("Room summary through canonical cursor " + str(summary_through) + ":\n" + summary)
        if peer_lines:
            sections.append("Recent group updates since your last turn:\n" + "\n".join(peer_lines))
        if not sections:
            return text
        return "\n\n".join(sections) + "\n\nCurrent user message:\n" + text

    def _run_member(self, room_id: str, profile: str, session_id: str, text: str,
                    mention_depth: int = 0, projection_boundary: int | None = None) -> None:
        key = (room_id, profile)
        with self._turn_locks_lock:
            turn_lock = self._turn_locks.setdefault(key, threading.Lock())
        with turn_lock:
            self._run_member_serialized(
                room_id, profile, session_id, text, mention_depth, projection_boundary
            )

    def _run_member_serialized(self, room_id: str, profile: str, session_id: str, text: str,
                               mention_depth: int = 0,
                               projection_boundary: int | None = None) -> None:
        if self.store.get_room(room_id) is None:
            return
        if self.recover_session:
            room = self.store.get_room(room_id)
            member = next((item for item in (room or {}).get("members", []) if item["profile"] == profile), None)
            if member:
                session_id = self.recover_session(
                    room_id, profile, member.get("runtime_session_id"), member.get("stored_session_id")
                ) or session_id
        timeline = self.store.timeline(room_id)
        boundary = projection_boundary if projection_boundary is not None else (
            timeline[-1]["seq"] if timeline else 0
        )
        current_room = self.store.get_room(room_id)
        if current_room is not None:
            self.compress_room(room_id)

        def project(event_type: str, payload: dict) -> dict:
            if event_type == "message.complete":
                payload = {**payload, "mention_depth": mention_depth}
            event = self.store.append_event(room_id, event_type, payload, profile)
            if event_type == "message.complete":
                self.store.set_projection_cursor(room_id, profile, boundary)
                room = self.store.get_room(room_id)
                reply = str(payload.get("text") or "")
                targets = [
                    member for member in route_mentions(reply, (room or {}).get("members", []))
                    if member["profile"] != profile
                ] if (
                    re.search(r"(?<!\w)@", reply)
                    and mention_depth < int((room or {}).get("max_mention_depth", 4))
                ) else []
                for member in targets:
                    target_session = self._ensure_member_session(room_id, member["profile"])
                    source_id = str(payload.get("message_id") or event["seq"])
                    dispatch_key = f"{profile}:{source_id}:{member['profile']}"
                    if target_session and self.store.claim_mention_dispatch(room_id, dispatch_key):
                        self._pool.submit(
                            self._run_member, room_id, member["profile"], target_session,
                            reply, mention_depth + 1, event["seq"] - 1,
                        )
            return event
        try:
            self.submit(session_id, self._incremental_prompt(room_id, profile, text, boundary), project)
        except Exception as exc:
            project("agent.error", {"message": str(exc)})

    def stop(self, room_id: str, profile: str) -> Any:
        room = self.store.get_room(room_id)
        member = next((m for m in (room or {}).get("members", []) if m["profile"] == profile), None)
        if not member or not member.get("runtime_session_id"):
            return False
        try:
            session_id = member["runtime_session_id"]
            if self.recover_session:
                session_id = self.recover_session(
                    room_id, profile, session_id, member.get("stored_session_id")
                ) or session_id
            result = self.interrupt(session_id)
        except Exception:
            return False
        self.store.append_event(room_id, "agent.stopped", {}, profile)
        return result

    def respond_approval(self, room_id: str, profile: str, choice: str, all_: bool = False) -> Any:
        room = self.store.get_room(room_id)
        member = next((m for m in (room or {}).get("members", []) if m["profile"] == profile), None)
        if not member or not member.get("runtime_session_id"):
            return False
        session_id = member["runtime_session_id"]
        if self.recover_session:
            session_id = self.recover_session(
                room_id, profile, session_id, member.get("stored_session_id")
            ) or session_id
        return self.approve(session_id, choice, all_)
