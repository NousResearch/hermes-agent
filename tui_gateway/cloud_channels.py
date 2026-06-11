"""Cloud channels (slice 4.0, hermes side): push a shared session's message
log to meshboard-cloud so the owner's other devices can read it even when
this gateway is offline.

Strictly additive to the zero-dependency core: nothing here runs unless BOTH
``HERMES_CLOUD_TOKEN`` is set and a session is explicitly shared (the
``session.cloud_share`` RPC).  No token → ``cloud_enabled()`` is False, the
desktop hides the action, and this module is never imported on a hot path.

Design contract (docs/channels-phase4-design.md, decisions D1-D6):
  * the hosting gateway is the single writer — this pusher only ever appends
    its own session's rows, in local-id order;
  * the cloud assigns the canonical per-channel ``seq`` and dedupes replays
    on (origin_device_id, origin_message_id), so re-pushing after a crash or
    reconnect is harmless — the watermark here is just an optimisation;
  * stdlib only (urllib), one daemon thread per shared session, low cadence.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_API = "https://api.meshboard.ai"
PUSH_INTERVAL_S = 5.0
BATCH_LIMIT = 200
_ROLES = {"user", "assistant", "tool", "system"}
_PERMISSIONS = {"read", "post", "admin"}


def cloud_api() -> str:
    return (os.environ.get("HERMES_CLOUD_API") or DEFAULT_API).rstrip("/")


def cloud_token() -> str:
    return (os.environ.get("HERMES_CLOUD_TOKEN") or "").strip()


def cloud_enabled() -> bool:
    """True only when the operator opted in by configuring a token."""
    return bool(cloud_token())


def _request(method: str, path: str, body: Optional[dict] = None, timeout: float = 15.0) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        f"{cloud_api()}{path}",
        data=data,
        method=method,
        headers={
            "authorization": f"Bearer {cloud_token()}",
            **({"content-type": "application/json"} if data is not None else {}),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8")[:300]
        except Exception:
            pass
        raise RuntimeError(f"cloud {method} {path} -> {e.code}: {detail}") from e


def promote_session(
    origin_session_key: str,
    *,
    title: str = "",
    model: str = "",
    source: str = "tui_gateway",
    origin_device_id: str = "",
) -> dict:
    """Idempotently promote a local session to a cloud channel.

    Returns the channel wire dict (incl. ``id`` and ``last_seq``).
    """
    return _request("POST", "/v1/channels", {
        "origin_session_key": origin_session_key,
        "title": title or None,
        "model": model or None,
        "source": source,
        "origin_device_id": origin_device_id or None,
    })


def invite_member(channel_id: str, email: str, permission: str = "read") -> dict:
    """Create a cloud-channel invite and return the cloud response."""
    channel = urllib.parse.quote(str(channel_id), safe="")
    perm = permission if permission in _PERMISSIONS else "read"
    return _request("POST", f"/v1/channels/{channel}/invites", {
        "email": email,
        "permission": perm,
    })


def list_members(channel_id: str) -> dict:
    """Return the current cloud-channel membership view."""
    channel = urllib.parse.quote(str(channel_id), safe="")
    return _request("GET", f"/v1/channels/{channel}/members")


def set_member_permission(channel_id: str, account_id: str, permission: str = "read") -> dict:
    """Change a cloud-channel member's permission."""
    channel = urllib.parse.quote(str(channel_id), safe="")
    account = urllib.parse.quote(str(account_id), safe="")
    perm = permission if permission in _PERMISSIONS else "read"
    return _request("PATCH", f"/v1/channels/{channel}/members/{account}", {
        "permission": perm,
    })


def remove_member(channel_id: str, account_id: str) -> dict:
    """Revoke a cloud-channel member grant."""
    channel = urllib.parse.quote(str(channel_id), safe="")
    account = urllib.parse.quote(str(account_id), safe="")
    return _request("DELETE", f"/v1/channels/{channel}/members/{account}")


def accept_invite(token: str) -> dict:
    """Redeem a cloud-channel invite token for the configured account."""
    invite_token = str(token or "").strip()
    if not invite_token:
        raise ValueError("token required")
    query = urllib.parse.urlencode({"token": invite_token})
    return _request("POST", f"/v1/channels/invites/accept?{query}")


def list_channels() -> dict:
    """Return owned and joined cloud channels for the configured account."""
    return _request("GET", "/v1/channels")


def list_messages(channel_id: str, *, since_seq: int = 0, limit: int = 100) -> dict:
    """Return a page of cloud-channel messages visible to the account."""
    channel = urllib.parse.quote(str(channel_id), safe="")
    query = urllib.parse.urlencode({
        "since_seq": max(0, int(since_seq or 0)),
        "limit": max(1, min(500, int(limit or 100))),
    })
    return _request("GET", f"/v1/channels/{channel}/messages?{query}")


def list_participants(channel_id: str) -> dict:
    """Return the live cloud-channel roster visible to the account."""
    channel = urllib.parse.quote(str(channel_id), safe="")
    return _request("GET", f"/v1/channels/{channel}/participants")


def _stream_request(path: str, timeout: float = 310.0):
    req = urllib.request.Request(
        f"{cloud_api()}{path}",
        method="GET",
        headers={
            "authorization": f"Bearer {cloud_token()}",
            "accept": "text/event-stream",
        },
    )
    return urllib.request.urlopen(req, timeout=timeout)


def stream_messages(
    channel_id: str,
    *,
    since_seq: int = 0,
    stop_event: Optional[threading.Event] = None,
) -> Any:
    """Yield ``(event, payload)`` tuples from the cloud-channel SSE tail."""
    channel = urllib.parse.quote(str(channel_id), safe="")
    query = urllib.parse.urlencode({"since_seq": max(0, int(since_seq or 0))})
    event = "message"
    data_lines: list[str] = []

    with _stream_request(f"/v1/channels/{channel}/stream?{query}") as resp:
        for raw in resp:
            if stop_event is not None and stop_event.is_set():
                break

            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                if data_lines:
                    data = "\n".join(data_lines)
                    try:
                        payload: Any = json.loads(data)
                    except json.JSONDecodeError:
                        payload = data
                    yield event, payload
                event = "message"
                data_lines = []
                continue

            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event = line[6:].strip() or "message"
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())

        if data_lines and (stop_event is None or not stop_event.is_set()):
            data = "\n".join(data_lines)
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                payload = data
            yield event, payload


def rows_to_batch(rows: list[dict], device_name: str) -> list[dict]:
    """Map local ``messages`` rows to the cloud push shape.

    Pure (unit-tested).  Rows with roles the cloud doesn't accept are
    skipped rather than failing the batch; ``sender_device`` falls back to
    this gateway's device name so cross-device attribution always lands.
    """
    batch: list[dict] = []
    for row in rows:
        role = str(row.get("role") or "").strip()
        if role not in _ROLES:
            continue
        batch.append({
            "origin_message_id": str(row.get("id")),
            "origin_device_id": device_name,
            "role": role,
            "content": row.get("content"),
            "sender_device": row.get("sender_device") or (device_name if role == "user" else None),
            "tool_name": row.get("tool_name"),
            "tool_calls": row.get("tool_calls"),
            "finish_reason": row.get("finish_reason"),
            "token_count": row.get("token_count"),
            "origin_ts": row.get("timestamp"),
        })
    return batch


def _read_rows_after(db_path: str, session_id: str, after_id: int) -> list[dict]:
    """Read message rows past the watermark via a fresh read-only connection
    (no coupling to the gateway's SessionDB handle or its locks)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """SELECT id, role, content, sender_device, tool_name, tool_calls,
                       finish_reason, token_count, timestamp
                  FROM messages
                 WHERE session_id = ? AND id > ?
                 ORDER BY id ASC LIMIT ?""",
            (session_id, after_id, BATCH_LIMIT),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


class CloudChannelPusher:
    """One daemon thread per shared session: tail the local message log past
    a watermark and push batches to the channel.  Errors back off quietly —
    the cloud being down must never affect the local session."""

    def __init__(self, *, db_path: str, session_key: str, channel_id: str, device_name: str):
        self.db_path = db_path
        self.session_key = session_key
        self.channel_id = channel_id
        self.device_name = device_name
        self.watermark = 0  # local messages.id high-water mark
        self.last_seq = 0
        self.last_error: str = ""
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name=f"cloud-push-{self.session_key[:18]}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def push_once(self) -> int:
        """One tail-and-push cycle. Returns rows accepted by the cloud."""
        rows = _read_rows_after(self.db_path, self.session_key, self.watermark)
        if not rows:
            return 0
        batch = rows_to_batch(rows, self.device_name)
        accepted = 0
        if batch:
            result = _request("POST", f"/v1/channels/{self.channel_id}/messages", {"messages": batch})
            accepted = int(result.get("accepted") or 0)
            self.last_seq = int(result.get("last_seq") or self.last_seq)
        # Advance past everything read (skipped roles included) — the cloud
        # dedupe makes a conservative watermark safe, a stuck one is not.
        self.watermark = max(self.watermark, int(rows[-1]["id"]))
        return accepted

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.push_once()
                self.last_error = ""
            except Exception as e:  # network/auth — log once per change, keep tailing
                msg = str(e)
                if msg != self.last_error:
                    logger.warning("cloud channel push (%s): %s", self.session_key, msg)
                    self.last_error = msg
            self._stop.wait(PUSH_INTERVAL_S)


# Registry of active pushers, keyed by stored session key.
_pushers: dict[str, CloudChannelPusher] = {}
_pushers_lock = threading.Lock()


def share_session(
    *, db_path: str, session_key: str, device_name: str,
    title: str = "", model: str = "",
) -> dict:
    """Promote + start (or return) the pusher for a session. Idempotent."""
    with _pushers_lock:
        existing = _pushers.get(session_key)
        if existing is not None:
            return {
                "channel_id": existing.channel_id,
                "already_shared": True,
                "pushed_seq": existing.last_seq,
            }
    channel = promote_session(
        session_key, title=title, model=model, origin_device_id=device_name,
    )
    pusher = CloudChannelPusher(
        db_path=db_path, session_key=session_key,
        channel_id=str(channel["id"]), device_name=device_name,
    )
    with _pushers_lock:
        _pushers[session_key] = pusher
    pusher.start()
    return {
        "channel_id": pusher.channel_id,
        "already_shared": bool(channel.get("already_promoted")),
        "pushed_seq": int(channel.get("last_seq") or 0),
    }


def unshare_session(session_key: str) -> bool:
    """Stop pushing (the cloud log is kept; deletion is an explicit cloud op)."""
    with _pushers_lock:
        pusher = _pushers.pop(session_key, None)
    if pusher is None:
        return False
    pusher.stop()
    return True


def shared_status(session_key: str) -> Optional[dict[str, Any]]:
    with _pushers_lock:
        pusher = _pushers.get(session_key)
    if pusher is None:
        return None
    return {
        "channel_id": pusher.channel_id,
        "watermark": pusher.watermark,
        "pushed_seq": pusher.last_seq,
        "last_error": pusher.last_error or None,
    }
