"""status_card.py — Real-time Claude session status card via Gateway adapter.

Monitors the session JSONL file and continuously updates a Telegram message
with the latest session state. Runs in a background thread, non-blocking.
Uses Gateway adapter callbacks for sending — no independent Bot instance needed.

Lifecycle:
    card = StatusCard(session_uuid, send_func=adapter.send, edit_func=adapter.edit_message, ...)
    card.start()                                   # sends initial message
    # ... Claude runs ...
    card.stop(summary="Done. 5 tools, 2m 30s")     # final update
"""

import asyncio
import json
import logging
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

def get_jsonl_path(session_uuid: str) -> Path:
    """Get the JSONL file path for a Claude session UUID.
    
    Searches ~/.claude/projects/ subdirectories for the matching file,
    avoiding hardcoded project directory names.
    """
    if not re.match(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        session_uuid,
    ):
        return Path(f"/nonexistent/{session_uuid}.jsonl")

    projects_dir = Path.home() / ".claude" / "projects"
    if not projects_dir.exists():
        return projects_dir / f"{session_uuid}.jsonl"

    target = f"{session_uuid}.jsonl"
    for subdir in sorted(projects_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        candidate = subdir / target
        if candidate.exists():
            return candidate

    # Fallback: return path even if it doesn't exist yet (first poll)
    # Use the most recently modified project dir as best guess
    subdirs = list(projects_dir.iterdir())
    if subdirs:
        most_recent = max(subdirs, key=lambda p: p.stat().st_mtime)
        return most_recent / target
    return projects_dir / target


# ------------------------------------------------------------------
# JSONL Parsing
# ------------------------------------------------------------------

def parse_jsonl(jsonl_path: Path) -> dict:
    """Parse JSONL and return structured session state."""
    if not jsonl_path.exists():
        return {"status": "no_session"}

    entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not entries:
        return {"status": "empty"}

    result = {
        "status": "active",
        "total_entries": len(entries),
        "assistant_count": sum(1 for e in entries if e.get("type") == "assistant"),
        "user_count": sum(1 for e in entries if e.get("type") == "user"),
    }

    # Walk backwards to find the latest meaningful entry
    for entry in reversed(entries):
        entry_type = entry.get("type", "")
        if entry_type not in ("assistant", "user"):
            continue

        result["entry_type"] = entry_type
        result["timestamp"] = entry.get("timestamp", "")

        msg = entry.get("message", {})
        content = msg.get("content", [])

        if entry_type == "assistant":
            result["model"] = msg.get("model", "")
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "tool_use":
                    result["tool"] = item.get("name", "")
                    result["tool_input"] = item.get("input", {})
                elif item.get("type") == "text" and "text" not in result:
                    result["text"] = item.get("text", "")[:200]
        else:
            # user content
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "tool_result":
                    result["tool_result"] = str(item.get("content", ""))[:200]
                elif item.get("type") == "text" and "text" not in result:
                    result["text"] = item.get("text", "")[:200]
        break

    return result


# ------------------------------------------------------------------
# Formatting
# ------------------------------------------------------------------

# Spinner line extraction — finds the most recent spinner line from tmux output
_SPINNER_LINE_RE = re.compile(
    r"^[" + re.escape("✶✽✻✢·*") + r"]\s+(.+)$",
)


def _extract_spinner_text(raw_output: str, max_len: int = 80) -> str:
    """Extract the most recent spinner text from raw tmux output.

    Returns e.g. "Boogieing… (12s · ↓ 88 tokens · thinking with xhigh effort)"
    """
    if not raw_output:
        return ""
    lines = raw_output.split("\n")
    for line in reversed(lines):
        line = line.strip()
        m = _SPINNER_LINE_RE.match(line)
        if m:
            text = m.group(1).strip()
            return text[:max_len]
    return ""


_TOOL_ICONS = {
    "Write": "✏️",
    "Edit": "📝",
    "Read": "📖",
    "Bash": "⚡",
    "Agent": "🤖",
    "TaskUpdate": "📋",
    "TaskCreate": "📋",
    "Grep": "🔍",
}


def _build_header(session_name: str, session_id: str, tmux_session: str = "") -> str:
    """Build status card header with optional session identification."""
    header = "📊 Claude Status"
    if session_name or session_id or tmux_session:
        parts = []
        if session_name:
            parts.append(session_name[:30])
        if tmux_session:
            parts.append(tmux_session)
        if session_id:
            parts.append(session_id)
        header += f" [{' · '.join(parts)}]"
    return header


def format_status_card(state: dict, observer_state: dict = None, max_length: int = 500,
                       session_name: str = "", session_id: str = "", tmux_session: str = "") -> str:
    """Format session state as a compact Telegram status card."""
    real_time = observer_state or {}
    # When observer reports a live state, prefer it over stale JSONL status.
    # This prevents "Starting session..." from sticking when the JSONL file
    # hasn't been written yet but the observer already detects IDLE/THINKING.
    observer_state_name = real_time.get("state")
    if observer_state_name and state["status"] in ("no_session", "empty"):
        # Observer has real data — build card using observer state instead
        header = _build_header(session_name, session_id, tmux_session)
        lines = [header]

        if observer_state_name == "THINKING":
            lines.append("🤔 Thinking...")
        elif observer_state_name == "TOOL_CALL":
            lines.append("🔧 Working...")
        elif observer_state_name == "PERMISSION":
            lines.append("⏸️ Waiting for permission")
        else:
            lines.append("✅ Idle")

        activity = real_time.get("current_activity", "")
        activity_detail = real_time.get("activity_detail", "")
        if activity and activity != "idle":
            if activity_detail:
                lines.append(f"⚡ {activity}: {activity_detail}")
            else:
                lines.append(f"⚡ {activity}")

        result = "\n".join(lines)
        if len(result) > max_length:
            result = result[:max_length - 3] + "..."
        return result

    if state["status"] == "no_session":
        h = _build_header(session_name, session_id, tmux_session)
        return f"{h}\n⏳ Starting session..."
    if state["status"] == "empty":
        h = _build_header(session_name, session_id, tmux_session)
        return f"{h}\n⏳ Waiting for activity..."

    header = _build_header(session_name, session_id, tmux_session)
    lines = [header]

    # State indicator — trust session state over observer when session is IDLE
    # Observer polls every 5s and may report stale THINKING while Claude is actually idle
    session_state = state.get("state", "IDLE")
    
    # During startup grace period (5s), trust session state over observer
    session_ready = state.get("session_ready", True)
    if not session_ready:
        # Startup phase: trust session state directly
        current_state = session_state
    elif session_state == "IDLE":
        # Trust session's IDLE — it's more accurate than stale observer state
        current_state = "IDLE"
    else:
        current_state = real_time.get("state") or session_state

    if current_state == "THINKING":
        lines.append("🤔 Thinking...")
    elif current_state == "TOOL_CALL":
        lines.append("🔧 Working...")
    elif current_state == "PERMISSION":
        perm_tool = real_time.get("tool_name") or state.get("tool")
        perm_target = real_time.get("tool_target") or state.get("tool_input", {})
        perm_detail = ""
        if perm_tool:
            icon = _TOOL_ICONS.get(perm_tool, "🔧")
            if isinstance(perm_target, dict):
                perm_detail = _format_tool_detail(perm_tool, perm_target)
            elif perm_target:
                perm_detail = str(perm_target)[:50]
            if perm_detail:
                lines.append(f"⏸️ {icon} {perm_tool}: {perm_detail}")
            else:
                lines.append(f"⏸️ {icon} {perm_tool}")
        else:
            lines.append("⏸️ Waiting for permission")
    else:
        lines.append("✅ Idle")

    # Current activity (from observer)
    activity = real_time.get("current_activity", "")
    activity_detail = real_time.get("activity_detail", "")
    if activity and activity != "idle":
        if activity_detail:
            lines.append(f"⚡ {activity}: {activity_detail}")
        else:
            lines.append(f"⚡ {activity}")

    # Message counts
    ac = real_time.get("assistant_count") or state.get("assistant_count", 0)
    uc = real_time.get("user_count") or state.get("user_count", 0)
    lines.append(f"💬 {ac} / {uc}")

    # Tool call - prefer observer's tool info
    tool = real_time.get("tool_name") or state.get("tool")
    if tool:
        icon = _TOOL_ICONS.get(tool, "🔧")
        detail = real_time.get("activity_detail") or state.get("tool_input", {})
        if isinstance(detail, dict):
            detail = _format_tool_detail(tool, detail)
        if detail:
            lines.append(f"{icon} {tool}: {detail}")
        else:
            lines.append(f"{icon} {tool}")

    # Text preview — only for THINKING (show spinner verb) and TOOL_CALL (show detail)
    if current_state == "THINKING":
        # Extract spinner verb from recent_output (e.g. "✻ Boogieing… (12s)")
        raw_output = real_time.get("recent_output", "")
        spinner_text = _extract_spinner_text(raw_output)
        if spinner_text:
            lines.append(f"💭 {spinner_text}")
    elif current_state == "TOOL_CALL":
        text = real_time.get("recent_output") or state.get("text") or state.get("recent_output")
        if text:
            preview = text[:80] + "..." if len(text) > 80 else text
            first_line = preview.split("\n", 1)[0].strip()
            if first_line:
                lines.append(f"💭 {first_line}")

    result = "\n".join(lines)

    if len(result) > max_length:
        result = result[:max_length - 3] + "..."

    return result


def _format_tool_detail(tool: str, tool_input: dict) -> str:
    """Extract a short detail string from tool input."""
    if tool in ("Write", "Edit", "Read"):
        path = tool_input.get("file_path", "")
        if "/" in path:
            path = path.rsplit("/", 2)[-1] if len(path.split("/")) > 3 else path
        return path
    if tool == "Bash":
        cmd = tool_input.get("command", "")
        return (cmd[:50] + "...") if len(cmd) > 50 else cmd
    if tool in ("TaskUpdate", "TaskCreate"):
        return f"#{tool_input.get('taskId', '?')} {tool_input.get('status', '')}"
    if tool == "Agent":
        return tool_input.get("description", "")
    return ""


# ------------------------------------------------------------------
# StatusCard — Gateway adapter-based message lifecycle
# ------------------------------------------------------------------

class StatusCard:
    """Manages a persistent Telegram status message via Gateway adapter.

    Uses async callbacks (send/edit/delete) from the Gateway's platform
    adapter instead of creating its own Bot instance. This ensures the
    status card uses the same proxy/auth config as the main message flow.

    Usage::

        card = StatusCard(
            session_uuid,
            loop=gateway_loop,
            send_func=adapter.send,
            edit_func=adapter.edit_message,
            delete_func=adapter.delete_message,
            chat_id=chat_id,
        )
        card.start()
        # ... Claude runs ...
        card.stop(summary="Done. 5 tools, 2m")
    """

    _POLL_INTERVAL = 3.0
    _MIN_EDIT_INTERVAL = 2.0
    _MAX_MSG_LEN = 4096
    _INIT_RETRIES = 3
    _RETRY_DELAY = 2.0
    _BUMP_INTERVAL = 60.0  # re-send card at bottom every 60s even if state unchanged

    def __init__(
        self,
        session_uuid: str,
        loop: asyncio.AbstractEventLoop,
        send_func: Callable,
        edit_func: Callable,
        delete_func: Callable,
        chat_id: str,
        poll_interval: float = 3.0,
        max_card_length: int = 500,
        bump_threshold: int = 3,
        session_name: str = "",
        session_id: str = "",
        tmux_session: str = "",
    ):
        self._session_uuid = session_uuid
        self._session_name = session_name
        self._session_id = session_id
        self._tmux_session = tmux_session
        self._loop = loop
        self._send_func = send_func
        self._edit_func = edit_func
        self._delete_func = delete_func
        self._chat_id = str(chat_id)
        self._jsonl_path = get_jsonl_path(session_uuid)
        self._poll_interval = poll_interval
        self._max_card_length = max_card_length
        self._bump_threshold = bump_threshold

        self._message_id: Optional[str] = None
        self._last_card_text: Optional[str] = None
        self._last_edit_time: float = 0.0
        self._bump_count: int = 0
        self._running = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Cached observer data for real-time status
        self._observer_state: dict = {}
        # State transition tracking (protected by _state_lock)
        self._state_lock = threading.Lock()
        self._send_lock = threading.Lock()  # protects _sending flag + message state
        self._sending = False  # True while an async message operation is in flight
        self._prev_state: Optional[str] = None
        self._pending_bump: bool = False
        self._prev_activity: Optional[str] = None
        self._STATE_TRANSITIONS = {
            ("IDLE", "THINKING"),
            ("THINKING", "TOOL_CALL"),
            ("TOOL_CALL", "THINKING"),
            ("THINKING", "IDLE"),
            ("TOOL_CALL", "IDLE"),
            ("IDLE", "TOOL_CALL"),
            ("PERMISSION", "IDLE"),
            ("PERMISSION", "THINKING"),
            ("PERMISSION", "TOOL_CALL"),
            ("TOOL_CALL", "PERMISSION"),
            ("THINKING", "PERMISSION"),
        }

    def _should_bump(self, new_state: str) -> bool:
        """Check if state transition warrants a bump (new message at bottom).

        When a valid transition is detected, sets ``_pending_bump`` so the bump
        survives across polls until it is actually performed.
        """
        with self._state_lock:
            if self._prev_state is None:
                self._prev_state = new_state
                return False
            transition = (self._prev_state, new_state)
            self._prev_state = new_state
            if transition in self._STATE_TRANSITIONS:
                self._pending_bump = True
                return True
            return False

    def _clear_pending_bump(self) -> None:
        """Clear the pending-bump flag after a bump has been performed."""
        with self._state_lock:
            self._pending_bump = False

    def update_from_observer(self, info: dict) -> None:
        """Update cached observer state from real-time session monitoring."""
        self._observer_state = {
            "status": "active",
            "state": info.get("state", "IDLE"),
            "current_activity": info.get("current_activity", ""),
            "activity_detail": info.get("activity_detail", ""),
            "recent_output": info.get("recent_output", ""),
            "tool_name": info.get("tool_name", ""),
            "tool_target": info.get("tool_target", ""),
            "assistant_count": info.get("assistant_count", 0),
            "user_count": info.get("user_count", 0),
        }

        # Trigger immediate update via Gateway adapter
        if self._running and self._loop and self._loop.is_running():
            try:
                now = time.monotonic()
                if (now - self._last_edit_time) < self._MIN_EDIT_INTERVAL:
                    return

                new_state = info.get("state", "IDLE")
                do_bump = self._should_bump(new_state)

                # Claim the sending slot (non-blocking)
                with self._send_lock:
                    if self._sending:
                        return  # _run_loop is already sending
                    self._sending = True

                async def _immediate_send():
                    try:
                        # Guard: don't send before initial message is sent
                        if self._message_id is None:
                            return
                        # Re-check: _run_loop may have already updated
                        state = parse_jsonl(self._jsonl_path)
                        card_text = format_status_card(state, observer_state=self._observer_state, max_length=self._max_card_length, session_name=self._session_name, session_id=self._session_id, tmux_session=self._tmux_session)
                        if card_text == self._last_card_text and not do_bump:
                            return
                        try:
                            if do_bump and self._message_id:
                                await self._bump_message(card_text)
                            else:
                                success = await self._edit_telegram(card_text)
                                if not success:
                                    await self._send_new_message(card_text)
                        except Exception as e:
                            logger.warning("Immediate status send failed: %s", e)
                        self._last_card_text = card_text
                        self._last_edit_time = time.monotonic()
                        self._bump_count = 0
                    finally:
                        with self._send_lock:
                            self._sending = False

                asyncio.run_coroutine_threadsafe(_immediate_send(), self._loop)
            except Exception as e:
                with self._send_lock:
                    self._sending = False
                logger.warning("Immediate status send failed: %s", e)

    @property
    def message_id(self) -> Optional[str]:
        return self._message_id

    def start(self) -> Optional[str]:
        """Send initial message and start background polling."""
        if self._running:
            return self._message_id

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"status-card-{self._session_uuid[:8]}",
            daemon=True,
        )
        self._thread.start()
        return None

    def stop(self, summary: Optional[str] = None) -> None:
        """Stop polling and optionally replace message with final summary."""
        self._running = False
        self._stop_event.set()

        if summary and self._message_id and self._loop and self._loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._send_or_edit(summary), self._loop
                )
                future.result(timeout=5.0)
            except Exception as e:
                logger.warning("Final status card update failed: %s", e)

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Background thread: polls JSONL and updates via Gateway adapter."""
        # Send initial message (with retries)
        initial_text = format_status_card(parse_jsonl(self._jsonl_path), max_length=self._max_card_length, session_name=self._session_name, session_id=self._session_id, tmux_session=self._tmux_session)
        sent = False
        for attempt in range(1, self._INIT_RETRIES + 1):
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._send_new_message(initial_text), self._loop
                )
                sent = future.result(timeout=10.0)
                if sent:
                    break
            except Exception as e:
                logger.warning(
                    "Status card: initial send attempt %d/%d failed: %s",
                    attempt, self._INIT_RETRIES, e,
                )
                if attempt < self._INIT_RETRIES:
                    self._stop_event.wait(timeout=self._RETRY_DELAY)

        if not sent:
            logger.error(
                "Status card: all %d initial send attempts failed, running without status card",
                self._INIT_RETRIES,
            )
            self._running = False
            return

        self._last_card_text = initial_text
        self._last_edit_time = time.monotonic()
        self._prev_state = None
        self._pending_bump = False
        self._prev_activity = None

        # Poll loop
        _poll_count = 0
        while self._running and not self._stop_event.is_set():
            try:
                state = parse_jsonl(self._jsonl_path)
                card_text = format_status_card(state, observer_state=self._observer_state, max_length=self._max_card_length, session_name=self._session_name, session_id=self._session_id, tmux_session=self._tmux_session)

                _poll_count += 1
                if _poll_count <= 5 or _poll_count % 20 == 0:
                    logger.debug(
                        "StatusCard poll #%d: jsonl=%s status=%s card_len=%d last=%s msg_id=%s running=%s",
                        _poll_count, self._jsonl_path.name, state.get("status"),
                        len(card_text), self._last_card_text[:30] if self._last_card_text else None,
                        self._message_id, self._running,
                    )

                current_state = (self._observer_state or {}).get("state", "IDLE")
                should_bump = self._should_bump(current_state) or self._pending_bump

                # Claim the sending slot (non-blocking)
                with self._send_lock:
                    if self._sending:
                        # Observer path is already sending — skip this poll
                        self._stop_event.wait(timeout=self._poll_interval)
                        continue
                    self._sending = True

                try:
                    if card_text != self._last_card_text:
                        logger.info(
                            "StatusCard poll: text changed (msg_id=%s, state=%s, observer=%s, bump=%s)",
                            self._message_id, state.get("status"), current_state, should_bump,
                        )
                        now = time.monotonic()
                        if (now - self._last_edit_time) >= self._MIN_EDIT_INTERVAL or should_bump:
                            if should_bump and self._message_id:
                                try:
                                    future = asyncio.run_coroutine_threadsafe(
                                        self._delete_func(self._chat_id, self._message_id), self._loop
                                    )
                                    future.result(timeout=5.0)
                                except Exception:
                                    pass
                                self._message_id = None
                                future = asyncio.run_coroutine_threadsafe(
                                    self._send_new_message(card_text), self._loop
                                )
                                sent_new = future.result(timeout=10.0)
                                if sent_new:
                                    self._last_card_text = card_text
                                    self._last_edit_time = time.monotonic()
                                    self._bump_count = 0
                                self._clear_pending_bump()
                            else:
                                future = asyncio.run_coroutine_threadsafe(
                                    self._edit_telegram(card_text), self._loop
                                )
                                success = future.result(timeout=10.0)
                                if success:
                                    self._last_card_text = card_text
                                    self._last_edit_time = time.monotonic()
                                    self._bump_count = 0
                                    self._clear_pending_bump()
                                else:
                                    logger.warning("StatusCard edit failed (bump_count=%d)", self._bump_count)
                                    self._bump_count += 1
                                    if self._bump_count >= self._bump_threshold:
                                        future = asyncio.run_coroutine_threadsafe(
                                            self._send_new_message(card_text), self._loop
                                        )
                                        future.result(timeout=10.0)
                                        self._last_card_text = card_text
                                        self._last_edit_time = time.monotonic()
                                        self._bump_count = 0
                                        self._clear_pending_bump()
                    else:
                        now = time.monotonic()
                        if self._message_id and (now - self._last_edit_time) >= self._BUMP_INTERVAL:
                            should_bump = True

                    # Periodic bump (text unchanged)
                    if should_bump and self._message_id and card_text == self._last_card_text:
                        try:
                            future = asyncio.run_coroutine_threadsafe(
                                self._delete_func(self._chat_id, self._message_id), self._loop
                            )
                            future.result(timeout=5.0)
                        except Exception:
                            pass
                        self._message_id = None
                        future = asyncio.run_coroutine_threadsafe(
                            self._send_new_message(card_text), self._loop
                        )
                        sent_new = future.result(timeout=10.0)
                        if sent_new:
                            self._last_card_text = card_text
                            self._last_edit_time = time.monotonic()
                            self._bump_count = 0
                        self._clear_pending_bump()
                finally:
                    with self._send_lock:
                        self._sending = False
            except Exception as e:
                logger.warning("Status card poll error: %s", e)

            self._stop_event.wait(timeout=self._poll_interval)

    async def _edit_telegram(self, text: str) -> bool:
        """Edit the status message via Gateway adapter. Returns True on success."""
        if not self._message_id:
            return False
        try:
            result = await self._edit_func(
                chat_id=self._chat_id,
                message_id=self._message_id,
                content=text[:self._MAX_MSG_LEN],
            )
            return getattr(result, "success", False)
        except Exception as e:
            logger.warning("Status card edit error: %s", e)
            return False

    async def _send_new_message(self, text: str) -> bool:
        """Send a new status message via Gateway adapter. Returns True on success."""
        try:
            result = await self._send_func(
                chat_id=self._chat_id,
                content=text[:self._MAX_MSG_LEN],
            )
            if getattr(result, "success", False) and getattr(result, "message_id", None):
                old_id = self._message_id
                self._message_id = str(result.message_id)
                logger.info("Status card sent: %s -> %s", old_id, self._message_id)
                return True
            return False
        except Exception as e:
            logger.warning("Status card send failed: %s", e)
            return False

    async def _bump_message(self, text: str) -> bool:
        """Delete old message and send new one at bottom."""
        if self._message_id:
            try:
                await self._delete_func(self._chat_id, self._message_id)
            except Exception as e:
                logger.warning("Status card delete old msg failed: %s", e)
        result = await self._send_new_message(text)
        return result

    async def _send_or_edit(self, text: str) -> None:
        """Used for the final summary — edit existing message."""
        if self._message_id:
            try:
                await self._edit_func(
                    chat_id=self._chat_id,
                    message_id=self._message_id,
                    content=text[:self._MAX_MSG_LEN],
                )
            except Exception as e:
                logger.warning("Status card final update error: %s", e)
