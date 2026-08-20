"""stream_output.py — Stream Claude Code tmux output to gateway in real-time.

Bridges the gap between the polling-based wait_for_idle and the gateway's
message editing capabilities.  Uses the session observer's callback path
to receive incremental output updates, then schedules async edits via the
gateway's event loop.

Usage (from claude_session_tool.py):
    streamer = SessionOutputStreamer(session, adapter_info)
    # Wire into observer callback chain
    old_cb = session._status_callback
    def chained(info):
        old_cb(info) if old_cb else None
        streamer.on_observer_update(info)
    session._status_callback = chained
    # ... wait_for_idle runs ...
    streamer.finish()  # final edit, removes cursor
"""

import asyncio
import logging
import time
import re
from typing import Optional

logger = logging.getLogger(__name__)


class SessionOutputStreamer:
    """Streams incremental Claude Code output to a chat via gateway adapter.

    Thread-safe: receives updates from the observer thread, schedules
    async edits on the gateway's event loop via loop.call_soon_threadsafe.
    """

    # Output noise to strip
    _NOISE_PATTERNS = [
        re.compile(r"^─{5,}$"),
        re.compile(r"bypass permissions (on|off)", re.IGNORECASE),
        re.compile(r"shift\+tab to cycle", re.IGNORECASE),
        re.compile(r"esc to interrupt", re.IGNORECASE),
        re.compile(r"/model|/mcp|/ide for Visual Studio Code", re.IGNORECASE),
    ]
    _SPINNER_RE = re.compile(
        r"^[" + re.escape("✶✽✻✢·*●") + r"]\s+\S",
    )

    def __init__(self, session, adapter_info: dict, config: Optional[dict] = None):
        self._session = session
        self._adapter_info = adapter_info
        self._loop = adapter_info.get("loop")
        self._chat_id = adapter_info.get("chat_id", "")
        self._send_func = adapter_info.get("send_func")
        self._edit_func = adapter_info.get("edit_func")

        # Config
        cfg = config or {}
        self._edit_interval = cfg.get("edit_interval", 3.0)
        self._max_length = cfg.get("max_length", 3800)
        self._cursor = cfg.get("cursor", " ⋯")
        self._min_delta_chars = cfg.get("min_delta_chars", 30)

        # State
        self._last_marker = session._send_marker
        self._message_id: Optional[str] = None
        self._last_sent_text = ""
        self._last_edit_time = 0.0
        self._accumulated = ""
        self._already_sent = False
        self._finished = False

    @property
    def already_sent(self) -> bool:
        return self._already_sent

    def on_observer_update(self, info: dict) -> None:
        """Called from observer thread. Reads incremental output and schedules edit."""
        if self._finished:
            return
        try:
            buf = self._session._buf
            current_marker = buf.total_count()
            if current_marker <= self._last_marker:
                return
            lines = buf.since(self._last_marker)
            self._last_marker = current_marker
            if not lines:
                return
            text = "\n".join(l.text for l in lines)
            cleaned = self._clean_output(text)
            if not cleaned.strip():
                return
            self._accumulated += ("\n" if self._accumulated else "") + cleaned

            # Check if enough time/new content to edit
            now = time.monotonic()
            elapsed = now - self._last_edit_time
            delta_chars = len(self._accumulated) - len(self._last_sent_text)
            if elapsed >= self._edit_interval or delta_chars >= self._min_delta_chars * 2:
                self._schedule_edit()
        except Exception as e:
            logger.debug("on_observer_update error: %s", e)

    def finish(self) -> None:
        """Final edit: send accumulated output without cursor."""
        self._finished = True
        if not self._accumulated or not self._loop or not self._loop.is_running():
            return
        display = self._truncate(self._accumulated)
        if display.strip():
            self._schedule_async(self._do_send_or_edit, display)

    def _schedule_edit(self) -> None:
        """Schedule an edit on the event loop."""
        if not self._loop or not self._loop.is_running():
            return
        display = self._truncate(self._accumulated) + self._cursor
        self._schedule_async(self._do_send_or_edit, display)

    def _truncate(self, text: str) -> str:
        if len(text) > self._max_length:
            return text[-self._max_length:]
        return text

    def _schedule_async(self, coro_fn, *args) -> None:
        """Schedule an async function on the gateway event loop (thread-safe)."""
        try:
            future = asyncio.run_coroutine_threadsafe(coro_fn(*args), self._loop)
            future.add_done_callback(self._on_edit_done)
        except Exception as e:
            logger.debug("schedule_async error: %s", e)

    def _on_edit_done(self, future) -> None:
        """Callback after async edit completes."""
        try:
            ok = future.result()
            if ok:
                self._last_sent_text = self._accumulated
                self._last_edit_time = time.monotonic()
        except Exception as e:
            logger.debug("edit done error: %s", e)

    async def _do_send_or_edit(self, text: str) -> bool:
        """Send or edit message. Returns True on success."""
        text = text.strip()
        if not text:
            return True
        try:
            if self._message_id and self._edit_func:
                result = await self._edit_func(
                    chat_id=self._chat_id,
                    message_id=self._message_id,
                    content=text,
                )
                if getattr(result, "success", False):
                    self._already_sent = True
                    return True

            # Send new message
            if self._send_func:
                result = await self._send_func(
                    chat_id=self._chat_id,
                    content=text,
                )
                if getattr(result, "success", False) and getattr(result, "message_id", None):
                    self._message_id = str(result.message_id)
                    self._already_sent = True
                    return True
            return False
        except Exception as e:
            logger.error("Stream send/edit error: %s", e)
            return False

    def _clean_output(self, text: str) -> str:
        """Strip tmux noise from output."""
        for pat in self._NOISE_PATTERNS:
            text = pat.sub("", text)
        lines = text.splitlines()
        cleaned = []
        for line in lines:
            if self._SPINNER_RE.match(line.strip()):
                continue
            stripped = line.strip()
            if stripped:
                cleaned.append(stripped)
        return "\n".join(cleaned)
