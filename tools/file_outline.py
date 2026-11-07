"""Bounded Markdown navigation, separate from body-read bookkeeping."""

import base64
import json
import re
import secrets
import time
from dataclasses import dataclass, replace
from pathlib import Path

from agent.redact import redact_sensitive_text
from tools.file_tools_read_tracking import _read_tracker, _read_tracker_lock, _task_data

WINDOW_BYTES = 256 * 1024       # one backend read; also the longest physical line
CALL_BYTES = 2 * 1024 * 1024    # scanning budget per tool call before a cursor is issued
PAGE_HEADINGS = 500
PAGE_CHARS = 90_000
HEADING_CHARS = 150
CURSOR_SECONDS = 600
CURSOR_COUNT = 8


@dataclass
class MarkdownPosition:
    byte: int = 0
    line: int = 0
    tail: bytes = b""
    paragraph: str = ""
    fence_char: str = ""
    fence_width: int = 0
    signature: tuple = ()
    headings: int = 0

    def heading(self, text):
        self.line += 1
        if self.line == 1:
            text = text.removeprefix("\ufeff")
        text = text.removesuffix("\r")
        stripped = text.lstrip(" ")
        indented = len(text) - len(stripped) > 3 or stripped.startswith("\t")
        marker = stripped[:1]
        width = len(stripped) - len(stripped.lstrip(marker)) if marker else 0
        remainder = stripped[width:]
        if self.fence_char:
            if (not indented and marker == self.fence_char
                    and width >= self.fence_width and not remainder.strip()):
                self.fence_char = ""
            return None
        previous, self.paragraph = self.paragraph, ""
        if indented:
            return None
        if marker in ("`", "~") and width >= 3:
            if marker == "~" or "`" not in remainder:
                self.fence_char, self.fence_width = marker, width
                return None
        if marker == "#" and width <= 6 and (not remainder or remainder[0] in " \t"):
            title = remainder.strip()
            without_close = title.rstrip("#")
            if not without_close or without_close.endswith((" ", "\t")):
                title = without_close.rstrip()
            return {"line": self.line, "level": width, "heading": title}
        if marker in ("=", "-") and width and not remainder.strip():
            if previous:
                return {"line": self.line - 1, "level": 1 if marker == "=" else 2,
                        "heading": previous.strip()}
            return None
        if stripped and not re.match(r"(?:>|[-+*]\s|\d+[.)]\s)", stripped):
            self.paragraph = stripped
        return None


def _page(position, data, eof, start, limit, entries, rendered):
    """Scan one window into *entries*; a checkpoint before each line means an
    output limit never loses a heading. Returns (position, rendered, stopped)."""
    consumed = 0
    while consumed < len(data):
        checkpoint = replace(position)
        newline = data.find(b"\n", consumed)
        stop = len(data) if newline < 0 else newline + 1
        line = position.tail + data[consumed:stop]
        if len(line) > WINDOW_BYTES:
            raise ValueError(f"Markdown line exceeds {WINDOW_BYTES // 1024} KiB; "
                             "use ordinary offset/limit reads")
        position.byte += stop - consumed
        consumed = stop
        if newline < 0 and not eof:
            position.tail = line
            break
        position.tail = b""
        entry = position.heading(line.removesuffix(b"\n").decode("utf-8", errors="replace"))
        if entry is None:
            continue
        position.headings += 1
        if position.headings < start:
            continue
        title = redact_sensitive_text(entry["heading"], file_read=True)
        entry["heading"] = title[:HEADING_CHARS]
        if len(title) > HEADING_CHARS:
            entry["heading_truncated"] = True
        size = len(json.dumps(entry, ensure_ascii=False)) + 2
        if entries and rendered + size > PAGE_CHARS:
            return checkpoint, rendered, True
        entries.append(entry)
        rendered += size
        if len(entries) >= min(limit, PAGE_HEADINGS):
            return position, rendered, True
    return position, rendered, False


def outline_page(ops, path, identity, task_id, start, limit, cursor):
    """Use existing task lifetime/cleanup; tokens expose no document contents."""
    if Path(path).suffix.lower() not in {".md", ".markdown", ".mkd", ".mdown"}:
        return {"mode": "outline", "outline": [], "body_read": False,
                "note": "Outline supports Markdown only; use an ordinary read for this file"}
    with _read_tracker_lock:
        task_data = _task_data(task_id)
        cache = task_data.setdefault("outline_cursors", {})
        now = time.monotonic()
        for token in list(cache):
            if now - cache[token][0] >= CURSOR_SECONDS:
                del cache[token]
        if cursor:
            saved = cache.get(cursor)
            if saved is None or saved[1] != identity:
                return {"error": "Invalid or expired outline cursor; restart without cursor"}
            position, start = replace(saved[2]), saved[3]
        else:
            position = MarkdownPosition()
    # Several backend windows per call keep typical documents to one round trip
    # while both the bytes read and the serialized output stay bounded.
    entries, rendered, scanned = [], 0, 0
    while True:
        window = ops.read_outline_window(path, position.byte, WINDOW_BYTES, position.signature)
        if "error" in window:
            return {"mode": "outline", "error": window["error"]}
        data = base64.b64decode(window["data"], validate=True)
        position.signature = tuple(window["signature"])
        try:
            position, rendered, stopped = _page(
                position, data, position.byte + len(data) >= window["size"],
                start, limit, entries, rendered)
        except ValueError as exc:
            return {"mode": "outline", "error": str(exc)}
        scanned += len(data)
        complete = position.byte == window["size"] and not position.tail
        if complete or stopped or not data or scanned >= CALL_BYTES:
            break
    result = {"mode": "outline", "outline": entries, "body_read": False,
              "scan_complete": complete, "truncated": not complete,
              "scanned_bytes": position.byte, "file_size": window["size"]}
    if complete:
        result.update(total_lines=position.line, total_headings=position.headings)
    else:
        token = secrets.token_urlsafe(18)
        with _read_tracker_lock:
            if _read_tracker.get(task_id) is not task_data:
                return {"error": "Outline task ended during the read; restart without cursor"}
            while len(cache) >= CURSOR_COUNT:
                del cache[next(iter(cache))]
            cache[token] = (time.monotonic(), identity, position, start)
        result.update(next_cursor=token, _hint=(
            "If a relevant heading is already listed, read its body with ordinary line "
            "offset/limit; you need not finish the outline. For more headings, call "
            "mode='outline' again with the same path and cursor=next_cursor (a page may "
            "contain no headings). Cursors expire after 10 minutes."))
    return result
