"""Bounded Markdown navigation, separate from body-read bookkeeping."""

import base64
import json
import re
import secrets
import time
from dataclasses import dataclass, replace
from pathlib import Path

from agent.redact import redact_sensitive_text
from tools.file_tools_read_tracking import _read_tracker_lock, _task_data

SCAN_BYTES = 64 * 1024
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


def _page(position, data, eof, start, limit):
    """Keep a checkpoint before each line so an output limit never loses a heading."""
    entries, rendered = [], 0
    consumed = 0
    while consumed < len(data):
        checkpoint = replace(position)
        newline = data.find(b"\n", consumed)
        stop = len(data) if newline < 0 else newline + 1
        line = position.tail + data[consumed:stop]
        if len(line) > SCAN_BYTES:
            raise ValueError("Markdown line exceeds 64 KiB; use ordinary offset/limit reads")
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
            position = checkpoint
            break
        entries.append(entry)
        rendered += size
        if len(entries) >= min(limit, PAGE_HEADINGS):
            break
    return position, entries


def outline_page(ops, path, identity, task_id, start, limit, cursor):
    """Use existing task lifetime/cleanup; tokens expose no document contents."""
    if Path(path).suffix.lower() not in {".md", ".markdown", ".mkd", ".mdown"}:
        return {"mode": "outline", "outline": [], "body_read": False,
                "note": "Outline supports Markdown only; use an ordinary read for this file"}
    with _read_tracker_lock:
        cache = _task_data(task_id).setdefault("outline_cursors", {})
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
    window = ops.read_outline_window(path, position.byte, SCAN_BYTES, position.signature)
    if "error" in window:
        return {"mode": "outline", "error": window["error"]}
    data = base64.b64decode(window["data"], validate=True)
    position.signature = tuple(window["signature"])
    try:
        position, entries = _page(position, data,
                                  position.byte + len(data) >= window["size"], start, limit)
    except ValueError as exc:
        return {"mode": "outline", "error": str(exc)}
    complete = position.byte == window["size"] and not position.tail
    result = {"mode": "outline", "outline": entries, "body_read": False,
              "scan_complete": complete, "truncated": not complete,
              "scanned_bytes": position.byte, "file_size": window["size"]}
    if complete:
        result.update(total_lines=position.line, total_headings=position.headings)
    else:
        token = secrets.token_urlsafe(18)
        with _read_tracker_lock:
            while len(cache) >= CURSOR_COUNT:
                del cache[next(iter(cache))]
            cache[token] = (time.monotonic(), identity, position, start)
        result.update(next_cursor=token, _hint=(
            "Continue with mode='outline', the same path, and cursor=next_cursor, "
            "even when this page has no headings. Cursors expire after 10 minutes. "
            "Read selected bodies with ordinary line offset/limit."))
    return result
