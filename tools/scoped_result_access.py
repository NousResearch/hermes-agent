"""Opt-in result paging for restricted bindings, using native spill storage.

This library registers no model tool. A binding supplies its reader name and
admitted producer names, enters one invocation scope and exposes ``read`` through
its own dispatch fence. Provider text never creates an access receipt.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import threading

from tools.spill_safety import ensure_spill_dir, write_text_exclusive

_ACTIVE: ContextVar[ScopedResultAccess | None] = ContextVar("scoped_result_access", default=None)
MAX_RESULT_BYTES = 2 * 1024 * 1024
MAX_INVOCATION_BYTES = 16 * 1024 * 1024
MAX_PAGE_CHARS = 4096
MAX_RESULTS = 128


@dataclass(frozen=True)
class _Receipt:
    path: Path
    device: int
    inode: int
    size: int
    digest: str


def current_result_access(tool_name: str, call_id: str,
                          parent_call_id: str | None = None) -> ScopedResultAccess | None:
    """Admit native results, including later budget spills of known producer calls."""
    access = _ACTIVE.get()
    if access is None or not access._active:
        return None
    with access._lock:
        if tool_name in access.producers:
            access._calls.add(call_id)
            return access
        if tool_name == "__budget_enforcement__" and (parent_call_id or call_id) in access._calls:
            return access
    return None


class ScopedResultAccess:
    """A single-use invocation capability; exit revokes access but retains spills."""

    def __init__(self, producers: set[str], reader_name: str):
        if not producers or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", reader_name):
            raise ValueError("producer names and a reader name are required")
        self.producers = frozenset(producers)
        self.reader_name = reader_name
        self._receipts: dict[str, _Receipt] = {}
        self._calls: set[str] = set()
        self._bytes = 0
        self._attempts = 0
        self._active = False
        self._used = False
        self._lock = threading.RLock()

    def __enter__(self):
        from tools.tool_result_storage import get_spillover_dir

        if self._used:
            raise RuntimeError("result access scopes cannot be reused")
        self._used = self._active = True
        self._root = get_spillover_dir().absolute()
        self._token = _ACTIVE.set(self)
        return self

    def __exit__(self, *_):
        with self._lock:
            self._active = False
            self._receipts.clear()
            self._calls.clear()
        _ACTIVE.reset(self._token)

    def persist(self, text: str) -> str:
        """Only called by native persistence after its inline size decision."""
        from tools.tool_result_storage import get_spillover_dir

        if len(text) > MAX_RESULT_BYTES:
            return json.dumps({"status": "result_limit_exceeded", "coverage": "incomplete"})
        data = text.encode("utf-8", errors="replace")
        with self._lock:
            if (not self._active or _ACTIVE.get() is not self
                    or get_spillover_dir().absolute() != self._root):
                return json.dumps({"status": "result_unavailable"})
            if (len(data) > MAX_RESULT_BYTES or self._bytes + len(data) > MAX_INVOCATION_BYTES
                    or self._attempts >= MAX_RESULTS):
                return json.dumps({"status": "result_limit_exceeded", "coverage": "incomplete"})
            # A failed write can retain a partial artifact. Charge admission before I/O.
            self._bytes += len(data)
            self._attempts += 1
            result_id = secrets.token_hex(24)
            path = self._root / f"scoped-{result_id}.txt"
            try:
                ensure_spill_dir(self._root)
                write_text_exclusive(path, data.decode("utf-8"))
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_size != len(data):
                    raise OSError("incomplete result write")
            except OSError:
                return json.dumps({"status": "result_unavailable", "coverage": "incomplete"})
            self._receipts[result_id] = _Receipt(
                path, info.st_dev, info.st_ino, len(data), hashlib.sha256(data).hexdigest(),
            )
            return json.dumps({
                "status": "stored", "reader": self.reader_name, "result_id": result_id,
                # Native retention tracks references in transcript text, independently of read authority.
                "archive": f"spillover/{path.name}",
                "offset": 0, "limit": MAX_PAGE_CHARS, "bytes": len(data),
                "instruction": "Read bounded pages of this saved result; do not repeat the provider request.",
            })

    def read(self, result_id: str, offset: int = 0, limit: int = MAX_PAGE_CHARS) -> dict:
        """Return a character page from an unchanged invocation-owned regular file."""
        from tools.tool_result_storage import get_spillover_dir

        if (not isinstance(result_id, str) or type(offset) is not int or type(limit) is not int
                or offset < 0 or not 1 <= limit <= MAX_PAGE_CHARS):
            return {"status": "invalid_request"}
        with self._lock:
            receipt = self._receipts.get(result_id)
            if (not self._active or _ACTIVE.get() is not self or receipt is None
                    or get_spillover_dir().absolute() != self._root):
                return {"status": "unavailable"}
            try:
                data = self._read_verified(receipt)
                text = data.decode("utf-8")
            except (OSError, ValueError):
                return {"status": "unavailable"}
            if offset > len(text):
                return {"status": "invalid_request"}
            end = min(offset + limit, len(text))
            return {"status": "ok", "result_id": result_id, "offset": offset,
                    "text": text[offset:end], "next_offset": end if end < len(text) else None,
                    "total_chars": len(text)}

    @staticmethod
    def _read_verified(receipt: _Receipt) -> bytes:
        before = receipt.path.lstat()
        if (not stat.S_ISREG(before.st_mode)
                or (before.st_dev, before.st_ino, before.st_size)
                != (receipt.device, receipt.inode, receipt.size)):
            raise ValueError("changed result")
        fd = os.open(receipt.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as stream:
            actual = os.fstat(stream.fileno())
            if (not stat.S_ISREG(actual.st_mode)
                    or (actual.st_dev, actual.st_ino, actual.st_size)
                    != (receipt.device, receipt.inode, receipt.size)):
                raise ValueError("changed result")
            data = stream.read(receipt.size + 1)
        if len(data) != receipt.size or hashlib.sha256(data).hexdigest() != receipt.digest:
            raise ValueError("changed result")
        return data
