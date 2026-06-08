"""Retrieval helpers for Hermes Wisdom Kernel."""

from __future__ import annotations

from wisdom.db import WisdomDB
from wisdom.models import CaptureRecord


def get_original(db: WisdomDB, capture_id: int) -> str | None:
    capture = db.get_capture(capture_id)
    return capture.original_text if capture else None


def inbox(db: WisdomDB, *, limit: int) -> list[CaptureRecord]:
    return db.list_captures(limit=limit, include_archived=False)


def search(db: WisdomDB, query: str, *, limit: int) -> list[CaptureRecord]:
    return db.search(query, limit=limit)
