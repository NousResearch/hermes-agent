#!/usr/bin/env python3
"""
Memory mutation ledger — append-only provenance for the durable store.

Every mutation of MEMORY.md / USER.md appends one JSON line to
``LEDGER.jsonl`` beside the memory files:

    {"ts": "...", "action": "add|replace|remove", "target": "memory|user",
     "old_sha256": "..."|null, "new_sha256": "..."|null,
     "override": false, "rationale": null, "warnings": [...]}

Why: memory entries are re-injected into every future system prompt, so a
polluted durable fact compounds across sessions. The ledger makes "when did
this fact enter memory, and why" answerable after the fact — the same
provenance standard applied everywhere else in this stack (source-grounded
research, verified commands). Tier-2 classification overrides are recorded
with their rationale so fuzzy accepts are auditable rather than silent.

The ledger is deliberately dumb: append-only, human-readable, no rotation
(one line per mutation is tiny). Failure to write a ledger line never
blocks a memory mutation — it is logged and the write proceeds (the ledger
is evidence, not a gate).
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LEDGER_NAME = "LEDGER.jsonl"


def _sha(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def ledger_path(memory_dir: Path) -> Path:
    return memory_dir / LEDGER_NAME


def record(memory_dir: Path, *, action: str, target: str,
           old_text: Optional[str] = None, new_content: Optional[str] = None,
           override: bool = False, rationale: Optional[str] = None,
           warnings: Optional[List[str]] = None) -> None:
    """Append one mutation record. Never raises — best-effort evidence."""
    try:
        memory_dir.mkdir(parents=True, exist_ok=True)
        entry: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "action": action,
            "target": target,
            "old_sha256": _sha(old_text),
            "new_sha256": _sha(new_content),
        }
        if old_text is not None:
            entry["old_preview"] = old_text[:80]
        if new_content is not None:
            entry["new_preview"] = new_content[:80]
        if override:
            entry["override"] = True
            entry["rationale"] = (rationale or "").strip() or None
            entry["warnings"] = warnings or []
        with ledger_path(memory_dir).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:  # pragma: no cover - ledger must not break memory writes
        logger.exception("memory ledger write failed (mutation already persisted)")
