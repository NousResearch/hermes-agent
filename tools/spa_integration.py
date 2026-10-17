"""SPA integration — preserve-before-lossy hook for the context compressor.

The fan-out's #1 convergence (6/6 lanes): the SPA only wins if the trigger
fires BEFORE lossy compression. This module provides the decision + the
preservation call, designed to hook into ContextCompressor.compress() at the
top (agent/context_compressor.py:5528), before the summary path runs.

Agnostic portability (deliberate composition): this module assumes only the
doc-chain interfaces exist (store.add with doc: targets + anchor reference).
It does NOT import the compressor; the compressor calls it. Any subset of the
hybrid composes in any order.
"""
from __future__ import annotations

import json
from typing import Any, Optional

# Load-bearing markers: content the lossy summary would flatten but the
# session needs exact (receipts, decisions, falsifiers, steers).
_LOAD_BEARING_MARKERS = (
    "verified", "passed", "refused", "receipt", "decision", "falsifier",
    "steer", "greenlight", "banked", "measured", "converged",
)


def should_preserve(messages: list[dict[str, Any]], *, threshold: int = 3) -> bool:
    """Decision: does the middle window carry load-bearing content worth
    preserving before the lossy summary runs?

    Counts load-bearing markers in the middle turns (the ones the summary
    would flatten). >= threshold -> preserve. Cheap, deterministic, no LLM.
    """
    hits = 0
    for msg in messages:
        text = str(msg.get("content", ""))
        if any(m in text.lower() for m in _LOAD_BEARING_MARKERS):
            hits += 1
            if hits >= threshold:
                return True
    return False


def preserve_session(
    store: Any,
    session_id: str,
    messages: list[dict[str, Any]],
    *,
    anchor_target: str = "memory",
) -> Optional[str]:
    """The SPA write: anchor first (the gate's requirement), then preserve the
    load-bearing content to a temp doc-chain attachment. Returns the doc name
    or None if nothing was preserved.

    Two-way door: the attachment is temporary; the dream cycle's banking is
    the prune trigger. Nothing here is a one-way door.
    """
    doc_name = f"session-{session_id}"
    load_bearing = [
        {"kind": "message", "role": m.get("role", "?"), "text": str(m.get("content", ""))[:500]}
        for m in messages
        if any(marker in str(m.get("content", "")).lower() for marker in _LOAD_BEARING_MARKERS)
    ]
    if not load_bearing:
        return None

    # Anchor first (the store's anchor-reference gate demands the pointer)
    store.add(anchor_target, f"@doc:{doc_name}")
    payload = json.dumps({"session_id": session_id, "load_bearing": load_bearing}, indent=1)
    store.add(f"doc:{doc_name}", payload)
    return doc_name
