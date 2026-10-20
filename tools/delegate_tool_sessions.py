"""The stored session of a delegated child, addressed by its ``subagent_id``.

``delegate_tool._build_child_agent`` stamps the child's ``model_config`` with ``SUBAGENT_ID_KEY``
beside ``_delegate_from``. Compression copies ``model_config`` onto every continuation, so each
segment of the child carries it. The live registry forgets a child when it ends; this marker is
how a viewer that only holds the ``subagent_id`` (the dispatch result, a ``subagent.list`` row)
still finds the child's stored transcript afterwards.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional

SUBAGENT_ID_KEY = "_subagent_id"


def stored_child_session(db: Any, subagent_id: str, conversation_ids: Iterable[str]) -> Optional[str]:
    """The newest segment of child *subagent_id* when it descends from one of *conversation_ids*,
    else None.

    A ``subagent_id`` is a random 32-bit suffix, so a match alone proves nothing: the row's
    ``parent_session_id`` chain must reach the conversation, and a hop is followed only when the
    row is its parent's delegate child or compression continuation. A branch or reset fork of the
    conversation, or any unrelated session, never lends its children. Every segment carries the
    marker, so the newest owned one is the child's tip (``get_compression_tip`` does not follow a
    delegate child's rotations: its chain step skips rows carrying ``_delegate_from``)."""
    from hermes_state_common import _sql_json_extract

    ancestors = frozenset(sid for sid in conversation_ids if sid)
    if not subagent_id or not ancestors:
        return None
    rows = db._read_all(
        "SELECT id FROM sessions WHERE source = 'subagent' AND "
        f"{_sql_json_extract('model_config', '$.' + SUBAGENT_ID_KEY)} = ? ORDER BY started_at DESC, id DESC",
        (subagent_id,),
    )
    return next((row["id"] for row in rows if _descends_from(db, row["id"], ancestors)), None)


def _descends_from(db: Any, session_id: str, ancestors: frozenset[str]) -> bool:
    # Each rotation is a hop, so a long-lived child's chain has no useful bound; ``seen`` ends a cycle.
    current, seen = db.get_session(session_id), set()
    while current is not None and current["id"] not in seen:
        if current["id"] in ancestors:
            return True
        seen.add(current["id"])
        parent = current.get("parent_session_id")
        if not parent or not (_delegated_from(current, parent) or db._is_compression_child_row(current)):
            return False
        current = db.get_session(parent)
    return False


def _delegated_from(row: dict[str, Any], parent: str) -> bool:
    """The marker only counts when it names the row's own parent: a compression continuation
    inherits the root's ``_delegate_from`` and reaches the delegator through its compression hops."""
    cfg = row.get("model_config")
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg)
        except ValueError:
            return False
    return isinstance(cfg, dict) and cfg.get("_delegate_from") == parent
