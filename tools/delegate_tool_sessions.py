"""The stored session of a delegated child, addressed by its ``subagent_id``.

``delegate_tool._build_child_agent`` stamps the child's ``model_config`` with ``SUBAGENT_ID_KEY``
beside ``_delegate_from``. Compression copies ``model_config`` onto every continuation, so each
segment of the child carries it. The live registry forgets a child when it ends; this marker is
how a viewer that only holds the ``subagent_id`` (the dispatch result, a ``subagent.list`` row)
still finds the child's stored transcript afterwards.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

SUBAGENT_ID_KEY = "_subagent_id"


def stored_child_session(store: Any, subagent_id: str, conversation_ids: Iterable[str]) -> Optional[str]:
    """The newest segment of child *subagent_id* when it descends from one of *conversation_ids*,
    else None.

    A ``subagent_id`` is a random 32-bit suffix, so a match alone proves nothing: the row's
    ``parent_session_id`` chain must reach the conversation, and a hop is followed only when the
    row is its parent's delegate child or compression continuation. A branch or reset fork of the
    conversation, or any unrelated session, never lends its children. Every segment carries the
    marker, so the tip is the owned segment that no owned segment continues by compression
    (``get_compression_tip`` does not follow a delegate child's rotations: its chain step skips rows
    carrying ``_delegate_from``)."""
    ancestors = frozenset(sid for sid in conversation_ids if sid)
    if not subagent_id or not ancestors:
        return None
    candidates = store.sessions.subagent_session_ids(subagent_id, SUBAGENT_ID_KEY)
    get, owns = _lineage_reader(store, ancestors)
    owned = [sid for sid in candidates if owns(sid)]
    # The tip is the owned segment no owned segment continues by compression; ``started_at`` only breaks
    # a tie, since the wall clock can step back between a rotation's two rows.
    owned_ids = frozenset(owned)
    continued = {seg.parent_session_id for seg in map(get, owned)
                 if seg is not None and seg.parent_session_id in owned_ids and _continues(seg, get)}
    return next((sid for sid in owned if sid not in continued), None)


def _lineage_reader(store: Any, ancestors: frozenset[str]):
    """``(get, owns)`` for one lookup. ``get`` reads each session row at most once; ``owns(sid)`` says whether
    *sid*'s chain reaches *ancestors* and records the verdict for every row it walked, since all of a
    child's segments share the chain above them (with ``compression.in_place: false`` every rotation is
    one more segment). Each rotation is a hop, so a long-lived child's chain has no useful bound; a row
    met twice on one walk ends a cycle."""
    rows: dict[str, Any] = {}
    verdicts: dict[str, bool] = {}

    def get(session_id: str) -> Any:
        if session_id not in rows:
            rows[session_id] = store.sessions.get(session_id)
        return rows[session_id]

    def owns(session_id: str) -> bool:
        path: dict[str, None] = {}
        current, verdict = get(session_id), False
        while current is not None:
            sid = current.session_id
            if sid in verdicts or sid in ancestors:
                verdict = verdicts.get(sid, True)
                break
            if sid in path:
                break
            path[sid] = None
            parent = current.parent_session_id
            if not parent or not (_delegated_from(current, parent) or _continues(current, get)):
                break
            current = get(parent)
        verdicts.update(dict.fromkeys(path, verdict))
        return verdict

    return get, owns


def _continues(row: Any, get) -> bool:
    """``lineage.is_compression_child`` over the lookup's cached rows: not a tool-sourced child, nor a branch,
    delegate or reset child of its parent (a config marker naming it), and the parent ended ``compression``."""
    parent = get(row.parent_session_id) if row.parent_session_id else None
    markers = tuple(row.model_config.get(key) for key in ("_branched_from", "_delegate_from", "_reset_from"))
    return bool(parent and row.source != "tool" and row.parent_session_id not in markers
                and parent.end_reason == "compression")


def _delegated_from(row: Any, parent: str) -> bool:
    """The marker only counts when it names the row's own parent: a compression continuation
    inherits the root's ``_delegate_from`` and reaches the delegator through its compression hops."""
    return row.model_config.get("_delegate_from") == parent
