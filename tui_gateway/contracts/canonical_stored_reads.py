"""Reads of a stored conversation that never attach the caller (``gateway/session_stored_reads.py``).

A picker previews a conversation before it opens it: ``session.resume`` would subscribe the
caller's pane, and ``session.events.since`` replays only what an attached viewer missed.
"""
from .base import Result
from .canonical_sessions import CanonicalSessionParams
from .common import TranscriptMessage
from .registry import canonical_method


class StoredTranscriptParams(CanonicalSessionParams):
    # ``next_cursor`` of the previous page; omitted for the first page.
    cursor: str | None = None
    limit: int = 100


class StoredTranscriptResult(Result):
    session_id: str
    # ``session.resume``'s ``display_messages`` projection of these stored rows.
    rows: list[TranscriptMessage]
    # Opaque; null after the last page.
    next_cursor: str | None


class StoredCompaction(Result):
    # When the compaction's handoff was written (Unix seconds).
    timestamp: float | None
    summary: str


class StoredCompactionsResult(Result):
    session_id: str
    compactions: list[StoredCompaction]


canonical_method('session.transcript', params=StoredTranscriptParams, result=StoredTranscriptResult,
                 doc="One page of a stored conversation's display rows, oldest first, across its "
                     "compression lineage (live rows and rows a compaction archived, each logical "
                     "message once). Never attaches the caller. A page reads at most `limit` stored "
                     "rows (1-500), so it can show fewer, even none; read on until `next_cursor` is "
                     "null. A cursor stays valid across later compactions and new turns. "
                     "Refusals: invalid_params (a cursor this owner did not issue), not_found, "
                     "permission_denied, profile_mismatch.")
canonical_method('session.compactions', params=CanonicalSessionParams, result=StoredCompactionsResult,
                 doc="The handoff summaries a stored conversation's compactions wrote, oldest "
                     "first, across its compression lineage. The display rows leave them out. "
                     "Never attaches the caller. Refusals: not_found, permission_denied, "
                     "profile_mismatch.")
