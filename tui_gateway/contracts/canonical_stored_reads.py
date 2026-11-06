"""Reads of a stored conversation that never attach the caller (``gateway/session_stored_reads.py``).

A picker previews a conversation before it opens it: ``session.resume`` would subscribe the caller's
pane, and ``session.events.since`` replays only what an attached viewer missed.
"""
from typing import Literal

from .base import Result
from .canonical_sessions import CanonicalSessionParams
from .common import TranscriptMessage
from .registry import canonical_method


class StoredTranscriptParams(CanonicalSessionParams):
    # ``next_cursor`` of the previous page of this session in this order; omitted for the first page.
    cursor: str | None = None
    limit: int = 100
    # ``latest``: the first page is the newest rows and each next page is older.
    order: Literal['oldest', 'latest'] = 'oldest'
    # Image parts as stored (data URLs); by default they read ``[image]``, as a byte-bounded resume shows them.
    inline_images: bool = False


class StoredTranscriptResult(Result):
    session_id: str
    # ``session.resume``'s ``display_messages`` projection of these stored rows, oldest first in either order.
    rows: list[TranscriptMessage]
    # Opaque, bound to this session and order; null after the last page.
    next_cursor: str | None


class StoredCompaction(Result):
    # When the compaction's handoff was written (Unix seconds).
    timestamp: float | None
    summary: str
    # How many messages the compaction summarized, when its handoff names it.
    compacted_messages: int | None


class StoredCompactionsResult(Result):
    session_id: str
    compactions: list[StoredCompaction]


canonical_method('session.transcript', params=StoredTranscriptParams, result=StoredTranscriptResult,
                 doc="One page of a stored conversation's display rows across its compression lineage "
                     "(live rows and rows a compaction archived, each logical message once), from the "
                     "oldest or the newest. Never attaches the caller. A page holds at most `limit` "
                     "(1-500) rows plus up to 64 tool results of the call at its edge, and at most 2 MiB "
                     "of JSON, cut between tool groups (a single larger row keeps only its role, time, ids "
                     "and the head of its text, and carries `truncated`); it can show "
                     "fewer rows, even none, so read on until `next_cursor` is null. A cursor stays valid "
                     "across later compactions and turns. A rotated lineage past "
                     "`sessions.max_resume_messages` pages only its current segment. "
                     "Refusals: invalid_params (a cursor issued for another session or order, or not by "
                     "this read), not_found, permission_denied, profile_mismatch.")
canonical_method('session.compactions', params=CanonicalSessionParams, result=StoredCompactionsResult,
                 doc="The summaries a stored conversation's compactions wrote, oldest first, across its "
                     "compression lineage, without the parts written for the model (preamble, end marker, "
                     "re-quoted user messages, recovery footer). The display rows leave them out. Never "
                     "attaches the caller. Refusals: not_found, permission_denied, profile_mismatch.")
