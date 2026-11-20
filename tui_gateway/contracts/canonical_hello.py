"""``client.hello`` — the post-upgrade handshake folded into ONE round trip (authority only).

The session authority serves it because the replay it batches IS owner state: on a native socket
``session.events.since`` reads the owner's own event stream (``AuthorityConnection.events_since``),
and the sidecar's in-process ring is a different numbering for a canonical session. A connection
with no session authority (or a backend older than this method) answers ``-32601``, and the shared
client keeps today's sequence on the SAME socket — a fallback, never a redial.

Fold, do not redesign: each session's window in the answer is the ``session.events.since`` result
verbatim (``truncated`` / ``snapshot_required`` / epoch rules unchanged), and answering it goes
through that same handler so its reconnect bookkeeping (``session.reconnects`` / ``replay_cursors``)
still happens.
"""

from __future__ import annotations

from pydantic import Field

from .base import Params, Result
from .canonical_sessions import CanonicalEventsResult
from .liveness import ClientCapabilitiesParams, ClientCapabilitiesResult
from .registry import canonical_method
from .sessions import OpenRequestEntry


class ClientHelloParams(Params):
    """What the reconnecting client holds, sent as its FIRST frame once ``gateway.ready`` (or
    ``gateway.capabilities``) advertised ``resumed_handshake``."""

    #: The client's own capabilities, applied exactly as ``client.capabilities`` applies them.
    capabilities: ClientCapabilitiesParams | None = None
    #: The legacy process epoch the client holds (``gateway.ready.replay_epoch``). Accepted and
    #: ignored for parity with ``session.events.since``; a canonical session's numbering is per
    #: session, and its qualifier travels in ``session_epochs``.
    replay_epoch: str | None = None
    #: ``session_id`` → the last event seq this client observed (its per-session watermark). Only
    #: sessions named here are replayed; a session without a watermark has nothing to catch up on.
    watermarks: dict[str, int] = Field(default_factory=dict)
    #: ``session_id`` → that session's own epoch, when the client holds one — the qualifier today's
    #: one ``session.events.since`` call per session sends as ``replay_epoch`` (an older legacy
    #: endpoint refuses any other key, so it is sent only when the client has it).
    session_epochs: dict[str, str] = Field(default_factory=dict)


class SessionReplaySnapshot(CanonicalEventsResult):
    """One session's replay window: the ``session.events.since`` answer batched, plus the session it
    belongs to. ``open_requests`` stays None here — the answer carries the deduped union at the top
    level, so a reconnecting client's questions are re-delivered once."""

    session_id: str
    #: The per-session refusal when this session's window could not be answered, else None.
    #: ``not_found``: the session is gone — the client drops its cursor, exactly as it does today;
    #: ``permission_denied`` / ``profile_mismatch`` / ``invalid_params``: the cursor is kept and
    #: re-asked on the next reconnect. Never a whole-call failure: one gone session must not cost
    #: every other session its replay.
    error: str | None = None


class SessionResumedResult(Result):
    """The ``client.hello`` answer — capabilities, skin identity, every session's replay and the open
    server→client requests, in ONE response."""

    capabilities_result: ClientCapabilitiesResult
    #: Content etag of the skin this connection's ``gateway.ready`` carried
    #: (``change_watcher.skin_etag``), so the client re-fetches the body only on a mismatch.
    skin_etag: str = ""
    per_session_replay: list[SessionReplaySnapshot] = Field(default_factory=list)
    #: Every unanswered server→client request across the replayed sessions, oldest first and deduped
    #: by request id (the shape ``session.events.since`` returns per session). The shared channel
    #: re-delivers these to its request handlers before the caller sees the result.
    open_requests: list[OpenRequestEntry] = Field(default_factory=list)


canonical_method("client.hello", params=ClientHelloParams, result=SessionResumedResult,
                 doc="The client's FIRST frame after a ``resumed_handshake`` ready: apply its "
                     "capabilities and answer the whole post-upgrade handshake in ONE response — "
                     "``capabilities_result`` (as ``client.capabilities`` answers), ``skin_etag``, "
                     "one replay window per watermark and the deduped ``open_requests``.")
