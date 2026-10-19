"""Client-turn identity: the registry behind ``prompt.submit.client_turn_id`` and ``turn.status``.

A turn is keyed by ``(profile scope, runtime session id, client_turn_id)`` — an opaque id the CLIENT
mints so a retry after a dropped socket resolves to the turn it already started instead of running
the same words twice. Same key + same text is a replay: the stored ack comes back with
``duplicate``/``replayed`` true and no second agent turn. Same key + DIFFERENT text is not a retry
(the client reused an id for different words): it runs, under a fresh server turn id.

Terminal truth comes from the turn, never from the request that arrives. The handler records
``accepted``; the turn thread records ``running`` when the agent actually starts; the turn loop's
``finally`` records ``terminal`` plus that run's own outcome. A replay of a finished turn therefore
reports what happened, not what was asked — the ack path has no other way to say "already finished,
badly", so ``outcome`` rides beside ``state``.

The registry is per-process and in-memory, like ``gateway/platforms/api_server.py``'s
``_IdempotencyCache``: a backend restart loses it and the client falls back to its own transcript
check. Bounded the same way — at most ``MAX_ENTRIES`` turns, each forgotten ``TTL_SECONDS`` after its
last transition. Two known bounds follow from that shape and are deliberate: a turn dispatched to an
isolated compute host cannot be observed finishing from this process (it stays ``accepted`` until
the TTL), and a steered/redirected submit whose live turn ends in the gap between the redirect ack
and the join stays ``running`` until the TTL. Both only cost a stale report; neither can run a turn
twice.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

#: Longest id the wire accepts (``prompt.submit`` / ``turn.status`` answer 4001 beyond it).
MAX_CLIENT_TURN_ID = 180
#: Live turns kept per process; the oldest is dropped once the cap is reached.
MAX_ENTRIES = 1000
#: A turn is forgotten this long after its last transition (accept / running / terminal).
TTL_SECONDS = 3600

#: ``state`` while the turn is not finished: the three values the ACK accepts.
ACCEPTED = "accepted"
RUNNING = "running"
TERMINAL = "terminal"
#: ``outcome`` once the turn is terminal: the three values a turn can actually end in.
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"


@dataclass
class ClientTurn:
    """One client-minted turn, and what the server knows about it."""

    key: tuple[str, str, str]
    client_turn_id: str
    server_turn_id: str
    fingerprint: str
    state: str = ACCEPTED
    outcome: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


def fingerprint(text) -> str:
    """Stable digest of the prompt text a turn id was first used with.

    Only equality matters (same words -> replay, different words -> a new submission), so the
    encoding just has to be deterministic for every payload ``prompt.submit`` carries (a string, a
    relay parts list).
    """
    try:
        payload = json.dumps(text, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = repr(text)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def profile_scope(session) -> str:
    """The profile a turn is keyed under.

    The live session's own home is authoritative (one serve process hosts several profiles); a
    sessionless caller is already inside ``_profile_scoped``, so the active home is its scope. Both
    ends of the key — this registry and ``turn.status`` — resolve through this one helper, so a
    lookup can never address a different bucket than the submit wrote.
    """
    home = session.get("profile_home") if isinstance(session, dict) else None
    if home:
        return str(home)
    try:
        from hermes_constants import get_hermes_home

        return str(get_hermes_home() or "")
    except Exception:
        return ""


def report_state(turn: ClientTurn) -> str:
    """The ``turn.status`` state for a turn: five values, wider than the ack's three.

    ``status`` is a report, the ack is a commitment: a finished failure reports ``failed`` here
    instead of ``terminal`` + ``outcome``, because the client asked "what happened", not "do you
    accept this".
    """
    if turn.state != TERMINAL:
        return turn.state
    return {FAILED: FAILED, CANCELLED: CANCELLED}.get(turn.outcome or "", TERMINAL)


class TurnRegistry:
    """Bounded, TTL'd, thread-safe map of client turn ids. See the module docstring."""

    def __init__(self, *, max_entries: int = MAX_ENTRIES, ttl_seconds: float = TTL_SECONDS):
        self._lock = threading.Lock()
        self._turns: OrderedDict[tuple[str, str, str], ClientTurn] = OrderedDict()
        self._max = max_entries
        self._ttl = ttl_seconds

    def _purge_locked(self, now: float) -> None:
        expired = [key for key, turn in self._turns.items() if now - turn.updated_at > self._ttl]
        for key in expired:
            self._turns.pop(key, None)
        while len(self._turns) > self._max:
            self._turns.popitem(last=False)

    def begin(self, *, profile: str, session_id: str, client_turn_id: str, text) -> tuple[ClientTurn, bool]:
        """Resolve a submit to its turn: ``(turn, replayed)``.

        ``replayed`` is True only for the same id with the same text — the caller must then answer
        the stored ack and start nothing. Different text under the same id replaces the entry: the
        words that are about to run are what the id now names.
        """
        key = (profile, session_id, client_turn_id)
        digest = fingerprint(text)
        with self._lock:
            now = time.time()
            self._purge_locked(now)
            turn = self._turns.get(key)
            if turn is not None and turn.fingerprint == digest:
                turn.updated_at = now
                self._turns.move_to_end(key)
                return turn, True
            turn = ClientTurn(key=key, client_turn_id=client_turn_id,
                              server_turn_id=f"turn_{uuid.uuid4().hex}", fingerprint=digest,
                              created_at=now, updated_at=now)
            self._turns[key] = turn
            self._turns.move_to_end(key)
            self._purge_locked(now)
            return turn, False

    def lookup(self, *, profile: str, session_id: str, client_turn_id: str) -> ClientTurn | None:
        with self._lock:
            self._purge_locked(time.time())
            return self._turns.get((profile, session_id, client_turn_id))

    def mark(self, turn: ClientTurn, state: str, outcome: str | None = None) -> bool:
        """Advance ``turn``, but only while it is still the record this registry holds.

        False means the entry was replaced (the client reused the id for different text) or already
        terminal: a late thread from an old turn must never overwrite a newer submission's state,
        and a finished turn is final.
        """
        with self._lock:
            stored = self._turns.get(turn.key)
            if stored is not turn or stored.state == TERMINAL:
                return False
            stored.state, stored.outcome = state, outcome
            stored.updated_at = time.time()
            return True

    def finish(self, turn: ClientTurn, outcome: str) -> bool:
        """Record that ``turn`` ended in ``outcome`` (succeeded / failed / cancelled)."""
        return self.mark(turn, TERMINAL, outcome)

    def __len__(self) -> int:
        with self._lock:
            self._purge_locked(time.time())
            return len(self._turns)


#: The process-wide registry. Replaced wholesale in tests; every helper below reads it per call.
REGISTRY = TurnRegistry()


def begin(*, profile: str, session_id: str, client_turn_id: str, text) -> tuple[ClientTurn, bool]:
    return REGISTRY.begin(profile=profile, session_id=session_id, client_turn_id=client_turn_id, text=text)


def lookup(*, profile: str, session_id: str, client_turn_id: str) -> ClientTurn | None:
    return REGISTRY.lookup(profile=profile, session_id=session_id, client_turn_id=client_turn_id)


def mark(turn: ClientTurn, state: str, outcome: str | None = None) -> bool:
    return REGISTRY.mark(turn, state, outcome)


def finish(turn: ClientTurn, outcome: str) -> bool:
    return REGISTRY.finish(turn, outcome)
