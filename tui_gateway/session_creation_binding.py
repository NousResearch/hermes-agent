"""Immutable origin metadata for authenticated create, not activation authority.

The scope token names this runtime's captured profile/store scope. It is not a
host path, credential, or a discovery key shared by independently created runtimes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4
from pathlib import Path
import threading


@dataclass(frozen=True)
class CreationBinding:
    session_id: str
    stored_session_id: str
    authenticated_owner: str
    store_path: str
    runtime_incarnation: str
    profile_store_scope: str
    # Retain the actual registry object: copied metadata cannot bless replacement.
    runtime_record: dict = field(repr=False, compare=False)

    @classmethod
    def mint(cls, session_id: str, stored_session_id: str, owner: str | None,
             store_path: str, runtime_record: dict) -> CreationBinding | None:
        # Anonymous/legacy transports cannot supply authenticated creation provenance.
        if any(not isinstance(value, str) or not value.strip() or len(value) > 256
               for value in (session_id, stored_session_id, owner)):
            return None
        return cls(session_id, stored_session_id, owner, store_path, uuid4().hex, uuid4().hex, runtime_record)

    def fields_for(self, owner: str | None, store_path: str) -> dict[str, dict[str, str]]:
        if owner != self.authenticated_owner or store_path != self.store_path:
            return {}
        # A fresh wire dict cannot mutate the server's retained origin. In particular,
        # retries do not restamp it from a later compression tip or current transport.
        return {"creation_binding": {
            "session_id": self.session_id,
            "stored_session_id": self.stored_session_id,
            "authenticated_owner": self.authenticated_owner,
            "runtime_incarnation": self.runtime_incarnation,
            "profile_store_scope": self.profile_store_scope,
        }}


@dataclass(frozen=True)
class EngineCreationBinding:
    """Original local engine and store witness; never refreshed by attachment retries."""
    engine: object = field(repr=False)
    database: object = field(repr=False)
    revision: int

    @classmethod
    def capture(cls, origin: CreationBinding, agent) -> EngineCreationBinding | None:
        from run_agent import AIAgent
        from hermes_state import SessionDB
        from agent.session_identity import SessionIdentityMixin

        if (not isinstance(agent, AIAgent)
                or type(agent).session_id is not SessionIdentityMixin.session_id
                or type(agent).session_identity_guard is not SessionIdentityMixin.session_identity_guard
                or type(agent).session_identity_revision is not SessionIdentityMixin.session_identity_revision
                or not isinstance(getattr(agent, "_hard_interrupt_requested", None), threading.Event)):
            return None
        guard = agent.session_identity_guard()
        if not guard.acquire(blocking=False):
            return None
        try:
            db = getattr(agent, "_session_db", None)
            if (not isinstance(db, SessionDB) or str(Path(db.db_path).resolve()) != origin.store_path
                    or agent.session_id != origin.stored_session_id):
                return None
            return cls(agent, db, agent.session_identity_revision)
        finally:
            guard.release()

    def matches(self, origin: CreationBinding, *, running: bool) -> bool:
        agent = self.engine
        holder = getattr(agent, "_active_session_turn_lease_holder", None)
        return (agent.session_identity_revision == self.revision
                and agent.session_id == origin.stored_session_id
                and agent._session_db is self.database
                and str(Path(self.database.db_path).resolve()) == origin.store_path
                and (not running or (isinstance(holder, str) and bool(holder)))
                and not getattr(agent, "_persist_disabled", False)
                and not agent._interrupt_requested and not agent._hard_interrupt_requested.is_set())
