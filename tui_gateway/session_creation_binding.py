"""Immutable origin metadata for authenticated create, not activation authority.

The scope token names this runtime's captured profile/store scope. It is not a
host path, credential, or a discovery key shared by independently created runtimes.
"""
from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4


@dataclass(frozen=True)
class CreationBinding:
    session_id: str
    stored_session_id: str
    authenticated_owner: str
    store_path: str
    runtime_incarnation: str
    profile_store_scope: str

    @classmethod
    def mint(cls, session_id: str, stored_session_id: str, owner: str | None, store_path: str) -> CreationBinding | None:
        # Anonymous/legacy transports cannot supply authenticated creation provenance.
        if any(not isinstance(value, str) or not value.strip() or len(value) > 256
               for value in (session_id, stored_session_id, owner)):
            return None
        return cls(session_id, stored_session_id, owner, store_path, uuid4().hex, uuid4().hex)

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
