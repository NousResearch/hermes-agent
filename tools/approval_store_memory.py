"""In-process reference implementation of :class:`ApprovalStore`.

**This implementation INTENTIONALLY VIOLATES the persistence + cross-instance
contract.** It exists for two reasons:

1. Documenting the gap: the contract test suite at
   ``tests/tools/test_approval_store_contract.py`` runs against this store
   to demonstrate which invariants in-process storage cannot satisfy.
   Persistence + cross-instance-atomicity tests will fail. That failure
   is intentional and is the proof that the contract is meaningful.

2. Process-local testing: when a test genuinely doesn't care about
   persistence (e.g. unit-testing the gateway's call into ``submit``),
   instantiating an ``InMemoryApprovalStore`` is cheaper than spinning
   up SQLite.

**Do NOT use in production.** Wire ``tools.approval_store_sqlite.SqliteApprovalStore``
(added in a follow-up commit) for any real gateway.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from tools.approval_store import (
    ApprovalProposal,
    ApprovalStore,
    ApprovalStoreError,
)


class InMemoryApprovalStore:
    """Thread-safe dict-backed approval store. Process-bound by design.

    Each instance has its own internal dict — two instances DO NOT share
    state, even if you give them the same identifier. This is the
    deliberate failure mode for cross-instance contract tests.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proposals: dict[str, ApprovalProposal] = {}

    # ----- Lifecycle -----

    def submit(self, proposal: ApprovalProposal) -> None:
        with self._lock:
            if proposal.approval_id in self._proposals:
                raise ValueError(
                    f"approval_id collision: {proposal.approval_id!r} "
                    "already exists in this store instance"
                )
            self._proposals[proposal.approval_id] = proposal

    def get(self, approval_id: str) -> Optional[ApprovalProposal]:
        with self._lock:
            return self._proposals.get(approval_id)

    def consume(self, approval_id: str, *, consumed_by: str,
                now: Optional[float] = None) -> Optional[ApprovalProposal]:
        ts = now if now is not None else time.time()
        with self._lock:
            proposal = self._proposals.get(approval_id)
            if proposal is None:
                return None
            if proposal.status != "pending":
                return None
            if proposal.is_expired_at(ts):
                self._proposals[approval_id] = proposal.with_status("expired")
                return None
            new = proposal.with_status(
                "consumed", consumed_by=consumed_by, consumed_at=ts,
            )
            self._proposals[approval_id] = new
            return new

    def deny(self, approval_id: str, *, denied_by: str,
             now: Optional[float] = None) -> bool:
        ts = now if now is not None else time.time()
        with self._lock:
            proposal = self._proposals.get(approval_id)
            if proposal is None or proposal.status != "pending":
                return False
            if proposal.is_expired_at(ts):
                # Already past TTL — call it expired, not denied.
                self._proposals[approval_id] = proposal.with_status("expired")
                return False
            self._proposals[approval_id] = proposal.with_status(
                "denied", consumed_by=denied_by, consumed_at=ts,
            )
            return True

    def mark_post_consume(self, approval_id: str, *, executed: bool,
                          reason: Optional[str] = None,
                          now: Optional[float] = None) -> bool:
        ts = now if now is not None else time.time()
        new_status = "executed" if executed else "blocked_after_consume"
        with self._lock:
            proposal = self._proposals.get(approval_id)
            if proposal is None or proposal.status != "consumed":
                return False
            from dataclasses import replace
            self._proposals[approval_id] = replace(
                proposal,
                execution_status=new_status,
                execution_reason=reason,
                execution_recorded_at=ts,
            )
            return True

    def expire_due(self, now: Optional[float] = None) -> int:
        ts = now if now is not None else time.time()
        count = 0
        with self._lock:
            for aid, proposal in list(self._proposals.items()):
                if proposal.status == "pending" and proposal.is_expired_at(ts):
                    self._proposals[aid] = proposal.with_status("expired")
                    count += 1
        return count


# Make a runtime-checkable conformance assertion explicit:
assert isinstance(InMemoryApprovalStore(), ApprovalStore), (
    "InMemoryApprovalStore must satisfy ApprovalStore Protocol"
)
