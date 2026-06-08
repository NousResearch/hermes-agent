"""Shared contract test mixin for :class:`tools.approval_store.ApprovalStore`.

Backend test modules subclass :class:`ApprovalStoreContract` and provide
``make_factory()`` returning a ``Callable[[], ApprovalStore]``. Two calls
to the factory MUST return store instances whose mutations are visible
to each other (i.e. they point at the same backing storage).

A backend that cannot satisfy this — e.g. the in-process reference
:class:`tools.approval_store_memory.InMemoryApprovalStore` — will fail
the cross-instance + persistence tests. That failure is the proof that
the contract is meaningful and that in-process storage is unsuitable
for the security invariants documented in ``tools/approval_store.py``.

The contract enforces:

  1. Persistence — submitted proposal is visible via a fresh store instance.
  2. Exactly-once consume — second consume of same id returns ``None``.
  3. Concurrent consume safety — same-process threads.
  4. Cross-instance consume safety — two store instances on same backing.
  5. Expired proposals cannot be consumed.
  6. Denied proposals cannot be consumed.
  7. Pinned policy payload survives round-trip through consume.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import pytest

from tools.approval_store import ApprovalProposal, ApprovalStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_proposal(
    approval_id: str = "appr-1",
    *,
    risk_level: str = "low",
    expires_in: float | None = None,
    command: str = "echo hi",
) -> ApprovalProposal:
    """Build a proposal with sensible defaults. Tests override what they care about."""
    created = time.time()
    return ApprovalProposal(
        approval_id=approval_id,
        created_at=created,
        expires_at=(created + expires_in) if expires_in is not None else None,
        session_key="session-A",
        requester="@tester:example",
        command=command,
        cwd="/tmp",
        backend="bash",
        risk_level=risk_level,
        risk_reason="contract-test",
        policy_decision="needs_approval",
        policy_version="contract-v1",
        requires_explicit_approval=True,
        default_decision="deny" if risk_level == "high" else "deny",
        diff_summary=None,
        display_text=f"Approve {command}?",
    )


# ---------------------------------------------------------------------------
# Contract mixin
# ---------------------------------------------------------------------------


class ApprovalStoreContract:
    """Subclass and provide :meth:`make_factory`. Pytest discovers the test
    methods on the subclass.

    The factory is a callable returning a NEW store instance bound to the
    same backing storage as previous factory calls. Production backends
    (e.g. SQLite tied to a file path) trivially satisfy this. The
    in-process reference is expected to fail the cross-instance tests.
    """

    # Backend tests override this.
    def make_factory(self) -> Callable[[], ApprovalStore]:  # pragma: no cover
        raise NotImplementedError("Backend test subclass must implement make_factory")

    # ----- Test methods -----

    def test_persists_pending_approval_across_store_instances(self) -> None:
        """A proposal submitted via one store instance must be visible from another."""
        factory = self.make_factory()
        store_a = factory()
        store_b = factory()

        proposal = _make_proposal("appr-persists")
        store_a.submit(proposal)

        loaded = store_b.get("appr-persists")
        assert loaded is not None, (
            "fresh store instance failed to see proposal submitted by sibling"
        )
        assert loaded.approval_id == "appr-persists"
        assert loaded.status == "pending"

    def test_consume_returns_proposal_exactly_once(self) -> None:
        """First consume returns the proposal; second returns None."""
        factory = self.make_factory()
        store = factory()

        store.submit(_make_proposal("appr-once"))
        first = store.consume("appr-once", consumed_by="@user")
        second = store.consume("appr-once", consumed_by="@user")

        assert first is not None
        assert first.approval_id == "appr-once"
        assert second is None, (
            "second consume of same id must return None — exactly-once violated"
        )

    def test_two_store_instances_cannot_both_consume(self) -> None:
        """Two store instances racing for same proposal: exactly one wins.

        Pre-condition: persistence holds (store_b actually SEES the proposal
        submitted by store_a). If a backend lacks persistence, this test is
        inapplicable — atomicity claims only make sense when both racers
        can independently locate the proposal. We assert persistence
        explicitly and let the failure surface (rather than skipping
        silently) so that a backend with broken persistence cannot pass
        this test by coincidence.
        """
        factory = self.make_factory()
        store_a = factory()
        store_b = factory()

        store_a.submit(_make_proposal("appr-race"))

        # Pre-condition: store_b must also see it. If this fails, the
        # backend lacks cross-instance persistence — that's a separate
        # contract violation but it also makes the atomicity claim
        # vacuous, so we assert here rather than skip.
        assert store_b.get("appr-race") is not None, (
            "cross-instance precondition failed: store_b cannot see "
            "proposal submitted via store_a (persistence broken). "
            "Atomicity test cannot proceed."
        )

        results: list = []
        barrier = threading.Barrier(2)

        def worker(store: ApprovalStore) -> None:
            barrier.wait()  # maximise contention
            results.append(store.consume("appr-race", consumed_by="@user"))

        t1 = threading.Thread(target=worker, args=(store_a,))
        t2 = threading.Thread(target=worker, args=(store_b,))
        t1.start(); t2.start()
        t1.join(); t2.join()

        winners = [r for r in results if r is not None]
        losers = [r for r in results if r is None]
        assert len(winners) == 1, (
            f"expected exactly one winner; got {len(winners)} winners and "
            f"{len(losers)} losers (results={results}). Storage is not "
            "atomic across instances."
        )
        assert len(losers) == 1

    def test_concurrent_threads_single_instance_consume_exactly_once(self) -> None:
        """Eight threads on one store instance racing for same id: exactly one wins."""
        factory = self.make_factory()
        store = factory()
        store.submit(_make_proposal("appr-thread-race"))

        results: list = []
        results_lock = threading.Lock()
        barrier = threading.Barrier(8)

        def worker() -> None:
            barrier.wait()
            r = store.consume("appr-thread-race", consumed_by="@user")
            with results_lock:
                results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()

        winners = [r for r in results if r is not None]
        assert len(winners) == 1, (
            f"thread-race: expected exactly one winner out of 8; got {len(winners)}"
        )

    def test_expired_proposal_cannot_be_consumed(self) -> None:
        """A proposal whose expires_at is in the past must not be consumable."""
        factory = self.make_factory()
        store = factory()

        past_proposal = _make_proposal("appr-expired", expires_in=-10.0)
        store.submit(past_proposal)

        result = store.consume("appr-expired", consumed_by="@user")
        assert result is None, "expired proposal MUST NOT be consumable"

    def test_denied_proposal_cannot_be_consumed(self) -> None:
        """After /deny, subsequent /approve must not execute."""
        factory = self.make_factory()
        store = factory()
        store.submit(_make_proposal("appr-denied"))

        denied_ok = store.deny("appr-denied", denied_by="@user")
        assert denied_ok is True

        result = store.consume("appr-denied", consumed_by="@user")
        assert result is None, "consume of denied proposal MUST return None"

    def test_missing_proposal_consume_returns_none(self) -> None:
        """Unknown approval_id must fail closed, not raise/proceed."""
        factory = self.make_factory()
        store = factory()
        assert store.consume("does-not-exist", consumed_by="@user") is None
        assert store.deny("does-not-exist", denied_by="@user") is False
        assert store.get("does-not-exist") is None

    def test_pinned_policy_payload_survives_roundtrip(self) -> None:
        """Pinned risk + policy fields submitted at creation are returned by consume."""
        factory = self.make_factory()
        store_a = factory()
        store_b = factory()

        proposal = ApprovalProposal(
            approval_id="appr-pinned",
            created_at=time.time(),
            session_key="session-pinned",
            command="rm -rf /etc",
            cwd="/",
            backend="bash",
            risk_level="high",
            risk_reason="HARDLINE: rm -rf rooted",
            policy_decision="needs_approval",
            policy_version="2026-06-07.1",
            requires_explicit_approval=True,
            default_decision="deny",
            diff_summary="3 files deleted, system irrecoverable",
            display_text="HIGH RISK: rm -rf /etc",
        )
        store_a.submit(proposal)

        loaded = store_b.consume("appr-pinned", consumed_by="@user")
        assert loaded is not None
        # All pinned fields preserved verbatim:
        assert loaded.risk_level == "high"
        assert loaded.risk_reason == "HARDLINE: rm -rf rooted"
        assert loaded.policy_decision == "needs_approval"
        assert loaded.policy_version == "2026-06-07.1"
        assert loaded.requires_explicit_approval is True
        assert loaded.default_decision == "deny"
        assert loaded.diff_summary == "3 files deleted, system irrecoverable"
        assert loaded.command == "rm -rf /etc"
        assert loaded.cwd == "/"
        assert loaded.backend == "bash"

    def test_expire_due_marks_pending_past_ttl(self) -> None:
        """expire_due moves past-TTL pending → expired in bulk."""
        factory = self.make_factory()
        store = factory()
        store.submit(_make_proposal("appr-exp-1", expires_in=-5))
        store.submit(_make_proposal("appr-exp-2", expires_in=-1))
        store.submit(_make_proposal("appr-live", expires_in=600))

        marked = store.expire_due()
        assert marked == 2, f"expected 2 expired; got {marked}"

        # Live one still consumable, expired ones not.
        assert store.consume("appr-exp-1", consumed_by="@u") is None
        assert store.consume("appr-exp-2", consumed_by="@u") is None
        live = store.consume("appr-live", consumed_by="@u")
        assert live is not None and live.approval_id == "appr-live"

    def test_collision_on_duplicate_submit(self) -> None:
        """Same approval_id submitted twice must fail closed (no silent overwrite)."""
        factory = self.make_factory()
        store = factory()
        store.submit(_make_proposal("appr-dup"))
        with pytest.raises(ValueError):
            store.submit(_make_proposal("appr-dup"))
