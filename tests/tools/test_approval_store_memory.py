"""Contract tests applied to :class:`InMemoryApprovalStore`.

This is the in-process reference implementation. By design it is
**process-bound** — two store instances do NOT share state.

The full contract suite runs here, but persistence + cross-instance
tests are marked ``xfail(strict=True)``: they MUST fail on this backend,
proving that the contract is meaningful and that in-process storage is
unsuitable for production. If those tests start passing it would mean
either the contract weakened or the backend now does persistence — both
require investigation.

Same-process exactly-once consume passes because of ``threading.Lock``.
That documents what the existing in-process storage does correctly,
and isolates what only transactional persistent storage can provide.
"""

from __future__ import annotations

from typing import Callable

import pytest

from tests.tools.approval_store_contract import ApprovalStoreContract
from tools.approval_store import ApprovalStore
from tools.approval_store_memory import InMemoryApprovalStore


class TestInMemoryApprovalStoreContract(ApprovalStoreContract):
    """In-process reference. Persistence/cross-instance tests EXPECTED to fail."""

    def make_factory(self) -> Callable[[], ApprovalStore]:
        # Each factory() call returns a NEW empty store instance — the
        # deliberate failure mode for cross-instance tests.
        return InMemoryApprovalStore

    # -- xfail(strict): these MUST fail. If they pass, contract weakened. --

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "InMemoryApprovalStore is process-bound by design; persistence "
            "across store instances requires durable backing (SQLite). "
            "This xfail documents that gap."
        ),
    )
    def test_persists_pending_approval_across_store_instances(self) -> None:
        super().test_persists_pending_approval_across_store_instances()

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Two in-memory store instances cannot coordinate consume — "
            "each has its own _proposals dict. Requires SQLite for "
            "cross-instance atomicity."
        ),
    )
    def test_two_store_instances_cannot_both_consume(self) -> None:
        super().test_two_store_instances_cannot_both_consume()

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Pinned policy must survive ROUND-TRIP through a second store "
            "instance; in-memory store loses this since instances do not "
            "share state."
        ),
    )
    def test_pinned_policy_payload_survives_roundtrip(self) -> None:
        super().test_pinned_policy_payload_survives_roundtrip()
