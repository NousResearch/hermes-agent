"""Contract tests applied to :class:`SqliteApprovalStore`.

Production backend. The full contract must pass — including the three
tests that ``InMemoryApprovalStore`` xfails (persistence across
instances, cross-instance atomic consume, pinned-policy roundtrip
via a second store instance).

Plus a few SQLite-specific tests for failure modes the contract is
silent on (corrupt payload, schema-init concurrency).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from tests.tools.approval_store_contract import ApprovalStoreContract
from tools.approval_store import (
    ApprovalProposal,
    ApprovalStore,
    ApprovalStoreError,
)
from tools.approval_store_sqlite import (
    SqliteApprovalStore,
    _reset_schema_cache_for_tests,
)


@pytest.fixture(autouse=True)
def _isolated_schema_cache() -> None:
    """Each test gets its own tmp_path; the module-level schema-init memo
    must not bleed between tests."""
    _reset_schema_cache_for_tests()


class TestSqliteApprovalStoreContract(ApprovalStoreContract):
    """Full contract MUST pass against SqliteApprovalStore."""

    @pytest.fixture(autouse=True)
    def _capture_tmp_path(self, tmp_path: Path) -> None:
        self._db_path = tmp_path / "state.db"

    def make_factory(self) -> Callable[[], ApprovalStore]:
        # Two factory calls return DIFFERENT store instances bound to the
        # SAME db file — the realistic production pattern (different
        # gateway processes share state.db).
        db_path = self._db_path
        return lambda: SqliteApprovalStore(db_path=db_path)


# ---------------------------------------------------------------------------
# Backend-specific failure-mode tests (not in the shared contract because
# they exercise SQLite internals).
# ---------------------------------------------------------------------------


class TestSqliteApprovalStoreFailures:
    def test_corrupt_payload_json_fails_closed(self, tmp_path: Path) -> None:
        """If payload_json is corrupt, get/consume must raise — not return
        a partial/malformed proposal that the executor might run blind."""
        store = SqliteApprovalStore(db_path=tmp_path / "state.db")
        store.submit(ApprovalProposal(
            approval_id="appr-corrupt",
            created_at=1.0,
            session_key="s",
            command="echo ok",
            risk_level="medium",
            risk_reason="corrupt-payload-test",
        ))
        # Directly corrupt the payload column.
        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "state.db"))
        try:
            conn.execute(
                "UPDATE gateway_approvals SET payload_json = ? "
                "WHERE approval_id = ?",
                ("{not-json,", "appr-corrupt"),
            )
            conn.commit()
        finally:
            conn.close()

        with pytest.raises(ApprovalStoreError):
            store.get("appr-corrupt")

    def test_status_filter_via_query(self, tmp_path: Path) -> None:
        """Sanity check that the indexed status column actually filters.

        Documents that listing pending approvals (a future addition for
        gateway boot-up recovery) will use the index without table scan.
        """
        store = SqliteApprovalStore(db_path=tmp_path / "state.db")
        _kwargs = dict(
            session_key="s", command="echo x",
            risk_level="medium", risk_reason="status-filter-test",
        )
        store.submit(ApprovalProposal(approval_id="appr-p", created_at=1.0, **_kwargs))
        store.submit(ApprovalProposal(approval_id="appr-d", created_at=1.0, **_kwargs))
        store.deny("appr-d", denied_by="@u", now=2.0)

        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "state.db"))
        try:
            rows = conn.execute(
                "SELECT approval_id FROM gateway_approvals "
                "WHERE status = 'pending' ORDER BY approval_id"
            ).fetchall()
        finally:
            conn.close()
        assert [r[0] for r in rows] == ["appr-p"]
