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

    def test_upgrade_from_pre_execution_columns_schema(self, tmp_path: Path) -> None:
        """Regression: SqliteApprovalStore must transparently upgrade an
        existing state.db that was created by an earlier hermes version
        which lacked the execution_status / execution_reason /
        execution_recorded_at columns.

        This is the production-upgrade path. Before the
        CREATE_TABLE_SQL / INDEX_SQL split + migrate-in-between fix,
        opening an old DB raised ``no such column: execution_status``
        because the partial index on execution_status was created
        BEFORE the ALTER TABLE that added the column.
        """
        import sqlite3
        db_path = tmp_path / "old_state.db"

        # Build the OLD schema (pre-Risk-1-fix) directly with raw SQL.
        # This is the exact shape an earlier hermes deploy would have
        # left on disk.
        OLD_SCHEMA = """
        CREATE TABLE IF NOT EXISTS gateway_approvals (
            approval_id  TEXT PRIMARY KEY,
            created_at   REAL NOT NULL,
            expires_at   REAL,
            status       TEXT NOT NULL DEFAULT 'pending',
            consumed_at  REAL,
            consumed_by  TEXT,
            payload_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_gateway_approvals_status
        ON gateway_approvals(status);
        CREATE INDEX IF NOT EXISTS idx_gateway_approvals_pending_expires
        ON gateway_approvals(expires_at)
        WHERE status = 'pending';
        """
        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(OLD_SCHEMA)
            # Seed a pre-existing pending row so we also verify the
            # migration doesn't drop data.
            conn.execute(
                "INSERT INTO gateway_approvals "
                "(approval_id, created_at, expires_at, status, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                ("appr-legacy", 1.0, 9999999999.0, "pending",
                 '{"approval_id":"appr-legacy","created_at":1.0,'
                 '"session_key":"s","command":"echo legacy",'
                 '"risk_level":"medium","risk_reason":"legacy-row",'
                 '"policy_decision":"needs_approval",'
                 '"default_decision":"deny","requires_explicit_approval":true}'),
            )
            conn.commit()
        finally:
            conn.close()

        # Now open via the NEW SqliteApprovalStore. _ensure_schema must
        # transparently ALTER TABLE to add execution_* columns AND
        # create the new partial index on the now-existing column.
        store = SqliteApprovalStore(db_path=db_path)

        # Verify the migration applied: the legacy row is still there
        # AND now exposes the new execution_* fields (at defaults).
        proposal = store.get("appr-legacy")
        assert proposal is not None
        assert proposal.command == "echo legacy"
        assert proposal.status == "pending"
        assert proposal.execution_status == "not_started"
        assert proposal.execution_reason is None
        assert proposal.execution_recorded_at is None

        # Verify the new partial index exists (would have failed CREATE
        # if the column wasn't there yet).
        conn = sqlite3.connect(str(db_path))
        try:
            indexes = {
                row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                )
            }
        finally:
            conn.close()
        assert "idx_gateway_approvals_blocked_after_consume" in indexes

        # End-to-end: consume the legacy proposal + record execution
        # outcome — confirms the new API works against the migrated DB.
        consumed = store.consume("appr-legacy", consumed_by="@upgrade-test")
        assert consumed is not None
        assert consumed.status == "consumed"
        ok = store.mark_post_consume("appr-legacy", executed=True)
        assert ok is True
        post = store.get("appr-legacy")
        assert post.execution_status == "executed"
        assert post.execution_recorded_at is not None

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
