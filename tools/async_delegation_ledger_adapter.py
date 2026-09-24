"""Backend-neutral async-delegation ledger port, factories, and runtime selector.

``selected_async_delegation_ledger`` resolves the active state-store backend and
returns the matching ledger: ``None`` for SQLite (the module keeps its legacy
path), or a configured PostgreSQL ledger otherwise. A PostgreSQL selection with a
missing/unreachable DSN raises instead of silently degrading to SQLite. Importing
this module never opens a database.
"""
from __future__ import annotations

from typing import Any, Protocol, cast, runtime_checkable

from hermes_constants import get_hermes_home


@runtime_checkable
class AsyncDelegationLedger(Protocol):
    """The durable async-delegation ledger surface; no DB handles escape it."""
    def persist_dispatch(self, record: dict[str, Any]) -> None: ...
    def persist_completion(self, event: dict[str, Any], result: dict[str, Any]) -> None: ...
    def prune_durable_records(self) -> None: ...
    def record_unit_child(self, delegation_id: str, entry: dict[str, Any]) -> None: ...
    def recover_abandoned_delegations(self) -> int: ...
    def restore_undelivered_completions(self, target_queue) -> int: ...
    def sweep_orphaned_completions(self, target_queue, *, now: float | None = None) -> int: ...
    def mark_completion_delivered(self, delegation_id: str) -> bool: ...
    def claim_completion_delivery(self, delegation_id: str, claim_id: str) -> bool: ...
    def release_completion_delivery(self, delegation_id: str, claim_id: str) -> bool: ...
    def defer_completion_delivery(self, delegation_id: str, claim_id: str) -> bool: ...
    def drop_completion_delivery(self, delegation_id: str, claim_id: str) -> bool: ...
    def complete_completion_delivery(self, delegation_id: str, claim_id: str) -> bool: ...
    def get_durable_delegation(self, delegation_id: str) -> dict[str, Any] | None: ...


def open_async_delegation_ledger(*, backend: str = "sqlite", dsn: str | None = None, **kwargs: Any):
    """Open an isolated adapter for conformance tests; never changes runtime routing."""
    if backend == "postgresql":
        if not dsn:
            raise ValueError("PostgreSQL async-delegation ledger requires a DSN")
        from tools.async_delegation_ledger_postgresql import PostgreSQLAsyncDelegationLedger
        return PostgreSQLAsyncDelegationLedger(dsn, **kwargs)
    if backend == "sqlite":
        return SqliteAsyncDelegationLedger()
    raise ValueError("async-delegation ledger backend must be sqlite or postgresql")


def open_configured_async_delegation_ledger(
    config: dict[str, Any], *, secret_lookup=None, **kwargs: Any,
) -> AsyncDelegationLedger:
    """Explicit factory for an injected consumer, never a global runtime selector.

    Shares only the validated state-store DSN policy. The caller controls lifetime
    and injection, so a selected PostgreSQL configuration never opens legacy
    ``state.db`` or activates gateway runtime.
    """
    from state_store import resolve_state_store_config
    resolved = resolve_state_store_config(config, secret_lookup=secret_lookup)
    if resolved.backend == "sqlite":
        return SqliteAsyncDelegationLedger()
    assert resolved.postgresql is not None
    lookup = secret_lookup
    if lookup is None:
        from state_store import _scoped_secret as lookup  # type: ignore[attr-defined]
    dsn = str(lookup(resolved.postgresql.dsn_env) or "").strip()
    if not dsn:
        raise ValueError("PostgreSQL async-delegation ledger requires configured DSN")
    from tools.async_delegation_ledger_postgresql import AsyncDelegationLedgerPostgreSQLConfig
    return cast(AsyncDelegationLedger, open_async_delegation_ledger(
        backend="postgresql", dsn=dsn,
        settings=AsyncDelegationLedgerPostgreSQLConfig(
            connect_timeout_seconds=resolved.postgresql.connect_timeout_seconds,
            pool_max_size=resolved.postgresql.pool_max_size,
        ), **kwargs,
    ))


_SELECTED_LEDGER_CACHE: dict[str, AsyncDelegationLedger | None] = {}


def selected_async_delegation_ledger() -> AsyncDelegationLedger | None:
    """Resolve the runtime async-delegation ledger from the selected backend.

    ``None`` means SQLite is selected and the module keeps its legacy path.
    Otherwise the configured PostgreSQL ledger is opened (and cached per Hermes
    home), raising on a missing or unreachable DSN rather than degrading to SQLite.
    """
    home = str(get_hermes_home())
    if home in _SELECTED_LEDGER_CACHE:
        return _SELECTED_LEDGER_CACHE[home]
    try:
        from hermes_cli.config import load_config
        config = load_config()
    except Exception:
        config = {}
    from state_store import resolve_state_store_config
    if resolve_state_store_config(config).backend == "sqlite":
        result: AsyncDelegationLedger | None = None
    else:
        result = open_configured_async_delegation_ledger(config)
    _SELECTED_LEDGER_CACHE[home] = result
    return result


class SqliteAsyncDelegationLedger:
    """Compatibility object for direct tests over the unchanged module contract."""

    def persist_dispatch(self, record):
        from tools import async_delegation
        return async_delegation._persist_dispatch(record)

    def persist_completion(self, event, result):
        from tools import async_delegation
        return async_delegation._persist_completion(event, result)

    def prune_durable_records(self):
        from tools import async_delegation
        return async_delegation._prune_durable_records()

    def record_unit_child(self, delegation_id, entry):
        from tools import async_delegation
        return async_delegation.record_unit_child(delegation_id, entry)

    def recover_abandoned_delegations(self):
        from tools import async_delegation
        return async_delegation.recover_abandoned_delegations()

    def restore_undelivered_completions(self, target_queue):
        from tools import async_delegation
        return async_delegation.restore_undelivered_completions(target_queue)

    def sweep_orphaned_completions(self, target_queue, *, now=None):
        from tools import async_delegation
        return async_delegation.sweep_orphaned_completions(target_queue, now=now)

    def mark_completion_delivered(self, delegation_id):
        from tools import async_delegation
        return async_delegation.mark_completion_delivered(delegation_id)

    def claim_completion_delivery(self, delegation_id, claim_id):
        from tools import async_delegation
        return async_delegation.claim_completion_delivery(delegation_id, claim_id)

    def release_completion_delivery(self, delegation_id, claim_id):
        from tools import async_delegation
        return async_delegation.release_completion_delivery(delegation_id, claim_id)

    def defer_completion_delivery(self, delegation_id, claim_id):
        from tools import async_delegation
        return async_delegation.defer_completion_delivery(delegation_id, claim_id)

    def drop_completion_delivery(self, delegation_id, claim_id):
        from tools import async_delegation
        return async_delegation.drop_completion_delivery(delegation_id, claim_id)

    def complete_completion_delivery(self, delegation_id, claim_id):
        from tools import async_delegation
        return async_delegation.complete_completion_delivery(delegation_id, claim_id)

    def get_durable_delegation(self, delegation_id):
        from tools import async_delegation
        return async_delegation.get_durable_delegation(delegation_id)

    def close(self):
        return None
