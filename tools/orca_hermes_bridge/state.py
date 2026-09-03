"""Pure state transitions for two-way Orca/Hermes account reconciliation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .accounts import (
    OrcaSnapshot,
    first_usable_provider_id,
    mapped_pool_rows,
)


PENDING_ECHO_SECONDS = 30.0


@dataclass(frozen=True)
class BridgeState:
    version: int = 1
    last_seen_orca_provider_id: str | None = None
    pending_orca_provider_id: str | None = None
    pending_started_at: float | None = None
    qwen_notified: bool = False


@dataclass(frozen=True)
class PoolMutation:
    provider_account_id: str
    clear_selected_status: bool


@dataclass(frozen=True)
class OrcaMutation:
    account_id: str | None
    provider_account_id: str


@dataclass(frozen=True)
class ReconcileDecision:
    state: BridgeState
    pool_mutation: PoolMutation | None = None
    orca_mutation: OrcaMutation | None = None
    notify_qwen: bool = False


def _pending_is_fresh(state: BridgeState, now: float) -> bool:
    return (
        state.pending_orca_provider_id is not None
        and state.pending_started_at is not None
        and 0 <= now - state.pending_started_at <= PENDING_ECHO_SECONDS
    )


def _row_provider_id(row: dict[str, Any], mapped: dict[str, dict[str, Any]]) -> str | None:
    return next((provider_id for provider_id, candidate in mapped.items() if candidate is row), None)


def reconcile(
    snapshot: OrcaSnapshot,
    rows: list[dict[str, Any]],
    state: BridgeState,
    now: float,
) -> ReconcileDecision:
    """Compute one deterministic reconciliation step without performing I/O."""
    mapped = mapped_pool_rows(rows)
    active_provider_id = snapshot.active.provider_account_id
    pending_fresh = _pending_is_fresh(state, now)

    if pending_fresh and active_provider_id == state.pending_orca_provider_id:
        acknowledged = replace(
            state,
            last_seen_orca_provider_id=active_provider_id,
            pending_orca_provider_id=None,
            pending_started_at=None,
            qwen_notified=False,
        )
        mutation = (
            PoolMutation(active_provider_id, clear_selected_status=False)
            if active_provider_id in mapped
            else None
        )
        return ReconcileDecision(state=acknowledged, pool_mutation=mutation)

    if (
        pending_fresh
        and active_provider_id == state.last_seen_orca_provider_id
        and active_provider_id != state.pending_orca_provider_id
    ):
        return ReconcileDecision(state=state)

    manual_change = (
        state.last_seen_orca_provider_id is None
        or active_provider_id != state.last_seen_orca_provider_id
    )
    if manual_change:
        manual_state = replace(
            state,
            last_seen_orca_provider_id=active_provider_id,
            pending_orca_provider_id=None,
            pending_started_at=None,
            qwen_notified=False,
        )
        mutation = (
            PoolMutation(active_provider_id, clear_selected_status=True)
            if active_provider_id in mapped
            else None
        )
        return ReconcileDecision(state=manual_state, pool_mutation=mutation)

    if active_provider_id not in mapped:
        return ReconcileDecision(
            state=replace(state, pending_orca_provider_id=None, pending_started_at=None)
        )

    active_usable = (
        first_usable_provider_id([mapped[active_provider_id]], now=now)
        == active_provider_id
    )
    if active_usable:
        return ReconcileDecision(
            state=replace(
                state,
                pending_orca_provider_id=None,
                pending_started_at=None,
                qwen_notified=False,
            )
        )

    known_rows = [
        row
        for row in rows
        if (_row_provider_id(row, mapped) in snapshot.accounts_by_provider_id)
    ]
    next_provider_id = first_usable_provider_id(known_rows, now=now)
    if next_provider_id is not None:
        target = snapshot.accounts_by_provider_id[next_provider_id]
        pending_state = replace(
            state,
            pending_orca_provider_id=next_provider_id,
            pending_started_at=now,
            qwen_notified=False,
        )
        return ReconcileDecision(
            state=pending_state,
            pool_mutation=PoolMutation(next_provider_id, clear_selected_status=False),
            orca_mutation=OrcaMutation(target.account_id, next_provider_id),
        )

    should_notify = not state.qwen_notified
    return ReconcileDecision(
        state=replace(
            state,
            pending_orca_provider_id=None,
            pending_started_at=None,
            qwen_notified=True,
        ),
        notify_qwen=should_notify,
    )
