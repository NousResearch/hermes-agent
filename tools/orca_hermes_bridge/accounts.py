"""Token-safe account identity mapping and Hermes pool mutations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.credential_pool import (
    PooledCredential,
    STATUS_DEAD,
    STATUS_EXHAUSTED,
    STATUS_OK,
    _exhausted_until,
)
from hermes_cli.auth import (
    _auth_store_lock,
    _decode_jwt_claims,
    _load_auth_store,
    _save_auth_store,
)


OPENAI_AUTH_CLAIM = "https://api.openai.com/auth"
STATUS_FIELDS = (
    "last_status",
    "last_status_at",
    "last_error_code",
    "last_error_reason",
    "last_error_message",
    "last_error_reset_at",
)


class InvalidOrcaSnapshotError(ValueError):
    """Raised when Orca did not provide an unambiguous active host account."""


class DuplicateProviderAccountError(ValueError):
    """Raised when two Hermes rows claim the same provider account identity."""


@dataclass(frozen=True)
class OrcaAccount:
    account_id: str | None
    provider_account_id: str
    email: str | None


@dataclass(frozen=True)
class OrcaSnapshot:
    active: OrcaAccount
    accounts_by_provider_id: dict[str, OrcaAccount]


def _nonempty_string(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def chatgpt_account_id(access_token: str) -> str | None:
    """Extract the stable ChatGPT account ID without validating or exposing a JWT."""
    claims = _decode_jwt_claims(access_token)
    auth_claim = claims.get(OPENAI_AUTH_CLAIM)
    if not isinstance(auth_claim, dict):
        return None
    return _nonempty_string(auth_claim.get("chatgpt_account_id"))


def parse_orca_accounts(payload: dict[str, Any]) -> OrcaSnapshot:
    """Normalize an accounts.list result or its outer RPC/CLI envelope."""
    snapshot: Any = payload
    if isinstance(snapshot, dict) and isinstance(snapshot.get("result"), dict):
        snapshot = snapshot["result"]
    codex = snapshot.get("codex") if isinstance(snapshot, dict) else None
    if not isinstance(codex, dict):
        raise InvalidOrcaSnapshotError("Orca snapshot has no Codex account state")

    by_provider: dict[str, OrcaAccount] = {}
    system = codex.get("systemDefault")
    if isinstance(system, dict):
        provider_id = _nonempty_string(system.get("providerAccountId"))
        if provider_id:
            by_provider[provider_id] = OrcaAccount(
                account_id=None,
                provider_account_id=provider_id,
                email=_nonempty_string(system.get("email")),
            )

    managed_by_id: dict[str, OrcaAccount] = {}
    for raw in codex.get("accounts") or []:
        if not isinstance(raw, dict):
            continue
        account_id = _nonempty_string(raw.get("id"))
        provider_id = _nonempty_string(raw.get("providerAccountId"))
        if not account_id or not provider_id:
            continue
        account = OrcaAccount(
            account_id=account_id,
            provider_account_id=provider_id,
            email=_nonempty_string(raw.get("email")),
        )
        existing = by_provider.get(provider_id)
        if existing is not None and existing.account_id != account_id:
            raise InvalidOrcaSnapshotError("Orca snapshot contains a duplicate provider account")
        by_provider[provider_id] = account
        managed_by_id[account_id] = account

    active_by_runtime = codex.get("activeAccountIdsByRuntime")
    if isinstance(active_by_runtime, dict) and "host" in active_by_runtime:
        active_id = active_by_runtime.get("host")
    else:
        active_id = codex.get("activeAccountId")
    if active_id is None:
        active = next((item for item in by_provider.values() if item.account_id is None), None)
    else:
        active = managed_by_id.get(str(active_id))
    if active is None:
        raise InvalidOrcaSnapshotError("Orca host selection cannot be resolved")
    return OrcaSnapshot(active=active, accounts_by_provider_id=by_provider)


def mapped_pool_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index Hermes pool rows by provider account, failing closed on ambiguity."""
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        provider_id = chatgpt_account_id(str(row.get("access_token") or ""))
        if provider_id is None:
            continue
        if provider_id in result:
            raise DuplicateProviderAccountError(
                f"Multiple Hermes credentials map to provider account {provider_id}"
            )
        result[provider_id] = row
    return result


def first_usable_provider_id(rows: list[dict[str, Any]], *, now: float) -> str | None:
    """Return the first usable mapped Codex identity using Hermes priority/cooldown rules."""
    ordered = sorted(enumerate(rows), key=lambda item: (item[1].get("priority", 0), item[0]))
    seen: set[str] = set()
    for _, row in ordered:
        provider_id = chatgpt_account_id(str(row.get("access_token") or ""))
        if provider_id is None:
            continue
        if provider_id in seen:
            raise DuplicateProviderAccountError(
                f"Multiple Hermes credentials map to provider account {provider_id}"
            )
        seen.add(provider_id)
        status = row.get("last_status")
        if status == STATUS_DEAD:
            continue
        if status == STATUS_EXHAUSTED:
            until = _exhausted_until(PooledCredential.from_dict("openai-codex", row))
            if until is not None and now < until:
                continue
        return provider_id
    return None


def reorder_codex_pool(
    provider_account_id: str,
    *,
    clear_selected_status: bool,
) -> bool:
    """Move one unambiguous Codex identity first without altering token material."""
    with _auth_store_lock():
        store = _load_auth_store()
        pool = store.get("credential_pool")
        rows = pool.get("openai-codex") if isinstance(pool, dict) else None
        if not isinstance(rows, list):
            return False
        mapped = mapped_pool_rows([row for row in rows if isinstance(row, dict)])
        selected = mapped.get(provider_account_id)
        if selected is None:
            return False

        indexed = list(enumerate(rows))
        remaining = [(index, row) for index, row in indexed if row is not selected]
        remaining.sort(key=lambda item: (item[1].get("priority", 0), item[0]))
        ordered = [selected, *(row for _, row in remaining)]
        changed = ordered != rows
        for priority, row in enumerate(ordered):
            if row.get("priority") != priority:
                row["priority"] = priority
                changed = True

        if clear_selected_status:
            desired = {
                "last_status": STATUS_OK,
                "last_status_at": None,
                "last_error_code": None,
                "last_error_reason": None,
                "last_error_message": None,
                "last_error_reset_at": None,
            }
            if any(selected.get(field) != value for field, value in desired.items()):
                selected.update(desired)
                changed = True

        if not changed:
            return False
        pool["openai-codex"] = ordered
        _save_auth_store(store)
        return True
