"""Dedicated read-only Postgres connection profile support for Rob.

This module implements CODE/CONFIG support only. It does NOT create the
``projectos_ro`` role anywhere, including production — that is a real,
reviewed, one-time DBA action requiring Nicolas's explicit authorization
(see ``scripts/rob_operator/create_projectos_ro_role.sql`` alongside this
module: a documented, human-run script, never auto-executed by this code).

Why a dedicated role is required at all: this engagement's own earlier
audit found the ONLY Postgres credential in production is the app's own
``projectos`` role, and it is a full superuser
(``rolsuper=t, rolcreatedb=t, rolcreaterole=t``) — confirmed directly on
NiPoGi. Reusing it for Rob's inspection tooling would mean every read-only
diagnostic query runs with unrestricted write/DDL/superuser capability.
The ``db_select``/``schema_inspect`` tools built on top of this module
refuse to run at all unless a connection profile explicitly marked as
read-only is configured — see ``ReadOnlyDsnProfile.assert_not_superuser``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class DbProfileError(Exception):
    pass


@dataclass(frozen=True)
class ReadOnlyDsnProfile:
    name: str
    dsn: str
    statement_timeout_ms: int = 5000
    row_limit: int = 1000


def load_profile(name: str) -> ReadOnlyDsnProfile:
    """Load a named read-only DSN profile from environment configuration.

    Expected env shape (never hardcoded, never the app's own superuser
    DSN): ``ROB_DB_PROFILE_<NAME>_DSN``, e.g.
    ``ROB_DB_PROFILE_PROJECTOS_DSN=postgresql://projectos_ro:...@127.0.0.1:5433/projectos``.
    Optional per-profile overrides:
    ``ROB_DB_PROFILE_<NAME>_STATEMENT_TIMEOUT_MS``,
    ``ROB_DB_PROFILE_<NAME>_ROW_LIMIT``.

    Raises DbProfileError if the profile isn't configured, or if the DSN's
    own username is literally ``projectos`` (the known superuser) — a
    deliberate, explicit refusal so a misconfiguration can never silently
    hand Rob the superuser credential under a read-only-sounding profile
    name.
    """
    key = name.upper().replace("-", "_")
    dsn_var = f"ROB_DB_PROFILE_{key}_DSN"
    dsn = os.environ.get(dsn_var)
    if not dsn:
        raise DbProfileError(
            f"No read-only DB profile configured for '{name}' ({dsn_var} is not set). "
            "Rob's DB tools refuse to fall back to any other credential — "
            "see db_readonly_profile.py and scripts/rob_operator/create_projectos_ro_role.sql."
        )
    _assert_not_superuser_dsn(dsn, name)

    timeout_ms = int(os.environ.get(f"ROB_DB_PROFILE_{key}_STATEMENT_TIMEOUT_MS", "5000"))
    row_limit = int(os.environ.get(f"ROB_DB_PROFILE_{key}_ROW_LIMIT", "1000"))
    return ReadOnlyDsnProfile(name=name, dsn=dsn, statement_timeout_ms=timeout_ms, row_limit=row_limit)


# Usernames known to be application superusers/write credentials — never
# acceptable as the identity behind a "read-only" profile, regardless of
# what the profile is named. Extend this list per-deployment, never
# remove entries silently.
_KNOWN_SUPERUSER_NAMES = frozenset({"projectos", "postgres"})


def _assert_not_superuser_dsn(dsn: str, profile_name: str) -> None:
    """Refuse a DSN whose username is a known superuser identity — pure
    string inspection, no network/DB access, so this check runs before
    any connection is ever attempted."""
    try:
        # postgresql://user:pass@host:port/db — extract user without a
        # full URI parser dependency; good enough for this narrow check.
        after_scheme = dsn.split("://", 1)[1]
        userinfo = after_scheme.split("@", 1)[0]
        username = userinfo.split(":", 1)[0]
    except IndexError as exc:
        raise DbProfileError(f"profile '{profile_name}' DSN is not a recognizable postgresql:// URI") from exc
    if username in _KNOWN_SUPERUSER_NAMES:
        raise DbProfileError(
            f"profile '{profile_name}' DSN authenticates as '{username}', a known superuser/write "
            "identity — refusing to use it as a read-only Rob profile. Configure a dedicated "
            "least-privilege role (see scripts/rob_operator/create_projectos_ro_role.sql) instead."
        )


def build_session_init_statements(profile: ReadOnlyDsnProfile) -> list:
    """SQL to run at the start of every Rob DB session/transaction,
    defense-in-depth ON TOP OF the role's own GRANT-level restrictions
    (belt and suspenders — a role misconfiguration and this guard would
    both have to fail for a write to slip through).

    Deliberately ``SET TRANSACTION READ ONLY``, not
    ``SET default_transaction_read_only = on``: psycopg opens an implicit
    transaction on the first ``cursor.execute()``, so a plain
    ``default_transaction_read_only`` SET issued as that first statement
    only takes effect for transactions that start AFTER it — it does
    nothing for the very transaction it runs in, which is also the one the
    actual query runs in here. ``SET TRANSACTION READ ONLY`` instead sets
    the CURRENT transaction's characteristics and is valid precisely
    because it is unconditionally the first statement executed in a fresh
    connection/transaction in this module's calling code."""
    return [
        "SET TRANSACTION READ ONLY",
        f"SET statement_timeout = {profile.statement_timeout_ms}",
        "SET search_path = public",
    ]
