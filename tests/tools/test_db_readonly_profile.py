"""Tests for tools.db_readonly_profile — the DB tool must refuse to run
against anything but a dedicated read-only credential, per the P0 spec's
explicit requirement: "The Rob DB tool must refuse to run if only the
application superuser credential is configured."
"""

import pytest

from tools.db_readonly_profile import DbProfileError, build_session_init_statements, load_profile


class TestMissingProfile:
    def test_unconfigured_profile_raises(self, monkeypatch):
        monkeypatch.delenv("ROB_DB_PROFILE_PROJECTOS_DSN", raising=False)
        with pytest.raises(DbProfileError, match="No read-only DB profile configured"):
            load_profile("projectos")


class TestSuperuserRefusal:
    def test_refuses_projectos_superuser_dsn(self, monkeypatch):
        # The exact real credential identity confirmed to be a superuser
        # in production this engagement — must never be usable here even
        # if someone points ROB_DB_PROFILE_PROJECTOS_DSN at it by mistake.
        monkeypatch.setenv(
            "ROB_DB_PROFILE_PROJECTOS_DSN",
            "postgresql://projectos:whatever@127.0.0.1:5432/projectos",
        )
        with pytest.raises(DbProfileError, match="known superuser"):
            load_profile("projectos")

    def test_refuses_postgres_superuser_dsn(self, monkeypatch):
        monkeypatch.setenv(
            "ROB_DB_PROFILE_TEST_DSN",
            "postgresql://postgres:whatever@127.0.0.1:5432/projectos",
        )
        with pytest.raises(DbProfileError, match="known superuser"):
            load_profile("test")

    def test_accepts_dedicated_readonly_role_dsn(self, monkeypatch):
        monkeypatch.setenv(
            "ROB_DB_PROFILE_PROJECTOS_DSN",
            "postgresql://projectos_ro:whatever@127.0.0.1:5432/projectos",
        )
        profile = load_profile("projectos")
        assert profile.dsn.startswith("postgresql://projectos_ro:")
        assert profile.name == "projectos"

    def test_malformed_dsn_rejected_not_crashed(self, monkeypatch):
        monkeypatch.setenv("ROB_DB_PROFILE_PROJECTOS_DSN", "not-a-uri-at-all")
        with pytest.raises(DbProfileError):
            load_profile("projectos")


class TestProfileDefaultsAndOverrides:
    def test_defaults(self, monkeypatch):
        monkeypatch.setenv("ROB_DB_PROFILE_PROJECTOS_DSN", "postgresql://projectos_ro:x@h/db")
        monkeypatch.delenv("ROB_DB_PROFILE_PROJECTOS_STATEMENT_TIMEOUT_MS", raising=False)
        monkeypatch.delenv("ROB_DB_PROFILE_PROJECTOS_ROW_LIMIT", raising=False)
        profile = load_profile("projectos")
        assert profile.statement_timeout_ms == 5000
        assert profile.row_limit == 1000

    def test_overrides(self, monkeypatch):
        monkeypatch.setenv("ROB_DB_PROFILE_PROJECTOS_DSN", "postgresql://projectos_ro:x@h/db")
        monkeypatch.setenv("ROB_DB_PROFILE_PROJECTOS_STATEMENT_TIMEOUT_MS", "2000")
        monkeypatch.setenv("ROB_DB_PROFILE_PROJECTOS_ROW_LIMIT", "50")
        profile = load_profile("projectos")
        assert profile.statement_timeout_ms == 2000
        assert profile.row_limit == 50


class TestSessionInitStatements:
    def test_forces_read_only_and_timeout(self, monkeypatch):
        monkeypatch.setenv("ROB_DB_PROFILE_PROJECTOS_DSN", "postgresql://projectos_ro:x@h/db")
        profile = load_profile("projectos")
        statements = build_session_init_statements(profile)
        # "SET TRANSACTION READ ONLY", not "SET default_transaction_read_only
        # = on" — the latter only affects transactions that start AFTER it,
        # which is never true here since this is always the first statement
        # of the transaction the actual query also runs in. See
        # build_session_init_statements' own docstring for the full reasoning
        # (this was a real, confirmed-ineffective bug, not a style choice).
        assert any("TRANSACTION READ ONLY" in s for s in statements)
        assert not any("default_transaction_read_only" in s for s in statements)
        assert any(f"statement_timeout = {profile.statement_timeout_ms}" in s for s in statements)

    def test_read_only_statement_runs_before_any_other_statement(self, monkeypatch):
        # SET TRANSACTION READ ONLY is only effective as the FIRST statement
        # of a transaction — order here isn't cosmetic.
        monkeypatch.setenv("ROB_DB_PROFILE_PROJECTOS_DSN", "postgresql://projectos_ro:x@h/db")
        profile = load_profile("projectos")
        statements = build_session_init_statements(profile)
        assert "TRANSACTION READ ONLY" in statements[0]
