"""Tests for the metadata-only JSONL query log.

Covers acceptance tests #14 (logs contain digest only; concurrent appends
remain valid JSONL; mode 0600) and #15 (unwritable required log returns
deterministic data_hold, not unlogged ok).
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
from pathlib import Path
from datetime import datetime, timezone

import pytest


from vault_retrieval.query_log import (
    QueryLogger,
    QueryLogEntry,
    normalize_terms,
    terms_sha256,
    QueryLogWriteError,
)


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    d = tmp_path / "state" / "vault-retrieval"
    d.mkdir(parents=True)
    return d


class TestNormalizeTerms:
    def test_lowercases(self):
        assert normalize_terms(["FOO", "Bar"]) == ["foo", "bar"]

    def test_strips_whitespace(self):
        assert normalize_terms(["  baz  ", "\tqux\n"]) == ["baz", "qux"]

    def test_dedupes_preserving_order(self):
        assert normalize_terms(["a", "b", "a", "c"]) == ["a", "b", "c"]

    def test_handles_empty(self):
        assert normalize_terms([]) == []


class TestTermsSha256:
    def test_stable(self):
        h1 = terms_sha256(["foo", "bar"])
        h2 = terms_sha256(["  FOO ", "bar"])
        assert h1 == h2

    def test_returns_64_hex(self):
        h = terms_sha256(["x"])
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestQueryLoggerBasic:
    def test_appends_one_valid_jsonl_line(self, log_dir):
        logger = QueryLogger(log_dir, query_log_enabled=True)
        entry = QueryLogEntry(
            query_id="RQL-TEST-001",
            queried_at=datetime.now(timezone.utc).isoformat(),
            task_id="t_x",
            run_id="1",
            project_topic="test",
            decision_horizon="today",
            search_terms=["alpha", "beta"],
            consulted=[{"path": "a.md", "range": "lines 1-2",
                         "source_mtime": "2026-10-01T00:00:00+00:00",
                         "source_last_updated": None, "characters_read": 12}],
            total_characters_read=12,
            budget_status="default",
            outcome="SUFFICIENT",
            data_hold_reason=None,
            redaction_counts={"secrets": 0, "pii": 0},
        )
        logger.append(entry)
        lines = (log_dir / "query-log.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["query_id"] == "RQL-TEST-001"
        # Raw terms MUST NOT be in the line; only the digest.
        assert "alpha" not in lines[0]
        assert "beta" not in lines[0]
        assert rec["search_terms_sha256"] == terms_sha256(["alpha", "beta"])
        assert rec["sensitive_payload_logged"] is False

    def test_log_file_mode_is_0600(self, log_dir):
        logger = QueryLogger(log_dir, query_log_enabled=True)
        entry = QueryLogEntry(
            query_id="RQL-MODE",
            queried_at=datetime.now(timezone.utc).isoformat(),
            task_id=None,
            run_id=None,
            project_topic="t",
            decision_horizon="t",
            search_terms=[],
            consulted=[{"path": "a", "range": "lines 1-1", "source_mtime": "2026-10-01T00:00:00+00:00",
                         "source_last_updated": None, "characters_read": 0}],
            total_characters_read=0,
            budget_status="default",
            outcome="SUFFICIENT",
            data_hold_reason=None,
            redaction_counts={"secrets": 0, "pii": 0},
        )
        logger.append(entry)
        st = os.stat(log_dir / "query-log.jsonl")
        mode = stat.S_IMODE(st.st_mode)
        assert mode == 0o600, f"expected 0o600, got {oct(mode)}"

    def test_state_dir_mode_is_0700(self, log_dir):
        QueryLogger(log_dir, query_log_enabled=True)
        st = os.stat(log_dir)
        mode = stat.S_IMODE(st.st_mode)
        assert mode == 0o700, f"expected 0o700, got {oct(mode)}"


class TestQueryLoggerDisabled:
    def test_disabled_appends_nothing(self, log_dir):
        logger = QueryLogger(log_dir, query_log_enabled=False)
        entry = QueryLogEntry(
            query_id="RQL-DISABLED",
            queried_at=datetime.now(timezone.utc).isoformat(),
            task_id=None,
            run_id=None,
            project_topic="t",
            decision_horizon="t",
            search_terms=[],
            consulted=[],
            total_characters_read=0,
            budget_status="default",
            outcome="SUFFICIENT",
            data_hold_reason=None,
            redaction_counts={"secrets": 0, "pii": 0},
        )
        # Disabled loggers raise instead of silently dropping — fail-closed.
        with pytest.raises(QueryLogWriteError):
            logger.append(entry)
        assert not (log_dir / "query-log.jsonl").exists()


class TestQueryLoggerConcurrency:
    def test_concurrent_appends_remain_valid_jsonl(self, log_dir):
        logger = QueryLogger(log_dir, query_log_enabled=True)

        def append_n(n: int):
            for i in range(n):
                entry = QueryLogEntry(
                    query_id=f"RQL-CONC-{n}-{i}",
                    queried_at=datetime.now(timezone.utc).isoformat(),
                    task_id=None,
                    run_id=None,
                    project_topic="t",
                    decision_horizon="t",
                    search_terms=[],
                    consulted=[{"path": "a", "range": "lines 1-1",
                                 "source_mtime": "2026-10-01T00:00:00+00:00",
                                 "source_last_updated": None, "characters_read": 0}],
                    total_characters_read=0,
                    budget_status="default",
                    outcome="SUFFICIENT",
                    data_hold_reason=None,
                    redaction_counts={"secrets": 0, "pii": 0},
                )
                logger.append(entry)

        threads = [threading.Thread(target=append_n, args=(5,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lines = (log_dir / "query-log.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 20
        for line in lines:
            json.loads(line)  # each line must parse


class TestQueryLoggerSensitiveAbsent:
    def test_no_raw_terms_in_log(self, log_dir):
        logger = QueryLogger(log_dir, query_log_enabled=True)
        entry = QueryLogEntry(
            query_id="RQL-NOSECRET",
            queried_at=datetime.now(timezone.utc).isoformat(),
            task_id=None,
            run_id=None,
            project_topic="t",
            decision_horizon="t",
            search_terms=["super-secret-secret", "ALSO_PRIVATE"],
            consulted=[],
            total_characters_read=0,
            budget_status="default",
            outcome="SUFFICIENT",
            data_hold_reason=None,
            redaction_counts={"secrets": 0, "pii": 0},
        )
        logger.append(entry)
        raw = (log_dir / "query-log.jsonl").read_bytes()
        assert b"super-secret-secret" not in raw
        assert b"ALSO_PRIVATE" not in raw
        # And no raw "query" or "prompt" field could leak either
        assert b'"query"' not in raw
        assert b'"prompt"' not in raw

    def test_no_extracted_text_in_log(self, log_dir):
        logger = QueryLogger(log_dir, query_log_enabled=True)
        entry = QueryLogEntry(
            query_id="RQL-NOTEXT",
            queried_at=datetime.now(timezone.utc).isoformat(),
            task_id=None,
            run_id=None,
            project_topic="t",
            decision_horizon="t",
            search_terms=[],
            consulted=[{"path": "a.md", "range": "lines 1-1",
                         "source_mtime": "2026-10-01T00:00:00+00:00",
                         "source_last_updated": None, "characters_read": 0,
                         "extracted_text": "this should NEVER appear"}],
            total_characters_read=0,
            budget_status="default",
            outcome="SUFFICIENT",
            data_hold_reason=None,
            redaction_counts={"secrets": 0, "pii": 0},
        )
        logger.append(entry)
        raw = (log_dir / "query-log.jsonl").read_bytes()
        assert b"this should NEVER appear" not in raw
        assert b"extracted_text" not in raw


class TestQueryLoggerUnwritable:
    def test_unwritable_dir_at_construction_raises(self, tmp_path):
        """If the log directory cannot be created (parent missing or unwritable
        in a way that even root cannot override), the constructor raises
        QueryLogWriteError. This makes the fail-closed behaviour deterministic
        — there is no partially-constructed logger that can silently swallow
        appends."""
        blocker = tmp_path / "blocker"
        blocker.write_text("not a dir")
        d = blocker / "vault-retrieval"
        with pytest.raises(QueryLogWriteError):
            QueryLogger(d, query_log_enabled=True)

    def test_disabled_logger_append_raises(self, tmp_path):
        """A logger instantiated with query_log_enabled=False raises on append
        — auditing off is fail-closed, not fail-open."""
        d = tmp_path / "state-disabled"
        d.mkdir()
        logger = QueryLogger(d, query_log_enabled=False)
        entry = QueryLogEntry(
            query_id="RQL-OFF",
            queried_at=datetime.now(timezone.utc).isoformat(),
            task_id=None,
            run_id=None,
            project_topic="t",
            decision_horizon="t",
            search_terms=[],
            consulted=[{"path": "a", "range": "lines 1-1",
                         "source_mtime": "2026-10-01T00:00:00+00:00",
                         "source_last_updated": None, "characters_read": 0}],
            total_characters_read=0,
            budget_status="default",
            outcome="SUFFICIENT",
            data_hold_reason=None,
            redaction_counts={"secrets": 0, "pii": 0},
        )
        with pytest.raises(QueryLogWriteError):
            logger.append(entry)
        # And no file was created.
        assert not (d / "query-log.jsonl").exists()
