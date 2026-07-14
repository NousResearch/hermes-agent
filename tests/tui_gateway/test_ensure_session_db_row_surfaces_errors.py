"""Regression tests for #63474 — Dashboard Chat tab sessions never persisted.

The reporter's diagnosis: ``_ensure_session_db_row`` swallows exceptions
in ``db.create_session`` at ``logger.debug`` level, so a transient
schema/connection failure during prompt.submit is silently dropped and
the session never appears in state.db.

This test enforces the new contract: when db.create_session raises, the
exception must surface (either re-raised or logged at WARNING+ level so
an operator scanning logs can see the failure). The previous behavior
swallowed at DEBUG, which is invisible in production deployments.
"""

import logging
import os
import sys
from pathlib import Path

import pytest


# Make repo root importable
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def srv_module():
    """Import tui_gateway.server via the real package path so logger names
    resolve to "tui_gateway.server" (matching caplog filters) AND other
    tests in the suite that monkeypatch `tui_gateway.server` keep working."""
    import importlib
    import tui_gateway  # ensure the package is importable
    return importlib.import_module("tui_gateway.server")


@pytest.fixture
def minimal_session():
    return {
        "session_key": "agent:main:tui:dm:test-session-key",
        "profile_home": None,
        "explicit_cwd": False,
    }


def test_create_session_failure_surfaces_at_warning_or_higher(srv_module, minimal_session):
    """#63474: when db.create_session raises, the failure must NOT be hidden
    at DEBUG level. The operator needs to see the error in normal production
    logs (WARNING+ is the contract). The exception is NOT re-raised because
    the 4 callers (system-message append, /title RPC, watcher setup,
    prompt.submit) all need to keep going if pre-creating the row failed —
    making it a hard error there would break user-facing flows."""
    class _DB:
        def create_session(self, *a, **kw):
            raise RuntimeError("synthetic db schema mismatch")

    srv_module._get_db = lambda: _DB()
    srv_module._session_cwd = lambda s: None
    srv_module._session_source = lambda s: "tui"

    # Attach a capture handler directly to the production logger. (We
    # deliberately avoid pytest's `caplog` fixture here — the project's
    # xonsh pytest plugin interferes with caplog's handler attachment
    # in subtle ways, and a direct handler is more reliable for verifying
    # this specific contract.)
    prod_logger = logging.getLogger("tui_gateway.server")
    captured = []

    class _Capture(logging.Handler):
        def emit(self, record):
            captured.append(record)

    handler = _Capture(level=logging.DEBUG)
    prod_logger.addHandler(handler)
    old_level = prod_logger.level
    prod_logger.setLevel(logging.DEBUG)
    try:
        # Must NOT raise. The function's contract is: log at WARNING+,
        # return cleanly, let the caller keep going.
        srv_module._ensure_session_db_row(minimal_session)
    finally:
        prod_logger.removeHandler(handler)
        prod_logger.setLevel(old_level)

    warning_records = [
        r for r in captured
        if r.levelno >= logging.WARNING and (
            "RuntimeError" in r.getMessage()
            or (r.exc_info and (
                "RuntimeError" in repr(r.exc_info[1])
                or r.exc_info[0] is RuntimeError
            ))
        )
    ]
    assert warning_records, (
        f"db.create_session raised but _ensure_session_db_row did not "
        f"log at WARNING+ level — silent failure mode is back (see #63474). "
        f"Captured records: {[(r.levelname, r.getMessage()[:80]) for r in captured]}"
    )


def test_create_session_failure_logs_exception_traceback(srv_module, minimal_session):
    """Companion test: when the create_session failure is logged, the actual
    exception type and message must be visible in the log message — not just
    'failed to persist' with no context. Operators need the traceback."""
    class _DB:
        def create_session(self, *a, **kw):
            raise RuntimeError("synthetic_unique_failure_marker_xyz")

    srv_module._get_db = lambda: _DB()
    srv_module._session_cwd = lambda s: None
    srv_module._session_source = lambda s: "tui"

    prod_logger = logging.getLogger("tui_gateway.server")
    captured = []

    class _Capture(logging.Handler):
        def emit(self, record):
            captured.append(record)

    handler = _Capture(level=logging.DEBUG)
    prod_logger.addHandler(handler)
    old_level = prod_logger.level
    prod_logger.setLevel(logging.DEBUG)
    try:
        # Must not raise — see WARNING-level test for rationale.
        srv_module._ensure_session_db_row(minimal_session)
    finally:
        prod_logger.removeHandler(handler)
        prod_logger.setLevel(old_level)

    # The unique marker must appear in a WARNING-level record (not just DEBUG).
    # On unfixed code the only record emitted is at DEBUG level — so this test
    # fails on unfixed because the level filter excludes it.
    warning_records = [
        r for r in captured
        if r.levelno >= logging.WARNING and (
            "synthetic_unique_failure_marker_xyz" in r.getMessage()
            or (
                r.exc_info
                and "synthetic_unique_failure_marker_xyz" in repr(r.exc_info[1])
            )
        )
    ]
    assert warning_records, (
        f"create_session raised with a unique marker but the marker did not "
        f"appear in any WARNING-level record (records: {[(r.levelname, r.getMessage()[:80]) for r in captured]}). "
        f"#63474: silent failure mode is back."
    )


def test_successful_path_still_writes_row(srv_module, minimal_session):
    """Regression guard: the happy path must continue to work after the fix."""
    captured = {}

    class _DB:
        def create_session(self, key, **kwargs):
            captured["key"] = key
            captured.update(kwargs)

    srv_module._get_db = lambda: _DB()
    srv_module._session_cwd = lambda s: None
    srv_module._session_source = lambda s: "tui"

    srv_module._ensure_session_db_row(minimal_session)

    assert captured.get("key") == minimal_session["session_key"]
    assert captured.get("source") == "tui"


def test_no_session_key_is_noop(srv_module):
    """Regression guard: session_key missing → no db call at all (the
    function must still short-circuit cleanly)."""
    called = {"count": 0}

    class _DB:
        def create_session(self, *a, **kw):
            called["count"] += 1

    srv_module._get_db = lambda: _DB()

    # No session_key → should return immediately
    srv_module._ensure_session_db_row({})
    assert called["count"] == 0
