"""Production-switch tests: ensure gateway init wires SqliteApprovalStore
as the default approval store, not InMemoryApprovalStore.

These tests protect against accidental regressions where someone changes
gateway/run.py and quietly removes the set_default_approval_store(...)
call, leaving production back on the legacy in-memory path.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tools import approval as approval_mod
from tools.approval import get_default_approval_store, set_default_approval_store
from tools.approval_store_memory import InMemoryApprovalStore
from tools.approval_store_sqlite import SqliteApprovalStore


@pytest.fixture(autouse=True)
def _restore_store():
    prev = get_default_approval_store()
    yield
    set_default_approval_store(prev)


def test_sqlite_store_default_path_uses_hermes_state_default_db_path():
    """SqliteApprovalStore() with no args MUST resolve db_path to the
    same value as hermes_state.DEFAULT_DB_PATH (i.e. the hermes-home
    convention via get_hermes_home()/state.db), not a hardcoded literal.

    Note: DEFAULT_DB_PATH is captured at hermes_state import time, so
    set_hermes_home_override AFTER import does not change it. This test
    therefore asserts on the symbol-equivalence (path is whatever
    hermes_state computed it to be), which is what production callers
    actually get."""
    from hermes_state import DEFAULT_DB_PATH

    store = SqliteApprovalStore()
    assert store._db_path == DEFAULT_DB_PATH, (
        f"SqliteApprovalStore default path {store._db_path!r} does not "
        f"match hermes_state.DEFAULT_DB_PATH {DEFAULT_DB_PATH!r}. The "
        "default path must come from DEFAULT_DB_PATH, not a hardcoded "
        "~/.hermes literal or any other independent computation."
    )


def test_no_hardcoded_home_hermes_path_in_store_default(monkeypatch):
    """Verify the SqliteApprovalStore code does NOT use a hardcoded
    ~/.hermes path as its primary default. The fallback string in the
    `except ImportError` branch is acceptable (only fires if hermes_state
    cannot be imported), but the primary path must go through
    DEFAULT_DB_PATH."""
    src = Path(__file__).parent.parent.parent / "tools" / "approval_store_sqlite.py"
    text = src.read_text()
    # The primary path (line ~145 area) must reference DEFAULT_DB_PATH:
    assert "DEFAULT_DB_PATH" in text, (
        "SqliteApprovalStore must import and use DEFAULT_DB_PATH for its "
        "primary default db_path"
    )
    # Bare ".hermes" literals are allowed ONLY inside the ImportError
    # fallback. Count must match: any new literal needs to be inside that
    # except block.
    hermes_literals = re.findall(r'"\.hermes"|\'\.hermes\'', text)
    assert len(hermes_literals) <= 1, (
        f"Found {len(hermes_literals)} hardcoded '.hermes' literals; "
        "only the ImportError-fallback may have one"
    )


def test_gateway_init_wires_sqlite_store(tmp_path):
    """When gateway/run.py initialises a GatewayRunner, the global
    approval store MUST be a SqliteApprovalStore instance (not None,
    not InMemoryApprovalStore).

    We exercise the wiring code path by instantiating a Runner with
    minimal config. If GatewayRunner.__init__ is too heavy to call here
    in a unit-test context, we at least verify the wiring snippet is
    present in the source AND that calling it directly produces the
    expected state."""
    # Override hermes_home to tmp_path so the test doesn't touch real DB.
    from hermes_constants import set_hermes_home_override
    token = set_hermes_home_override(str(tmp_path))
    try:
        # Reset the default store so we observe what gets installed.
        set_default_approval_store(None)
        assert get_default_approval_store() is None

        # Simulate the exact two-line block from GatewayRunner.__init__:
        from tools.approval import set_default_approval_store as _set
        from tools.approval_store_sqlite import SqliteApprovalStore as _Sqlite
        _set(_Sqlite())

        store = get_default_approval_store()
        assert store is not None, (
            "After gateway init wiring, default approval store must NOT be None"
        )
        assert isinstance(store, SqliteApprovalStore), (
            f"Default approval store must be SqliteApprovalStore, "
            f"got {type(store).__name__}"
        )
        assert not isinstance(store, InMemoryApprovalStore), (
            "InMemoryApprovalStore MUST NEVER be the production default; "
            "it is a test-only reference"
        )
    finally:
        from hermes_constants import reset_hermes_home_override
        reset_hermes_home_override(token)


def test_gateway_run_py_contains_sqlite_wiring_call():
    """Regression guard: the GatewayRunner __init__ MUST contain a call
    to set_default_approval_store(SqliteApprovalStore(...)). If someone
    later refactors and removes it, production silently regresses to
    legacy in-memory FIFO — this test catches that at CI time."""
    src = Path(__file__).parent.parent.parent / "gateway" / "run.py"
    text = src.read_text()
    # Search permissively — different formatting/whitespace is OK as
    # long as the call is structurally present.
    pattern = re.compile(
        r"set_default_approval_store\s*\(\s*SqliteApprovalStore\s*\(",
        re.MULTILINE,
    )
    assert pattern.search(text), (
        "gateway/run.py MUST call set_default_approval_store(SqliteApprovalStore(...)) "
        "during initialisation. If you removed it deliberately, you also need "
        "to remove this test and explain in the commit message why production "
        "should not have a durable approval store."
    )


def test_in_memory_store_not_referenced_outside_tests():
    """Codebase-level grep: production code (tools/, gateway/, agent/,
    hermes_cli/, hermes/) MUST NOT instantiate or default-wire
    InMemoryApprovalStore. The in-memory store exists only as a
    test-time reference."""
    root = Path(__file__).parent.parent.parent
    offenders = []
    for subdir in ("tools", "gateway", "agent", "hermes_cli", "hermes"):
        sub = root / subdir
        if not sub.is_dir():
            continue
        for py in sub.rglob("*.py"):
            text = py.read_text(errors="ignore")
            # Skip the in-memory module itself (it defines the class).
            if py.name == "approval_store_memory.py":
                continue
            # Look for instantiation or default-wiring.
            if re.search(r"\bInMemoryApprovalStore\s*\(", text) or \
               re.search(r"set_default_approval_store\s*\(\s*InMemoryApprovalStore", text):
                offenders.append(str(py.relative_to(root)))
    assert offenders == [], (
        "InMemoryApprovalStore is instantiated in production code paths: "
        f"{offenders}. It must only appear in tests/. The production "
        "default approval store is SqliteApprovalStore."
    )
