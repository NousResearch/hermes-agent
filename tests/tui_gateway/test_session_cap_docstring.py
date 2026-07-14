"""Tests for the LRU session-cap configuration (#63552).

The cap is documented in tui_gateway/server.py directly above
_max_live_sessions(). The docstring previously claimed "Default-on" while the
code returns 0 (off) when unconfigured — actively misleading for future
maintainers. The cap is also gated on _transport_is_dead() which is never true
while a WebSocket is alive, making it doubly inert for connected clients.

These tests read the docstring directly (so a wrong docstring fails on unfixed
code) and verify a single behavioral contract: when no max_live_sessions is
configured, _max_live_sessions() returns 0 (default-off).
"""

import importlib.util
import inspect
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_PY = REPO_ROOT / "tui_gateway" / "server.py"


def _load_server_module():
    """Import tui_gateway.server without triggering its heavy app init."""
    spec = importlib.util.spec_from_file_location("tui_gateway_server_under_test", SERVER_PY)
    assert spec is not None and spec.loader is not None, f"failed to load {SERVER_PY}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _docstring_block_above_max_live_sessions():
    """Return the comment block immediately above _max_live_sessions()."""
    src = SERVER_PY.read_text(encoding="utf-8")
    lines = src.splitlines()
    idx = None
    for i, line in enumerate(lines):
        if line.startswith("def _max_live_sessions"):
            idx = i
            break
    assert idx is not None, "_max_live_sessions() not found in tui_gateway/server.py"
    block = []
    # Walk backwards collecting comment lines.
    for j in range(idx - 1, -1, -1):
        line = lines[j]
        if line.startswith("#"):
            block.insert(0, line)
        else:
            break
    return "\n".join(block)


def test_docstring_does_not_claim_default_on():
    """#63552: the misleading 'Default-on' phrase must be gone from the docstring."""
    block = _docstring_block_above_max_live_sessions()
    assert "Default-on" not in block, (
        "docstring still claims 'Default-on' but code returns 0 when unconfigured; "
        "see #63552"
    )


def test_docstring_uses_opt_in_or_default_off_language():
    """The docstring must state that the cap is opt-in / default-off, so a future
    maintainer reading the code doesn't assume the cap runs by default."""
    block = _docstring_block_above_max_live_sessions().lower()
    assert (
        "default-off" in block
        or "opt-in" in block
        or "opt in" in block
    ), "docstring must clearly mark the cap as opt-in / default-off (see #63552)"


def test_docstring_documents_inertness_with_live_transport():
    """The dead-transport-gate inertness is a known limitation that future
    maintainers should not have to re-derive from reading _session_is_lru_evictable.
    The docstring should at least mention the inertness/limitation."""
    block = _docstring_block_above_max_live_sessions().lower()
    assert (
        "inert" in block
        or "doubly" in block
        or "_transport_is_dead" in block
        or "transport" in block
    ), "docstring should reference the dead-transport gate / inertness limitation"


def test_default_off_when_unconfigured():
    """Behavioral contract: with no config and no env override, the cap is 0."""
    import os
    # Make sure no env-var override accidentally enables the cap.
    for k in ("HERMES_MAX_LIVE_SESSIONS", "MAX_LIVE_SESSIONS"):
        os.environ.pop(k, None)

    server = _load_server_module()
    cap = server._max_live_sessions()
    assert cap == 0, f"expected cap=0 (off by default), got {cap}"


def test_max_live_sessions_is_callable_int_returning():
    """Sanity: the function signature should still return an int."""
    server = _load_server_module()
    sig = inspect.signature(server._max_live_sessions)
    assert sig.return_annotation is int
