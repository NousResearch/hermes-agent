"""Unit tests for per-run MCP ``_meta`` ContextVar helpers."""

import contextvars

from tools.mcp_run_meta import get_mcp_run_meta, reset_mcp_run_meta, set_mcp_run_meta


def test_default_is_none():
    assert get_mcp_run_meta() is None


def test_set_and_reset_restore_prior_value():
    token = set_mcp_run_meta({"a": 1})
    try:
        assert get_mcp_run_meta() == {"a": 1}
    finally:
        reset_mcp_run_meta(token)
    assert get_mcp_run_meta() is None


def test_context_isolation_across_copied_contexts():
    """Concurrent runs must not share mcp_meta via ContextVar leakage."""
    seen = {}

    def _worker(label, meta):
        token = set_mcp_run_meta(meta)
        try:
            seen[label] = get_mcp_run_meta()
        finally:
            reset_mcp_run_meta(token)

    ctx_a = contextvars.copy_context()
    ctx_b = contextvars.copy_context()
    ctx_a.run(_worker, "a", {"run": "A"})
    ctx_b.run(_worker, "b", {"run": "B"})

    assert seen == {"a": {"run": "A"}, "b": {"run": "B"}}
    assert get_mcp_run_meta() is None
