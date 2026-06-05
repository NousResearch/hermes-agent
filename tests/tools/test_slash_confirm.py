"""Tests for tools/slash_confirm.py — the generic slash-command confirmation primitive.

Covers register/resolve/clear lifecycle, stale-entry behavior, confirm_id
mismatch, handler exceptions, and async resolution.
"""

import asyncio
import time

import pytest

from tools import slash_confirm


@pytest.fixture(autouse=True)
def _clean_pending():
    """Every test gets a clean primitive state."""
    slash_confirm._pending.clear()
    yield
    slash_confirm._pending.clear()


class TestRegisterAndGetPending:
    def test_register_stores_entry(self):
        async def handler(choice):
            return f"got {choice}"

        slash_confirm.register("sess1", "cid1", "reload-mcp", handler)

        pending = slash_confirm.get_pending("sess1")
        assert pending is not None
        assert pending["confirm_id"] == "cid1"
        assert pending["command"] == "reload-mcp"
        assert pending["handler"] is handler
        assert "created_at" in pending

    def test_get_pending_missing_returns_none(self):
        assert slash_confirm.get_pending("nobody") is None

    def test_register_supersedes_prior_entry(self):
        async def h1(choice):
            return "first"

        async def h2(choice):
            return "second"

        slash_confirm.register("sess1", "cid1", "reload-mcp", h1)
        slash_confirm.register("sess1", "cid2", "reload-mcp", h2)

        pending = slash_confirm.get_pending("sess1")
        assert pending["confirm_id"] == "cid2"
        assert pending["handler"] is h2

    def test_get_pending_returns_copy_not_reference(self):
        async def h(choice):
            return "x"

        slash_confirm.register("sess1", "cid1", "cmd", h)

        p1 = slash_confirm.get_pending("sess1")
        p1["command"] = "mutated"

        p2 = slash_confirm.get_pending("sess1")
        assert p2["command"] == "cmd"


class TestResolve:
    @pytest.mark.asyncio
    async def test_resolve_runs_handler_and_pops_entry(self):
        calls = []

        async def handler(choice):
            calls.append(choice)
            return f"resolved {choice}"

        slash_confirm.register("sess1", "cid1", "reload-mcp", handler)

        result = await slash_confirm.resolve("sess1", "cid1", "once")
        assert result == "resolved once"
        assert calls == ["once"]

        # Entry should be popped.
        assert slash_confirm.get_pending("sess1") is None

    @pytest.mark.asyncio
    async def test_resolve_no_pending_returns_none(self):
        result = await slash_confirm.resolve("sess1", "cid1", "once")
        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_confirm_id_mismatch_returns_none(self):
        async def handler(choice):
            return "should not run"

        slash_confirm.register("sess1", "cid_real", "cmd", handler)

        result = await slash_confirm.resolve("sess1", "cid_wrong", "once")
        assert result is None

        # Stale entry should still be present (mismatch doesn't pop).
        assert slash_confirm.get_pending("sess1") is not None

    @pytest.mark.asyncio
    async def test_resolve_stale_entry_returns_none(self):
        async def handler(choice):
            return "should not run"

        slash_confirm.register("sess1", "cid1", "cmd", handler)
        # Force entry age past timeout
        slash_confirm._pending["sess1"]["created_at"] = time.time() - 10000

        result = await slash_confirm.resolve("sess1", "cid1", "once")
        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_handler_exception_returns_error_string(self):
        async def handler(choice):
            raise RuntimeError("boom")

        slash_confirm.register("sess1", "cid1", "cmd", handler)

        result = await slash_confirm.resolve("sess1", "cid1", "once")
        assert result is not None
        assert "boom" in result
        # Entry should still be popped even when handler raises.
        assert slash_confirm.get_pending("sess1") is None

    @pytest.mark.asyncio
    async def test_resolve_non_string_return_becomes_none(self):
        async def handler(choice):
            return {"not": "a string"}

        slash_confirm.register("sess1", "cid1", "cmd", handler)
        result = await slash_confirm.resolve("sess1", "cid1", "once")
        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_double_click_only_runs_handler_once(self):
        calls = []

        async def handler(choice):
            calls.append(choice)
            return "ran"

        slash_confirm.register("sess1", "cid1", "cmd", handler)

        # Simulate two near-simultaneous button clicks.
        r1, r2 = await asyncio.gather(
            slash_confirm.resolve("sess1", "cid1", "once"),
            slash_confirm.resolve("sess1", "cid1", "once"),
        )
        # Exactly one should have run the handler.
        assert calls == ["once"]
        assert (r1 == "ran") ^ (r2 == "ran")


class TestClear:
    def test_clear_removes_entry(self):
        async def h(c):
            return "x"

        slash_confirm.register("sess1", "cid1", "cmd", h)
        assert slash_confirm.get_pending("sess1") is not None

        slash_confirm.clear("sess1")
        assert slash_confirm.get_pending("sess1") is None

    def test_clear_missing_is_noop(self):
        # Should not raise.
        slash_confirm.clear("nobody")


class TestClearIfStale:
    def test_clears_stale_entry(self):
        async def h(c):
            return "x"

        slash_confirm.register("sess1", "cid1", "cmd", h)
        slash_confirm._pending["sess1"]["created_at"] = time.time() - 10000

        cleared = slash_confirm.clear_if_stale("sess1", timeout=300)
        assert cleared is True
        assert slash_confirm.get_pending("sess1") is None

    def test_preserves_fresh_entry(self):
        async def h(c):
            return "x"

        slash_confirm.register("sess1", "cid1", "cmd", h)

        cleared = slash_confirm.clear_if_stale("sess1", timeout=300)
        assert cleared is False
        assert slash_confirm.get_pending("sess1") is not None

    def test_returns_false_for_missing_entry(self):
        cleared = slash_confirm.clear_if_stale("nobody")
        assert cleared is False
