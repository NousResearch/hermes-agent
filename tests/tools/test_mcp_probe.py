"""Tests for the parallel-discovery outer-timeout policy in tools.mcp_tool.

Covers ``probe_mcp_server_tools()`` and the sibling site in
``register_mcp_servers()``: both gather per-server connects in parallel under
an enclosing ``_run_on_mcp_loop`` bound, and both must scale that bound off the
largest *sanitized* ``connect_timeout`` so a slow-but-legitimately-configured
server is not cancelled before its own budget elapses.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_mcp_state():
    """Ensure clean MCP module state before/after each test."""
    import tools.mcp_tool as mcp
    old_loop = mcp._mcp_loop
    old_thread = mcp._mcp_thread
    old_servers = dict(mcp._servers)
    yield
    mcp._servers.clear()
    mcp._servers.update(old_servers)
    mcp._mcp_loop = old_loop
    mcp._mcp_thread = old_thread


class TestProbeMcpServerTools:
    """Tests for the lightweight probe_mcp_server_tools function."""

    def test_returns_empty_when_mcp_not_available(self):
        with patch("tools.mcp_tool._MCP_AVAILABLE", False):
            from tools.mcp_tool import probe_mcp_server_tools
            result = probe_mcp_server_tools()
        assert result == {}


    def test_failed_server_omitted_from_results(self):
        """Servers that fail to connect are silently skipped."""
        config = {
            "github": {"command": "npx", "connect_timeout": 5},
            "broken": {"command": "nonexistent", "connect_timeout": 5},
        }
        mock_tool = SimpleNamespace(name="create_issue", description="Create")
        mock_server = MagicMock()
        mock_server._tools = [mock_tool]
        mock_server.shutdown = AsyncMock()

        async def fake_connect(name, cfg):
            if name == "broken":
                raise ConnectionError("Server not found")
            return mock_server

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._load_mcp_config", return_value=config), \
             patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("tools.mcp_tool._ensure_mcp_loop"), \
             patch("tools.mcp_tool._run_on_mcp_loop") as mock_run, \
             patch("tools.mcp_tool._stop_mcp_loop"):

            def run_coro(coro_or_factory, timeout=120):
                coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()

            mock_run.side_effect = run_coro

            from tools.mcp_tool import probe_mcp_server_tools
            result = probe_mcp_server_tools()

        assert "github" in result
        assert "broken" not in result


    def test_outer_timeout_respects_large_connect_timeout(self):
        """A server with connect_timeout > 120 must not be cancelled by the
        outer probe bound before its own inner wait_for budget elapses."""
        config = {
            "slow": {"command": "npx", "connect_timeout": 300},
        }
        mock_tool = SimpleNamespace(name="do_thing", description="Do a thing")
        mock_server = MagicMock()
        mock_server._tools = [mock_tool]
        mock_server.shutdown = AsyncMock()

        async def fake_connect(name, cfg):
            return mock_server

        captured = {}

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._load_mcp_config", return_value=config), \
             patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("tools.mcp_tool._ensure_mcp_loop"), \
             patch("tools.mcp_tool._run_on_mcp_loop") as mock_run, \
             patch("tools.mcp_tool._stop_mcp_loop"):

            def run_coro(coro_or_factory, timeout=120):
                captured["timeout"] = timeout
                coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()

            mock_run.side_effect = run_coro

            from tools.mcp_tool import probe_mcp_server_tools
            probe_mcp_server_tools()

        # Outer bound must scale to the configured connect_timeout: max(120, 300 + 10).
        assert captured.get("timeout", 0) >= 310

    @pytest.mark.parametrize(
        "bad_value",
        [
            float("nan"),   # YAML `.nan` -> Python float nan
            float("inf"),   # YAML `.inf` -> Python float inf
            float("-inf"),  # YAML `-.inf`
            0,              # non-positive
            -5,             # negative
        ],
        ids=["nan", "inf", "-inf", "zero", "negative"],
    )
    def test_outer_timeout_finite_for_non_finite_connect_timeout(self, bad_value):
        """A YAML ``.nan``/``.inf`` (or non-positive) connect_timeout must not
        poison the outer probe bound.

        A YAML loader parses ``.nan``/``.inf`` into real Python ``float('nan')``/
        ``float('inf')`` values, so they reach ``cfg.get('connect_timeout')`` as
        finite-passing floats. ``float(nan)`` raises no ``TypeError``/``ValueError``,
        so the old guard let them through and ``max(connect_timeouts) + 10``
        became non-finite — a non-finite outer deadline never trips
        ``_run_on_mcp_loop``'s ``remaining <= 0`` branch, so the probe would
        never time out. The poisoned value must be sanitized to the default
        while a legitimately-large peer server still drives the bound.

        A second, valid ``connect_timeout=300`` server is co-configured so the
        correct outer bound is ``max(120, 300 + 10) = 310``. Pre-fix, an ``inf``
        member makes ``max()`` return ``inf`` and a ``nan`` member silently
        collapses the bound back to the ``120`` floor — both distinguishable
        from the sanitized 310.
        """
        import math as _math

        config = {
            "poisoned": {"command": "npx", "connect_timeout": bad_value},
            "slow": {"command": "npx", "connect_timeout": 300},
        }
        mock_tool = SimpleNamespace(name="do_thing", description="Do a thing")
        mock_server = MagicMock()
        mock_server._tools = [mock_tool]
        mock_server.shutdown = AsyncMock()

        async def fake_connect(name, cfg):
            return mock_server

        captured = {}

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._load_mcp_config", return_value=config), \
             patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("tools.mcp_tool._ensure_mcp_loop"), \
             patch("tools.mcp_tool._run_on_mcp_loop") as mock_run, \
             patch("tools.mcp_tool._stop_mcp_loop"):

            def run_coro(coro_or_factory, timeout=120):
                captured["timeout"] = timeout
                coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()

            mock_run.side_effect = run_coro

            from tools.mcp_tool import probe_mcp_server_tools
            probe_mcp_server_tools()

        outer = captured.get("timeout")
        assert outer is not None
        assert _math.isfinite(outer), f"outer_timeout was non-finite: {outer!r}"
        # Sanitized poisoned value drops out; the valid 300s peer drives the bound.
        assert outer == max(120.0, 300.0 + 10.0)

    def test_skips_disabled_servers(self):
        """Disabled servers are not probed."""
        config = {
            "github": {"command": "npx", "connect_timeout": 5},
            "disabled_one": {"command": "npx", "enabled": False},
        }
        mock_tool = SimpleNamespace(name="create_issue", description="Create")
        mock_server = MagicMock()
        mock_server._tools = [mock_tool]
        mock_server.shutdown = AsyncMock()

        connect_calls = []

        async def fake_connect(name, cfg):
            connect_calls.append(name)
            return mock_server

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._load_mcp_config", return_value=config), \
             patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("tools.mcp_tool._ensure_mcp_loop"), \
             patch("tools.mcp_tool._run_on_mcp_loop") as mock_run, \
             patch("tools.mcp_tool._stop_mcp_loop"):

            def run_coro(coro_or_factory, timeout=120):
                coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()

            mock_run.side_effect = run_coro

            from tools.mcp_tool import probe_mcp_server_tools
            result = probe_mcp_server_tools()

        assert "github" in result
        assert "disabled_one" not in result
        assert "disabled_one" not in connect_calls


class TestRegisterMcpServersOuterTimeout:
    """``register_mcp_servers`` runs the same parallel-discovery shape as the
    probe, so it needs the same largest-configured-timeout outer bound.

    Its inner per-server budget lives in ``_discover_and_register_server``,
    which honors each server's ``connect_timeout``; the enclosing
    ``_run_on_mcp_loop`` call was hard-capped at 120s, so a server configured
    above that floor was cancelled by the outer cap before its own connect
    budget elapsed and was silently dropped from discovery.
    """

    @staticmethod
    def _register(config, captured):
        """Drive ``register_mcp_servers`` with the real ``_discover_all``
        coroutine, recording the outer timeout it was handed."""
        import tools.mcp_tool as mcp

        async def fake_discover(name, cfg):
            return [f"mcp__{name}__tool_a"]

        def run_coro(coro_or_factory, timeout=120):
            captured["timeout"] = timeout
            coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._discover_and_register_server", side_effect=fake_discover), \
             patch("tools.mcp_tool._ensure_mcp_loop"), \
             patch("tools.mcp_tool._run_on_mcp_loop", side_effect=run_coro), \
             patch("tools.mcp_tool._existing_tool_names", return_value=[]):
            mcp.register_mcp_servers(config)

    def test_outer_timeout_respects_large_connect_timeout(self):
        """A server with ``connect_timeout`` above the 120s floor drives the bound."""
        captured = {}
        self._register({"slow": {"command": "npx", "connect_timeout": 300}}, captured)

        assert captured.get("timeout") == max(120.0, 300.0 + 10.0)

    def test_outer_timeout_keeps_120s_floor_for_small_timeouts(self):
        """Many small servers keep the generous 120s floor — the bound only grows."""
        captured = {}
        self._register(
            {
                "a": {"command": "npx", "connect_timeout": 5},
                "b": {"command": "npx", "connect_timeout": 10},
            },
            captured,
        )

        assert captured.get("timeout") == 120.0

    @pytest.mark.parametrize(
        "bad_value",
        [
            float("nan"),   # YAML `.nan` -> Python float nan
            float("inf"),   # YAML `.inf` -> Python float inf
            float("-inf"),  # YAML `-.inf`
            0,              # non-positive
            -5,             # negative
            "not-a-number",  # unparseable
        ],
        ids=["nan", "inf", "-inf", "zero", "negative", "unparseable"],
    )
    def test_outer_timeout_finite_for_bad_connect_timeout(self, bad_value):
        """A poisoned ``connect_timeout`` must not make the outer bound
        non-finite (``nan``/``inf`` deadlines never trip ``_run_on_mcp_loop``'s
        ``remaining <= 0`` branch, so discovery would never time out).

        A valid 300s peer is co-configured so the sanitized bound is a
        distinguishable 310: an ``inf`` member would otherwise return ``inf``
        from ``max()`` and a ``nan`` member would collapse it to the floor.
        """
        import math as _math

        captured = {}
        self._register(
            {
                "poisoned": {"command": "npx", "connect_timeout": bad_value},
                "slow": {"command": "npx", "connect_timeout": 300},
            },
            captured,
        )

        outer = captured.get("timeout")
        assert outer is not None
        assert _math.isfinite(outer), f"outer_timeout was non-finite: {outer!r}"
        assert outer == max(120.0, 300.0 + 10.0)

class TestSanitizedConnectTimeout:
    """``_sanitized_connect_timeout`` is the single source both the inner
    ``asyncio.wait_for`` budgets and the outer bound read, so the two can never
    disagree about what a server's connect budget is."""

    @pytest.mark.parametrize(
        "bad_value",
        [
            float("nan"),
            float("inf"),
            float("-inf"),
            0,
            -5,
            "not-a-number",
            None,
            [30],
        ],
        ids=["nan", "inf", "-inf", "zero", "negative", "unparseable", "none", "list"],
    )
    def test_unusable_values_fall_back_to_the_default(self, bad_value):
        import tools.mcp_tool as mcp

        assert mcp._sanitized_connect_timeout({"connect_timeout": bad_value}) == float(
            mcp._DEFAULT_CONNECT_TIMEOUT
        )

    def test_missing_value_uses_the_default(self):
        import tools.mcp_tool as mcp

        assert mcp._sanitized_connect_timeout({}) == float(mcp._DEFAULT_CONNECT_TIMEOUT)

    @pytest.mark.parametrize("good_value", [1, 5, 30.5, 300, "45"])
    def test_usable_values_pass_through_as_floats(self, good_value):
        import tools.mcp_tool as mcp

        result = mcp._sanitized_connect_timeout({"connect_timeout": good_value})
        assert result == float(good_value)
        assert isinstance(result, float)
