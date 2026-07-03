"""Tests for terminal/file tool availability in local dev environments."""

import importlib

import pytest

from model_tools import get_tool_definitions

terminal_tool_module = importlib.import_module("tools.terminal_tool")


@pytest.fixture(autouse=True)
def _clear_caches():
    """Invalidate check_fn and tool-definitions caches before each test
    so that monkeypatched env vars / config take effect."""
    from tools.registry import invalidate_check_fn_cache
    from model_tools import _clear_tool_defs_cache
    invalidate_check_fn_cache()
    _clear_tool_defs_cache()
    yield
    invalidate_check_fn_cache()
    _clear_tool_defs_cache()


class TestTerminalRequirements:
    def test_local_backend_requirements(self, monkeypatch):
        monkeypatch.setattr(
            terminal_tool_module,
            "_get_env_config",
            lambda: {"env_type": "local"},
        )
        assert terminal_tool_module.check_terminal_requirements() is True

    def test_terminal_and_file_tools_resolve_for_local_backend(self, monkeypatch):
        monkeypatch.setattr(
            terminal_tool_module,
            "_get_env_config",
            lambda: {"env_type": "local"},
        )
        tools = get_tool_definitions(enabled_toolsets=["terminal", "file"], quiet_mode=True)
        names = {tool["function"]["name"] for tool in tools}
        assert "terminal" in names
        assert {"read_file", "write_file", "patch", "search_files"}.issubset(names)

    def test_terminal_and_execute_code_tools_resolve_for_managed_modal(self, monkeypatch, tmp_path):
        monkeypatch.setattr("tools.tool_backend_helpers.managed_nous_tools_enabled", lambda: True)
        monkeypatch.setattr(terminal_tool_module, "managed_nous_tools_enabled", lambda: True)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
        monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
        monkeypatch.setattr(
            terminal_tool_module,
            "_get_env_config",
            lambda: {"env_type": "modal", "modal_mode": "managed"},
        )
        monkeypatch.setattr(
            terminal_tool_module,
            "is_managed_tool_gateway_ready",
            lambda _vendor: True,
        )
        tools = get_tool_definitions(enabled_toolsets=["terminal", "code_execution"], quiet_mode=True)
        names = {tool["function"]["name"] for tool in tools}

        assert "terminal" in names
        assert "execute_code" in names


class TestCheckFnTransientFailureSuppression:
    """The check_fn TTL cache should absorb transient probe failures.

    Regression coverage for #21658 / #5304: a single flaky
    ``check_terminal_requirements()`` (Docker daemon busy, probe timeout)
    must not silently strip the terminal/file toolset from a subagent. After
    a recent success, a transient False is treated as a flake; a failure with
    no recent success — or past the grace window — is honored.
    """

    @pytest.fixture(autouse=True)
    def _reset(self):
        from tools.registry import invalidate_check_fn_cache

        invalidate_check_fn_cache()
        yield
        invalidate_check_fn_cache()

    def test_transient_failure_after_success_is_suppressed(self, monkeypatch):
        import tools.registry as reg

        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            # First call succeeds, second flakes (False).
            return calls["n"] == 1

        # Pin the cache clock so the TTL doesn't serve a stale entry between
        # the two probes — we want both to actually run.
        t = {"now": 1000.0}
        monkeypatch.setattr(reg.time, "monotonic", lambda: t["now"])

        assert reg._check_fn_cached(flaky) is True  # records last-good
        t["now"] += reg._CHECK_FN_TTL_SECONDS + 1  # expire the TTL cache
        # Within grace window of the success → flake suppressed, stays True.
        assert reg._check_fn_cached(flaky) is True
        assert calls["n"] == 2  # the probe actually ran (not just cached)

    def test_persistent_failure_after_grace_is_honored(self, monkeypatch):
        import tools.registry as reg

        def good():
            return True

        def bad():
            return False

        t = {"now": 1000.0}
        monkeypatch.setattr(reg.time, "monotonic", lambda: t["now"])

        assert reg._check_fn_cached(good) is True
        # Advance past the failure grace window, then fail.
        t["now"] += reg._CHECK_FN_FAILURE_GRACE_SECONDS + 1
        # Different fn so last-good for `good` doesn't apply; bad has no success.
        assert reg._check_fn_cached(bad) is False

    def test_failure_with_no_prior_success_is_honored(self, monkeypatch):
        import tools.registry as reg

        def never():
            return False

        t = {"now": 1000.0}
        monkeypatch.setattr(reg.time, "monotonic", lambda: t["now"])
        assert reg._check_fn_cached(never) is False

    def test_grace_expiry_lets_real_outage_through(self, monkeypatch):
        import tools.registry as reg

        state = {"ok": True}

        def probe():
            return state["ok"]

        t = {"now": 1000.0}
        monkeypatch.setattr(reg.time, "monotonic", lambda: t["now"])

        assert reg._check_fn_cached(probe) is True
        state["ok"] = False
        # Just past TTL, within grace → flake suppressed.
        t["now"] += reg._CHECK_FN_TTL_SECONDS + 1
        assert reg._check_fn_cached(probe) is True
        # Now move well past the grace window since the last success → honored.
        t["now"] += reg._CHECK_FN_FAILURE_GRACE_SECONDS + 1
        assert reg._check_fn_cached(probe) is False

    def test_subagent_keeps_file_tools_through_docker_flake(self, monkeypatch):
        """End-to-end: a docker probe that flakes on the 2nd build keeps the
        file/terminal toolset available for the subagent being constructed."""
        import tools.registry as reg

        flake = {"first": True}

        def flaky_terminal_check():
            if flake["first"]:
                flake["first"] = False
                return True
            return False  # transient flake on the subagent build

        monkeypatch.setattr(
            terminal_tool_module, "check_terminal_requirements", flaky_terminal_check
        )
        # file tools delegate to the same check via tools.check_file_requirements.
        import tools as tools_pkg

        monkeypatch.setattr(
            tools_pkg, "check_file_requirements", flaky_terminal_check
        )

        t = {"now": 5000.0}
        monkeypatch.setattr(reg.time, "monotonic", lambda: t["now"])

        from model_tools import get_tool_definitions, _clear_tool_defs_cache

        reg.invalidate_check_fn_cache()
        _clear_tool_defs_cache()
        # Parent build (probe ok) → records last-good.
        parent = get_tool_definitions(enabled_toolsets=["terminal", "file"], quiet_mode=True)
        assert "read_file" in {x["function"]["name"] for x in parent}

        # Subagent build moments later: TTL expired, probe flakes False, but
        # within grace → file/terminal tools must still resolve.
        t["now"] += reg._CHECK_FN_TTL_SECONDS + 1
        _clear_tool_defs_cache()
        child = get_tool_definitions(enabled_toolsets=["terminal", "file"], quiet_mode=True)
        child_names = {x["function"]["name"] for x in child}
        assert {"read_file", "write_file", "patch", "search_files", "terminal"}.issubset(
            child_names
        )
