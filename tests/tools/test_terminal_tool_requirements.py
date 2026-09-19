"""Tests for terminal/file tool availability in local dev environments."""

import importlib
import logging

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

    def test_file_requirements_resolve_from_the_current_package(self, monkeypatch):
        import tools
        from tools import file_tools

        monkeypatch.setattr(
            terminal_tool_module,
            "check_terminal_requirements",
            lambda: True,
        )

        def fail_package_export():
            raise AssertionError("file requirements re-imported the tools package export")

        monkeypatch.setattr(tools, "check_file_requirements", fail_package_export)
        assert file_tools._check_file_reqs() is True


    def test_terminal_and_execute_code_tools_resolve_for_managed_modal(self, monkeypatch, tmp_path):
        monkeypatch.setattr("tools.tool_backend_helpers.managed_nous_tools_enabled", lambda: True)
        monkeypatch.setattr("tools.terminal_tool_backends.managed_nous_tools_enabled", lambda: True)
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
            "tools.terminal_tool_backends.is_managed_tool_gateway_ready",
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

    def test_expected_false_reprobe_logs_info_but_probe_exception_stays_warning(
        self, monkeypatch, caplog
    ):
        import tools.registry as reg

        calls = {"count": 0}

        def unavailable():
            calls["count"] += 1
            return False

        def broken():
            raise RuntimeError("probe failed")

        clock = {"now": 1000.0}
        monkeypatch.setattr(reg.time, "monotonic", lambda: clock["now"])

        with caplog.at_level(logging.INFO, logger="tools.registry"):
            assert reg._check_fn_cached(unavailable) is False
            clock["now"] += reg._CHECK_FN_TTL_SECONDS + 1
            assert reg._check_fn_cached(unavailable) is False
            assert reg._check_fn_cached(broken) is False

        expected_false = [
            record for record in caplog.records if "returned False" in record.getMessage()
        ]
        raised = [record for record in caplog.records if "raised" in record.getMessage()]
        assert calls["count"] == 2
        assert [record.levelno for record in expected_false] == [logging.INFO, logging.INFO]
        assert len(raised) == 1
        assert raised[0].levelno == logging.WARNING
        assert raised[0].exc_info is not None


    def test_core_tool_drop_after_success_warns_once_never_configured_stays_info(self, monkeypatch, caplog):
        """A check_fn that drops a ``_HERMES_CORE_TOOLS`` member it had previously admitted logs at
        WARNING naming the tool (#112649 atom 4): core tools are non-deferrable, so a dropped one
        leaves neither the schema nor the tool_search catalog and the model's "no such tool" is
        accurate. A core tool whose probe never succeeded (unconfigured browser/image_gen/HA on a
        stock home) keeps the INFO verdict of ae5666f7fc4 — the live pass counted 13 WARNINGs per
        process when every unconfigured core tool warned. The WARNING fires once per probe per
        process, not on every TTL re-probe.
        """
        import tools.registry as reg

        state = {"core": True}

        def core_gate():
            return state["core"]

        def never_gate():
            return False

        import toolsets

        clock = {"now": 1000.0}
        monkeypatch.setattr(reg.time, "monotonic", lambda: clock["now"])
        monkeypatch.setattr(toolsets, "_HERMES_CORE_TOOLS",
                            [*toolsets._HERMES_CORE_TOOLS, "core_probe_tool", "never_probe_tool"])
        for name, gate in (("core_probe_tool", core_gate), ("never_probe_tool", never_gate)):
            reg.registry.register(name=name, toolset=name, schema={"name": name},
                                  handler=lambda **kw: None, check_fn=gate)
        reg.invalidate_check_fn_cache()
        try:
            with caplog.at_level(logging.INFO, logger="tools.registry"):
                assert reg._check_fn_cached(core_gate) is True
                assert reg._check_fn_cached(never_gate) is False
                state["core"] = False
                clock["now"] += reg._CHECK_FN_FAILURE_GRACE_SECONDS + reg._CHECK_FN_TTL_SECONDS + 1
                assert reg._check_fn_cached(core_gate) is False
                assert reg._check_fn_cached(never_gate) is False
                clock["now"] += reg._CHECK_FN_TTL_SECONDS + 1
                assert reg._check_fn_cached(core_gate) is False
        finally:
            reg.registry.deregister("core_probe_tool")
            reg.registry.deregister("never_probe_tool")
            reg.invalidate_check_fn_cache()

        core = [r for r in caplog.records if "core_gate" in r.getMessage()]
        never = [r for r in caplog.records if "never_gate" in r.getMessage()]
        assert [r.levelno for r in core] == [logging.WARNING, logging.INFO]
        assert "core_probe_tool" in core[0].getMessage()
        assert never and all(r.levelno == logging.INFO for r in never)

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

    def test_profile_scoped_availability_does_not_cross_multiplex_profiles(
        self, tmp_path
    ):
        """Both availability caches must use the active profile as a key."""
        import tools.registry as reg
        from agent.secret_scope import (
            get_secret,
            reset_secret_scope,
            set_multiplex_active,
            set_secret_scope,
        )
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )
        from model_tools import _clear_tool_defs_cache, get_tool_definitions

        profile_a = tmp_path / "profiles" / "a"
        profile_b = tmp_path / "profiles" / "b"
        profile_a.mkdir(parents=True)
        profile_b.mkdir(parents=True)
        tool_name = "profile_scoped_availability_probe"

        def probe():
            return bool(get_secret("PROFILE_CACHE_TEST_TOKEN"))

        reg.registry.register(
            name=tool_name,
            toolset="profile-cache-test",
            schema={
                "name": tool_name,
                "description": "test-only profile-scoped availability probe",
                "parameters": {"type": "object", "properties": {}},
            },
            handler=lambda _args: "ok",
            check_fn=probe,
        )
        set_multiplex_active(True)
        try:
            home_a = set_hermes_home_override(str(profile_a))
            secrets_a = set_secret_scope({"PROFILE_CACHE_TEST_TOKEN": "token-a"})
            try:
                tools_a = get_tool_definitions(
                    enabled_toolsets=["profile-cache-test"],
                    quiet_mode=True,
                    skip_tool_search_assembly=True,
                )
            finally:
                reset_secret_scope(secrets_a)
                reset_hermes_home_override(home_a)

            home_b = set_hermes_home_override(str(profile_b))
            secrets_b = set_secret_scope({})
            try:
                tools_b = get_tool_definitions(
                    enabled_toolsets=["profile-cache-test"],
                    quiet_mode=True,
                    skip_tool_search_assembly=True,
                )
            finally:
                reset_secret_scope(secrets_b)
                reset_hermes_home_override(home_b)

            assert tool_name in {tool["function"]["name"] for tool in tools_a}
            assert tool_name not in {tool["function"]["name"] for tool in tools_b}
        finally:
            set_multiplex_active(False)
            reg.registry.deregister(tool_name)
            reg.invalidate_check_fn_cache()
            _clear_tool_defs_cache()

    def test_unscoped_multiplex_request_bypasses_cache(self, monkeypatch):
        """An unknown profile must never share another request's cache entry."""
        import model_tools
        import tools.registry as reg
        from agent.secret_scope import set_multiplex_active

        values = iter([True, False])
        definition_calls = {"n": 0}

        def probe():
            return next(values)

        def compute_definitions(*_args, **_kwargs):
            definition_calls["n"] += 1
            return []

        set_multiplex_active(True)
        monkeypatch.setattr(model_tools, "_compute_tool_definitions", compute_definitions)
        try:
            assert reg._check_fn_cached(probe) is True
            assert reg._check_fn_cached(probe) is False
            assert not reg._check_fn_cache

            model_tools.get_tool_definitions(quiet_mode=True)
            model_tools.get_tool_definitions(quiet_mode=True)
            assert definition_calls["n"] == 2
            assert not model_tools._tool_defs_cache
        finally:
            set_multiplex_active(False)
            reg.invalidate_check_fn_cache()
            model_tools._clear_tool_defs_cache()

    def test_profile_scoped_check_cache_is_bounded(self, monkeypatch):
        """Many multiplex profiles must not grow registry caches forever."""
        import tools.registry as reg

        scopes = iter(f"/profiles/{index}" for index in range(1_000))
        monkeypatch.setattr(reg, "check_fn_cache_scope", lambda: next(scopes))

        def available():
            return True

        for _ in range(1_000):
            assert reg._check_fn_cached(available) is True

        assert len(reg._check_fn_cache) <= reg._CHECK_FN_CACHE_MAX
        assert len(reg._check_fn_last_good) <= reg._CHECK_FN_CACHE_MAX

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
        # File tools resolve the terminal check from their own package. This
        # stays reliable when a workspace also contains a top-level tools/ dir.

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
    def test_terminal_and_execute_code_tools_resolve_for_vercel_sandbox(self, monkeypatch):
        monkeypatch.setenv("VERCEL_OIDC_TOKEN", "oidc-token")
        monkeypatch.setattr(
            terminal_tool_module,
            "_get_env_config",
            lambda: {"env_type": "vercel_sandbox", "container_disk": 51200},
        )
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda _name: object(),
        )
        tools = get_tool_definitions(enabled_toolsets=["terminal", "code_execution"], quiet_mode=True)
        names = {tool["function"]["name"] for tool in tools}

        assert "terminal" in names
        assert "execute_code" in names

    def test_terminal_and_execute_code_tools_hide_for_unsupported_vercel_runtime(self, monkeypatch):
        monkeypatch.setenv("VERCEL_OIDC_TOKEN", "oidc-token")
        monkeypatch.setattr(
            terminal_tool_module,
            "_get_env_config",
            lambda: {
                "env_type": "vercel_sandbox",
                "container_disk": 51200,
                "vercel_runtime": "node20",
            },
        )
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda _name: object(),
        )
        tools = get_tool_definitions(enabled_toolsets=["terminal", "code_execution"], quiet_mode=True)
        names = {tool["function"]["name"] for tool in tools}

        assert "terminal" not in names
        assert "execute_code" not in names

    def test_terminal_and_execute_code_tools_hide_for_vercel_without_auth(self, monkeypatch):
        monkeypatch.delenv("VERCEL_OIDC_TOKEN", raising=False)
        monkeypatch.delenv("VERCEL_TOKEN", raising=False)
        monkeypatch.delenv("VERCEL_PROJECT_ID", raising=False)
        monkeypatch.delenv("VERCEL_TEAM_ID", raising=False)
        monkeypatch.setattr(
            terminal_tool_module,
            "_get_env_config",
            lambda: {
                "env_type": "vercel_sandbox",
                "container_disk": 51200,
                "vercel_runtime": "node22",
            },
        )
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda _name: object(),
        )
        tools = get_tool_definitions(enabled_toolsets=["terminal", "code_execution"], quiet_mode=True)
        names = {tool["function"]["name"] for tool in tools}

        assert "terminal" not in names
        assert "execute_code" not in names


class TestUnscopedSecretReadLogging:
    """#100697: with multiplexing on, boot-time check_fns run before any
    profile secret scope exists, so get_secret fails closed with
    UnscopedSecretError. That expected signal must not be logged like a
    crashed check_fn (WARNING + traceback); an unscoped read reported while
    the scope was *resolved* is a genuinely lost scope and stays loud."""

    def test_raising_check_fn_logs_traceback_on_cached_path(self, caplog):
        """A check_fn that raises is a probe bug, not "nothing configured": the verdict log must
        carry the traceback so a silently stripped toolset is diagnosable from agent.log (#87950)."""
        import logging

        import tools.registry as reg

        def probe():
            raise RuntimeError("resolver exploded")

        with caplog.at_level(logging.WARNING, logger="tools.registry"):
            assert reg._check_fn_cached(probe) is False
        verdicts = [r for r in caplog.records if "dependent tools will be unavailable" in r.getMessage()]
        assert verdicts and all(r.exc_info and r.exc_info[0] is RuntimeError for r in verdicts)

    def test_expected_fail_closed_probe_is_quiet_but_lost_scope_stays_loud(self, caplog):
        """The verdict comes from the LIVE scope at the catch site (#110635): unscoped caller →
        expected boot-time miss (DEBUG, no traceback); scoped caller whose probe still hit the
        fail-closed path → a dropped scope inside the probe (WARNING + traceback)."""
        import logging
        import threading

        import tools.registry as reg
        from agent.secret_scope import (
            get_secret, reset_secret_scope, set_multiplex_active, set_secret_scope,
        )

        def probe():
            return bool(get_secret("REGISTRY_LOG_PROBE_TOKEN", ""))

        def probe_on_bare_thread():
            # A bare thread starts with an EMPTY context: the scope the caller holds never reaches
            # the get_secret inside — the spawn-site bug the loud branch exists to expose.
            out = []
            t = threading.Thread(target=lambda: out.append(probe()))
            t.start()
            t.join()
            return out[0]

        set_multiplex_active(True)
        try:
            with caplog.at_level(logging.DEBUG, logger="tools.registry"):
                assert reg._run_check_fn_uncached(probe) is False
                boot = [r for r in caplog.records if r.name == "tools.registry"]
                caplog.clear()
                token = set_secret_scope({})
                try:
                    assert reg._run_check_fn_uncached(probe_on_bare_thread) is False
                finally:
                    reset_secret_scope(token)
                lost = [r for r in caplog.records if r.name == "tools.registry"]
        finally:
            set_multiplex_active(False)

        assert boot and all(r.levelno == logging.DEBUG and r.exc_info is None for r in boot)
        assert any(r.levelno >= logging.WARNING and r.exc_info for r in lost)

    def test_uncached_probe_at_boot_is_not_reported_as_a_lost_scope(self, caplog):
        """#110635: a ``no_cache_check_fn`` probe (the browser vault gate) skips the cache-scope
        lookup, so on the old branch-derived hint every gateway start under multiplexing logged
        WARNING + traceback for an EXPECTED fail-closed read. Same probe, cached or uncached,
        same DEBUG verdict."""
        import logging

        import tools.registry as reg
        from agent.secret_scope import get_secret, set_multiplex_active

        def probe():
            return bool(get_secret("REGISTRY_LOG_PROBE_TOKEN", ""))

        reg.no_cache_check_fn(probe)
        set_multiplex_active(True)
        try:
            with caplog.at_level(logging.DEBUG, logger="tools.registry"):
                assert reg._check_fn_cached(probe) is False
        finally:
            set_multiplex_active(False)
            reg._NO_CACHE_CHECK_FNS.discard(probe)
        records = [r for r in caplog.records if r.name == "tools.registry"]
        assert records and all(r.levelno == logging.DEBUG and r.exc_info is None for r in records)
