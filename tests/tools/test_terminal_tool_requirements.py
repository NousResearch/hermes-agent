"""Tests for terminal/file tool availability in local dev environments."""

import importlib
import os

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

    def test_tenki_requires_auth_only_for_plain_sessions(self, monkeypatch, tmp_path):
        original_find_spec = terminal_tool_module.importlib.util.find_spec

        def fake_find_spec(name):
            if name == "tenki":
                return object()
            return original_find_spec(name)

        monkeypatch.setattr(terminal_tool_module.importlib.util, "find_spec", fake_find_spec)
        monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "missing.yaml"))
        monkeypatch.delenv("TENKI_WORKSPACE_ID", raising=False)
        monkeypatch.delenv("TENKI_WORKSPACE", raising=False)
        monkeypatch.setattr(
            terminal_tool_module,
            "_get_env_config",
            lambda: {
                "env_type": "tenki",
                "tenki_workspace_id": "",
            },
        )

        assert terminal_tool_module.check_terminal_requirements() is False

        monkeypatch.setenv("TENKI_AUTH_TOKEN", "tok")
        assert terminal_tool_module.check_terminal_requirements() is True

    def test_tool_definition_cache_isolates_profiles_and_rechecks_credentials(
        self,
        monkeypatch,
        tmp_path,
    ):
        import model_tools
        import tools.registry as registry_module
        from agent.secret_scope import is_multiplex_active, set_multiplex_active
        from gateway.run import _profile_runtime_scope
        from hermes_cli import config as hermes_config

        home_a = tmp_path / "a"
        home_b = tmp_path / "b"
        config_text = "terminal:\n  backend: tenki\n  cwd: /home/tenki\n"
        for home in (home_a, home_b):
            home.mkdir()
            (home / "config.yaml").write_text(config_text, encoding="utf-8")
        (home_a / ".env").write_text(
            "TENKI_AUTH_TOKEN=token-a\n",
            encoding="utf-8",
        )
        (home_b / ".env").write_text("", encoding="utf-8")
        # Same config contents and fingerprint make profile identity, not mtime
        # luck, the distinguishing cache-key invariant.
        stat_a = (home_a / "config.yaml").stat()
        os.utime(
            home_b / "config.yaml",
            ns=(stat_a.st_atime_ns, stat_a.st_mtime_ns),
        )

        monkeypatch.setenv("TERMINAL_ENV", "local")
        monkeypatch.setenv("TENKI_AUTH_TOKEN", "process-token-must-not-leak")
        clock = {"now": 1000.0}
        monkeypatch.setattr(model_tools.time, "monotonic", lambda: clock["now"])
        monkeypatch.setattr(
            registry_module.time,
            "monotonic",
            lambda: clock["now"],
        )
        previous = is_multiplex_active()
        set_multiplex_active(True)
        model_tools._clear_tool_defs_cache()
        registry_module.invalidate_check_fn_cache()
        hermes_config._LOAD_CONFIG_CACHE.clear()
        hermes_config._RAW_CONFIG_CACHE.clear()
        try:
            with _profile_runtime_scope(home_a):
                names_a = {
                    tool["function"]["name"]
                    for tool in get_tool_definitions(
                        enabled_toolsets=["terminal"],
                        quiet_mode=True,
                    )
                }
            with _profile_runtime_scope(home_b):
                names_b = {
                    tool["function"]["name"]
                    for tool in get_tool_definitions(
                        enabled_toolsets=["terminal"],
                        quiet_mode=True,
                    )
                }

            assert "terminal" in names_a
            assert "terminal" not in names_b
            assert len(model_tools._tool_defs_cache) == 2

            (home_b / ".env").write_text(
                "TENKI_AUTH_TOKEN=token-b\n",
                encoding="utf-8",
            )
            clock["now"] += registry_module._CHECK_FN_TTL_SECONDS + 1
            with _profile_runtime_scope(home_b):
                names_b_after_rotation = {
                    tool["function"]["name"]
                    for tool in get_tool_definitions(
                        enabled_toolsets=["terminal"],
                        quiet_mode=True,
                    )
                }
            assert "terminal" in names_b_after_rotation
        finally:
            set_multiplex_active(previous)
            model_tools._clear_tool_defs_cache()
            registry_module.invalidate_check_fn_cache()


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


    def test_cache_is_scoped_to_profile_secret_context(self, tmp_path):
        """A success under profile A cannot make profile B's unavailable
        backend appear ready through the shared check_fn TTL cache."""
        import tools.registry as reg
        from agent.secret_scope import reset_secret_scope, set_secret_scope
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        home_a = tmp_path / "a"
        home_b = tmp_path / "b"
        home_a.mkdir()
        home_b.mkdir()

        def profile_probe():
            from hermes_constants import get_hermes_home

            return get_hermes_home() == home_a

        home_token = set_hermes_home_override(home_a)
        secret_token = set_secret_scope({"TENKI_AUTH_TOKEN": "token-a"})
        try:
            assert reg._check_fn_cached(profile_probe) is True
        finally:
            reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)

        home_token = set_hermes_home_override(home_b)
        secret_token = set_secret_scope({})
        try:
            assert reg._check_fn_cached(profile_probe) is False
        finally:
            reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)

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
    def test_terminal_and_execute_code_tools_resolve_for_vercel_sandbox(self, monkeypatch):
        monkeypatch.setenv("VERCEL_OIDC_TOKEN", "oidc-token")
        monkeypatch.setattr(
            terminal_tool_module,
            "_get_env_config",
            lambda: {"env_type": "vercel_sandbox", "container_disk": 51200},
        )
        monkeypatch.setattr(
            terminal_tool_module.importlib.util,
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
            terminal_tool_module.importlib.util,
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
            terminal_tool_module.importlib.util,
            "find_spec",
            lambda _name: object(),
        )
        tools = get_tool_definitions(enabled_toolsets=["terminal", "code_execution"], quiet_mode=True)
        names = {tool["function"]["name"] for tool in tools}

        assert "terminal" not in names
        assert "execute_code" not in names
