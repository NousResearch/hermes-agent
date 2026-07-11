"""Tests for named browser profiles (account / cookie-jar isolation).

``browser.profiles`` maps a profile name to a CDP endpoint, each typically a
separate persistent Chrome with its own user-data-dir (cookies, logins). This
lets the agent operate distinct identities — its own browser vs. the user's
authenticated accounts — without cross-contaminating sessions.

``browser_navigate(profile="<name>")`` binds the navigation, and every
follow-up snapshot/click/type on the same task, to that profile's endpoint via
a composite session key ``{task_id}::profile:{name}`` — mirroring the
``::local`` hybrid-routing precedent.

These tests cover the resolution layer (profiles map read, CDP resolution,
legacy fallback, unknown-profile error), session-key composition/parsing,
session creation routing, the navigate-level validation gate, and cleanup
fan-out. They assert behavior contracts (how the pieces relate), not frozen
snapshots.
"""
from unittest.mock import Mock

import pytest

import tools.browser_tool as browser_tool


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Clear module-level caches so each test starts clean."""
    monkeypatch.setattr(browser_tool, "_active_sessions", {})
    monkeypatch.setattr(browser_tool, "_last_active_session_key", {})
    monkeypatch.setattr(browser_tool, "_start_browser_cleanup_thread", lambda: None)
    monkeypatch.setattr(browser_tool, "_update_session_activity", lambda t: None)
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    # No global override / cloud provider unless a test sets one.
    monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: "")
    monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: None)
    # Same-profile concurrency registries — reset so tests don't leak tabs/locks.
    monkeypatch.setattr(browser_tool, "_endpoint_locks", {})
    monkeypatch.setattr(browser_tool, "_session_owned_tab", {})
    monkeypatch.setattr(browser_tool, "_session_endpoint", {})


def _set_profiles(monkeypatch, mapping):
    """Point _get_browser_profiles at a fixed mapping, identity CDP resolve."""
    monkeypatch.setattr(browser_tool, "_get_browser_profiles", lambda: dict(mapping))
    monkeypatch.setattr(browser_tool, "_resolve_cdp_override", lambda u: u)


class TestProfilesMapRead:
    """_get_browser_profiles reads browser.profiles, coercing/filtering."""

    def test_reads_mapping_from_config(self, monkeypatch):
        def fake_cfg():
            return {"browser": {"profiles": {
                "default": "http://localhost:9224",
                "work": "http://localhost:9225",
            }}}

        import hermes_cli.config as cfg
        monkeypatch.setattr(cfg, "read_raw_config", fake_cfg)
        out = browser_tool._get_browser_profiles()
        assert out == {
            "default": "http://localhost:9224",
            "work": "http://localhost:9225",
        }

    def test_missing_profiles_returns_empty(self, monkeypatch):
        import hermes_cli.config as cfg
        monkeypatch.setattr(cfg, "read_raw_config", lambda: {"browser": {}})
        assert browser_tool._get_browser_profiles() == {}

    def test_drops_empty_and_nonstring_values(self, monkeypatch):
        import hermes_cli.config as cfg
        monkeypatch.setattr(cfg, "read_raw_config", lambda: {"browser": {"profiles": {
            "good": "http://localhost:9225",
            "blank": "   ",
            "nullish": None,
            "": "http://localhost:9999",
        }}})
        assert browser_tool._get_browser_profiles() == {"good": "http://localhost:9225"}


class TestProfileResolution:
    """_resolve_profile_cdp maps a name to a concrete CDP URL or raises."""

    def test_named_profile_resolves(self, monkeypatch):
        _set_profiles(monkeypatch, {"work": "http://localhost:9225"})
        assert browser_tool._resolve_profile_cdp("work") == "http://localhost:9225"

    def test_default_falls_back_to_legacy_cdp_url(self, monkeypatch):
        """default profile with no profiles entry uses legacy browser.cdp_url."""
        monkeypatch.setattr(browser_tool, "_get_browser_profiles", lambda: {})
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: "ws://localhost:9224/x")
        assert browser_tool._resolve_profile_cdp("default") == "ws://localhost:9224/x"

    def test_default_explicit_entry_wins_over_legacy(self, monkeypatch):
        _set_profiles(monkeypatch, {"default": "http://localhost:9230"})
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: "ws://localhost:9224/x")
        assert browser_tool._resolve_profile_cdp("default") == "http://localhost:9230"

    def test_default_with_nothing_configured_returns_empty(self, monkeypatch):
        monkeypatch.setattr(browser_tool, "_get_browser_profiles", lambda: {})
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: "")
        assert browser_tool._resolve_profile_cdp("default") == ""

    def test_unknown_profile_raises_no_silent_fallback(self, monkeypatch):
        """Crossing an account boundary by accident is the failure we prevent."""
        _set_profiles(monkeypatch, {"work": "http://localhost:9225"})
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: "ws://localhost:9224/x")
        with pytest.raises(ValueError) as ei:
            browser_tool._resolve_profile_cdp("nonexistent")
        # Error names the bad profile and lists configured ones.
        assert "nonexistent" in str(ei.value)
        assert "work" in str(ei.value)


class TestSessionKeyComposition:
    """Composite session-key build/parse round-trips and isolation invariants."""

    def test_compose_and_parse_round_trip(self):
        key = browser_tool._compose_profile_session_key("task-1", "work")
        assert key == "task-1::profile:work"
        assert browser_tool._profile_from_session_key(key) == "work"

    def test_bare_key_has_no_profile(self):
        assert browser_tool._profile_from_session_key("task-1") is None

    def test_local_sidecar_key_has_no_profile(self):
        assert browser_tool._profile_from_session_key("task-1::local") is None

    def test_profile_key_is_not_a_local_sidecar(self):
        """Profile keys must not be mistaken for ::local sidecars."""
        key = browser_tool._compose_profile_session_key("task-1", "work")
        assert not browser_tool._is_local_sidecar_key(key)

    def test_distinct_profiles_yield_distinct_keys(self):
        a = browser_tool._compose_profile_session_key("t", "alice")
        b = browser_tool._compose_profile_session_key("t", "bob")
        assert a != b  # isolation: two accounts never collide on one key


class TestSessionCreationRouting:
    """_get_session_info binds a profile key to that profile's CDP endpoint."""

    def test_profile_key_creates_cdp_session_for_its_endpoint(self, monkeypatch):
        _set_profiles(monkeypatch, {"work": "ws://localhost:9225/devtools/browser/abc"})
        monkeypatch.setattr(browser_tool, "_ensure_cdp_supervisor", lambda t: None)

        key = browser_tool._compose_profile_session_key("default", "work")
        session = browser_tool._get_session_info(key)

        assert session["cdp_url"] == "ws://localhost:9225/devtools/browser/abc"
        assert session["features"].get("cdp_override") is True

    def test_two_profiles_get_independent_sessions(self, monkeypatch):
        _set_profiles(monkeypatch, {
            "alice": "ws://localhost:9225/a",
            "bob": "ws://localhost:9226/b",
        })
        monkeypatch.setattr(browser_tool, "_ensure_cdp_supervisor", lambda t: None)

        ka = browser_tool._compose_profile_session_key("default", "alice")
        kb = browser_tool._compose_profile_session_key("default", "bob")
        sa = browser_tool._get_session_info(ka)
        sb = browser_tool._get_session_info(kb)

        # Distinct endpoints — no cookie-jar cross-contamination.
        assert sa["cdp_url"] != sb["cdp_url"]
        assert browser_tool._active_sessions[ka]["cdp_url"] == "ws://localhost:9225/a"
        assert browser_tool._active_sessions[kb]["cdp_url"] == "ws://localhost:9226/b"

    def test_profile_session_skips_cloud_provider(self, monkeypatch):
        """A profile pins an explicit endpoint — cloud provider is bypassed."""
        provider = Mock()
        provider.create_session.return_value = {"session_name": "x", "cdp_url": "wss://cloud/ws"}
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: provider)
        _set_profiles(monkeypatch, {"work": "ws://localhost:9225/w"})
        monkeypatch.setattr(browser_tool, "_ensure_cdp_supervisor", lambda t: None)

        key = browser_tool._compose_profile_session_key("default", "work")
        session = browser_tool._get_session_info(key)

        assert provider.create_session.call_count == 0
        assert session["cdp_url"] == "ws://localhost:9225/w"


class TestNavigateValidationGate:
    """browser_navigate(profile=...) validates before doing any work."""

    def test_unknown_profile_returns_error_not_raise(self, monkeypatch):
        _set_profiles(monkeypatch, {"work": "ws://localhost:9225/w"})
        out = browser_tool.browser_navigate("https://example.com", profile="ghost")
        import json
        data = json.loads(out)
        assert data["success"] is False
        assert "ghost" in data["error"]

    def test_profile_resolving_to_empty_returns_error(self, monkeypatch):
        """default profile with nothing configured → actionable error, no nav."""
        monkeypatch.setattr(browser_tool, "_get_browser_profiles", lambda: {})
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: "")
        out = browser_tool.browser_navigate("https://example.com", profile="default")
        import json
        data = json.loads(out)
        assert data["success"] is False
        assert "default" in data["error"]


class TestBackwardsCompat:
    """No profile arg → byte-identical behavior to pre-profiles."""

    def test_no_profile_does_not_compose_profile_key(self, monkeypatch):
        """A nav without profile keeps the bare task_id (legacy behavior)."""
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: None)
        key = browser_tool._navigation_session_key("default", "https://example.com")
        assert browser_tool._profile_from_session_key(key) is None

    def test_legacy_cdp_url_still_resolves_as_default(self, monkeypatch):
        """Configs with only browser.cdp_url (no profiles) still work."""
        monkeypatch.setattr(browser_tool, "_get_browser_profiles", lambda: {})
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: "ws://localhost:9224/legacy")
        assert browser_tool._resolve_profile_cdp("default") == "ws://localhost:9224/legacy"


class TestCleanupFanOut:
    """cleanup_browser reaps profile sessions correctly."""

    def test_bare_task_cleanup_reaps_profile_sessions(self, monkeypatch):
        reaped = []
        monkeypatch.setattr(browser_tool, "_cleanup_single_browser_session", lambda k: reaped.append(k))
        monkeypatch.setattr(browser_tool, "_active_sessions", {
            "default": {"session_name": "primary"},
            "default::profile:work": {"session_name": "work_sess"},
            "default::profile:home": {"session_name": "home_sess"},
        })
        monkeypatch.setattr(browser_tool, "_last_active_session_key",
                            {"default": "default::profile:work"})

        browser_tool.cleanup_browser("default")

        assert set(reaped) == {"default", "default::profile:work", "default::profile:home"}
        assert "default" not in browser_tool._last_active_session_key

    def test_explicit_profile_key_reaps_only_itself(self, monkeypatch):
        reaped = []
        monkeypatch.setattr(browser_tool, "_cleanup_single_browser_session", lambda k: reaped.append(k))
        monkeypatch.setattr(browser_tool, "_active_sessions", {
            "default": {"session_name": "primary"},
            "default::profile:work": {"session_name": "work_sess"},
        })
        monkeypatch.setattr(browser_tool, "_last_active_session_key",
                            {"default": "default::profile:work"})

        browser_tool.cleanup_browser("default::profile:work")

        assert reaped == ["default::profile:work"]
        # The last-active pointer named the exact session we just reaped, so it
        # must be dropped — leaving it would let a follow-up click/snapshot
        # resurrect a cleaned session on about:blank. A pointer at a *different*
        # (still-live) session would be preserved; see below.
        assert "default" not in browser_tool._last_active_session_key

    def test_explicit_profile_key_preserves_unrelated_pointer(self, monkeypatch):
        reaped = []
        monkeypatch.setattr(browser_tool, "_cleanup_single_browser_session", lambda k: reaped.append(k))
        monkeypatch.setattr(browser_tool, "_active_sessions", {
            "default": {"session_name": "primary"},
            "default::profile:work": {"session_name": "work_sess"},
        })
        # Pointer names the still-live primary session, not the one being reaped.
        monkeypatch.setattr(browser_tool, "_last_active_session_key",
                            {"default": "default"})

        browser_tool.cleanup_browser("default::profile:work")

        assert reaped == ["default::profile:work"]
        # Reaping one profile session must not disturb a pointer at a live session.
        assert browser_tool._last_active_session_key.get("default") == "default"


class TestEndpointLock:
    """Per-endpoint serialization: one lock per CDP url, shared across sessions."""

    def test_same_endpoint_shares_one_lock(self):
        l1 = browser_tool._endpoint_lock_for("http://localhost:9225")
        l2 = browser_tool._endpoint_lock_for("http://localhost:9225")
        assert l1 is l2  # same Chrome -> same lock -> commands serialize

    def test_distinct_endpoints_get_distinct_locks(self):
        l_work = browser_tool._endpoint_lock_for("http://localhost:9225")
        l_home = browser_tool._endpoint_lock_for("http://localhost:9226")
        assert l_work is not l_home  # different Chrome -> parallel, never block


class TestOwnedTab:
    """Each profile session acquires and caches its own labeled tab."""

    def test_first_call_opens_tab_second_call_caches(self, monkeypatch):
        calls = []

        def fake_raw(session_name, cdp_url, argv, timeout=15):
            calls.append(argv)
            return {"success": True, "data": {"tabId": "t7"}}

        monkeypatch.setattr(browser_tool, "_run_raw_agent_browser", fake_raw)

        ref1 = browser_tool._ensure_owned_tab("s::profile:work", "cdp_abc", "http://x:9225")
        ref2 = browser_tool._ensure_owned_tab("s::profile:work", "cdp_abc", "http://x:9225")

        assert ref1 == "t7" and ref2 == "t7"
        # Only ONE `tab new` — the second call is served from cache.
        tab_new_calls = [c for c in calls if c[:2] == ["tab", "new"]]
        assert len(tab_new_calls) == 1
        assert browser_tool._session_owned_tab["s::profile:work"] == "t7"
        assert browser_tool._session_endpoint["s::profile:work"] == "http://x:9225"

    def test_acquire_failure_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            browser_tool, "_run_raw_agent_browser",
            lambda *a, **k: {"success": False, "error": "endpoint down"},
        )
        ref = browser_tool._ensure_owned_tab("s::profile:work", "cdp_abc", "http://x:9225")
        assert ref is None
        assert "s::profile:work" not in browser_tool._session_owned_tab

    def test_release_closes_and_forgets_tab(self, monkeypatch):
        closed = []
        monkeypatch.setattr(
            browser_tool, "_run_raw_agent_browser",
            lambda sn, url, argv, timeout=15: closed.append(argv) or {"success": True},
        )
        monkeypatch.setattr(browser_tool, "_active_sessions",
                            {"s::profile:work": {"session_name": "cdp_abc"}})
        browser_tool._session_owned_tab["s::profile:work"] = "t7"
        browser_tool._session_endpoint["s::profile:work"] = "http://x:9225"

        browser_tool._release_owned_tab("s::profile:work")

        assert ["tab", "close", "t7"] in closed
        assert "s::profile:work" not in browser_tool._session_owned_tab
        assert "s::profile:work" not in browser_tool._session_endpoint


class TestSameProfileCommandRouting:
    """_run_browser_command activates the owned tab (under lock) for profile
    sessions, and skips all of it for non-profile sessions."""

    def _stub_command_env(self, monkeypatch):
        """Stub out everything _run_browser_command needs except the profile path."""
        monkeypatch.setattr(browser_tool, "_safe_command_timeout", lambda: 5)
        monkeypatch.setattr(browser_tool, "_find_agent_browser", lambda: "agent-browser")
        monkeypatch.setattr(browser_tool, "_requires_real_termux_browser_install", lambda c: False)
        monkeypatch.setattr(browser_tool, "_is_local_mode", lambda: False)
        monkeypatch.setattr(browser_tool, "_get_browser_engine", lambda: "auto")
        monkeypatch.setattr(browser_tool, "_build_browser_env", lambda: {})
        monkeypatch.setattr(browser_tool, "_merge_browser_path", lambda p: p)
        monkeypatch.setattr(browser_tool, "_socket_safe_tmpdir", lambda: "/tmp")
        monkeypatch.setattr(browser_tool, "_write_owner_pid", lambda d, n: None)
        monkeypatch.setattr(browser_tool, "_needs_chromium_sandbox_bypass", lambda: False)
        monkeypatch.setattr(browser_tool, "_lightpanda_fallback_reason", lambda *a: None)
        from tools import interrupt
        monkeypatch.setattr(interrupt, "is_interrupted", lambda: False)

    def test_profile_command_activates_owned_tab_before_running(self, monkeypatch):
        self._stub_command_env(monkeypatch)
        order = []

        # Session bound to a profile endpoint.
        monkeypatch.setattr(browser_tool, "_get_session_info",
                            lambda tid: {"session_name": "cdp_abc",
                                         "cdp_url": "http://x:9225"})

        def fake_raw(session_name, cdp_url, argv, timeout=15):
            order.append(("raw", tuple(argv)))
            return {"success": True, "data": {"tabId": "t7"}}
        monkeypatch.setattr(browser_tool, "_run_raw_agent_browser", fake_raw)

        # Intercept the real subprocess spawn — record when the MAIN command runs.
        class FakeProc:
            returncode = 0
            def wait(self, timeout=None): order.append(("main-command", "snapshot"))
        monkeypatch.setattr(browser_tool.subprocess, "Popen", lambda *a, **k: FakeProc())
        # File I/O around the spawn: make reads return a valid JSON result.
        monkeypatch.setattr(browser_tool.os, "open", lambda *a, **k: 0)
        monkeypatch.setattr(browser_tool.os, "close", lambda fd: None)
        monkeypatch.setattr(browser_tool.os, "makedirs", lambda *a, **k: None)
        monkeypatch.setattr(browser_tool.os, "unlink", lambda p: None)
        import builtins, io
        real_open = builtins.open
        def fake_open(path, *a, **k):
            if isinstance(path, str) and "_stdout_" in path:
                return io.StringIO('{"success": true, "data": {}}')
            if isinstance(path, str) and "_stderr_" in path:
                return io.StringIO("")
            return real_open(path, *a, **k)
        monkeypatch.setattr(builtins, "open", fake_open)

        browser_tool._run_browser_command("s::profile:work", "snapshot")

        # The owned-tab activate (a raw `tab t7`) must appear BEFORE the main command.
        activate_idx = next(i for i, e in enumerate(order)
                            if e[0] == "raw" and e[1] == ("tab", "t7"))
        main_idx = next(i for i, e in enumerate(order) if e[0] == "main-command")
        assert activate_idx < main_idx
        # And the lock must have been released (available to re-acquire).
        lock = browser_tool._endpoint_lock_for("http://x:9225")
        assert lock.acquire(blocking=False)
        lock.release()

    def test_non_profile_command_skips_tab_machinery(self, monkeypatch):
        self._stub_command_env(monkeypatch)
        raw_calls = []
        monkeypatch.setattr(browser_tool, "_run_raw_agent_browser",
                            lambda *a, **k: raw_calls.append(a) or {"success": True})
        # Bare (non-profile) session on the global override endpoint.
        monkeypatch.setattr(browser_tool, "_get_session_info",
                            lambda tid: {"session_name": "cdp_abc",
                                         "cdp_url": "http://x:9224"})

        class FakeProc:
            returncode = 0
            def wait(self, timeout=None): pass
        monkeypatch.setattr(browser_tool.subprocess, "Popen", lambda *a, **k: FakeProc())
        monkeypatch.setattr(browser_tool.os, "open", lambda *a, **k: 0)
        monkeypatch.setattr(browser_tool.os, "close", lambda fd: None)
        monkeypatch.setattr(browser_tool.os, "makedirs", lambda *a, **k: None)
        monkeypatch.setattr(browser_tool.os, "unlink", lambda p: None)
        import builtins, io
        real_open = builtins.open
        def fake_open(path, *a, **k):
            if isinstance(path, str) and "_stdout_" in path:
                return io.StringIO('{"success": true, "data": {}}')
            if isinstance(path, str) and "_stderr_" in path:
                return io.StringIO("")
            return real_open(path, *a, **k)
        monkeypatch.setattr(builtins, "open", fake_open)

        browser_tool._run_browser_command("bare-task", "snapshot")

        # No tab machinery fired for a non-profile session.
        assert raw_calls == []
        assert browser_tool._session_owned_tab == {}

