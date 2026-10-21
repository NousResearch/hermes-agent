"""An unbound thread on a multiplexed host resolves no CDP endpoint (#134889 GW-26).

The live ``/browser connect`` endpoint already failed closed for a context with no profile bound.
The *configured* endpoint did not: an unbound thread still resolved the launch profile's
``browser.cdp_url`` from config, so a thread that never carried a profile could reach the launch
profile's browser (a remote-control endpoint).
"""
import pytest

from tools import browser_tool_cdp


class _Origin:
    """The launch profile's config: one ``browser.cdp_url`` every unbound thread could read."""

    def _browser_cfg(self, key, default, cast, label):
        assert key == "cdp_url"
        return "ws://launch.invalid/devtools/browser/LAUNCH"


@pytest.fixture
def multiplex_host(monkeypatch):
    """This process serves several profiles, and this context has no profile bound."""
    from agent import secret_scope
    import hermes_constants
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True, raising=False)
    monkeypatch.setattr(hermes_constants, "get_hermes_home_override", lambda: None, raising=False)
    monkeypatch.setattr(browser_tool_cdp, "_origin", lambda: _Origin(), raising=False)
    monkeypatch.setattr(browser_tool_cdp, "env_cdp_url", lambda: "", raising=False)
    monkeypatch.setattr(browser_tool_cdp, "_live_overrides", {}, raising=False)
    return secret_scope


def test_an_unbound_thread_resolves_no_configured_endpoint(multiplex_host):
    """The launch profile's ``browser.cdp_url`` is not an unbound thread's to connect to."""
    assert browser_tool_cdp._get_cdp_override_raw() == ""


def test_the_live_endpoint_fails_closed_the_same_way(multiplex_host, monkeypatch):
    """The live endpoint keeps its own fail-closed answer (the guard the config path now shares)."""
    monkeypatch.setattr(browser_tool_cdp, "_live_overrides", {"default": "http://127.0.0.1:9444"}, raising=False)
    assert browser_tool_cdp.live_cdp_override() == ""


def test_a_bound_profile_still_resolves_its_own_endpoint(monkeypatch):
    """A profile-bound context on the same host is unaffected."""
    from agent import secret_scope
    import hermes_constants
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True, raising=False)
    monkeypatch.setattr(hermes_constants, "get_hermes_home_override", lambda: "/home/bee", raising=False)
    monkeypatch.setattr(browser_tool_cdp, "_origin", lambda: _Origin(), raising=False)
    monkeypatch.setattr(browser_tool_cdp, "env_cdp_url", lambda: "", raising=False)
    monkeypatch.setattr(browser_tool_cdp, "_live_overrides", {}, raising=False)
    assert browser_tool_cdp._get_cdp_override_raw() == "ws://launch.invalid/devtools/browser/LAUNCH"


def test_a_single_profile_process_is_unchanged(monkeypatch):
    """A process that serves one profile resolves the configured endpoint as before."""
    from agent import secret_scope
    import hermes_constants
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False, raising=False)
    monkeypatch.setattr(hermes_constants, "get_hermes_home_override", lambda: None, raising=False)
    monkeypatch.setattr(browser_tool_cdp, "_origin", lambda: _Origin(), raising=False)
    monkeypatch.setattr(browser_tool_cdp, "env_cdp_url", lambda: "", raising=False)
    monkeypatch.setattr(browser_tool_cdp, "_live_overrides", {}, raising=False)
    assert browser_tool_cdp._get_cdp_override_raw() == "ws://launch.invalid/devtools/browser/LAUNCH"


def test_a_bound_context_keeps_its_own_live_endpoint(monkeypatch):
    """The live endpoint of a bound profile is still served on a multiplexed host."""
    from agent import secret_scope
    import hermes_constants
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True, raising=False)
    monkeypatch.setattr(hermes_constants, "get_hermes_home_override", lambda: "/home/bee", raising=False)
    monkeypatch.setattr(browser_tool_cdp, "_live_overrides", {"default": "http://127.0.0.1:9444"}, raising=False)
    monkeypatch.setattr(hermes_constants, "hermes_home_key", lambda *a, **k: "default", raising=False)
    assert browser_tool_cdp.live_cdp_override() == "http://127.0.0.1:9444"


def test_the_guard_is_a_plain_predicate(monkeypatch):
    """The guard is the same test the live endpoint used, not a second rule."""
    from agent import secret_scope
    import hermes_constants
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True, raising=False)
    monkeypatch.setattr(hermes_constants, "get_hermes_home_override", lambda: None, raising=False)
    assert browser_tool_cdp._unbound_on_multiplex() is True
    monkeypatch.setattr(hermes_constants, "get_hermes_home_override", lambda: "/home/bee", raising=False)
    assert browser_tool_cdp._unbound_on_multiplex() is False
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False, raising=False)
    monkeypatch.setattr(hermes_constants, "get_hermes_home_override", lambda: None, raising=False)
    assert browser_tool_cdp._unbound_on_multiplex() is False
