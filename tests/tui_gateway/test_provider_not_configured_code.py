"""A session that cannot start for lack of a provider says so in a machine-readable way.

The Desktop routes this failure to its provider-setup flow — the one thing that fixes it — and had
to recognise it by matching the sentence. The sentences moved (``agent_init`` says "No LLM provider
configured", ``missing_provider_credentials_message`` says "Provider 'x' is set in config.yaml
but …"), the matcher did not, and a blank install got a dead-end toast instead of onboarding.
"""

import pytest

from agent.auxiliary_unavailable import ProviderNotConfiguredError
from tui_gateway import server


@pytest.fixture
def emitted(monkeypatch):
    events = []
    monkeypatch.setattr(server, "_emit", lambda kind, sid, payload: events.append((kind, payload)))
    return events


def _fail_build(monkeypatch, exc):
    """Run the deferred agent build against a registered session, failing at ``_make_agent``."""
    import threading

    def _raise(*_a, **_kw):
        raise exc

    monkeypatch.setattr(server, "_make_agent", _raise)
    monkeypatch.setattr(server, "_deferred_build_agent_kwargs", lambda *a, **kw: {})
    monkeypatch.setattr(server, "_await_resume_history", lambda *a, **kw: True)
    monkeypatch.setattr(server, "_set_session_context", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_clear_session_context", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_session_cwd", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_bind_build_profile_scopes", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_finish_agent_build", lambda *a, **kw: None)

    session = {"session_key": "key", "agent_ready": threading.Event()}
    monkeypatch.setitem(server._sessions, "sid", session)
    return session


def _run_build(session):
    server._start_agent_build("sid", session)
    session["_agent_build_thread"].join(timeout=30)


def test_provider_not_configured_is_named_in_the_error_event(monkeypatch, emitted):
    session = _fail_build(monkeypatch, ProviderNotConfiguredError(
        "No LLM provider configured. Run `hermes model` to select a provider, or run `hermes setup` "
        "for first-time configuration."))

    _run_build(session)

    kind, payload = next((k, p) for k, p in emitted if k == "error")
    assert payload["code"] == "provider_not_configured"
    assert "No LLM provider configured" in payload["message"]  # the sentence still reaches the reader


def test_an_unrelated_build_failure_carries_no_code(monkeypatch, emitted):
    session = _fail_build(monkeypatch, RuntimeError("the session database is locked"))

    _run_build(session)

    _, payload = next((k, p) for k, p in emitted if k == "error")
    assert "code" not in payload


class TestAgentInitRaisesTheNamedError:
    """Both "cannot serve this session" paths in agent_init raise the type, not a bare RuntimeError."""

    def test_no_provider_configured_at_all(self, monkeypatch, tmp_path):
        from types import SimpleNamespace

        from agent.agent_init import _routed_client_kwargs

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Nothing resolves and no fallback chain: the end of the ladder.
        monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", lambda *a, **kw: (None, None))
        agent = SimpleNamespace(provider="auto", model="m", base_url=None, api_key=None,
                                _fallback_activated=False, _explicit_provider="")

        with pytest.raises(ProviderNotConfiguredError, match="No LLM provider configured"):
            _routed_client_kwargs(agent, None, 60)

    def test_explicit_provider_without_credentials(self, monkeypatch, tmp_path):
        from types import SimpleNamespace

        from agent.agent_init import _routed_client_kwargs

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        agent = SimpleNamespace(provider="minimax-oauth", model="m", base_url=None, api_key=None,
                                _fallback_activated=False, _explicit_provider="minimax-oauth")

        with pytest.raises(ProviderNotConfiguredError, match=r"hermes auth add minimax-oauth"):
            _routed_client_kwargs(agent, None, 60)

    def test_the_type_is_a_runtime_error(self):
        """Existing `except RuntimeError` handlers keep catching it."""
        assert issubclass(ProviderNotConfiguredError, RuntimeError)
