"""The openai-native marker provider is never picked by the registry's automatic walks (#135684).

``has_codex_credentials()`` probes the chat OAuth grant, not a search credential: on a default
install with Codex signed in and no web-search key it made openai-native the single eligible
provider, and the Codex transport then declared the hosted ``{"type": "web_search"}`` tool —
which the Codex backend fails with ``server_error`` on every searching turn. Automatic
selection must fall through (keeping the working client-side ``web_search`` tool), while an
explicit ``web.search_backend: openai-native`` still selects it.
"""

import agent.web_search_registry as wsr
import plugins.web.openai_native.provider as native


def _sole_native_registry(monkeypatch):
    """Registry snapshot where openai-native is the only registered, available provider
    (availability forced True — the environment's auth store is not consulted)."""
    provider = native.OpenAINativeWebSearchProvider()
    monkeypatch.setattr(wsr._registry, "merged", lambda *a, **k: {"openai-native": provider})
    monkeypatch.setattr(native.OpenAINativeWebSearchProvider, "is_available", lambda self: True)
    return provider


def test_sole_available_native_is_not_auto_selected(monkeypatch):
    provider = _sole_native_registry(monkeypatch)
    monkeypatch.setattr(wsr, "_keyless_tier_enabled", lambda: False)

    assert wsr._resolve(None, capability="search") is not provider


def test_explicit_native_config_still_selects_it(monkeypatch):
    provider = _sole_native_registry(monkeypatch)

    assert wsr._resolve("openai-native", capability="search") is provider


def test_transport_keeps_client_tool_when_native_not_auto_selected(monkeypatch):
    """End-to-end contract of #135684: with no explicit backend the Codex transport must not
    swap the client ``web_search`` function for the hosted built-in."""
    from agent.transports import codex as codex_transport
    from tools import web_tools

    _sole_native_registry(monkeypatch)
    monkeypatch.setattr(wsr, "_keyless_tier_enabled", lambda: False)
    monkeypatch.setattr(web_tools, "_get_search_backend", lambda: "")

    assert codex_transport._openai_prefers_native_web_search() is False

    def _tools():
        return [{"type": "function", "name": "web_search", "parameters": {"type": "object"}}]

    swapped, aliases = codex_transport._alias_wire_tools(_tools(), {}, is_xai_responses=False, is_codex_backend=True)

    assert swapped == _tools()
    assert aliases == {}
