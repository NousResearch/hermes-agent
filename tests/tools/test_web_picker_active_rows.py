"""Picker row highlighting for per-capability web backends.

``web_search``/``web_extract`` resolve ``web.search_backend`` / ``web.extract_backend``
first and only then fall back to the shared ``web.backend`` (tools/web_tools.py
``_get_search_backend`` / ``_get_extract_backend``).  The picker row's ``is_active``
must follow the same precedence, otherwise a vendor that is genuinely serving one
capability is displayed as inactive (#132511).
"""

import types

import pytest

from hermes_cli.tools_config_providers import _is_provider_active


@pytest.fixture(autouse=True)
def _no_web_env(monkeypatch):
    """Keep tier auto-detection out of the picture: no web credentials present."""
    for var in ("EXA_API_KEY", "PARALLEL_API_KEY", "TAVILY_API_KEY", "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("agent.web_search_provider.get_provider_env", lambda name: "", raising=True)


def _row(backend: str, **extra) -> dict:
    return {"name": backend, "web_backend": backend, "env_vars": [], **extra}


@pytest.mark.parametrize("web,active,inactive", [
    # The reported config: search on firecrawl, extract on tavily — both rows serve something.
    ({"search_backend": "firecrawl", "extract_backend": "tavily", "backend": "firecrawl"}, ["firecrawl", "tavily"], ["ddgs"]),
    # An override shadows the shared key for its own capability only: shadowed twice, it serves nothing.
    ({"search_backend": "searxng", "extract_backend": "tavily", "backend": "firecrawl"}, ["searxng", "tavily"], ["firecrawl"]),
    # No overrides (or blank ones): the shared key decides, as before.
    ({"search_backend": "", "extract_backend": None, "backend": "searxng"}, ["searxng"], ["firecrawl"]),
    # Dispatch lower-cases and strips the configured name; highlighting must agree.
    ({"search_backend": " FireCrawl ", "backend": "tavily"}, ["firecrawl", "tavily"], []),
])
def test_row_active_iff_it_serves_a_capability(web, active, inactive):
    config = {"web": web}
    assert [b for b in active if not _is_provider_active(_row(b), config)] == []
    assert [b for b in inactive if _is_provider_active(_row(b), config)] == []
    # Tiered rows keep the free/paid discriminator on the capability keys too.
    tiered = {"web": {"search_backend": "parallel", "provider_tier": {"parallel": "paid"}}}
    assert _is_provider_active(_row("parallel", web_tier="paid"), tiered) is True
    assert _is_provider_active(_row("parallel", web_tier="free"), tiered) is False


def test_managed_row_active_for_a_nous_capability_pin(monkeypatch):
    """``extract_backend: nous`` (Desktop "Use for Extract" on Nous Subscription): the managed row serves
    extract and must be active alongside the BYOK row serving search; a vendor-only config leaves it off."""
    import hermes_cli.tools_config as tools_config

    monkeypatch.setattr(tools_config, "get_nous_subscription_features", lambda *a, **k: types.SimpleNamespace(
        features={"web": types.SimpleNamespace(managed_by_nous=True)}))
    nous = _row("firecrawl", managed_nous_feature="web")
    split = {"web": {"search_backend": "firecrawl", "extract_backend": "nous"}}
    assert _is_provider_active(nous, split) is True
    assert _is_provider_active(_row("firecrawl"), split) is True
    assert _is_provider_active(nous, {"web": {"search_backend": "firecrawl", "extract_backend": "tavily"}}) is False


def _web_rows():
    from hermes_cli.tools_config import TOOL_CATEGORIES, _visible_providers

    return {p["name"]: p for p in _visible_providers(TOOL_CATEGORIES["web"], {"web": {}})}


@pytest.mark.parametrize("env,active", [
    ({"FIRECRAWL_API_KEY": "fc-x"}, {"Firecrawl"}),
    ({"FIRECRAWL_API_URL": "http://localhost:3002"}, {"Firecrawl Self-Hosted"}),
    ({}, {"Firecrawl"}),  # explicit selection, no credentials: anonymous cloud
])
def test_firecrawl_rows_split_by_the_credential_set(monkeypatch, env, active):
    """Cloud and Self-Hosted rows share ``web_backend: firecrawl``; only the one whose env var is set
    is the configured provider (#112022), so the picker cursor does not land on an unconfigured row."""
    import hermes_cli.tools_config as tools_config

    monkeypatch.setattr(tools_config, "get_env_value", env.get)
    rows = _web_rows()
    config = {"web": {"backend": "firecrawl"}}
    lit = {name for name in ("Firecrawl", "Firecrawl Self-Hosted") if _is_provider_active(rows[name], config)}
    assert lit == active


def test_keyless_row_readiness_follows_registry_availability(monkeypatch):
    """No-key rows used to read "ready" vacuously (#132526): OpenAI Native needs an openai-codex login,
    while free-tier keyless rows really run on the public ring."""
    import plugins.web.openai_native.provider as native
    from hermes_cli.tools_config_providers import provider_readiness_status

    rows = _web_rows()
    native_row = next(p for name, p in rows.items() if p.get("web_backend") == "openai-native")
    free_rows = [p for p in rows.values() if p.get("web_tier") == "free"]
    assert free_rows
    monkeypatch.setattr(native, "has_codex_credentials", lambda: False)
    assert provider_readiness_status(native_row, {"web": {}}) == "needs_auth"
    assert {provider_readiness_status(p, {"web": {}}) for p in free_rows} == {"ready"}
    monkeypatch.setattr(native, "has_codex_credentials", lambda: True)
    assert provider_readiness_status(native_row, {"web": {}}) == "ready"
