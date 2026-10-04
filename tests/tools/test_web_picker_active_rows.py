"""Picker row highlighting for per-capability web backends.

``web_search``/``web_extract`` resolve ``web.search_backend`` / ``web.extract_backend``
first and only then fall back to the shared ``web.backend`` (tools/web_tools.py
``_get_search_backend`` / ``_get_extract_backend``).  The picker row's ``is_active``
must follow the same precedence, otherwise a vendor that is genuinely serving one
capability is displayed as inactive.
"""

import pytest

from hermes_cli.tools_config_providers import _is_provider_active


@pytest.fixture(autouse=True)
def _no_web_env(monkeypatch):
    """Keep tier auto-detection out of the picture: no web credentials present."""
    for var in (
        "EXA_API_KEY",
        "PARALLEL_API_KEY",
        "TAVILY_API_KEY",
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        "agent.web_search_provider.get_provider_env", lambda name: "", raising=True
    )


def _row(backend: str, **extra) -> dict:
    return {"name": backend, "web_backend": backend, "env_vars": [], **extra}


class TestSplitCapabilityRowsAreActive:
    def test_extract_only_backend_is_active(self):
        """The reported config: search on firecrawl, extract on tavily."""
        config = {
            "web": {
                "search_backend": "firecrawl",
                "extract_backend": "tavily",
                "backend": "firecrawl",
            }
        }
        assert _is_provider_active(_row("tavily"), config) is True

    def test_search_only_backend_is_active(self):
        config = {
            "web": {
                "search_backend": "searxng",
                "extract_backend": "firecrawl",
                "backend": "firecrawl",
            }
        }
        assert _is_provider_active(_row("searxng"), config) is True

    def test_shared_key_fallback_still_highlights(self):
        """Neither per-capability key set → shared ``web.backend`` decides, as before."""
        config = {"web": {"backend": "tavily"}}
        assert _is_provider_active(_row("tavily"), config) is True
        assert _is_provider_active(_row("firecrawl"), config) is False

    def test_unrelated_backend_stays_inactive(self):
        config = {
            "web": {
                "search_backend": "firecrawl",
                "extract_backend": "tavily",
                "backend": "firecrawl",
            }
        }
        assert _is_provider_active(_row("ddgs"), config) is False
        assert _is_provider_active(_row("searxng"), config) is False

    def test_backend_naming_one_capability_does_not_claim_the_other(self):
        """``search_backend: tavily`` alone: tavily serves search, not extract."""
        config = {"web": {"search_backend": "tavily", "backend": "firecrawl"}}
        assert _is_provider_active(_row("tavily"), config) is True
        assert _is_provider_active(_row("firecrawl"), config) is True

    def test_shared_key_shadowed_by_both_overrides(self):
        """An override shadows the shared key for its own capability only — when both
        capabilities are overridden, the shared vendor serves neither and must not be
        highlighted."""
        config = {
            "web": {
                "search_backend": "searxng",
                "extract_backend": "tavily",
                "backend": "firecrawl",
            }
        }
        assert _is_provider_active(_row("searxng"), config) is True
        assert _is_provider_active(_row("tavily"), config) is True
        assert _is_provider_active(_row("firecrawl"), config) is False

    def test_empty_capability_keys_fall_back_to_shared(self):
        """Blank overrides are what a cleared selection leaves behind."""
        config = {"web": {"search_backend": "", "extract_backend": None, "backend": "searxng"}}
        assert _is_provider_active(_row("searxng"), config) is True

    def test_case_and_whitespace_match_dispatch(self):
        """The dispatchers lower-case and strip the config value; highlighting must agree."""
        config = {"web": {"search_backend": " FireCrawl ", "backend": "tavily"}}
        assert _is_provider_active(_row("firecrawl"), config) is True

    def test_tier_gate_still_applies_to_capability_keys(self):
        """Tiered rows (exa/parallel free vs paid) keep the tier discriminator."""
        config = {
            "web": {
                "search_backend": "parallel",
                "provider_tier": {"parallel": "paid"},
            }
        }
        assert _is_provider_active(_row("parallel", web_tier="paid"), config) is True
        assert _is_provider_active(_row("parallel", web_tier="free"), config) is False


class TestNonWebProvidersUnaffected:
    def test_row_without_web_backend_is_never_active(self):
        assert _is_provider_active({"name": "openai", "env_vars": []}, {"web": {}}) is False