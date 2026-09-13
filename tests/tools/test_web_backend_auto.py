"""Tests for the hand-edited ``web.backend`` value ``"auto"``.

``"auto"`` is never written by ``hermes tools``, but other Hermes config
sections use ``auto`` as a legitimate sentinel (computer_use, auxiliary
providers), so hand edits land on ``web.backend`` regularly. Before the
strict-selection change these values were silently ignored; after it they
surfaced as hard "no registered provider 'auto'" errors. Contract under
test: ``"auto"`` means "pick for me" everywhere a backend name is read —

  * ``_get_backend()`` — ``auto`` counts as unset: run the credential
    ladder, including past the ``selection_exists`` firecrawl sentinel.
  * ``_get_capability_backend()`` — ``auto`` per-capability override falls
    through to the shared selection instead of erroring.
  * ``check_web_api_key()`` — readiness ignores ``auto`` as a name and
    probes the backends the user actually has credentials for.
  * Genuine typos stay strict — only the exact token ``auto`` is special.
"""

import pytest

import agent.web_search_registry as registry
import tools.tool_backend_helpers as helpers
import tools.web_tools as wt


@pytest.fixture
def web_config(monkeypatch):
    """Patch tools.web_tools._load_web_config with a caller-owned dict."""
    cfg = {}
    monkeypatch.setattr(wt, "_load_web_config", lambda: cfg)
    return cfg


def _neutralize_host_credentials(monkeypatch):
    """Pin every credential probe so the host's own .env cannot leak in."""
    monkeypatch.setattr(wt, "_env_value", lambda name: "")
    monkeypatch.setattr(wt, "check_firecrawl_api_key", lambda: False)
    monkeypatch.setattr("tools.xai_http.has_xai_credentials", lambda: False)
    monkeypatch.setattr(wt, "_ddgs_package_importable", lambda: False)
    monkeypatch.setattr(wt, "_is_tool_gateway_ready", lambda: False)
    monkeypatch.setattr(wt, "_list_registered_web_providers", list)
    monkeypatch.setattr(wt, "_registered_web_provider", lambda name: None)
    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    # Empty the process-global registry so earlier tests' providers and the
    # host's plugin set cannot satisfy availability walks.
    monkeypatch.setattr(registry, "_providers", {})
    monkeypatch.setattr(registry, "_scoped_providers", {})
    monkeypatch.setattr(registry, "_keyless_tier_enabled", lambda: False)


def _with_credential(monkeypatch, env_var):
    """Make exactly one env-var credential present in the ladder."""
    monkeypatch.setattr(
        wt, "_env_value", lambda name: "sk-test" if name == env_var else ""
    )


class TestSharedBackendAuto:
    """web.backend: auto behaves like an unset shared name."""

    def test_auto_runs_credential_ladder(self, web_config, monkeypatch):
        web_config["backend"] = "auto"
        _neutralize_host_credentials(monkeypatch)
        _with_credential(monkeypatch, "TAVILY_API_KEY")
        assert wt._get_backend() == "tavily"

    def test_auto_bypasses_selection_exists_firecrawl_sentinel(
        self, web_config, monkeypatch
    ):
        # read_selection() sees "auto" on disk, so selection_exists("web")
        # reports True; the ladder must still run instead of pinning
        # firecrawl.
        web_config["backend"] = "auto"
        _neutralize_host_credentials(monkeypatch)
        monkeypatch.setattr(helpers, "selection_exists", lambda section: True)
        _with_credential(monkeypatch, "EXA_API_KEY")
        assert wt._get_backend() == "exa"

    def test_auto_with_no_credentials_reaches_default(self, web_config, monkeypatch):
        web_config["backend"] = "auto"
        _neutralize_host_credentials(monkeypatch)
        assert wt._get_backend() == "firecrawl"  # backward-compat default

    def test_case_and_whitespace_variants_of_auto(self, web_config, monkeypatch):
        web_config["backend"] = "  AUTO "
        _neutralize_host_credentials(monkeypatch)
        _with_credential(monkeypatch, "TAVILY_API_KEY")
        assert wt._get_backend() == "tavily"

    def test_genuine_typo_stays_strict(self, web_config, monkeypatch):
        # Only the exact token "auto" is special; unknown names keep
        # surfacing verbatim so the vendor path raises its honest error.
        web_config["backend"] = "autodetect"
        _neutralize_host_credentials(monkeypatch)
        _with_credential(monkeypatch, "TAVILY_API_KEY")
        assert wt._get_backend() == "autodetect"

    def test_never_configured_still_ladders(self, web_config, monkeypatch):
        # No web section at all: unchanged pre-existing behavior.
        _neutralize_host_credentials(monkeypatch)
        _with_credential(monkeypatch, "TAVILY_API_KEY")
        assert wt._get_backend() == "tavily"


class TestPerCapabilityBackendAuto:
    """search_backend/extract_backend: auto falls through to the shared pick."""

    def test_auto_search_backend_falls_through_to_shared(self, web_config):
        web_config.update({"backend": "tavily", "search_backend": "auto"})
        assert wt._get_search_backend() == "tavily"

    def test_auto_extract_backend_falls_through_to_shared(self, web_config):
        web_config.update({"backend": "firecrawl", "extract_backend": "AUTO"})
        assert wt._get_extract_backend() == "firecrawl"

    def test_per_capability_only_selection_keeps_firecrawl_sentinel(
        self, web_config, monkeypatch
    ):
        # Shared name unset but an override exists: the "configured"
        # firecrawl sentinel is intentional design and stays.
        web_config.update({"backend": "", "extract_backend": "searxng"})
        _neutralize_host_credentials(monkeypatch)
        monkeypatch.setattr(helpers, "selection_exists", lambda section: True)
        _with_credential(monkeypatch, "TAVILY_API_KEY")
        assert wt._get_backend() == "firecrawl"


class TestCheckWebApiKeyWithAuto:
    def test_auto_counts_backends_with_credentials(self, web_config, monkeypatch):
        web_config["backend"] = "auto"
        _neutralize_host_credentials(monkeypatch)
        _with_credential(monkeypatch, "TAVILY_API_KEY")
        assert wt.check_web_api_key() is True

    def test_auto_without_any_credentials_is_not_ready(self, web_config, monkeypatch):
        web_config["backend"] = "auto"
        _neutralize_host_credentials(monkeypatch)
        monkeypatch.setattr(registry, "_read_config_key", lambda *keys: None)
        assert wt.check_web_api_key() is False


class TestNormalizationHelpers:
    """The single normalization point for stored backend names."""

    def test_raw_name_strips_lowercases_and_blanks(self):
        assert wt._raw_backend_name({"backend": "  Tavily "}) == "tavily"
        assert wt._raw_backend_name({"backend": None}) == ""
        assert wt._raw_backend_name({}) == ""

    def test_auto_sentinel_matches_exact_token_only(self):
        assert wt._is_auto_sentinel("auto") is True
        assert wt._is_auto_sentinel("AUTO") is False  # raw names are pre-lowered
        assert wt._is_auto_sentinel("autodetect") is False
        assert wt._is_auto_sentinel("") is False

    def test_configured_name_normalizes_auto_to_unset(self):
        assert wt._configured_backend_name({"backend": "auto"}) == ""
        assert wt._configured_backend_name({"backend": "  AUTO "}) == ""
        assert wt._configured_backend_name({"backend": "tavily"}) == "tavily"

    def test_per_capability_key_reads_through_same_helper(self):
        cfg = {"search_backend": "AUTO", "extract_backend": "exa"}
        assert wt._configured_backend_name(cfg, "search_backend") == ""
        assert wt._configured_backend_name(cfg, "extract_backend") == "exa"


class TestRegistryPathUnaffected:
    def test_registry_resolver_still_degrades_unknown_names(self, monkeypatch):
        # The registry resolver already treats unknown configured names as
        # "fall back to the ladder"; document that contract so the tools
        # layer and registry stay aligned.
        class _FakeProvider(registry.WebSearchProvider):
            @property
            def name(self) -> str:
                return "vendor-x"

            def is_available(self) -> bool:
                return True

            def supports_search(self) -> bool:
                return True

            def supports_extract(self) -> bool:
                return False

            def search(self, query, limit=5):
                return {}

        monkeypatch.setitem(registry._providers, "vendor-x", _FakeProvider())
        monkeypatch.setattr(registry, "_scoped_providers", {})
        resolved = registry.get_provider("does-not-exist")
        assert resolved is None
