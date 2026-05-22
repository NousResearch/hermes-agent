"""Shared fixtures for tests/tools/ web-provider tests.

Per-file subprocess isolation means each test file gets a fresh interpreter,
so module-level state (like the web-search-provider registry) is empty when
a file starts.  The ``web_registry_populated`` fixture registers all bundled
providers before each test and resets the registry afterwards — tests that
depend on the registry being populated should use it explicitly or via
``@pytest.mark.usefixtures("web_registry_populated")``.
"""

import pytest


def register_all_web_providers():
    """Register all bundled web-search providers into the global registry.

    This is the single source of truth for the provider list used by
    test classes that need the registry populated for dispatch checks.
    """
    from agent.web_search_registry import register_provider, _reset_for_tests
    from plugins.web.brave_free.provider import BraveFreeWebSearchProvider
    from plugins.web.ddgs.provider import DDGSWebSearchProvider
    from plugins.web.exa.provider import ExaWebSearchProvider
    from plugins.web.firecrawl.provider import FirecrawlWebSearchProvider
    from plugins.web.parallel.provider import ParallelWebSearchProvider
    from plugins.web.searxng.provider import SearXNGWebSearchProvider
    from plugins.web.tavily.provider import TavilyWebSearchProvider
    from plugins.web.xai.provider import XAIWebSearchProvider

    _reset_for_tests()
    for cls in (
        BraveFreeWebSearchProvider,
        DDGSWebSearchProvider,
        ExaWebSearchProvider,
        FirecrawlWebSearchProvider,
        ParallelWebSearchProvider,
        SearXNGWebSearchProvider,
        TavilyWebSearchProvider,
        XAIWebSearchProvider,
    ):
        register_provider(cls())


@pytest.fixture
def web_registry_populated():
    """Populate the web-search-provider registry for one test, then reset."""
    register_all_web_providers()
    yield
    from agent.web_search_registry import _reset_for_tests
    _reset_for_tests()


@pytest.fixture(autouse=True)
def _no_lazy_install_stt(request, monkeypatch):
    """Prevent transcription tests from triggering real faster-whisper install.

    ``tools.transcription_tools._try_lazy_install_stt()`` calls
    ``tools.lazy_deps.ensure("stt.faster_whisper")`` and re-probes via
    ``importlib.util.find_spec``. In a dev environment with the package already
    installed, the probe returns True and ``_get_provider()`` reports ``"local"``
    even when tests have patched ``_HAS_FASTER_WHISPER`` to ``False`` to simulate
    the unavailable state. That breaks ~15 tests that expect cloud fallback or
    ``"none"`` when local is unavailable.

    Default the helper to ``False`` everywhere; tests that specifically exercise
    the lazy-install path can still re-patch it locally.
    """
    try:
        import tools.transcription_tools as _tt
    except Exception:
        return
    monkeypatch.setattr(_tt, "_try_lazy_install_stt", lambda: False, raising=False)
