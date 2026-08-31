"""Shared fixtures for tests/tools/ web-provider tests.

Per-file subprocess isolation means each test file gets a fresh interpreter,
so module-level state (like the web-search-provider registry) is empty when
a file starts.  The ``web_registry_populated`` fixture registers all bundled
providers before each test and resets the registry afterwards — tests that
depend on the registry being populated should use it explicitly or via
``@pytest.mark.usefixtures("web_registry_populated")``.
"""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _no_host_browser_use_cli():
    """Keep the host's browser-use/uvx install out of tests.

    Browser Use mode is default-on when the CLI is runnable, so a developer
    machine with uvx on PATH would silently flip every built-in-browser test
    into CLI mode. Pin discovery to "not installed"; tests that exercise the
    CLI path monkeypatch ``bu_cli._find_cli`` themselves.
    """
    try:
        import tools.browser_use_cli as bu_cli
    except Exception:
        yield
        return
    # Keep a handle to the real discovery function so TestFindCli (and any
    # test that wants genuine PATH probing) can restore it explicitly.
    if not hasattr(bu_cli, "_find_cli_unpatched"):
        bu_cli._find_cli_unpatched = bu_cli._find_cli
    with patch.object(bu_cli, "_find_cli", lambda: None):
        yield


@pytest.fixture(autouse=True)
def _materialize_mcp_sdk_symbols():
    """Materialize the lazily-imported MCP SDK before each tools test.

    ``tools/mcp_tool.py`` defers the ~260ms ``mcp`` SDK import until first
    real use (CLI startup perf). Tests in this directory patch SDK symbols
    (``ClientSession``, ``stdio_client``, ``_MCP_HTTP_AVAILABLE``, ...) on
    the module and expect the pre-lazy eager-import world: symbols bound,
    availability flags reflecting the installed SDK. Ensure that state up
    front so ``mock.patch`` sees real originals and ``_ensure_mcp_sdk()``
    can never clobber a patched flag mid-test (it no-ops once attempted).
    """
    try:
        from tools import mcp_tool
        mcp_tool._ensure_mcp_sdk()
    except Exception:
        pass
    yield


@pytest.fixture(autouse=True)
def _clear_web_result_cache():
    """Reset the web_search TTL memo between tests.

    The memo is module-global state in tools/web_result_cache.py; without
    this, a test that exercised web_search_tool leaves a cached response
    that a later test with the same query would receive instead of its own
    mocked provider result.
    """
    from tools.web_result_cache import search_memo
    search_memo.clear()
    yield
    search_memo.clear()


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
    from plugins.web.keenable.provider import KeenableWebSearchProvider
    from plugins.web.searxng.provider import SearXNGWebSearchProvider
    from plugins.web.xai.provider import XAIWebSearchProvider

    _reset_for_tests()
    for cls in (
        BraveFreeWebSearchProvider,
        DDGSWebSearchProvider,
        ExaWebSearchProvider,
        FirecrawlWebSearchProvider,
        ParallelWebSearchProvider,
        KeenableWebSearchProvider,
        SearXNGWebSearchProvider,
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


@pytest.fixture
def disable_lazy_stt_install():
    """Disarm the runtime lazy-install probe so static ``_HAS_FASTER_WHISPER``
    patches accurately simulate 'faster-whisper not installed'.

    Without this, ``_try_lazy_install_stt()`` calls
    ``importlib.util.find_spec("faster_whisper")``, which returns truthy
    whenever the package is installed in the dev / CI environment —
    defeating the test's ``_HAS_FASTER_WHISPER=False`` patch.

    Opt in at module scope with
    ``pytestmark = pytest.mark.usefixtures("disable_lazy_stt_install")``.
    """
    with patch("tools.transcription_tools._try_lazy_install_stt", return_value=False):
        yield
