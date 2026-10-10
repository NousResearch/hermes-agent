"""Shared fixtures for tests/tools/ web-provider tests.

Per-file subprocess isolation means each test file gets a fresh interpreter,
so module-level state (like the web-search-provider registry) is empty when
a file starts.  The ``web_registry_populated`` fixture registers all bundled
providers before each test and resets the registry afterwards — tests that
depend on the registry being populated should use it explicitly or via
``@pytest.mark.usefixtures("web_registry_populated")``.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _no_host_browser_use_cli():
    """Keep the host's PM-managed browser-use install out of tests.

    Browser Use mode is default-on when the CLI is runnable, so a developer
    machine with the CLI installed would silently flip every built-in-browser test
    into CLI mode. Pin discovery to "not installed"; tests that exercise the
    CLI path monkeypatch ``bu_cli._find_cli`` themselves.
    """
    try:
        import tools.browser_use_cli as bu_cli
    except Exception:
        yield
        return
    # Keep a handle to the real discovery function so TestFindCli (and any
    # test that wants genuine PM discovery) can restore it explicitly.
    if not hasattr(bu_cli, "_find_cli_unpatched"):
        bu_cli._find_cli_unpatched = bu_cli._find_cli
    with patch.object(bu_cli, "_find_cli", lambda: None):
        yield


@pytest.fixture(autouse=True)
def _no_host_bot_desktop_autostart():
    """Keep the host's TigerVNC/Xfce install out of tests.

    ``computer_use`` auto-starts the profile's Bot Desktop on a headless Linux
    host with the packages installed, so a developer box that has them would
    launch a real Xvnc + Xfce session per test. Pin the binaries to "missing";
    tests that exercise the desktop path monkeypatch ``runtime`` themselves.
    """
    try:
        from tools.bot_desktop import runtime as bd_runtime
    except Exception:
        yield
        return
    with patch.object(bd_runtime, "missing_binaries", lambda: ["Xvnc"]):
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
def _sandbox_real_hermes_home(tmp_path, monkeypatch, request):
    """Keep machine-real PM/home probes out of the REAL Hermes home.

    On a default-install checkout this repo lives INSIDE the real Hermes home
    (``%LOCALAPPDATA%\\hermes\\hermes-agent``), so probes anchored to the repo
    path and the launch home — not ``HERMES_HOME`` — reach real-home state the
    test guard rightly refuses (tests/home_io_guard.py):

    - ``pm.environments.payload_venv`` stats ``<repo>.parent/manifest.json``
      and ``store_root`` probes the same location for the PM store; the
      import-time ``activate_dependencies`` chain makes this fire on plain
      imports of modules that pull in ``hermes_bootstrap``;
    - ``tools.environments.local._resolve_hermes_bin_dir`` walks the process
      PATH into the real install, and callers ``isdir`` the real ``<home>/bin``.

    A normal CI checkout (sibling of the native home) gets "not a payload"
    answers and usually no PATH-resolvable install, so the sandbox emulates
    exactly that normal-checkout world; nothing here weakens a test that
    already passes on CI. ``hermes_constants.get_hermes_home`` and
    ``get_default_hermes_root`` readers are covered by the home redirect in
    ``tests/conftest.py::_hermetic_environment``.

    Stubs, in the order a test can hit them:
    - ``pm.environments.payload_venv`` -> ``None`` (the tests that build
      synthetic payloads re-stub it themselves and keep their coverage);
    - ``pm.environments.store_root`` -> ``HERMES_RUNTIME_DIR`` when set, else
      ``<tmp>/pm-runtime`` — the real resolver's documented-override contract,
      resolved eagerly so the probe never walks the real parent;
    - ``hermes_constants.get_hermes_home`` -> ``<tmp>/hermes-home`` (attribute
      pin; ``from hermes_constants import get_hermes_home`` importers instead
      read the redirected ``HERMES_HOME`` env vars, which is equally sandboxed);
    - ``tools.environments.local._resolve_hermes_bin_dir`` -> cache-faithful
      stub mirroring ``_HERMES_BIN_DIR``'s real contract — a set value wins,
      ``_SENTINEL`` means "resolve now" and yields the fixture's fake bin dir,
      ``None`` means no injection — so tests expressing injection intent via
      the cache keep their semantics. Per-file sandboxes that patch the same
      seams (test_local_env_blocklist's ``_sandbox``) stack on top as the
      inner monkeypatch and win.

    Tests asserting the REAL resolver's disk walk opt out via the
    ``real_bin_resolver`` marker. A test that legitimately needs the host
    install (none today — the ones that spawn real children pin
    ``HERMES_RUNTIME_DIR`` and stay sandbox-compatible) opts out entirely
    via the ``real_home`` marker; the real-home tripwire stays armed for
    everything else on purpose.
    """
    pm_runtime = tmp_path / "pm-runtime"
    pm_runtime.mkdir(exist_ok=True)

    import hermes_constants
    from pm import environments as pm_env
    from tools.environments import local as local_mod

    if "real_home" not in request.keywords:
        if "real_bin_resolver" not in request.keywords:
            bin_dir = tmp_path / "sandbox-bin"
            bin_dir.mkdir(exist_ok=True)
            shim = "hermes.exe" if os.name == "nt" else "hermes"
            (bin_dir / shim).write_bytes(b"@echo fake-hermes\n")
            sentinel = local_mod._SENTINEL

            def _fake_resolver():
                if local_mod._HERMES_BIN_DIR is sentinel:
                    return str(bin_dir)  # "resolve now" -> the fake install
                return local_mod._HERMES_BIN_DIR  # None -> no injection; str -> inject it

            monkeypatch.setattr(local_mod, "_resolve_hermes_bin_dir", _fake_resolver)

        def _fake_payload_venv(project_root):
            return None  # normal checkout: no sibling manifest.json

        def _fake_store_root(project_root):
            override = os.environ.get("HERMES_RUNTIME_DIR")
            if override:
                return Path(override).resolve()
            return pm_runtime

        monkeypatch.setattr(pm_env, "payload_venv", _fake_payload_venv)
        monkeypatch.setattr(pm_env, "store_root", _fake_store_root)

        # Attribute-identity matters: live connector ops are keyed by
        # hermes_home_key() on one thread and get_process_hermes_home() on
        # another, so the pin must FOLLOW HERMES_HOME exactly like the real
        # resolver and only substitute the fallback for tests that exercise
        # default-home resolution (env cleared). The override branch (context-
        # local set_hermes_home_override) must also keep the real precedence —
        # tests assert get_hermes_home() equals their override, not this pin.
        def _pinned_home():
            override = hermes_constants.get_hermes_home_override()
            if override:
                return hermes_constants._expand_hermes_home(override)
            env_home = os.environ.get("HERMES_HOME", "").strip()
            return Path(env_home) if env_home else tmp_path / "hermes-home"

        monkeypatch.setattr(hermes_constants, "get_hermes_home", _pinned_home)

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
    from plugins.web.tavily.provider import TavilyWebSearchProvider
    from plugins.web.perplexity.provider import PerplexityWebSearchProvider
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
        TavilyWebSearchProvider,
        PerplexityWebSearchProvider,
        SearXNGWebSearchProvider,
        XAIWebSearchProvider,
    ):
        register_provider(cls())


@pytest.fixture
def grant_computer_use_approvals(monkeypatch):
    """Answer every computer_use approval prompt with "once" through the shared gate.

    computer_use fails CLOSED when nobody can answer (no interactive user, no
    gateway), so dispatch tests that only care about routing must present an
    interactive CLI with a granting callback. "once" persists nothing, so no
    grant leaks into ``tools.approval``'s session/permanent stores.
    """
    from tools.computer_use import tool as cu_tool

    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    cu_tool.set_approval_callback(lambda command, description, **kw: "once")
    yield
    cu_tool.set_approval_callback(None)


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
