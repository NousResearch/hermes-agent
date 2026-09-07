import os
import sys
import tempfile
import threading
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.nous_account import NousPortalAccountInfo


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
PLUGINS_DIR = REPO_ROOT / "plugins"


def _load_tool_module(module_name: str, filename: str):
    spec = spec_from_file_location(module_name, TOOLS_DIR / filename)
    assert spec and spec.loader
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_plugin_module(module_name: str, relpath: str):
    """Load a plugin module by file path from ``plugins/``.

    Mirror of :func:`_load_tool_module` for the plugin tree. Used by tests
    that exercise the per-vendor browser plugins' session-lifecycle
    behaviour after the PR #25214 migration.
    """
    spec = spec_from_file_location(module_name, PLUGINS_DIR / relpath)
    assert spec and spec.loader
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _reset_modules(prefixes: tuple[str, ...]):
    for name in list(sys.modules):
        if name.startswith(prefixes):
            sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def _restore_tool_and_agent_modules():
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "tools"
        or name.startswith("tools.")
        or name == "agent"
        or name.startswith("agent.")
    }
    try:
        yield
    finally:
        _reset_modules(("tools", "agent"))
        sys.modules.update(original_modules)


@pytest.fixture(autouse=True)
def _enable_managed_nous_tools(monkeypatch):
    """Ensure managed_nous_tools_enabled() returns True even after module reloads.

    The _install_fake_tools_package() helper resets and reimports tool modules,
    so a simple monkeypatch on tool_backend_helpers doesn't survive.  We patch
    the *source* modules that the reimported modules will import from — both
    hermes_cli.nous_account — so the function body returns True.
    """
    monkeypatch.setattr(
        "hermes_cli.nous_account.get_nous_portal_account_info",
        lambda: NousPortalAccountInfo(
            logged_in=True,
            source="jwt",
            fresh=False,
            paid_service_access=True,
        ),
    )


def _install_fake_tools_package():
    _reset_modules(("tools", "agent"))

    tools_package = types.ModuleType("tools")
    tools_package.__path__ = [str(TOOLS_DIR)]  # type: ignore[attr-defined]
    sys.modules["tools"] = tools_package

    env_package = types.ModuleType("tools.environments")
    env_package.__path__ = [str(TOOLS_DIR / "environments")]  # type: ignore[attr-defined]
    sys.modules["tools.environments"] = env_package

    agent_package = types.ModuleType("agent")
    agent_package.__path__ = []  # type: ignore[attr-defined]
    sys.modules["agent"] = agent_package
    sys.modules["agent.auxiliary_client"] = types.SimpleNamespace(
        call_llm=lambda *args, **kwargs: "",
    )
    # The fake `agent` package has an empty __path__, so every real
    # agent.* submodule that production code imports needs an explicit
    # stand-in here. tools.browser_tool imports redact_cdp_url;
    # hermes_cli.auth (imported transitively by nous_account /
    # tool_backend_helpers) imports sanitize_borrowed_credential_payload.
    sys.modules["agent.redact"] = types.SimpleNamespace(
        redact_cdp_url=lambda value: str(value),
    )
    sys.modules["agent.credential_persistence"] = types.SimpleNamespace(
        sanitize_borrowed_credential_payload=lambda entry, provider_id=None: entry,
    )

    # Stubs for the browser-provider plugin layer introduced in PR #25214.
    # The fake `agent` package has an empty __path__ so real submodules
    # aren't reachable; we install just enough stand-ins to satisfy
    # ``tools.browser_tool``'s top-level imports. The actual lifecycle
    # tests instantiate the real plugin classes via _load_tool_module
    # below, so the stubs only need to satisfy import + isinstance.
    class _StubBrowserProvider:
        """Minimal BrowserProvider stub for ``from agent.browser_provider import BrowserProvider``."""

    sys.modules["agent.browser_provider"] = types.SimpleNamespace(
        BrowserProvider=_StubBrowserProvider,
    )
    sys.modules["agent.browser_registry"] = types.SimpleNamespace(
        get_provider=lambda name: None,
        list_providers=lambda: [],
        register_provider=lambda provider: None,
        _resolve=lambda configured: None,
    )

    # Plugin module stubs — the real plugin classes are loaded from disk by
    # the lifecycle tests below via _load_tool_module(). For the import
    # phase, we just need the class names to exist on the right module path.
    plugins_package = types.ModuleType("plugins")
    plugins_package.__path__ = []  # type: ignore[attr-defined]
    sys.modules["plugins"] = plugins_package
    plugins_browser_package = types.ModuleType("plugins.browser")
    plugins_browser_package.__path__ = []  # type: ignore[attr-defined]
    sys.modules["plugins.browser"] = plugins_browser_package

    for _name, _classname in (
        ("browserbase", "BrowserbaseBrowserProvider"),
        ("browser_use", "BrowserUseBrowserProvider"),
        ("firecrawl", "FirecrawlBrowserProvider"),
    ):
        _vendor_pkg = types.ModuleType(f"plugins.browser.{_name}")
        _vendor_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules[f"plugins.browser.{_name}"] = _vendor_pkg
        _provider_stub_cls = type(_classname, (_StubBrowserProvider,), {})
        sys.modules[f"plugins.browser.{_name}.provider"] = types.SimpleNamespace(
            **{_classname: _provider_stub_cls},
        )
    # The fake ``plugins.browser`` package has an empty __path__, so the shared
    # base the real vendor modules import must be loaded by path as well.
    _load_plugin_module("plugins.browser._common", "browser/_common.py")

    sys.modules["tools.managed_tool_gateway"] = _load_tool_module(
        "tools.managed_tool_gateway",
        "managed_tool_gateway.py",
    )

    interrupt_event = threading.Event()
    sys.modules["tools.interrupt"] = types.SimpleNamespace(
        set_interrupt=lambda value=True: interrupt_event.set() if value else interrupt_event.clear(),
        is_interrupted=lambda: interrupt_event.is_set(),
        _interrupt_event=interrupt_event,
    )
    sys.modules["tools.approval"] = types.SimpleNamespace(
        detect_dangerous_command=lambda *args, **kwargs: None,
        check_dangerous_command=lambda *args, **kwargs: {"approved": True},
        check_all_command_guards=lambda *args, **kwargs: {"approved": True},
        load_permanent_allowlist=lambda *args, **kwargs: [],
        DANGEROUS_PATTERNS=[],
    )

    class _Registry:
        def register(self, **kwargs):
            return None

    from tools.registry import tool_error

    sys.modules["tools.registry"] = types.SimpleNamespace(
        registry=_Registry(), tool_error=tool_error,
    )

    class _DummyEnvironment:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def cleanup(self):
            return None

    class _DummyConnectionError(Exception):
        def __init__(self, message="", backend="", detail="", *args):
            super().__init__(message, *args)
            self.backend = backend
            self.detail = detail

    sys.modules["tools.environments.base"] = types.SimpleNamespace(
        BaseEnvironment=_DummyEnvironment,
        EnvironmentConnectionError=_DummyConnectionError,
    )
    sys.modules["tools.environments.local"] = types.SimpleNamespace(LocalEnvironment=_DummyEnvironment)
    sys.modules["tools.environments.singularity"] = types.SimpleNamespace(
        _get_scratch_dir=lambda: Path(tempfile.gettempdir()),
        SingularityEnvironment=_DummyEnvironment,
    )
    sys.modules["tools.environments.ssh"] = types.SimpleNamespace(SSHEnvironment=_DummyEnvironment)
    sys.modules["tools.environments.docker"] = types.SimpleNamespace(DockerEnvironment=_DummyEnvironment)
    sys.modules["tools.environments.modal"] = types.SimpleNamespace(ModalEnvironment=_DummyEnvironment)
    sys.modules["tools.environments.managed_modal"] = types.SimpleNamespace(ManagedModalEnvironment=_DummyEnvironment)


def test_browser_use_explicit_local_mode_stays_local_even_when_managed_gateway_is_ready(tmp_path):
    _install_fake_tools_package()
    (tmp_path / "config.yaml").write_text("browser:\n  cloud_provider: local\n", encoding="utf-8")
    env = os.environ.copy()
    env.pop("BROWSER_USE_API_KEY", None)
    env.update({
        "HERMES_HOME": str(tmp_path),
        "TOOL_GATEWAY_USER_TOKEN": "nous-token",
        "BROWSER_USE_GATEWAY_URL": "http://127.0.0.1:3009",
    })

    with patch.dict(os.environ, env, clear=True):
        _load_tool_module("tools.browser_tool", "browser_tool.py")
        browser_tool_cloud = sys.modules["tools.browser_tool_cloud"]

        local_mode = browser_tool_cloud._is_local_mode()
        provider = browser_tool_cloud._get_cloud_provider()

    assert local_mode is True
    assert provider is None


def test_browserbase_does_not_use_gateway_only_configuration():
    _install_fake_tools_package()
    env = os.environ.copy()
    env.pop("BROWSERBASE_API_KEY", None)
    env.pop("BROWSERBASE_PROJECT_ID", None)
    env.update({
        "TOOL_GATEWAY_USER_TOKEN": "nous-token",
        "BROWSERBASE_GATEWAY_URL": "http://127.0.0.1:3009",
    })

    with patch.dict(os.environ, env, clear=True):
        browserbase_module = _load_plugin_module(
            "plugins.browser.browserbase.provider",
            "browser/browserbase/provider.py",
        )
        provider = browserbase_module.BrowserbaseBrowserProvider()

    assert provider.is_available() is False


def test_terminal_tool_respects_direct_modal_mode_without_falling_back_to_managed():
    _install_fake_tools_package()
    env = os.environ.copy()
    env.pop("MODAL_TOKEN_ID", None)
    env.pop("MODAL_TOKEN_SECRET", None)

    with patch.dict(os.environ, env, clear=True):
        _load_tool_module("tools.terminal_tool", "terminal_tool.py")
        terminal_tool_backends = sys.modules["tools.terminal_tool_backends"]

        with (
            patch.object(terminal_tool_backends, "is_managed_tool_gateway_ready", return_value=True),
            patch.object(Path, "exists", return_value=False),
        ):
            with pytest.raises(ValueError, match="direct Modal credentials"):
                terminal_tool_backends._create_environment(
                    env_type="modal",
                    image="python:3.11",
                    cwd="/root",
                    timeout=60,
                    container_config={
                        "container_cpu": 1,
                        "container_memory": 2048,
                        "container_disk": 1024,
                        "container_persistent": True,
                        "modal_mode": "direct",
                    },
                    task_id="task-modal-direct-only",
                )


class TestShellEscapeBypass:
    """Regression for #36846/#36847: backslash escapes and empty-string
    literals split tokens so a denylisted command (rm) slips past detection
    while the shell still executes it."""

    def test_backslash_escape_bypass_caught(self):
        from tools.approval import detect_dangerous_command
        # literal: r-backslash-m -rf /  (shell collapses r\m -> rm)
        assert detect_dangerous_command("r\\m -rf /")[0] is True

    def test_empty_string_literal_bypass_caught(self):
        from tools.approval import detect_dangerous_command
        assert detect_dangerous_command("r''m -rf /")[0] is True
        assert detect_dangerous_command('r""m -rf /')[0] is True

    def test_plain_dangerous_still_caught(self):
        from tools.approval import detect_dangerous_command
        assert detect_dangerous_command("rm -rf /")[0] is True

    def test_benign_command_not_flagged(self):
        from tools.approval import detect_dangerous_command
        assert detect_dangerous_command("ls -la")[0] is False
