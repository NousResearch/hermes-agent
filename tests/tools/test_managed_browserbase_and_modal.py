import os
import sys
import tempfile
import threading
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch

import pytest


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
    hermes_cli.auth and hermes_cli.models — so the function body returns True.
    """
    monkeypatch.setattr("hermes_cli.auth.get_nous_auth_status", lambda: {"logged_in": True})
    monkeypatch.setattr("hermes_cli.models.check_nous_free_tier", lambda: False)


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
        get_active_browser_provider=lambda: None,
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

    sys.modules["tools.environments.base"] = types.SimpleNamespace(BaseEnvironment=_DummyEnvironment)
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
        browser_tool = _load_tool_module("tools.browser_tool", "browser_tool.py")

        local_mode = browser_tool._is_local_mode()
        provider = browser_tool._get_cloud_provider()

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


def test_browser_use_managed_gateway_adds_idempotency_key_and_persists_external_call_id():
    _install_fake_tools_package()
    env = os.environ.copy()
    env.pop("BROWSER_USE_API_KEY", None)
    env.update({
        "TOOL_GATEWAY_USER_TOKEN": "nous-token",
        "BROWSER_USE_GATEWAY_URL": "http://127.0.0.1:3009",
    })

    class _Response:
        status_code = 200
        ok = True
        text = ""
        headers = {"x-external-call-id": "call-browser-use-1"}

        def json(self):
            return {
                "id": "bu_local_session_1",
                "connectUrl": "wss://connect.browser-use.example/session",
            }

    with patch.dict(os.environ, env, clear=True):
        browser_use_module = _load_plugin_module(
            "plugins.browser.browser_use.provider",
            "browser/browser_use/provider.py",
        )

        with patch.object(browser_use_module.requests, "post", return_value=_Response()) as post:
            provider = browser_use_module.BrowserUseBrowserProvider()
            session = provider.create_session("task-browser-use-managed")

    sent_headers = post.call_args.kwargs["headers"]
    assert sent_headers["X-Browser-Use-API-Key"] == "nous-token"
    assert sent_headers["X-Idempotency-Key"].startswith("browser-use-session-create:")
    sent_payload = post.call_args.kwargs["json"]
    assert sent_payload["timeout"] == 5
    assert sent_payload["proxyCountryCode"] == "us"
    assert session["external_call_id"] == "call-browser-use-1"


def test_browser_use_managed_gateway_reuses_pending_idempotency_key_after_timeout():
    _install_fake_tools_package()
    env = os.environ.copy()
    env.pop("BROWSER_USE_API_KEY", None)
    env.update({
        "TOOL_GATEWAY_USER_TOKEN": "nous-token",
        "BROWSER_USE_GATEWAY_URL": "http://127.0.0.1:3009",
    })

    class _Response:
        status_code = 200
        ok = True
        text = ""
        headers = {"x-external-call-id": "call-browser-use-2"}

        def json(self):
            return {
                "id": "bu_local_session_2",
                "connectUrl": "wss://connect.browser-use.example/session2",
            }

    with patch.dict(os.environ, env, clear=True):
        browser_use_module = _load_plugin_module(
            "plugins.browser.browser_use.provider",
            "browser/browser_use/provider.py",
        )
        provider = browser_use_module.BrowserUseBrowserProvider()
        timeout = browser_use_module.requests.Timeout("timed out")

        with patch.object(
            browser_use_module.requests,
            "post",
            side_effect=[timeout, _Response()],
        ) as post:
            try:
                provider.create_session("task-browser-use-timeout")
            except browser_use_module.requests.Timeout:
                pass
            else:
                raise AssertionError("Expected Browser Use create_session to propagate timeout")

            provider.create_session("task-browser-use-timeout")

    first_headers = post.call_args_list[0].kwargs["headers"]
    second_headers = post.call_args_list[1].kwargs["headers"]
    assert first_headers["X-Idempotency-Key"] == second_headers["X-Idempotency-Key"]


def test_browser_use_managed_gateway_preserves_pending_idempotency_key_for_in_progress_conflicts():
    _install_fake_tools_package()
    env = os.environ.copy()
    env.pop("BROWSER_USE_API_KEY", None)
    env.update({
        "TOOL_GATEWAY_USER_TOKEN": "nous-token",
        "BROWSER_USE_GATEWAY_URL": "http://127.0.0.1:3009",
    })

    class _ConflictResponse:
        status_code = 409
        ok = False
        text = '{"error":{"code":"CONFLICT","message":"Managed Browser Use session creation is already in progress for this idempotency key"}}'
        headers = {}

        def json(self):
            return {
                "error": {
                    "code": "CONFLICT",
                    "message": "Managed Browser Use session creation is already in progress for this idempotency key",
                }
            }

    class _SuccessResponse:
        status_code = 200
        ok = True
        text = ""
        headers = {"x-external-call-id": "call-browser-use-4"}

        def json(self):
            return {
                "id": "bu_local_session_4",
                "connectUrl": "wss://connect.browser-use.example/session4",
            }

    with patch.dict(os.environ, env, clear=True):
        browser_use_module = _load_plugin_module(
            "plugins.browser.browser_use.provider",
            "browser/browser_use/provider.py",
        )
        provider = browser_use_module.BrowserUseBrowserProvider()

        with patch.object(
            browser_use_module.requests,
            "post",
            side_effect=[_ConflictResponse(), _SuccessResponse()],
        ) as post:
            try:
                provider.create_session("task-browser-use-conflict")
            except RuntimeError:
                pass
            else:
                raise AssertionError("Expected Browser Use create_session to propagate the in-progress conflict")

            provider.create_session("task-browser-use-conflict")

    first_headers = post.call_args_list[0].kwargs["headers"]
    second_headers = post.call_args_list[1].kwargs["headers"]
    assert first_headers["X-Idempotency-Key"] == second_headers["X-Idempotency-Key"]


def test_browser_use_managed_gateway_uses_new_idempotency_key_for_a_new_session_after_success():
    _install_fake_tools_package()
    env = os.environ.copy()
    env.pop("BROWSER_USE_API_KEY", None)
    env.update({
        "TOOL_GATEWAY_USER_TOKEN": "nous-token",
        "BROWSER_USE_GATEWAY_URL": "http://127.0.0.1:3009",
    })

    class _Response:
        status_code = 200
        ok = True
        text = ""
        headers = {"x-external-call-id": "call-browser-use-3"}

        def json(self):
            return {
                "id": "bu_local_session_3",
                "connectUrl": "wss://connect.browser-use.example/session3",
            }

    with patch.dict(os.environ, env, clear=True):
        browser_use_module = _load_plugin_module(
            "plugins.browser.browser_use.provider",
            "browser/browser_use/provider.py",
        )
        provider = browser_use_module.BrowserUseBrowserProvider()

        with patch.object(browser_use_module.requests, "post", side_effect=[_Response(), _Response()]) as post:
            provider.create_session("task-browser-use-new")
            provider.create_session("task-browser-use-new")

    first_headers = post.call_args_list[0].kwargs["headers"]
    second_headers = post.call_args_list[1].kwargs["headers"]
    assert first_headers["X-Idempotency-Key"] != second_headers["X-Idempotency-Key"]


def test_terminal_tool_prefers_managed_modal_when_gateway_ready_and_no_direct_creds():
    _install_fake_tools_package()
    env = os.environ.copy()
    env.pop("MODAL_TOKEN_ID", None)
    env.pop("MODAL_TOKEN_SECRET", None)

    with patch.dict(os.environ, env, clear=True):
        terminal_tool = _load_tool_module("tools.terminal_tool", "terminal_tool.py")

        with (
            patch.object(terminal_tool, "is_managed_tool_gateway_ready", return_value=True),
            patch.object(terminal_tool, "_ManagedModalEnvironment", return_value="managed-modal-env") as managed_ctor,
            patch.object(terminal_tool, "_ModalEnvironment", return_value="direct-modal-env") as direct_ctor,
            patch.object(Path, "exists", return_value=False),
        ):
            result = terminal_tool._create_environment(
                env_type="modal",
                image="python:3.11",
                cwd="/root",
                timeout=60,
                container_config={
                    "container_cpu": 1,
                    "container_memory": 2048,
                    "container_disk": 1024,
                    "container_persistent": True,
                    "modal_mode": "auto",
                },
                task_id="task-modal-managed",
            )

    assert result == "managed-modal-env"
    assert managed_ctor.called
    assert not direct_ctor.called


def test_terminal_tool_auto_mode_prefers_managed_modal_when_available():
    _install_fake_tools_package()
    env = os.environ.copy()
    env.update({
        "MODAL_TOKEN_ID": "tok-id",
        "MODAL_TOKEN_SECRET": "tok-secret",
    })

    with patch.dict(os.environ, env, clear=True):
        terminal_tool = _load_tool_module("tools.terminal_tool", "terminal_tool.py")

        with (
            patch.object(terminal_tool, "is_managed_tool_gateway_ready", return_value=True),
            patch.object(terminal_tool, "_ManagedModalEnvironment", return_value="managed-modal-env") as managed_ctor,
            patch.object(terminal_tool, "_ModalEnvironment", return_value="direct-modal-env") as direct_ctor,
        ):
            result = terminal_tool._create_environment(
                env_type="modal",
                image="python:3.11",
                cwd="/root",
                timeout=60,
                container_config={
                    "container_cpu": 1,
                    "container_memory": 2048,
                    "container_disk": 1024,
                    "container_persistent": True,
                    "modal_mode": "auto",
                },
                task_id="task-modal-auto",
            )

    assert result == "managed-modal-env"
    assert managed_ctor.called
    assert not direct_ctor.called


def test_terminal_tool_auto_mode_falls_back_to_direct_modal_when_managed_unavailable():
    _install_fake_tools_package()
    env = os.environ.copy()
    env.update({
        "MODAL_TOKEN_ID": "tok-id",
        "MODAL_TOKEN_SECRET": "tok-secret",
    })

    with patch.dict(os.environ, env, clear=True):
        terminal_tool = _load_tool_module("tools.terminal_tool", "terminal_tool.py")

        with (
            patch.object(terminal_tool, "is_managed_tool_gateway_ready", return_value=False),
            patch.object(terminal_tool, "_ManagedModalEnvironment", return_value="managed-modal-env") as managed_ctor,
            patch.object(terminal_tool, "_ModalEnvironment", return_value="direct-modal-env") as direct_ctor,
        ):
            result = terminal_tool._create_environment(
                env_type="modal",
                image="python:3.11",
                cwd="/root",
                timeout=60,
                container_config={
                    "container_cpu": 1,
                    "container_memory": 2048,
                    "container_disk": 1024,
                    "container_persistent": True,
                    "modal_mode": "auto",
                },
                task_id="task-modal-direct-fallback",
            )

    assert result == "direct-modal-env"
    assert direct_ctor.called
    assert not managed_ctor.called


def test_terminal_tool_respects_direct_modal_mode_without_falling_back_to_managed():
    _install_fake_tools_package()
    env = os.environ.copy()
    env.pop("MODAL_TOKEN_ID", None)
    env.pop("MODAL_TOKEN_SECRET", None)

    with patch.dict(os.environ, env, clear=True):
        terminal_tool = _load_tool_module("tools.terminal_tool", "terminal_tool.py")

        with (
            patch.object(terminal_tool, "is_managed_tool_gateway_ready", return_value=True),
            patch.object(Path, "exists", return_value=False),
        ):
            with pytest.raises(ValueError, match="direct Modal credentials"):
                terminal_tool._create_environment(
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
