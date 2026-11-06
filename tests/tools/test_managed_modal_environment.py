import sys
import tempfile
import threading
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"


def _load_tool_module(module_name: str, filename: str):
    spec = spec_from_file_location(module_name, TOOLS_DIR / filename)
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
    """Save and restore sys.modules entries so fakes don't leak to other tests."""
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name in {"tools", "agent", "hermes_cli"}
        or name.startswith("tools.")
        or name.startswith("agent.")
        or name.startswith("hermes_cli.")
    }
    try:
        yield
    finally:
        _reset_modules(("tools", "agent", "hermes_cli"))
        sys.modules.update(original_modules)


def _install_fake_tools_package(*, credential_mounts=None):
    _reset_modules(("tools", "agent", "hermes_cli"))

    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.__path__ = []  # type: ignore[attr-defined]
    sys.modules["hermes_cli"] = hermes_cli
    sys.modules["hermes_cli.config"] = types.SimpleNamespace(
        get_hermes_home=lambda: Path(tempfile.gettempdir()) / "hermes-home",
    )

    tools_package = types.ModuleType("tools")
    tools_package.__path__ = [str(TOOLS_DIR)]  # type: ignore[attr-defined]
    sys.modules["tools"] = tools_package

    env_package = types.ModuleType("tools.environments")
    env_package.__path__ = [str(TOOLS_DIR / "environments")]  # type: ignore[attr-defined]
    sys.modules["tools.environments"] = env_package

    interrupt_event = threading.Event()
    sys.modules["tools.interrupt"] = types.SimpleNamespace(
        set_interrupt=lambda value=True: interrupt_event.set() if value else interrupt_event.clear(),
        is_interrupted=lambda: interrupt_event.is_set(),
        _interrupt_event=interrupt_event,
    )

    class _DummyBaseEnvironment:
        def __init__(self, cwd: str, timeout: int, env=None):
            self.cwd = cwd
            self.timeout = timeout
            self.env = env or {}

        def _prepare_command(self, command: str):
            return command, None

    sys.modules["tools.environments.base"] = types.SimpleNamespace(BaseEnvironment=_DummyBaseEnvironment)
    sys.modules["tools.managed_tool_gateway"] = types.SimpleNamespace(
        resolve_managed_tool_gateway=lambda vendor: types.SimpleNamespace(
            vendor=vendor,
            gateway_origin="https://modal-gateway.example.com",
            nous_user_token="user-token",
            managed_mode=True,
        )
    )
    sys.modules["tools.credential_files"] = types.SimpleNamespace(
        get_credential_file_mounts=lambda: list(credential_mounts or []),
    )

    return interrupt_event


class _FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def test_managed_modal_execute_polls_until_completed(monkeypatch):
    _install_fake_tools_package()
    managed_modal = _load_tool_module("tools.environments.managed_modal", "environments/managed_modal.py")

    calls = []
    poll_count = {"value": 0}

    def fake_request(method, url, headers=None, json=None, timeout=None):
        calls.append((method, url, json, timeout))
        if method == "POST" and url.endswith("/v1/sandboxes"):
            return _FakeResponse(200, {"id": "sandbox-1"})
        if method == "POST" and url.endswith("/execs"):
            return _FakeResponse(202, {"execId": json["execId"], "status": "running"})
        if method == "GET" and "/execs/" in url:
            poll_count["value"] += 1
            if poll_count["value"] == 1:
                return _FakeResponse(200, {"execId": url.rsplit("/", 1)[-1], "status": "running"})
            return _FakeResponse(200, {
                "execId": url.rsplit("/", 1)[-1],
                "status": "completed",
                "output": "hello",
                "returncode": 0,
            })
        if method == "POST" and url.endswith("/terminate"):
            return _FakeResponse(200, {"status": "terminated"})
        raise AssertionError(f"Unexpected request: {method} {url}")

    monkeypatch.setattr(managed_modal.requests, "request", fake_request)
    monkeypatch.setattr(managed_modal.time, "sleep", lambda _: None)

    env = managed_modal.ManagedModalEnvironment(image="python:3.11")
    result = env.execute("echo hello")
    env.cleanup()

    assert result == {"output": "hello", "returncode": 0}
    assert any(call[0] == "POST" and call[1].endswith("/execs") for call in calls)


def test_managed_modal_rejects_host_credential_passthrough():
    _install_fake_tools_package(
        credential_mounts=[{
            "host_path": "/tmp/token.json",
            "container_path": "/root/.hermes/token.json",
        }]
    )
    managed_modal = _load_tool_module("tools.environments.managed_modal", "environments/managed_modal.py")

    with pytest.raises(ValueError, match="credential-file passthrough"):
        managed_modal.ManagedModalEnvironment(image="python:3.11")


def test_managed_modal_execute_times_out_and_cancels(monkeypatch):
    _install_fake_tools_package()
    managed_modal = _load_tool_module("tools.environments.managed_modal", "environments/managed_modal.py")

    calls = []
    monotonic_values = iter([0.0, 0.0, 0.0, 12.5, 12.5])

    def fake_request(method, url, headers=None, json=None, timeout=None):
        calls.append((method, url, json, timeout))
        if method == "POST" and url.endswith("/v1/sandboxes"):
            return _FakeResponse(200, {"id": "sandbox-1"})
        if method == "POST" and url.endswith("/execs"):
            return _FakeResponse(202, {"execId": json["execId"], "status": "running"})
        if method == "GET" and "/execs/" in url:
            return _FakeResponse(200, {"execId": url.rsplit("/", 1)[-1], "status": "running"})
        if method == "POST" and url.endswith("/cancel"):
            return _FakeResponse(202, {"status": "cancelling"})
        if method == "POST" and url.endswith("/terminate"):
            return _FakeResponse(200, {"status": "terminated"})
        raise AssertionError(f"Unexpected request: {method} {url}")

    monkeypatch.setattr(managed_modal.requests, "request", fake_request)
    monkeypatch.setattr(managed_modal.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(managed_modal.time, "sleep", lambda _: None)

    env = managed_modal.ManagedModalEnvironment(image="python:3.11")
    result = env.execute("sleep 30", timeout=2)
    env.cleanup()

    assert result == {
        "output": "Managed Modal exec timed out after 2s",
        "returncode": 124,
    }
    assert any(call[0] == "POST" and call[1].endswith("/cancel") for call in calls)
