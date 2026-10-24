"""Post-approval terminal admission through real profile-scoped plugin discovery."""
import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from agent.tool_execution_context import bind_tool_execution_context
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli import plugins
from tools import terminal_tool as terminal

_PLUGIN = '''
from hermes_cli.authorized_tool_execution import hold_foreground_execution
EVENTS = []
MODE = "wrap"
def execute(next_call, args, **context):
    EVENTS.append((dict(context), dict(args)))
    if MODE == "deny":
        raise RuntimeError("resource admission unavailable")
    if args.get("background"):
        raise RuntimeError("resource lifetime requires a foreground command")
    if not context["lineage_valid"]:
        raise RuntimeError("trusted supervisor identity unavailable")
    try:
        with hold_foreground_execution():
            return next_call()
    finally:
        EVENTS.append("released")
def register(ctx):
    ctx.register_middleware("authorized_tool_execution", execute)
'''


@dataclass
class Agent:
    session_id: str = "root"
    _delegate_depth: int = 0


@pytest.fixture
def homes(tmp_path):
    plugins._reset_plugin_managers_for_tests()
    result = []
    for name in ("a", "b"):
        home = tmp_path / name
        plugin = home / "plugins" / "admission"
        plugin.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text("name: admission\nversion: 0.1.0\n", encoding="utf-8")
        (plugin / "__init__.py").write_text(_PLUGIN, encoding="utf-8")
        (home / "config.yaml").write_text("plugins:\n  enabled: [admission]\n", encoding="utf-8")
        result.append(home)
    yield result
    plugins._reset_plugin_managers_for_tests()


def _module():
    manager = plugins._delivery_manager()
    return manager._plugins["admission"].module


@pytest.fixture
def backend(tmp_path, monkeypatch):
    calls = []
    config = {"env_type": "local", "timeout": 60, "cwd": str(tmp_path), "host_cwd": None}
    monkeypatch.setattr(terminal, "_get_env_config", lambda: config)
    env = SimpleNamespace(host_cwd=None)
    def execute(command, **kwargs):
        calls.append((command, kwargs))
        return {"output": "finished", "returncode": 0}
    env.execute = execute
    monkeypatch.setattr(terminal, "_acquire_env", lambda *args: env)
    monkeypatch.setattr(terminal, "_pre_exec_block", lambda *args, **kwargs: None)
    monkeypatch.setattr(terminal, "_check_all_guards", lambda *args, **kwargs: {"approved": True})
    monkeypatch.setattr(terminal, "_resolve_command_cwd", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(terminal, "finalize_foreground_result", lambda **kwargs: json.dumps(kwargs["result"]))
    return calls


def test_profile_a_b_a_dispatch_is_scoped_and_foreground_cannot_yield(homes, backend, monkeypatch):
    monkeypatch.setattr(terminal, "yield_to_background_handler", lambda **kwargs: pytest.fail("protected command yielded"))
    modules = []
    for home in (homes[0], homes[1], homes[0]):
        token = set_hermes_home_override(home)
        try:
            with bind_tool_execution_context(Agent(home.name)):
                result = json.loads(terminal.terminal_tool("echo ok", task_id="task"))
            assert result["returncode"] == 0
            module = _module()
            modules.append(module)
            context = module.EVENTS[-2][0]
            assert context["profile_home"] == str(home)
            assert context["root_session_id"] == home.name
        finally:
            reset_hermes_home_override(token)
    assert modules[0] is modules[2] and modules[0] is not modules[1]
    assert len(modules[0].EVENTS) == 4 and len(modules[1].EVENTS) == 2
    assert len(backend) == 3 and all("yield_handler" not in kw for _, kw in backend)


@pytest.mark.parametrize("approval", [False, True])
def test_approval_precedes_admission_and_denial_cannot_execute(homes, backend, monkeypatch, approval):
    token = set_hermes_home_override(homes[0])
    try:
        module = _module()
        module.MODE = "deny"
        monkeypatch.setattr(terminal, "_check_all_guards", lambda *args, **kwargs: {"approved": approval})
        with bind_tool_execution_context(Agent()):
            result = json.loads(terminal.terminal_tool("echo ok"))
        assert result.get("error")
        assert len(module.EVENTS) == int(approval)
        assert not backend
    finally:
        reset_hermes_home_override(token)


@pytest.mark.parametrize("background,timeout", [(True, 60), (False, 9999)])
def test_background_and_promoted_commands_cannot_escape_resource_policy(homes, backend, monkeypatch, background, timeout):
    token = set_hermes_home_override(homes[0])
    try:
        monkeypatch.setattr(terminal, "spawn_background_process", lambda **kwargs: pytest.fail("detached"))
        with bind_tool_execution_context(Agent()):
            result = json.loads(terminal.terminal_tool("echo ok", background=background, timeout=timeout))
        assert "foreground" in result["error"]
        assert _module().EVENTS[0][1]["background"] is True
        assert not backend
    finally:
        reset_hermes_home_override(token)


def test_unbound_direct_caller_cannot_claim_root(homes, backend):
    token = set_hermes_home_override(homes[0])
    try:
        result = json.loads(terminal.terminal_tool("echo ok", session_id="forged-root"))
        assert "supervisor" in result["error"]
        assert not backend
    finally:
        reset_hermes_home_override(token)


def test_protected_backend_error_after_dispatch_is_not_retried(homes, backend, monkeypatch):
    calls = []
    def execute(*args, **kwargs):
        calls.append(True)
        raise OSError("backend disconnected after dispatch")
    monkeypatch.setattr(terminal, "_acquire_env", lambda *args: SimpleNamespace(execute=execute, host_cwd=None))
    monkeypatch.setattr(terminal.time, "sleep", lambda seconds: pytest.fail("protected command retried"))
    token = set_hermes_home_override(homes[0])
    try:
        with bind_tool_execution_context(Agent()):
            result = json.loads(terminal.terminal_tool("echo ok"))
        assert result["error"] and len(calls) == 1
        assert _module().EVENTS[-1] == "released"
    finally:
        reset_hermes_home_override(token)


def test_real_local_command_with_registered_middleware(homes, tmp_path, monkeypatch):
    token = set_hermes_home_override(homes[0])
    try:
        monkeypatch.setattr(terminal, "_get_env_config", lambda: {
            "env_type": "local", "timeout": 30, "cwd": str(tmp_path), "host_cwd": None,
        })
        monkeypatch.setattr(terminal, "_start_cleanup_thread", lambda: None)
        with bind_tool_execution_context(Agent()):
            result = json.loads(terminal.terminal_tool("echo RESOURCE_READY", task_id="resource-smoke"))
        assert result["exit_code"] == 0 and "RESOURCE_READY" in result["output"]
        assert _module().EVENTS[-1] == "released"
    finally:
        terminal._evict_environment_for_task("resource-smoke")
        reset_hermes_home_override(token)
