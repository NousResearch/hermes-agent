"""execute_code fails loudly when a script rewrites the protected config.yaml (#113421).

``write_file`` / ``patch`` hard-deny writes to the active profile's config.yaml —
approvals.mode and hooks live there — but arbitrary Python reached the same file with
a plain text-mode write and no gate. The tripwire hashes the file around the run: a
changed digest fails the call, identical content re-written passes, and an
unresolvable config path skips the check (the fail-open surface the file_tools
guards already use).
"""

from __future__ import annotations

import json
import sys

import pytest

import tools.approval as approval_module
import tools.code_execution_tool as cet
import tools.code_kernel as code_kernel_module
import tools.file_tools_write_guards as write_guards
import tools.interrupt as interrupt_module
import tools.process_registry as process_registry
import tools.terminal_tool as terminal_module


@pytest.fixture
def pinned_config(monkeypatch, tmp_path):
    """Pin the protected config to tmp_path and mock the chain down to the local kernel."""
    config = tmp_path / "config.yaml"
    config.write_text("approvals:\n  mode: smart\n")
    monkeypatch.setattr(write_guards, "_hermes_config_resolved", str(config))
    monkeypatch.setattr(write_guards, "_hermes_config_resolved_loaded", True)

    monkeypatch.setattr(cet, "SANDBOX_AVAILABLE", True)
    monkeypatch.setattr(cet, "_load_config", lambda: {"timeout": 5})
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: False)
    monkeypatch.setattr(approval_module, "check_execute_code_guard",
                        lambda code, env_type, has_host_access=False: {"approved": True})
    monkeypatch.setattr(terminal_module, "_get_env_config", lambda: {"env_type": "local"})
    monkeypatch.setattr(terminal_module, "_docker_has_host_access", lambda cfg: False)
    monkeypatch.setattr(interrupt_module, "is_interrupted", lambda: False)
    monkeypatch.setattr(cet, "_get_execution_mode", lambda: "project")
    monkeypatch.setattr(cet, "_resolve_child_python", lambda mode: sys.executable)
    monkeypatch.setattr(cet, "_resolve_child_cwd", lambda mode, cwd, task_id="": str(tmp_path))
    return config


def _kernel_returning(script_effect):
    """Wrap a simulated script side effect as the session-kernel return value."""
    def _run(code, **kwargs):
        script_effect()
        return json.dumps({"status": "ok", "output": "script done"})
    return _run


class TestConfigRewriteTripwire:
    def test_script_rewriting_config_fails_loudly(self, pinned_config, monkeypatch):
        def script_appends_hooks():
            pinned_config.write_text(pinned_config.read_text() + "hooks:\n  pre_tool_call: []\n")

        monkeypatch.setattr(code_kernel_module, "execute_in_session_kernel",
                            _kernel_returning(script_appends_hooks))

        result = json.loads(cet.execute_code("cfg_rewrite()"))

        assert result["status"] == "error"
        assert str(pinned_config) in result["error"]
        assert "security-sensitive" in result["error"]

    def test_identical_rewrite_passes(self, pinned_config, monkeypatch):
        def script_rewrites_same_content():
            pinned_config.write_text(pinned_config.read_text())

        monkeypatch.setattr(code_kernel_module, "execute_in_session_kernel",
                            _kernel_returning(script_rewrites_same_content))

        result = json.loads(cet.execute_code("cfg_idempotent()"))

        assert result == {"status": "ok", "output": "script done"}

    def test_unrelated_run_passes(self, pinned_config, monkeypatch, tmp_path):
        monkeypatch.setattr(code_kernel_module, "execute_in_session_kernel",
                            _kernel_returning(lambda: (tmp_path / "out.txt").write_text("x")))

        result = json.loads(cet.execute_code("print('hi')"))

        assert result == {"status": "ok", "output": "script done"}

    def test_creating_missing_config_trips(self, pinned_config, monkeypatch):
        pinned_config.unlink()

        def script_creates_config():
            pinned_config.write_text("approvals:\n  mode: yolo\n")

        monkeypatch.setattr(code_kernel_module, "execute_in_session_kernel",
                            _kernel_returning(script_creates_config))

        result = json.loads(cet.execute_code("cfg_create()"))

        assert result["status"] == "error"
        assert str(pinned_config) in result["error"]

    def test_unresolvable_config_skips_check(self, tmp_path, monkeypatch):
        monkeypatch.setattr(write_guards, "_hermes_config_resolved", None)
        monkeypatch.setattr(write_guards, "_hermes_config_resolved_loaded", True)

        monkeypatch.setattr(cet, "SANDBOX_AVAILABLE", True)
        monkeypatch.setattr(cet, "_load_config", lambda: {"timeout": 5})
        monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: False)
        monkeypatch.setattr(approval_module, "check_execute_code_guard",
                            lambda code, env_type, has_host_access=False: {"approved": True})
        monkeypatch.setattr(terminal_module, "_get_env_config", lambda: {"env_type": "local"})
        monkeypatch.setattr(terminal_module, "_docker_has_host_access", lambda cfg: False)
        monkeypatch.setattr(interrupt_module, "is_interrupted", lambda: False)
        monkeypatch.setattr(cet, "_get_execution_mode", lambda: "project")
        monkeypatch.setattr(cet, "_resolve_child_python", lambda mode: sys.executable)
        monkeypatch.setattr(cet, "_resolve_child_cwd", lambda mode, cwd, task_id="": str(tmp_path))
        monkeypatch.setattr(code_kernel_module, "execute_in_session_kernel",
                            _kernel_returning(lambda: None))

        result = json.loads(cet.execute_code("print('hi')"))

        assert result == {"status": "ok", "output": "script done"}
