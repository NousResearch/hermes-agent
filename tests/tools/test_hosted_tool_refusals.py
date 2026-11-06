"""Hosted scopes refuse every tool family that would run or read on this host.

``tools/hosted_tool_policy.py`` lists the families with no sandbox route yet. The registry-dispatched
ones are covered by the conformance check ``host_tools_refused``; this file drives the paths that
skip the registry (browser harness entry, checkpoints, inline skill shell, ``@`` references, the
workspace probe, cron scripts, MCP stdio spawns, media delivery, transcription, delegation, connector
file inputs, document extraction) through their real entry points, under a hosted scope and with no
scope bound, and checks that the hosted call fails closed before any host effect.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys

import pytest

from agent.runtime_scope import bind_state
from hermes_scope import Principal, Scope
from tools.hosted_tool_policy import HostedToolRefused

A = Scope("acme", "agent-a", Principal("u1"), owner=Principal("u1"), can_administer=True)


@pytest.fixture
def no_spawn(monkeypatch):
    """Record (and refuse) every host subprocess a hosted call would start."""
    spawned = []

    def refuse(*args, **kwargs):
        spawned.append(args[0] if args else kwargs.get("args"))
        raise AssertionError(f"a host process was started: {spawned[-1]}")
    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(subprocess, "run", refuse)
    return spawned


def test_browser_exec_runs_no_tenant_python_on_the_host(monkeypatch, no_spawn):
    import tools.browser_use_cli as browser
    from tools import browser_use_shot

    setup = []
    monkeypatch.setattr(browser, "_find_cli", lambda: [sys.executable, "-"])
    monkeypatch.setattr(browser, "_base_subprocess_env", lambda: setup.append("env") or {})
    monkeypatch.setattr(browser, "_route_exec_current", lambda env, *a: setup.append("route") or (env, None, False, 0))
    monkeypatch.setattr(browser, "_workspace_dir", lambda task: setup.append("workspace"))
    monkeypatch.setattr(browser_use_shot, "harness_command", lambda cmd, **kw: cmd)
    with bind_state(A):
        result = browser.browser_exec("print('HOST-EXECUTED')", task_id="t")
    assert "not available to a hosted agent" in result and "HOST-EXECUTED" not in result.replace("print('HOST", "")
    assert setup == [] and no_spawn == []


def test_checkpoints_take_and_restore_nothing_on_the_host(tmp_path, no_spawn):
    from tools.checkpoint_manager import CheckpointManager

    mgr = CheckpointManager(enabled=True)
    (tmp_path / "f.txt").write_text("x")
    with bind_state(A):
        assert mgr.ensure_checkpoint(str(tmp_path), "before write") is False
        restored = mgr.restore(str(tmp_path), "deadbeef")
    assert restored["success"] is False and "hosted agent" in restored["error"]
    assert no_spawn == []


def test_inline_skill_shell_does_not_run_on_the_host(tmp_path, no_spawn):
    from agent.skill_preprocessing import run_inline_shell

    with bind_state(A):
        out = run_inline_shell("echo HOST", tmp_path, 5)
    assert "refused" in out and no_spawn == []


def test_context_references_read_nothing_on_the_host(tmp_path, monkeypatch):
    from agent.context_references import preprocess_context_references
    from tools.environments import sandbox_conformance as conformance

    planted = tmp_path / "secret.txt"
    planted.write_text("HOST-ONLY")
    with conformance._watch_host(tmp_path) as touched, bind_state(A):
        result = preprocess_context_references("look at @file:secret.txt and @diff", cwd=tmp_path,
                                               context_length=100_000)
    assert result.blocked and "HOST-ONLY" not in result.message and not touched


def test_the_workspace_probe_runs_no_git_on_the_host(tmp_path, no_spawn):
    from agent.coding_context import build_coding_workspace_block

    (tmp_path / ".git").mkdir()
    with bind_state(A):
        assert build_coding_workspace_block(tmp_path) == ""
    assert no_spawn == []


def test_a_cron_script_does_not_run_on_the_host(tmp_path, no_spawn):
    from cron.scheduler_script import _run_job_script

    with bind_state(A):
        ok, out = _run_job_script("job.py")
    assert ok is False and "hosted agent" in out and no_spawn == []


def test_an_mcp_stdio_server_is_not_started_on_the_host(no_spawn):
    from tools.mcp_tool import MCPServerTask

    server = MCPServerTask("user-server")
    with bind_state(A), pytest.raises(HostedToolRefused):
        asyncio.run(server._run_stdio({"command": "touch", "args": ["/tmp/x"]}))  # no-tmp: ok — never run
    assert no_spawn == []


def test_media_delivery_never_attaches_a_host_file(tmp_path):
    from gateway.platforms.base import validate_media_delivery_path

    planted = tmp_path / "report.png"
    planted.write_bytes(b"x")
    assert validate_media_delivery_path(str(planted)) == str(planted.resolve())
    with bind_state(A):
        assert validate_media_delivery_path(str(planted)) is None


def test_transcription_reads_no_host_audio(tmp_path, no_spawn):
    from tools.environments import sandbox_conformance as conformance
    from tools.transcription_tools import transcribe_audio

    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"OggS")
    with conformance._watch_host(tmp_path) as touched, bind_state(A):
        result = transcribe_audio(str(audio))
    assert result["success"] is False and "hosted agent" in result["error"]
    assert not touched and no_spawn == []


def test_delegation_spawns_no_child_agent():
    from tools.delegate_tool import delegate_task

    class Parent:
        def __getattr__(self, name):
            raise AssertionError(f"the parent agent was used to spawn a child ({name})")
    with bind_state(A):
        result = json.loads(delegate_task(goal="do it", parent_agent=Parent()))
    assert "not available to a hosted agent" in result["error"]


def test_a_connector_file_input_never_reads_the_host(tmp_path):
    from tools.connectors.gateway.files import FileInputError, _readable_file

    planted = tmp_path / "doc.pdf"
    planted.write_bytes(b"%PDF")
    with bind_state(A), pytest.raises(FileInputError, match="hosted agent"):
        _readable_file(str(planted), "t")


def test_document_extraction_is_not_done_on_the_host(monkeypatch):
    from tools import file_tools, read_extract

    monkeypatch.setattr(read_extract, "extract_document_bytes",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("extracted on the host")))
    with bind_state(A):
        assert file_tools._read_extracted_document("report.pdf", __import__("pathlib").PurePosixPath(
            "/workspace/report.pdf"), 1, 100, "t") is None


def test_local_calls_still_reach_every_guarded_path(tmp_path):
    """With no scope bound the same entry points behave as before (spot checks of the cheap ones)."""
    from agent.coding_context import build_coding_workspace_block
    from agent.skill_preprocessing import run_inline_shell
    from tools.checkpoint_manager import CheckpointManager

    assert run_inline_shell("echo LOCAL", tmp_path, 5) == "LOCAL"
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    assert build_coding_workspace_block(tmp_path) != ""
    (tmp_path / "f.txt").write_text("x")
    assert CheckpointManager(enabled=True).ensure_checkpoint(str(tmp_path), "t") is True


def test_an_mcp_servers_tool_is_refused_in_a_hosted_scope():
    """A tool an MCP server registered (toolset ``mcp-<server>``) runs for the profile, not the agent."""
    from tools.registry import registry

    calls = []
    registry.register(name="mcp_probe_server_echo", toolset="mcp-probe-server",
                      schema={"name": "mcp_probe_server_echo", "description": "echo", "parameters": {}},
                      handler=lambda args, **kw: calls.append(args) or json.dumps({"ok": True}))
    try:
        with bind_state(A):
            result = json.loads(registry.dispatch("mcp_probe_server_echo", {}))
        assert "not available to a hosted agent" in result["error"] and calls == []
        assert json.loads(registry.dispatch("mcp_probe_server_echo", {})) == {"ok": True}
    finally:
        registry.deregister("mcp_probe_server_echo")
