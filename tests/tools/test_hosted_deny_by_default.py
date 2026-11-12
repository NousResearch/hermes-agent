"""A hosted scope is deny-by-default: an allowlist of sandbox-only tools, one switch for host features.

``tools/hosted_tool_policy.py`` holds both decisions. These tests drive every route a tool call can take
(the agent loop's sequential and concurrent paths, inline agent tools, memory-provider tools, the
registry, ``handle_function_call``) and every non-tool feature that reads or writes this host's
profile state, under a hosted scope, and check nothing reaches the host. The same entry points with no
scope, or the local scope, behave exactly as before.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.runtime_scope import bind_state
from hermes_scope import Principal, Scope
from tests.tools.test_sandbox_scope_conformance import A, B
from tests.tools.test_sandbox_scope_conformance import provider as _conformance_provider
from tools.environments import sandbox_conformance as c
from tools.environments.sandbox_scope import sandbox_identity

provider = _conformance_provider  # the recording-sandbox fixture, by its name

def _local_scope() -> Scope:
    """The local install's scope, homed at the test's HERMES_HOME (a relative agent path would
    make ``bind_state`` stamp a home under the working directory)."""
    from hermes_constants import get_hermes_home
    return Scope("local", str(get_hermes_home()), Principal("local"))


def _png() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), "red").save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------------------------------
# The allowlist


def test_the_allowlist_is_exactly_the_sandbox_only_tools():
    """A tool joins only by editing this set (with a reason in hosted_tool_policy.py): anything new
    stays refused in a hosted scope until then."""
    from tools.hosted_tool_policy import HOSTED_TOOL_ALLOWLIST
    assert HOSTED_TOOL_ALLOWLIST == {
        "terminal", "process_manage", "execute_code", "read_file", "write_file", "patch", "search_files",
    }


def test_a_newly_registered_tool_is_refused_until_allowlisted():
    from tools.registry import registry
    reached = []
    registry.register(name="brand_new_probe_tool", toolset="brand-new",
                      schema={"name": "brand_new_probe_tool", "description": "x", "parameters": {}},
                      handler=lambda args, **kw: reached.append(args) or json.dumps({"ok": True}))
    try:
        with bind_state(A):
            assert "not available to a hosted agent" in registry.dispatch("brand_new_probe_tool", {})
        assert reached == []
        assert json.loads(registry.dispatch("brand_new_probe_tool", {})) == {"ok": True}
    finally:
        registry.deregister("brand_new_probe_tool")


def test_in_a_local_scope_the_allowlist_is_never_consulted(monkeypatch):
    """Unscoped and local: the policy answers "allowed" without looking at the allowlist, and every
    tool dispatches to its handler as before."""
    import tools.hosted_tool_policy as policy
    from tools.registry import registry

    class Exploding(frozenset):
        def __contains__(self, item):
            raise AssertionError("the hosted allowlist was consulted outside a hosted scope")
    monkeypatch.setattr(policy, "HOSTED_TOOL_ALLOWLIST", Exploding())
    names = sorted(registry.get_all_tool_names())
    reached, originals = [], {}
    for name in names:
        entry = registry.get_entry(name)
        originals[name] = (entry, entry.handler, entry.is_async)
        entry.handler, entry.is_async = (lambda args, _n=name, **kw: reached.append(_n) or "{}"), False
    try:
        for scope in (None, _local_scope()):
            reached.clear()
            with bind_state(scope) if scope else _null():
                for name in names:
                    assert policy.hosted_tool_refusal(name) is None
                    registry.dispatch(name, {})
                for feature in policy.HOST_FEATURES:
                    assert policy.host_feature_enabled(feature)
            assert reached == names
    finally:
        for entry, handler, is_async in originals.values():
            entry.handler, entry.is_async = handler, is_async


def _null():
    import contextlib
    return contextlib.nullcontext()


# --------------------------------------------------------------------------------------------------
# Every dispatch route meets the allowlist


def _make_agent(tmp_path: Path):
    from run_agent import AIAgent
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("run_agent._hermes_home", tmp_path),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=True)
    agent._flush_messages_to_session_db = MagicMock(return_value=True)
    agent._tool_result_content_for_active_model = MagicMock(side_effect=lambda _n, result: result)
    return agent


class _ProviderTools:
    """A memory provider's tools: dispatched by the agent loop, never through the registry."""

    def __init__(self):
        self.reached = []

    def has_tool(self, name):
        return name == "provider_recall"

    def handle_tool_call(self, name, args, **kwargs):
        self.reached.append(name)
        return json.dumps({"recalled": "HOST-PROVIDER-PRIVATE"})


def _call(call_id: str, name: str, args: dict | None = None):
    return SimpleNamespace(id=call_id, type="function",
                           function=SimpleNamespace(name=name, arguments=json.dumps(args or {})))


@pytest.mark.parametrize("path", ["sequential", "concurrent"])
def test_every_agent_loop_route_is_refused_before_its_executor(tmp_path, monkeypatch, path):
    """Memory-provider tools, inline agent tools (catalog, memory, session_search, todo) and registry
    tools all reach their executor only through the one chokepoint, on both execution paths."""
    import tools.connectors.catalog_tool as catalog
    from agent import tool_executor as te

    agent = _make_agent(tmp_path)
    agent._memory_manager = provider_tools = _ProviderTools()
    agent._todo_store = MagicMock()
    reached = []
    monkeypatch.setattr(catalog, "search", lambda *a, **k: reached.append("catalog") or {})
    calls = [_call("c1", "provider_recall"), _call("c2", "manage_catalog", {"action": "search"}),
             _call("c3", "todo_list", {"todos": []}), _call("c4", "session_search", {"query": "x"})]
    messages: list[dict] = []
    run = te.execute_tool_calls_sequential if path == "sequential" else te.execute_tool_calls_concurrent
    with bind_state(A):
        run(agent, SimpleNamespace(tool_calls=calls), messages, "task")
    assert provider_tools.reached == [] and reached == [], (provider_tools.reached, reached)
    agent._todo_store.write.assert_not_called()
    assert [m["tool_call_id"] for m in messages] == ["c1", "c2", "c3", "c4"]
    for message in messages:
        assert "not available to a hosted agent" in message["content"], message
        assert "HOST-PROVIDER-PRIVATE" not in message["content"]


def test_a_local_agent_loop_still_reaches_every_executor(tmp_path):
    from agent import tool_executor as te

    agent = _make_agent(tmp_path)
    agent._memory_manager = provider_tools = _ProviderTools()
    messages: list[dict] = []
    with bind_state(_local_scope()):
        te.execute_tool_calls_sequential(agent, SimpleNamespace(tool_calls=[_call("c1", "provider_recall")]),
                                         messages, "task")
    assert provider_tools.reached == ["provider_recall"]
    assert "HOST-PROVIDER-PRIVATE" in messages[0]["content"]


def test_the_inline_executor_table_refuses_on_its_own(monkeypatch):
    """Callers that use the inline table directly (outside the loop) meet the same allowlist."""
    import tools.connectors.catalog_tool as catalog
    import tools.connectors.tool as connector
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext

    reached = []
    monkeypatch.setattr(catalog, "search", lambda *a, **k: reached.append("host catalog") or {})
    monkeypatch.setattr(connector, "run_mcp_operation", lambda *a, **k: reached.append(a) or "{}")
    agent = SimpleNamespace(session_id="s", platform="desktop", connection_callback=lambda x: None,
                            enabled_toolsets=None, _memory_store=None, _memory_manager=None)
    with bind_state(A):
        for name, args in (("manage_catalog", {"action": "search"}),
                           ("manage_connections", {"action": "install", "connectors": [{"name": "linear", "mcp": True}]}),
                           ("setup_mcp", {"server": "linear"})):
            result = INLINE_TOOL_EXECUTORS[name](agent, args, InlineToolContext("t"))
            assert "not available to a hosted agent" in result, (name, result)
    assert not reached, f"an inline executor reached the host: {reached}"


def test_handle_function_call_refuses_connectors_and_bridges(monkeypatch):
    """Connector calls skip the registry (``dispatch_connector_call``); the entry point refuses them."""
    import model_tools
    from tools import connectors

    reached = []
    monkeypatch.setattr(connectors, "dispatch_connector_call", lambda *a, **k: reached.append(a) or "{}")
    with bind_state(A):
        for name in ("connectors__linear__list_issues", "tool_search", "memory"):
            result = model_tools.handle_function_call(name, {})
            assert "not available to a hosted agent" in result, (name, result)
    assert reached == []


# --------------------------------------------------------------------------------------------------
# Tools that used to reach host state, now refused (round-2 findings 9, 10, 11, 13, 14)


def test_inline_memory_writes_no_host_store():
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
    from tools.memory_tool import MemoryStore, get_memory_dir
    with bind_state(A):
        store = MemoryStore()
        store.load_from_disk()
        agent = SimpleNamespace(_memory_store=store, _memory_manager=None)
        result = INLINE_TOOL_EXECUTORS["memory"](agent, {"target": "memory", "operations": [
            {"action": "add", "content": "A-PRIVATE-ROUND2"}]}, InlineToolContext("t"))
    path = get_memory_dir() / "MEMORY.md"
    assert not path.exists() or "A-PRIVATE-ROUND2" not in path.read_text(), result
    with bind_state(B):
        other = MemoryStore()
        other.load_from_disk()
    assert "A-PRIVATE-ROUND2" not in other.memory_entries


def test_the_memory_store_itself_never_touches_the_host_file_in_a_hosted_scope():
    from tools.memory_tool import MemoryStore, get_memory_dir
    path = get_memory_dir() / "MEMORY.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("HOST-MEMORY-PRIVATE\n")
    with bind_state(A):
        store = MemoryStore()
        store.load_from_disk()
        assert store.memory_entries == [] and store.format_for_system_prompt("memory") is None
        assert store.add("memory", "A-PRIVATE")["success"] is False
    assert path.read_text() == "HOST-MEMORY-PRIVATE\n"
    local = MemoryStore()
    local.load_from_disk()
    assert local.memory_entries == ["HOST-MEMORY-PRIVATE"]


def test_session_search_reads_no_host_history():
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    from tools.registry import registry
    db = SessionDB(get_hermes_home() / "state.db")
    db.create_session("private-host-history", source="cli")
    db.append_message("private-host-history", role="user", content="HOST-HISTORY-PRIVATE")
    try:
        with bind_state(A):
            result = registry.dispatch("session_search", {"session_id": "private-host-history"})
        assert "HOST-HISTORY-PRIVATE" not in result, result
    finally:
        db.close()


def test_recall_never_opens_the_profile_state_db(monkeypatch):
    import hermes_state_registry
    from run_agent import AIAgent
    opened = []
    monkeypatch.setattr(hermes_state_registry, "acquire", lambda *a, **k: opened.append(1) or object())
    agent = SimpleNamespace(_persist_disabled=False, _session_db=None)
    with bind_state(A):
        assert AIAgent._get_session_db_for_recall(agent) is None
    assert opened == []


def test_web_extract_never_spills_to_host(provider, monkeypatch):
    import tools.web_tools as web
    from hermes_constants import get_hermes_home
    from tools.registry import registry

    async def safe(url):
        return True

    async def extract(*args):
        return [{"url": "https://example.org/round2-private", "content": "WEB-TENANT-CONTENT\n" * 1000}]
    monkeypatch.setattr(web, "async_is_safe_url", safe)
    monkeypatch.setattr(web, "_get_extract_backend", lambda: "stub")
    monkeypatch.setattr(web, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(web, "_resolve_extract_provider", lambda backend: (object(), None))
    monkeypatch.setattr(web, "_extract_safe_urls", extract)
    with bind_state(A):
        result = registry.dispatch("web_extract", {"urls": ["https://example.org/round2-private"], "char_limit": 2000})
    paths = list((get_hermes_home() / "cache" / "web").glob("*.md"))
    leaked = [str(p) for p in paths if "WEB-TENANT-CONTENT" in p.read_text()]
    assert not leaked, f"host web spill {leaked}; response={result[-500:]}"
    assert "not available to a hosted agent" in result


def test_vision_preparation_does_not_materialize_on_host(provider, monkeypatch):
    from tools import vision_tools as vision
    png = _png()
    original = c.RecordingSandboxEnvironment.execute

    def execute(self, command, **kwargs):
        original(self, command, **kwargs)
        return {"returncode": 0, "output": base64.b64encode(png).decode()}
    monkeypatch.setattr(c.RecordingSandboxEnvironment, "execute", execute)
    cache = vision._secure_cache_dir("cache/vision", "temp_vision_images")
    before = set(cache.iterdir())
    with bind_state(A), pytest.raises(vision._ImagePrepError, match="hosted agent"):
        asyncio.run(vision._prepare_image("/workspace/a.png", "t", None, validate_decode=True))
    assert set(cache.iterdir()) == before, "a sandbox image was materialized on the host"


def test_video_is_not_materialized_on_host():
    from tools import vision_tools as vision
    temp: list = []
    with bind_state(A), pytest.raises(ValueError, match="hosted agent"):
        asyncio.run(vision._materialize_video("/workspace/a.mp4", "t", temp))
    assert temp == []


# --------------------------------------------------------------------------------------------------
# The host feature switch (round-2 findings 4, 12, 15 and the earlier per-site refusals)


def test_startup_context_reads_no_host_file(tmp_path, monkeypatch):
    from agent.prompt_builder import build_context_files_prompt, load_soul_md
    from hermes_constants import get_hermes_home
    (tmp_path / "AGENTS.md").write_text("HOST-STARTUP-PRIVATE")
    (get_hermes_home() / "SOUL.md").write_text("HOST-SOUL-PRIVATE")
    with c._watch_host(tmp_path) as touched, bind_state(A):
        assert build_context_files_prompt(cwd=str(tmp_path), skip_soul=False) == ""
        assert load_soul_md() is None
    assert not touched
    assert "HOST-STARTUP-PRIVATE" in build_context_files_prompt(cwd=str(tmp_path), skip_soul=True)


def test_subdirectory_hints_read_no_host_file(tmp_path):
    from agent.subdirectory_hints import SubdirectoryHintTracker
    sub = tmp_path / "src"
    sub.mkdir()
    (sub / "AGENTS.md").write_text("HOST-INSTRUCTIONS-PRIVATE")
    with c._watch_host(tmp_path) as touched, bind_state(A):
        tracker = SubdirectoryHintTracker(str(tmp_path))
        assert tracker.check_tool_call("read_file", {"path": str(sub / "main.py")}) is None
        tracker.rebind_working_dir(str(sub))
    assert not touched, touched
    built_unscoped = SubdirectoryHintTracker(str(tmp_path))  # an agent built before its scope was bound
    with c._watch_host(sub) as touched, bind_state(A):
        assert built_unscoped.check_tool_call("read_file", {"path": str(sub / "main.py")}) is None
    assert not touched, touched
    local = SubdirectoryHintTracker(str(tmp_path))
    assert "HOST-INSTRUCTIONS-PRIVATE" in (local.check_tool_call("read_file", {"path": str(sub / "main.py")}) or "")


def test_built_in_memory_and_providers_are_not_initialized(tmp_path, monkeypatch):
    from agent import agent_init
    resolved = []
    monkeypatch.setattr(agent_init, "resolve_memory_provider", lambda *a, **k: resolved.append(a))
    agent = SimpleNamespace(enabled_toolsets=["memory"], disabled_toolsets=None, tools=[],
                            _emit_startup_warning=lambda *a: None)
    with bind_state(A):
        agent_init._init_memory(agent, {"memory": {"provider": "honcho"}}, False, "cli")
    assert agent._memory_store is None and agent._memory_manager is None and resolved == []


def test_background_review_is_not_spawned(monkeypatch):
    from run_agent import AIAgent
    spawned = []
    agent = SimpleNamespace(_spawn_background_review_now=lambda **kw: spawned.append(kw), _delegate_depth=0)
    with bind_state(A):
        AIAgent._spawn_background_review(agent, [], review_memory=True, explicit=True)
    assert spawned == []


def test_trajectory_is_not_written_on_host(tmp_path, monkeypatch):
    from agent.session_persistence import SessionPersistenceMixin
    monkeypatch.chdir(tmp_path)
    agent = SimpleNamespace(save_trajectories=True, model="stub",
                            _convert_to_trajectory_format=lambda *a: [{"from": "human", "value": "TENANT-TRANSCRIPT"}])
    with bind_state(A):
        SessionPersistenceMixin._save_trajectory(agent, [], "hello", True)
    assert not (tmp_path / "trajectory_samples.jsonl").exists()
    SessionPersistenceMixin._save_trajectory(agent, [], "hello", True)
    assert "TENANT-TRANSCRIPT" in (tmp_path / "trajectory_samples.jsonl").read_text()


def test_failed_write_bookkeeping_does_not_stat_host(provider, tmp_path):
    from agent.turn_explainers import TurnExplainersMixin
    target = tmp_path / "target.txt"
    target.write_text("HOST-ONLY")
    agent = SimpleNamespace(_turn_failed_file_mutations={})
    with c._watch_host(target) as touched, bind_state(A):
        c.call_tool("terminal", {"command": "true"}, task_id="t")
        TurnExplainersMixin._record_file_mutation_result(agent, "write_file", {"path": str(target)},
                                                         '{"error":"write failed"}', True, task_id="t")
    assert not touched, f"host bookkeeping: {touched}; recorded={agent._turn_failed_file_mutations}"
    assert agent._turn_failed_file_mutations == {}


def test_no_checkpoint_preflight_in_a_hosted_scope(tmp_path):
    """The preflight's sandbox-path check and ``CheckpointManager``'s ``checkpoints`` feature consult
    (mutation-checked in test_hosted_tool_refusals) keep a hosted call from snapshotting anything."""
    from agent.tool_executor import _ToolCallRef, _begin_tool_execution
    mgr = MagicMock(enabled=True)
    agent = SimpleNamespace(_checkpoint_mgr=mgr, quiet_mode=True, verbose_logging=False, tool_progress_callback=None,
                            tool_start_callback=None, _touch_activity=lambda *_: None)
    with bind_state(A):
        _begin_tool_execution(agent, _ToolCallRef("write_file", {"path": str(tmp_path / "x")}, "t", "c", []), None)
        _begin_tool_execution(agent, _ToolCallRef("terminal", {"command": "rm -rf /w"}, "t", "c", []), None)
    assert mgr.ensure_checkpoint.call_count == 0 and mgr.get_working_dir_for_path.call_count == 0


def test_every_host_feature_is_off_in_a_hosted_scope_and_on_locally():
    from tools.hosted_tool_policy import HOST_FEATURES, host_feature_enabled
    with bind_state(A):
        assert not any(host_feature_enabled(f) for f in HOST_FEATURES)
    with bind_state(_local_scope()):
        assert all(host_feature_enabled(f) for f in HOST_FEATURES)
    with pytest.raises(KeyError):
        host_feature_enabled("not_a_feature")


def test_no_backend_probe_environment_is_built_in_a_hosted_scope(monkeypatch):
    from agent import prompt_builder
    ran = []
    monkeypatch.setattr(prompt_builder, "_run_backend_probe", lambda *a: ran.append(a) or "")
    monkeypatch.setattr(prompt_builder, "_BACKEND_PROBE_CACHE", {})
    with bind_state(A):
        assert prompt_builder._probe_remote_backend("docker") is None
    assert ran == []
    prompt_builder._probe_remote_backend("docker")
    assert ran, "a local probe no longer runs"


# Idle suspension (round-2 finding 6): stop_idle is part of the required hosted interface.


def test_a_hosted_backend_without_stop_idle_is_refused_at_creation(provider, monkeypatch):
    monkeypatch.delattr(c.RecordingSandboxEnvironment, "stop_idle")
    with bind_state(A):
        result = c.call_tool("terminal", {"command": "true"}, task_id="t")
    assert "missing stop_idle" in json.dumps(result), result
    assert not [call for call in provider.calls if call.op == "execute"]


def test_an_idle_hosted_sandbox_is_never_destroyed(provider, monkeypatch):
    """Even if a backend lost ``stop_idle`` after creation, the reaper suspends (or fails) rather than
    destroying the workspace."""
    from tools import terminal_tool
    from tools.terminal_tool_lifecycle import _cleanup_inactive_envs
    with bind_state(A):
        c.call_tool("terminal", {"command": "true"}, task_id="t")
    terminal_tool._last_activity[sandbox_identity(A).key] = time.time() - 1000
    monkeypatch.delattr(c.RecordingSandboxEnvironment, "stop_idle")
    _cleanup_inactive_envs(lifetime_seconds=300)
    assert not [call for call in provider.calls if call.op == "cleanup"], provider.calls
