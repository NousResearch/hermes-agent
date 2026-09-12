from __future__ import annotations

import asyncio
import hashlib
import json
import multiprocessing
import threading
import time
from pathlib import Path

import pytest

from tools.mcp_skills_protocol import SkillEntry


def _activate_in_child(home: str, record: dict, session: str, barrier) -> None:
    """Spawn-safe worker for the durable cross-process activation regression."""
    from tools.mcp_skills_registry import mark_active
    barrier.wait(timeout=10)
    mark_active(record, home, session)


def _register_lazy_in_child(home: str, session: str, entry: dict, barrier) -> None:
    from tools.mcp_skills_registry import register_lazy_entry
    if barrier is not None:
        barrier.wait(timeout=10)
    register_lazy_entry(home, session, "fixture", "config-a", entry)


def _mark_verified_in_child(home: str, session: str, record: dict, barrier) -> None:
    from tools.mcp_skills_registry import mark_get_verified
    barrier.wait(timeout=10)
    mark_get_verified(record, home, session)


def _manifest(name: str = "remote-demo", body: bytes | None = None, support: bytes = b"support"):
    body = body or f"---\nname: {name}\ndescription: Remote demo\n---\n\n# Demo\n".encode()
    root = f"skill://fixture/{name}"
    entry = SkillEntry.model_validate({
        "uri": f"{root}/SKILL.md",
        "frontmatter": {"name": name, "description": "Remote demo"},
        "resources": [
            {"uri": f"{root}/SKILL.md", "digest": "sha256:" + hashlib.sha256(body).hexdigest(), "size": len(body)},
            {"uri": f"{root}/references/info.md", "digest": "sha256:" + hashlib.sha256(support).hexdigest(), "size": len(support)},
        ],
    })
    return entry, body, support


@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    home.mkdir()
    workspace.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.chdir(workspace)
    from tools import mcp_skills_registry as registry
    from tools.terminal_tool import set_approval_callback
    registry.clear_runtime_state()
    set_approval_callback(lambda *_args, **_kwargs: "once")
    yield home, workspace
    set_approval_callback(None)
    registry.clear_runtime_state()


def _publish(home: Path, session: str = "session-a"):
    from tools.mcp_skills_registry import (
        mark_get_verified, pin_session, publish_live_catalog, server_config_fingerprint,
    )
    entry, body, support = _manifest()
    publish_live_catalog(
        home, "fixture", server_config_fingerprint({"skills": {"enabled": True}}), [entry])
    records = pin_session(home, session)
    mark_get_verified(records[0], home, session)
    return records[0], body, support


def test_session_snapshot_is_pinned_and_cold_resume_is_origin_bound(_state):
    home, _workspace = _state
    from tools.mcp_skills_registry import active_skills, clear_runtime_state, mark_active, pin_session, publish_live_catalog

    first, _body, _support = _publish(home)
    changed, _, _ = _manifest("new-demo")
    publish_live_catalog(home, "fixture", "config-a", [changed])
    assert [row["uri"] for row in pin_session(home, "session-a")] == [first["uri"]]
    assert [row["uri"] for row in pin_session(home, "session-b")] == [changed.uri]

    mark_active(first, home, "session-a")

    clear_runtime_state()
    resumed = pin_session(home, "session-a")
    assert resumed[0]["uri"] == first["uri"]
    assert resumed[0]["connected"] is False
    assert active_skills(home, "session-a")[0]["uri"] == first["uri"]
    snapshot_files = list((home / "cache" / "mcp-skills" / "session-snapshots").glob("*.json"))
    assert len(snapshot_files) == 2
    assert "session-a" not in snapshot_files[0].name


def test_verified_cache_rehashes_and_rejects_corruption(_state, monkeypatch):
    home, _workspace = _state
    from tools import mcp_skills_cache as cache
    record, body, _support = _publish(home)
    resource = record["resources"][0]
    calls = []
    monkeypatch.setattr(cache, "_fetch", lambda r, s, home=None: (calls.append(s["uri"]) or body, "text/markdown", True))

    raw, mime, is_text, path, scan = cache.get_verified_resource(record, resource, home)
    assert (raw, mime, is_text) == (body, "text/markdown", True)
    assert scan["scope"] == "single_resource_text"
    assert calls == [resource["uri"]]
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="size mismatch|digest mismatch"):
        cache.get_verified_resource(record, resource, home)
    assert calls == [resource["uri"]]


def test_materialization_is_non_overwriting_and_keeps_per_file_provenance(_state, monkeypatch):
    home, workspace = _state
    from tools import mcp_skills_cache as cache
    monkeypatch.setattr(cache, "resolve_agent_cwd", lambda: workspace)
    record, _body, support = _publish(home)
    resource = record["resources"][1]

    created = cache.materialize_resource(record, resource, support)
    target = Path(created["path"])
    assert created["status"] == "created"
    assert target.read_bytes() == support
    assert target.stat().st_mode & 0o111 == 0
    provenance = json.loads(Path(created["provenance_path"]).read_text())
    assert provenance["resource_uri"] == resource["uri"]

    target.write_bytes(b"user edit")
    preserved = cache.materialize_resource(record, resource, support)
    assert preserved["status"] == "user_modified"
    assert target.read_bytes() == b"user edit"


def test_materialization_rejects_escape_and_remote_backend(_state, monkeypatch):
    home, workspace = _state
    from tools import mcp_skills_cache as cache
    monkeypatch.setattr(cache, "resolve_agent_cwd", lambda: workspace)
    record, _body, support = _publish(home)
    resource = record["resources"][1]
    with pytest.raises(ValueError, match="normalized path"):
        cache.materialize_resource(record, resource, support, "../escape")
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    with pytest.raises(RuntimeError, match="unavailable for terminal backend"):
        cache.materialize_resource(record, resource, support)


@pytest.mark.parametrize("config_key", ["create_dir", "external_dirs"])
def test_materialization_rejects_absent_configured_discovery_root(_state, monkeypatch, config_key):
    home, workspace = _state
    from agent import skill_utils
    from tools import mcp_skills_cache as cache

    future_root = workspace / ".hermes" / "mcp-skills" / "materialized"
    value = str(future_root) if config_key == "create_dir" else [str(future_root)]
    (home / "config.yaml").write_text(json.dumps({"skills": {config_key: value}}))
    skill_utils._external_dirs_cache_clear()
    monkeypatch.setattr(cache, "resolve_agent_cwd", lambda: workspace)
    record, _body, support = _publish(home)
    resource = record["resources"][1]

    assert not future_root.exists()
    with pytest.raises(ValueError, match="local skill discovery root"):
        cache.materialize_resource(record, resource, support)
    assert not future_root.exists()


def test_materialization_rejects_cwd_nested_in_actual_project_skill_root(_state, monkeypatch):
    home, workspace = _state
    from agent import skill_utils
    from tools import mcp_skills_cache as cache

    project = workspace / "project"
    project_skills = project / ".hermes" / "skills"
    nested_cwd = project_skills / "maintenance"
    nested_cwd.mkdir(parents=True)
    (project / ".git").mkdir()
    (home / "config.yaml").write_text(json.dumps({"skills": {"trusted_project_dirs": [str(project)]}}))
    skill_utils._external_dirs_cache_clear()
    monkeypatch.setattr(cache, "resolve_agent_cwd", lambda: nested_cwd)
    monkeypatch.chdir(nested_cwd)
    record, _body, support = _publish(home)

    assert skill_utils.get_project_skills_dirs() == [project_skills]
    with pytest.raises(ValueError, match="local skill discovery root"):
        cache.materialize_resource(record, record["resources"][1], support)
    assert not (nested_cwd / ".hermes").exists()


def test_managed_filesystem_operations_fail_before_creation_without_descriptor_safety(_state, monkeypatch):
    home, _workspace = _state
    from tools import mcp_skills_fs as managed_fs

    monkeypatch.setattr(managed_fs, "_descriptor_safety_available", lambda: False)
    with pytest.raises(RuntimeError, match="POSIX root-descriptor"):
        managed_fs.secure_atomic_bytes(home, "cache/mcp-skills/content", b"no")
    assert not (home / "cache").exists()

    # No managed state exists in an MCP-disabled session, so ordinary prompt
    # assembly and compression continuation remain no-ops on this platform.
    from tools import mcp_skills_registry as registry
    monkeypatch.setattr(registry, "_descriptor_safety_available", lambda: False)
    assert registry.pin_session(home, "ordinary") == ()
    assert registry.active_skills(home, "ordinary") == []
    registry.continue_session(home, "ordinary", "ordinary-compressed")
    assert not (home / "cache").exists()


def test_remote_skill_view_uses_get_verified_list_manifest(_state, monkeypatch):
    home, _workspace = _state
    from tools import mcp_skills_cache as cache
    from tools.mcp_skills_registry import qualified_name
    from tools.mcp_skills_view import serve_remote_skill
    record, body, support = _publish(home)

    payloads = {record["resources"][0]["uri"]: (body, "text/markdown", True),
                record["resources"][1]["uri"]: (support, "text/markdown", True)}
    monkeypatch.setattr(cache, "_fetch", lambda _record, resource, home=None: payloads[resource["uri"]])
    loaded = json.loads(serve_remote_skill(
        qualified_name(record), file_path=None, task_id=None, session_id="session-a",
        materialize=False, destination=None))
    assert loaded["success"] is True
    assert loaded["origin"] == {"type": "mcp", "server": "fixture", "uri": record["uri"]}
    assert "REMOTE MCP SKILL" in loaded["content"]
    assert loaded["skill_dir"] is None
    assert loaded["permissions_inert"] is True
    assert loaded["linked_files"] == {"references": ["references/info.md"]}

    support_loaded = json.loads(serve_remote_skill(
        qualified_name(record), file_path="references/info.md", task_id=None, session_id="session-a",
        materialize=False, destination=None))
    assert support_loaded["content"] == "support"


async def _async_value(value):
    return value


def test_skills_get_must_match_pinned_list_manifest(_state, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from tools import mcp_skills_cache as cache
    from tools import mcp_tool_discovery as discovery
    from tools import mcp_tool_loop
    from tools.mcp_skills_registry import pin_session, publish_live_catalog, server_config_fingerprint

    home, _workspace = _state
    entry, _body, _support = _manifest()
    config = {"skills": {"enabled": True}}
    fingerprint = server_config_fingerprint(config)
    publish_live_catalog(home, "fixture", fingerprint, [entry])
    record = pin_session(home, "get-session")[0]
    server = SimpleNamespace(
        session=object(), _rpc_lock=asyncio.Lock(), tool_timeout=10,
        _skills_home=str(home), _skills_config_fingerprint=fingerprint, _config=config,
        _skills_local_opt_in=True, _skills_advertised=True,
        _skills_list_completed=True, _skills_connected=True)
    server._skills_ready_session = server.session
    server._skills_ready_epoch = 0
    monkeypatch.setattr(discovery, "_get_connected_server_for_call", lambda _name: server)
    monkeypatch.setattr(mcp_tool_loop, "_run_on_mcp_loop", lambda factory, timeout: asyncio.run(factory()))
    monkeypatch.setattr(cache, "_get_on_session", lambda _lock, _session, _uri: _async_value(entry))
    cache._ensure_get_verified(record, home, "get-session")
    assert record["get_verified"] is True

    changed, _, _ = _manifest("changed-name")
    other = pin_session(home, "other-session")[0]
    monkeypatch.setattr(cache, "_get_on_session", lambda _lock, _session, _uri: _async_value(changed))
    with pytest.raises(ValueError, match="does not match"):
        cache._ensure_get_verified(other, home, "other-session")


def test_support_file_requires_parent_activation(_state, monkeypatch):
    home, _workspace = _state
    from tools import mcp_skills_cache as cache
    from tools.mcp_skills_registry import qualified_name
    from tools.mcp_skills_view import serve_remote_skill
    record, _body, support = _publish(home)
    cache_path = home / "cache" / "mcp-skills" / "active-origins"
    from tools import mcp_skills_registry as registry
    registry.clear_runtime_state()
    result = json.loads(serve_remote_skill(
        qualified_name(record), file_path="references/info.md", task_id=None,
        session_id="session-a", materialize=False, destination=None))
    assert result["success"] is False
    assert "main SKILL.md first" in result["error"]
    assert not cache_path.exists()


def test_cache_and_registry_reject_symlinked_parent_escape(_state, monkeypatch):
    home, workspace = _state
    outside = workspace / "outside"
    outside.mkdir()
    (home / "cache").symlink_to(outside, target_is_directory=True)
    from tools import mcp_skills_registry as registry
    entry, _body, _support = _manifest()
    registry.publish_live_catalog(home, "fixture", "config-a", [entry])
    with pytest.raises(ValueError, match="non-directory|symlink"):
        registry.pin_session(home, "escaped")
    assert list(outside.iterdir()) == []


def test_live_origin_must_match_before_any_rpc(_state, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from tools import mcp_skills_cache as cache
    from tools import mcp_tool_discovery as discovery
    home, _workspace = _state
    record, _body, _support = _publish(home, "origin-check")
    record["get_verified"] = False
    calls = []
    server = SimpleNamespace(
        session=object(), _rpc_lock=asyncio.Lock(), tool_timeout=10,
        _skills_home=str(home), _skills_config_fingerprint="different-config",
        _config={"skills": {"enabled": True}}, _skills_local_opt_in=True,
        _skills_advertised=True, _skills_list_completed=True, _skills_connected=True)
    server._skills_ready_session = server.session
    server._skills_ready_epoch = 0
    monkeypatch.setattr(discovery, "_get_connected_server_for_call", lambda _name: server)
    monkeypatch.setattr(cache, "_get_on_session", lambda *_args: calls.append("rpc"))
    with pytest.raises(RuntimeError, match="profile/configuration"):
        cache._ensure_get_verified(record, home, "origin-check")
    assert calls == []


def test_symlinked_profile_home_is_rejected_before_any_rpc(_state, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from tools import mcp_skills_cache as cache
    from tools import mcp_tool_discovery as discovery
    home, workspace = _state
    record, _body, _support = _publish(home, "symlink-origin")
    record["get_verified"] = False
    alias = workspace / "profile-alias"
    alias.symlink_to(home, target_is_directory=True)
    calls = []
    server = SimpleNamespace(
        session=object(), _rpc_lock=asyncio.Lock(), tool_timeout=10,
        _skills_home=str(alias), _skills_config_fingerprint="config-a",
        _config={"skills": {"enabled": True}}, _skills_local_opt_in=True,
        _skills_advertised=True, _skills_list_completed=True, _skills_connected=True)
    server._skills_ready_session = server.session
    server._skills_ready_epoch = 0
    monkeypatch.setattr(discovery, "_get_connected_server_for_call", lambda _name: server)
    monkeypatch.setattr(cache, "_get_on_session", lambda *_args: calls.append("rpc"))
    with pytest.raises(ValueError, match="managed root"):
        cache._ensure_get_verified(record, alias, "symlink-origin")
    assert calls == []


def test_materialization_is_collision_free_and_sidecar_owned(_state, monkeypatch):
    home, workspace = _state
    from tools import mcp_skills_cache as cache
    monkeypatch.setattr(cache, "resolve_agent_cwd", lambda: workspace)
    first, _body, support = _publish(home, "first")
    from tools.mcp_skills_protocol import SkillEntry
    second_raw = _manifest()[0].model_dump(mode="json")
    second_raw["uri"] = second_raw["uri"].replace(
        "skill://fixture/remote-demo", "skill://fixture/team-b/remote-demo")
    for resource in second_raw["resources"]:
        resource["uri"] = resource["uri"].replace(
            "skill://fixture/remote-demo", "skill://fixture/team-b/remote-demo")
    second_entry = SkillEntry.model_validate(second_raw)
    from tools.mcp_skills_registry import _entry_record
    second = _entry_record("fixture", "config-a", second_entry, connected=True, get_verified=True)
    a = cache.materialize_resource(first, first["resources"][1], support)
    b = cache.materialize_resource(second, second["resources"][1], support)
    assert a["path"] != b["path"]
    sidecar = Path(a["provenance_path"])
    data = json.loads(sidecar.read_text())
    data["skill_uri"] = second["uri"]
    sidecar.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="does not own"):
        cache.materialize_resource(first, first["resources"][1], support)


def test_active_state_transaction_serializes_concurrent_writers(_state, monkeypatch):
    home, _workspace = _state
    from tools import mcp_skills_registry as registry
    one, _, _ = _manifest("one")
    two, _, _ = _manifest("two")
    registry.publish_live_catalog(home, "fixture", "config-a", [one, two])
    records = registry.pin_session(home, "race")
    entered = 0
    maximum = 0
    guard = threading.Lock()
    original = registry._write_active_locked

    def observed(*args, **kwargs):
        nonlocal entered, maximum
        with guard:
            entered += 1
            maximum = max(maximum, entered)
        time.sleep(0.03)
        try:
            return original(*args, **kwargs)
        finally:
            with guard:
                entered -= 1

    monkeypatch.setattr(registry, "_write_active_locked", observed)
    start = threading.Barrier(3)
    threads = [threading.Thread(target=lambda row=row: (start.wait(), registry.mark_active(row, home, "race")))
               for row in records]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()
    assert maximum == 1
    registry.clear_runtime_state()
    assert {row["uri"] for row in registry.active_skills(home, "race")} == {one.uri, two.uri}


def test_active_state_transaction_merges_cross_process_writers_and_cold_reload(_state):
    home, _workspace = _state
    from tools import mcp_skills_registry as registry
    one, _, _ = _manifest("process-one")
    two, _, _ = _manifest("process-two")
    registry.publish_live_catalog(home, "fixture", "config-a", [one, two])
    records = registry.pin_session(home, "process-race")
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    children = [context.Process(
        target=_activate_in_child, args=(str(home), row, "process-race", barrier)) for row in records]
    for child in children:
        child.start()
    for child in children:
        child.join(timeout=15)
        assert child.exitcode == 0
    registry.clear_runtime_state()
    assert {row["uri"] for row in registry.active_skills(home, "process-race")} == {one.uri, two.uri}


@pytest.mark.parametrize("operation", ["point_get", "listed_get", "resource_read"])
@pytest.mark.parametrize(
    "mutation", ["server", "session", "profile", "config", "epoch", "ready_session", "ready_epoch"])
def test_remote_results_are_discarded_when_complete_source_token_changes(
        _state, monkeypatch, operation, mutation):
    import asyncio
    from types import SimpleNamespace
    from tools import mcp_skills_cache as cache
    from tools import mcp_tool_discovery as discovery
    from tools import mcp_tool_loop
    from tools.mcp_skills_registry import pin_session, publish_live_catalog, server_config_fingerprint

    home, _workspace = _state
    entry, body, _support = _manifest()
    config = {"skills": {"enabled": True}, "command": "stable"}
    fingerprint = server_config_fingerprint(config)
    publish_live_catalog(home, "fixture", fingerprint, [entry] if operation != "point_get" else [])
    sid = f"token-{operation}-{mutation}"
    record = pin_session(home, sid)[0] if operation != "point_get" else None
    if record is not None:
        record["get_verified"] = operation == "resource_read"
    server = SimpleNamespace(
        session=object(), _rpc_lock=asyncio.Lock(), tool_timeout=10, _skills_epoch=7,
        _skills_home=str(home), _skills_config_fingerprint=fingerprint, _config=config,
        _skills_local_opt_in=True, _skills_advertised=True,
        _skills_list_completed=True, _skills_connected=True)
    server._skills_ready_session = server.session
    server._skills_ready_epoch = server._skills_epoch
    current = {"server": server}
    monkeypatch.setattr(discovery, "_get_connected_server_for_call", lambda _name: current["server"])
    monkeypatch.setattr(cache, "_get_on_session", lambda *_args: _async_value(entry))
    monkeypatch.setattr(cache, "_read_on_session", lambda *_args: _async_value(
        SimpleNamespace(contents=[SimpleNamespace(uri=entry.uri, text=body.decode(), blob=None,
                                                  mimeType="text/markdown")])))

    def settle(factory, timeout):
        result = asyncio.run(factory())
        if mutation == "server":
            current["server"] = SimpleNamespace(**vars(server))
        elif mutation == "session":
            server.session = object()
        elif mutation == "profile":
            server._skills_home = str(home / "other-profile")
        elif mutation == "config":
            server._config = {"skills": {"enabled": True}, "command": "changed"}
            server._skills_config_fingerprint = server_config_fingerprint(server._config)
        elif mutation == "epoch":
            server._skills_epoch += 1
        elif mutation == "ready_session":
            server._skills_ready_session = object()
        else:
            server._skills_ready_epoch += 1
        return result

    monkeypatch.setattr(mcp_tool_loop, "_run_on_mcp_loop", settle)
    with pytest.raises((RuntimeError, ValueError), match="changed|extension-ready|managed root"):
        if operation == "point_get":
            cache.register_skill_uri("fixture", entry.uri, home, sid)
        elif operation == "listed_get":
            assert record is not None
            cache._ensure_get_verified(record, home, sid)
        else:
            assert record is not None
            cache._fetch(record, record["resources"][0], home)
    if operation == "point_get":
        assert pin_session(home, sid) == ()
    elif operation == "listed_get":
        assert record is not None and record["get_verified"] is False


@pytest.mark.parametrize("operation", ["point_get", "listed_get", "resource_read"])
def test_reconnect_generation_rejects_remote_skill_rpc_before_skills_discovery(
        _state, monkeypatch, operation):
    from types import SimpleNamespace

    from tools import mcp_skills_cache as cache
    from tools import mcp_tool_discovery as discovery
    from tools import mcp_tool_loop
    from tools.mcp_skills_protocol import SKILLS_EXTENSION, SkillsListResult
    from tools.mcp_skills_registry import (
        pin_session, publish_live_catalog, server_config_fingerprint,
    )
    from tools.mcp_tool import MCPServerTask

    home, _workspace = _state
    entry, _body, _support = _manifest()
    config = {"skills": {"enabled": True}, "command": "fixture"}
    fingerprint = server_config_fingerprint(config)
    publish_live_catalog(home, "fixture", fingerprint, [] if operation == "point_get" else [entry])
    session_id = f"reconnect-{operation}"
    record = pin_session(home, session_id)[0] if operation != "point_get" else None
    if record is not None:
        record["get_verified"] = operation == "resource_read"

    class NewSession:
        def __init__(self):
            self.methods = []

        async def send_request(self, request, adapter):
            self.methods.append(request.method)
            return SkillsListResult(skills=[entry])

    old_session = object()
    new_session = NewSession()
    server = MCPServerTask("fixture")
    server._config = config
    server.session = old_session
    server._skills_home = str(home)
    server._skills_config_fingerprint = fingerprint
    server._skills_local_opt_in = True
    server._skills_advertised = True
    server._skills_list_completed = True
    server._skills_connected = True
    server._skills_epoch = 4
    reached = asyncio.Event()
    release = asyncio.Event()

    async def negotiate(_session, _timeout):
        return SimpleNamespace(capabilities=SimpleNamespace(extensions={SKILLS_EXTENSION: {}}))

    async def paused_discover_tools():
        reached.set()
        await release.wait()

    async def no_wait():
        return "shutdown"

    server._negotiate_session = negotiate
    server._discover_tools = paused_discover_tools
    server._wait_for_lifecycle_event = no_wait
    monkeypatch.setattr(discovery, "_get_connected_server_for_call", lambda _name: server)
    rpc_calls = []
    monkeypatch.setattr(
        mcp_tool_loop, "_run_on_mcp_loop",
        lambda *_args, **_kwargs: rpc_calls.append("rpc"),
    )

    async def run_reconnect():
        task = asyncio.create_task(server._serve_session(new_session, 1))
        await reached.wait()
        assert server.session is new_session
        assert server._skills_epoch == 5
        with pytest.raises(RuntimeError, match="extension-ready"):
            if operation == "point_get":
                cache.register_skill_uri("fixture", entry.uri, home, session_id)
            elif operation == "listed_get":
                assert record is not None
                cache._ensure_get_verified(record, home, session_id)
            else:
                assert record is not None
                cache._fetch(record, record["resources"][0], home)
        assert rpc_calls == []
        release.set()
        await task

    asyncio.run(run_reconnect())
    assert new_session.methods == ["skills/list"]


def test_snapshot_reads_observe_cross_process_quarantine(_state):
    home, _workspace = _state
    from tools import mcp_skills_registry as registry
    child, child_body, _ = _manifest("child")
    child_raw = child.model_dump(mode="json")
    child_root = "skill://fixture/parent/child"
    old_root = "skill://fixture/child"
    child_raw["uri"] = child_raw["uri"].replace(old_root, child_root)
    for resource in child_raw["resources"]:
        resource["uri"] = resource["uri"].replace(old_root, child_root)
    parent_body = b"---\nname: parent\ndescription: Remote demo\n---\n\n# Parent\n"
    parent = SkillEntry.model_validate({
        "uri": "skill://fixture/parent/SKILL.md",
        "frontmatter": {"name": "parent", "description": "Remote demo"},
        "resources": [
            {"uri": "skill://fixture/parent/SKILL.md",
             "digest": "sha256:" + hashlib.sha256(parent_body).hexdigest(), "size": len(parent_body)},
            {"uri": child_raw["uri"], "digest": "sha256:" + hashlib.sha256(child_body).hexdigest(),
             "size": len(child_body)},
        ],
    })
    registry.publish_live_catalog(home, "fixture", "config-a", [parent])
    assert [row["frontmatter"]["name"] for row in registry.pin_session(home, "quarantine-race")] == ["parent"]
    child_raw["resources"][0]["digest"] = "sha256:" + "0" * 64
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_register_lazy_in_child,
        args=(str(home), "quarantine-race", child_raw, None))
    process.start()
    process.join(timeout=15)
    assert process.exitcode == 0
    assert registry.pin_session(home, "quarantine-race") == ()
    blocked = registry.resolve_catalog_resource("fixture", child_raw["uri"], home, "quarantine-race")
    assert blocked is not None and blocked["metadata_allowed"] is False


def test_cross_process_lazy_writers_and_get_verified_merge_losslessly(_state):
    home, _workspace = _state
    from tools import mcp_skills_registry as registry
    base, _, _ = _manifest("base")
    one, _, _ = _manifest("lazy-one")
    two, _, _ = _manifest("lazy-two")
    registry.publish_live_catalog(home, "fixture", "config-a", [base])
    base_record = registry.pin_session(home, "lazy-race")[0]
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(3)
    children = [
        context.Process(target=_register_lazy_in_child,
                        args=(str(home), "lazy-race", item.model_dump(mode="json"), barrier))
        for item in (one, two)
    ]
    children.append(context.Process(
        target=_mark_verified_in_child,
        args=(str(home), "lazy-race", base_record, barrier)))
    for child_process in children:
        child_process.start()
    for child_process in children:
        child_process.join(timeout=15)
        assert child_process.exitcode == 0
    rows = registry.pin_session(home, "lazy-race")
    assert {row["uri"] for row in rows} == {base.uri, one.uri, two.uri}
    assert next(row for row in rows if row["uri"] == base.uri)["get_verified"] is True
