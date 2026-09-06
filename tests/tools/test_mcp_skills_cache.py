from __future__ import annotations

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
    registry.clear_runtime_state()
    yield home, workspace
    registry.clear_runtime_state()


def _publish(home: Path, session: str = "session-a"):
    from tools.mcp_skills_registry import mark_get_verified, pin_session, publish_live_catalog
    entry, body, support = _manifest()
    publish_live_catalog(home, "fixture", "config-a", [entry])
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
    from tools.mcp_skills_registry import pin_session, publish_live_catalog

    home, _workspace = _state
    entry, _body, _support = _manifest()
    publish_live_catalog(home, "fixture", "config-a", [entry])
    record = pin_session(home, "get-session")[0]
    server = SimpleNamespace(
        session=object(), _rpc_lock=asyncio.Lock(), tool_timeout=10,
        _skills_home=str(home), _skills_config_fingerprint="config-a")
    monkeypatch.setattr(discovery, "_get_connected_server_for_call", lambda _name: server)
    monkeypatch.setattr(mcp_tool_loop, "_run_on_mcp_loop", lambda factory, timeout: asyncio.run(factory()))
    monkeypatch.setattr(cache, "_get_on_session", lambda _server, _uri: _async_value(entry))
    cache._ensure_get_verified(record, home, "get-session")
    assert record["get_verified"] is True

    changed, _, _ = _manifest("changed-name")
    other = pin_session(home, "other-session")[0]
    monkeypatch.setattr(cache, "_get_on_session", lambda _server, _uri: _async_value(changed))
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
        _skills_home=str(home), _skills_config_fingerprint="different-config")
    monkeypatch.setattr(discovery, "_get_connected_server_for_call", lambda _name: server)
    monkeypatch.setattr(cache, "_get_on_session", lambda *_args: calls.append("rpc"))
    with pytest.raises(RuntimeError, match="origin no longer matches"):
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
        _skills_home=str(alias), _skills_config_fingerprint="config-a")
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
