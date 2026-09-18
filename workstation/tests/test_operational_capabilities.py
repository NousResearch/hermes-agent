"""Rigorously test Progressive Operational Compilation / Operational Capability Runtime."""
import json
import os
from pathlib import Path
import pytest

from workstation.artifacts import ArtifactStore
from workstation.operational_capabilities import (
    CapabilityCycleError,
    CapabilityDependency,
    CapabilityDepthExceededError,
    CapabilityDriftError,
    CapabilityLifecycle,
    CapabilityNotFoundError,
    CapabilityResolver,
    OperationalCapability,
    OperationalCapabilityRegistry,
    learn_operational_capability,
    matches_version,
)
from workstation.operational_kernel import OperationalKernel, interpolate_variables


@pytest.fixture
def clean_workstation(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    artifacts = ArtifactStore(root_dir=home / "artifacts")
    registry = OperationalCapabilityRegistry(artifacts=artifacts, root=home / "operational_capabilities")
    resolver = CapabilityResolver(registry=registry)
    kernel = OperationalKernel(registry=registry, resolver=resolver, artifacts=artifacts)
    return {
        "home": home,
        "artifacts": artifacts,
        "registry": registry,
        "resolver": resolver,
        "kernel": kernel,
    }


def test_operational_capability_roundtrip():
    dep = CapabilityDependency(capability_id="cap_login", version_constraint="^1.0.0", input_mappings={"user": "$inputs.username"})
    cap = OperationalCapability(
        id="cap_search",
        name="Search Operation",
        version="1.2.0",
        route="native_browser",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        dependencies=[dep],
        implementation={"steps": [{"tool": "browser_type", "args": {"text": "$inputs.query"}}]},
        lifecycle=CapabilityLifecycle.DISCOVERED,
        semantic_fingerprint="sem_fp_123",
    )
    d = cap.to_dict()
    assert d["id"] == "cap_search"
    assert d["dependencies"][0]["capability_id"] == "cap_login"
    assert d["lifecycle"] == "discovered"

    rebuilt = OperationalCapability.from_dict(d)
    assert rebuilt.id == "cap_search"
    assert rebuilt.version == "1.2.0"
    assert len(rebuilt.dependencies) == 1
    assert rebuilt.dependencies[0].capability_id == "cap_login"
    assert rebuilt.lifecycle == CapabilityLifecycle.DISCOVERED


def test_semver_matching():
    assert matches_version("1.2.3", "*")
    assert matches_version("1.2.3", None)
    assert matches_version("1.2.3", "1.2.3")
    assert matches_version("1.2.3", "^1.0.0")
    assert not matches_version("2.0.0", "^1.0.0")
    assert matches_version("1.5.0", ">=1.2.0")
    assert not matches_version("1.1.0", ">=1.2.0")


def test_registry_registration_and_promotion(clean_workstation):
    reg = clean_workstation["registry"]
    cap = OperationalCapability(
        id="cap_export",
        name="Export CSV",
        version="1.0.0",
        route="filesystem",
        implementation={"steps": [{"primitive": "fs_write", "args": {"path": "out.csv", "content": "a,b\n1,2"}}]},
        semantic_fingerprint="sem_export_1",
    )
    reg.register(cap)

    fetched = reg.get("cap_export")
    assert fetched is not None
    assert fetched.name == "Export CSV"
    assert fetched.lifecycle == CapabilityLifecycle.DISCOVERED

    # Validate once -> VALIDATED
    reg.record_validation("cap_export", {"test": "ok_1"}, auto_promote_threshold=2)
    assert reg.get("cap_export").lifecycle == CapabilityLifecycle.VALIDATED

    # Validate second time -> auto promotes to PROMOTED
    reg.record_validation("cap_export", {"test": "ok_2"}, auto_promote_threshold=2)
    assert reg.get("cap_export").lifecycle == CapabilityLifecycle.PROMOTED


def test_dependency_resolution_and_linearization(clean_workstation):
    reg = clean_workstation["registry"]
    resolver = clean_workstation["resolver"]

    # Leaf capability
    cap_c = OperationalCapability(
        id="cap_c",
        name="C",
        version="1.0.0",
        route="filesystem",
        implementation={"steps": []},
        lifecycle=CapabilityLifecycle.PROMOTED,
    )
    # Intermediate capability depending on C
    cap_b = OperationalCapability(
        id="cap_b",
        name="B",
        version="1.0.0",
        route="filesystem",
        dependencies=[CapabilityDependency(capability_id="cap_c")],
        implementation={"steps": []},
        lifecycle=CapabilityLifecycle.PROMOTED,
    )
    # Root capability depending on B
    cap_a = OperationalCapability(
        id="cap_a",
        name="A",
        version="1.0.0",
        route="filesystem",
        dependencies=[CapabilityDependency(capability_id="cap_b")],
        implementation={"steps": []},
        lifecycle=CapabilityLifecycle.PROMOTED,
    )

    reg.register(cap_c)
    reg.register(cap_b)
    reg.register(cap_a)

    resolved = resolver.resolve("cap_a")
    assert resolved.id == "cap_a"

    linear = resolver.linearize("cap_a")
    order = [c.id for c in linear]
    assert order == ["cap_c", "cap_b", "cap_a"]


def test_cycle_detection(clean_workstation):
    reg = clean_workstation["registry"]
    resolver = clean_workstation["resolver"]

    # cap_1 -> cap_2 -> cap_1 (cycle!)
    cap_1 = OperationalCapability(
        id="cap_1",
        name="1",
        version="1.0.0",
        route="filesystem",
        dependencies=[CapabilityDependency(capability_id="cap_2")],
    )
    cap_2 = OperationalCapability(
        id="cap_2",
        name="2",
        version="1.0.0",
        route="filesystem",
        dependencies=[CapabilityDependency(capability_id="cap_1")],
    )
    reg.register(cap_1)
    reg.register(cap_2)

    with pytest.raises(CapabilityCycleError) as excinfo:
        resolver.resolve("cap_1")
    assert "Cycle detected" in str(excinfo.value)


def test_depth_exceeded_error(clean_workstation):
    reg = clean_workstation["registry"]
    # create resolver with max_depth=3
    resolver = CapabilityResolver(registry=reg, max_depth=3)

    for i in range(5):
        dep = [CapabilityDependency(capability_id=f"deep_{i+1}")] if i < 4 else []
        reg.register(OperationalCapability(
            id=f"deep_{i}",
            name=f"Deep {i}",
            version="1.0.0",
            route="filesystem",
            dependencies=dep,
        ))

    with pytest.raises(CapabilityDepthExceededError):
        resolver.resolve("deep_0")


def test_filesystem_primitives_execution(clean_workstation, tmp_path):
    kernel = clean_workstation["kernel"]
    base_dir = tmp_path / "sandbox"
    base_dir.mkdir()

    # 1. mkdir
    res_mkdir = kernel.execute_primitive("mkdir", {"path": "sub", "base_dir": str(base_dir)})
    assert res_mkdir["success"]

    # 2. write
    res_write = kernel.execute_primitive("write", {"path": "sub/hello.txt", "content": "Hello World", "base_dir": str(base_dir)})
    assert res_write["success"]

    # 3. stat
    res_stat = kernel.execute_primitive("stat", {"path": "sub/hello.txt", "base_dir": str(base_dir)})
    assert res_stat["exists"]
    assert res_stat["size"] == 11

    # 4. read
    content = kernel.execute_primitive("read", {"path": "sub/hello.txt", "base_dir": str(base_dir)})
    assert content == "Hello World"

    # 5. patch
    res_patch = kernel.execute_primitive("patch", {"path": "sub/hello.txt", "target_content": "World", "replacement_content": "Hermes", "base_dir": str(base_dir)})
    assert res_patch["success"]
    assert kernel.execute_primitive("read", {"path": "sub/hello.txt", "base_dir": str(base_dir)}) == "Hello Hermes"

    # 6. copy
    res_copy = kernel.execute_primitive("copy", {"src": "sub/hello.txt", "dst": "sub/copy.txt", "base_dir": str(base_dir)})
    assert res_copy["success"]

    # 7. hash
    h = kernel.execute_primitive("hash_file", {"path": "sub/copy.txt", "base_dir": str(base_dir)})
    assert len(h) == 64

    # 8. move
    res_move = kernel.execute_primitive("move", {"src": "sub/copy.txt", "dst": "sub/moved.txt", "base_dir": str(base_dir)})
    assert res_move["success"]
    assert not (base_dir / "sub" / "copy.txt").exists()
    assert (base_dir / "sub" / "moved.txt").exists()


def test_composition_and_atomic_sub_capability_reuse(clean_workstation, tmp_path):
    kernel = clean_workstation["kernel"]
    reg = clean_workstation["registry"]
    base_dir = tmp_path / "composition_sandbox"
    base_dir.mkdir()

    # Base capability: creates a header file
    cap_header = OperationalCapability(
        id="cap_create_header",
        name="Create Header",
        version="1.0.0",
        route="filesystem",
        implementation={
            "steps": [
                {
                    "primitive": "fs_write",
                    "args": {
                        "path": "$inputs.header_file",
                        "content": "# Project Header: $inputs.title\n",
                        "base_dir": str(base_dir),
                    },
                }
            ],
            "output": {"created_path": "$inputs.header_file"},
        },
        lifecycle=CapabilityLifecycle.PROMOTED,
    )
    reg.register(cap_header)

    # Composite 1: creates README by reusing cap_create_header then appending body
    cap_readme = OperationalCapability(
        id="cap_create_readme",
        name="Create Readme",
        version="1.0.0",
        route="filesystem",
        dependencies=[
            CapabilityDependency(
                capability_id="cap_create_header",
                input_mappings={"header_file": "$inputs.file", "title": "$inputs.title"},
                output_alias="header",
            )
        ],
        implementation={
            "steps": [
                {
                    "primitive": "fs_patch",
                    "args": {
                        "path": "$deps.header.output.created_path",
                        "target_content": "# Project Header: $inputs.title\n",
                        "replacement_content": "# Project Header: $inputs.title\n\nReadme Content Here.\n",
                        "base_dir": str(base_dir),
                    },
                }
            ],
            "output": {"status": "readme_created", "file": "$inputs.file"},
        },
        lifecycle=CapabilityLifecycle.PROMOTED,
    )
    reg.register(cap_readme)

    # Composite 2: creates CHANGELOG by reusing cap_create_header
    cap_changelog = OperationalCapability(
        id="cap_create_changelog",
        name="Create Changelog",
        version="1.0.0",
        route="filesystem",
        dependencies=[
            CapabilityDependency(
                capability_id="cap_create_header",
                input_mappings={"header_file": "$inputs.file", "title": "$inputs.title"},
                output_alias="header",
            )
        ],
        implementation={
            "steps": [
                {
                    "primitive": "fs_patch",
                    "args": {
                        "path": "$deps.header.output.created_path",
                        "target_content": "# Project Header: $inputs.title\n",
                        "replacement_content": "# Project Header: $inputs.title\n\n- Initial release.\n",
                        "base_dir": str(base_dir),
                    },
                }
            ],
            "output": {"status": "changelog_created", "file": "$inputs.file"},
        },
        lifecycle=CapabilityLifecycle.PROMOTED,
    )
    reg.register(cap_changelog)

    # Execute Composite 1
    res1 = kernel.execute_capability("cap_create_readme", {"file": "README.md", "title": "My Doc"})
    assert res1["success"]
    readme_text = (base_dir / "README.md").read_text(encoding="utf-8")
    assert "Readme Content Here." in readme_text

    # Execute Composite 2 (reuses cap_create_header atomically)
    res2 = kernel.execute_capability("cap_create_changelog", {"file": "CHANGELOG.md", "title": "My Changes"})
    assert res2["success"]
    changelog_text = (base_dir / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "- Initial release." in changelog_text

    # cap_create_header was executed twice deterministically, no LLM calls
    header_cap = reg.get("cap_create_header")
    assert header_cap.success_count >= 2
    assert header_cap.savings["llm_calls_saved"] >= 2


def test_drift_quarantine_and_reasoning_handoff(clean_workstation, tmp_path):
    kernel = clean_workstation["kernel"]
    reg = clean_workstation["registry"]
    base_dir = tmp_path / "drift_sandbox"
    base_dir.mkdir()

    # Capability has postcondition that file must contain 'ExpectedToken'
    cap_drift = OperationalCapability(
        id="cap_drift_demo",
        name="Drift Demo",
        version="1.0.0",
        route="filesystem",
        postconditions=[
            json.dumps({"type": "file_contains", "path": "test.txt", "text": "ExpectedToken", "base_dir": str(base_dir)})
        ],
        implementation={
            "steps": [
                {
                    "primitive": "fs_write",
                    "args": {
                        "path": "test.txt",
                        "content": "Unexpected Content!",
                        "base_dir": str(base_dir),
                    },
                }
            ]
        },
        lifecycle=CapabilityLifecycle.PROMOTED,
    )
    reg.register(cap_drift)

    # When owner is passed, it creates reasoning handoff artifact and returns NEEDS_REASONING
    result = kernel.execute_capability("cap_drift_demo", {}, owner="session_123")
    assert result.get("status") == "NEEDS_REASONING"
    assert "state_ref" in result

    # Capability is quarantined in registry!
    quarantined = reg.get("cap_drift_demo")
    assert quarantined.drift_state == "quarantined"
    assert quarantined.lifecycle == CapabilityLifecycle.VALIDATED


def test_work_execute_deterministic_capability_integration(clean_workstation, tmp_path):
    reg = clean_workstation["registry"]
    artifacts = clean_workstation["artifacts"]
    base_dir = tmp_path / "work_exec_sandbox"
    base_dir.mkdir()

    cap = OperationalCapability(
        id="cap_batch_report",
        name="Batch Report Generator",
        version="1.0.0",
        route="filesystem",
        implementation={
            "steps": [
                {
                    "primitive": "fs_write",
                    "args": {
                        "path": "$inputs.report_path",
                        "content": "Report summary for $inputs.name",
                        "base_dir": str(base_dir),
                    },
                }
            ],
            "output": {"report_file": "$inputs.report_path"},
        },
        lifecycle=CapabilityLifecycle.PROMOTED,
        semantic_fingerprint="sem_report_100",
        scope={"route": "filesystem"},
    )
    reg.register(cap)

    from workstation.task_compiler import TaskCompiler
    compiler = TaskCompiler(artifacts=artifacts)

    # Test 1: Direct execution via capability_id
    req_direct = {
        "capability_id": "cap_batch_report",
        "capability_inputs": {"report_path": "rep1.txt", "name": "Task Alpha"},
    }
    res_direct = compiler.execute(req_direct, task_id="task_1", session_id="sess_1", dispatch=lambda tool, args: {})
    assert res_direct["status"] == "COMPLETED"
    assert res_direct["capability_id"] == "cap_batch_report"
    assert (base_dir / "rep1.txt").read_text(encoding="utf-8") == "Report summary for Task Alpha"

    # Test 2: Automatic reuse via operation_fingerprint matching promoted capability
    req_auto = {
        "operation_fingerprint": "sem_report_100",
        "recipe_scope": {"route": "filesystem"},
        "capability_inputs": {"report_path": "rep2.txt", "name": "Task Beta"},
    }
    res_auto = compiler.execute(req_auto, task_id="task_2", session_id="sess_1", dispatch=lambda tool, args: {})
    assert res_auto["status"] == "COMPLETED"
    assert res_auto.get("reused_capability") is True
    assert (base_dir / "rep2.txt").read_text(encoding="utf-8") == "Report summary for Task Beta"


def test_cross_backend_composite(clean_workstation, tmp_path):
    kernel = clean_workstation["kernel"]
    reg = clean_workstation["registry"]
    base_dir = tmp_path / "cross_backend"
    base_dir.mkdir()

    (base_dir / "payload.json").write_text(json.dumps({"query": "Hermes Autonomous"}), encoding="utf-8")

    dispatched_browser_calls = []
    def mock_dispatch(tool_name, tool_args):
        dispatched_browser_calls.append((tool_name, tool_args))
        if tool_name == "browser_navigate":
            return {"url": tool_args["url"], "status": "loaded"}
        if tool_name == "browser_type":
            return {"typed": tool_args["text"]}
        return {}

    cap_fs = OperationalCapability(
        id="cap_fs_query_reader",
        name="Read Query",
        version="1.0.0",
        route="filesystem",
        implementation={
            "steps": [
                {
                    "primitive": "fs_read",
                    "args": {"path": "payload.json", "base_dir": str(base_dir)},
                }
            ],
            "output": {"raw": "$prev"},
        },
        lifecycle=CapabilityLifecycle.PROMOTED,
    )
    reg.register(cap_fs)

    cap_cross = OperationalCapability(
        id="cap_browser_search",
        name="Browser Search From File",
        version="1.0.0",
        route="composite",
        dependencies=[
            CapabilityDependency(capability_id="cap_fs_query_reader", output_alias="reader")
        ],
        implementation={
            "steps": [
                {
                    "primitive": "browser_navigate",
                    "args": {"url": "https://search.example.com"},
                },
                {
                    "primitive": "browser_type",
                    "args": {"ref": "@e1", "text": "Hermes Autonomous"},
                },
            ],
            "output": {"status": "search_completed"},
        },
        lifecycle=CapabilityLifecycle.PROMOTED,
    )
    reg.register(cap_cross)

    res = kernel.execute_capability("cap_browser_search", {}, dispatch=mock_dispatch)
    assert res["success"]
    assert len(dispatched_browser_calls) == 2
    assert dispatched_browser_calls[0] == ("browser_navigate", {"url": "https://search.example.com"})
    assert dispatched_browser_calls[1] == ("browser_type", {"ref": "@e1", "text": "Hermes Autonomous", "clear": True})


def test_zero_llm_calls_replay(clean_workstation, tmp_path):
    kernel = clean_workstation["kernel"]
    reg = clean_workstation["registry"]
    base_dir = tmp_path / "zero_llm"
    base_dir.mkdir()

    cap = OperationalCapability(
        id="cap_deterministic_replay",
        name="Deterministic Replay",
        version="1.0.0",
        route="filesystem",
        implementation={
            "steps": [
                {"primitive": "fs_write", "args": {"path": "step1.txt", "content": "1", "base_dir": str(base_dir)}},
                {"primitive": "fs_write", "args": {"path": "step2.txt", "content": "2", "base_dir": str(base_dir)}},
            ]
        },
        lifecycle=CapabilityLifecycle.PROMOTED,
    )
    reg.register(cap)

    llm_call_count = 0
    def mock_llm(*args, **kwargs):
        nonlocal llm_call_count
        llm_call_count += 1
        raise RuntimeError("LLM must not be called during deterministic capability execution!")

    res = kernel.execute_capability("cap_deterministic_replay", {})
    assert res["success"]
    assert llm_call_count == 0
    assert res["savings"]["llm_calls_saved"] >= 1

