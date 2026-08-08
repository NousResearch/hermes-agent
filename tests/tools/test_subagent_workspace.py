from pathlib import Path

import pytest

from tools.subagent_workspace import (
    _split_workspace_volume,
    build_workspace_overrides,
    resolve_parent_workspace_root,
)


def test_full_ro_mounts_parent_workspace_read_only(tmp_path):
    plan = build_workspace_overrides(
        visibility="full_ro",
        mappings=None,
        workspace_root=tmp_path,
        child_token="child-1",
        backend="docker",
    )

    assert plan["task_env_overrides"]["cwd"] == "/workspace"
    assert plan["task_env_overrides"]["docker_volumes"] == [f"{tmp_path.resolve()}:/workspace:ro"]
    assert "read-only" in plan["prompt_note"]


def test_temp_rw_creates_isolated_subworkspace(tmp_path):
    plan = build_workspace_overrides(
        visibility="temp_rw",
        mappings=None,
        workspace_root=tmp_path,
        child_token="child-2",
        backend="docker",
    )

    mount = plan["task_env_overrides"]["docker_volumes"][0]
    host_path = Path(mount.split(":", 1)[0])
    assert host_path.is_dir()
    assert host_path.parent == tmp_path / ".hermes-subagents"
    assert mount.endswith(":/workspace")
    assert "isolated writable subworkspace" in plan["prompt_note"]


def test_mapped_rejects_workspace_escape(tmp_path):
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)

    with pytest.raises(ValueError, match="escapes the parent workspace"):
        build_workspace_overrides(
            visibility="mapped",
            mappings=[{"source": str(outside), "target": "shared"}],
            workspace_root=tmp_path,
            child_token="child-3",
            backend="docker",
        )


def test_mapped_supports_multiple_targets_and_read_only(tmp_path):
    (tmp_path / "pkg-a").mkdir()
    (tmp_path / "pkg-b").mkdir()

    plan = build_workspace_overrides(
        visibility="mapped",
        mappings=[
            {"source": "pkg-a", "target": "work/a"},
            {"source": "pkg-b", "target": "/workspace/work/b", "read_only": True},
        ],
        workspace_root=tmp_path,
        child_token="child-4",
        backend="docker",
    )

    assert plan["task_env_overrides"]["cwd"] == "/workspace"
    assert plan["task_env_overrides"]["docker_volumes"] == [
        f"{(tmp_path / 'pkg-a').resolve()}:/workspace/work/a",
        f"{(tmp_path / 'pkg-b').resolve()}:/workspace/work/b:ro",
    ]
    assert "/workspace/work/a" in plan["prompt_note"]
    assert "/workspace/work/b" in plan["prompt_note"]


def test_mapped_rejects_container_targets_outside_workspace(tmp_path):
    (tmp_path / "pkg-a").mkdir()

    with pytest.raises(ValueError, match="within /workspace"):
        build_workspace_overrides(
            visibility="mapped",
            mappings=[{"source": "pkg-a", "target": "/workspace2"}],
            workspace_root=tmp_path,
            child_token="child-6",
            backend="docker",
        )


def test_restricted_modes_require_docker_backend(tmp_path):
    with pytest.raises(ValueError, match="requires the docker terminal backend"):
        build_workspace_overrides(
            visibility="full_ro",
            mappings=None,
            workspace_root=tmp_path,
            child_token="child-5",
            backend="local",
        )


def test_resolve_parent_workspace_root_prefers_task_overrides(monkeypatch, tmp_path):
    host_root = tmp_path / "workspace-root"
    host_root.mkdir()

    import tools.terminal_tool as terminal_tool

    monkeypatch.setattr(
        terminal_tool,
        "_task_env_overrides",
        {"task-1": {"host_cwd": str(host_root)}},
    )

    assert resolve_parent_workspace_root("task-1") == host_root.resolve()


def test_resolve_parent_workspace_root_uses_runtime_config(monkeypatch, tmp_path):
    import tools.terminal_tool as terminal_tool

    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {"cwd": str(tmp_path), "host_cwd": None, "docker_volumes": []},
    )

    assert resolve_parent_workspace_root() == tmp_path.resolve()


def test_resolve_parent_workspace_root_rejects_docker_container_cwd_fallback(monkeypatch):
    import tools.terminal_tool as terminal_tool

    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": "docker",
            "cwd": "/workspace",
            "host_cwd": None,
            "docker_volumes": [],
        },
    )

    with pytest.raises(ValueError, match="unable to prove the parent host workspace root"):
        resolve_parent_workspace_root()


def test_resolve_parent_workspace_root_uses_single_nested_workspace_mount(monkeypatch, tmp_path):
    host_root = tmp_path / "pkg"
    host_root.mkdir()

    import tools.terminal_tool as terminal_tool

    monkeypatch.setattr(
        terminal_tool,
        "_task_env_overrides",
        {"task-2": {"docker_volumes": [f"{host_root.resolve()}:/workspace/pkg:ro"]}},
    )

    assert resolve_parent_workspace_root("task-2") == host_root.resolve()


def test_resolve_parent_workspace_root_rejects_ambiguous_nested_workspace_mounts(monkeypatch, tmp_path):
    pkg_a = tmp_path / "pkg-a"
    pkg_b = tmp_path / "pkg-b"
    pkg_a.mkdir()
    pkg_b.mkdir()

    import tools.terminal_tool as terminal_tool

    monkeypatch.setattr(
        terminal_tool,
        "_task_env_overrides",
        {
            "task-3": {
                "docker_volumes": [
                    f"{pkg_a.resolve()}:/workspace/a",
                    f"{pkg_b.resolve()}:/workspace/b:ro",
                ]
            }
        },
    )

    with pytest.raises(ValueError, match="no single host workspace root"):
        resolve_parent_workspace_root("task-3")


def test_split_workspace_volume_supports_windows_style_host_paths():
    host, container = _split_workspace_volume(r"C:/repo:/workspace/pkg:ro")

    assert host == "C:/repo"
    assert container == "/workspace/pkg"
