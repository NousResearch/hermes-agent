"""Resolved mutation targets are captured once and reused end to end."""

from __future__ import annotations

import json
from contextlib import nullcontext

import pytest

from tools import file_tools
from tools.file_operations import (
    LintResult,
    PatchResult,
    ReadResult,
    ResolvedMutationTarget,
    ShellFileOperations,
    WriteResult,
)


class RecordingOperations:
    def __init__(self, *, cwd: str = "/backend/work", remote_home: str | None = None):
        self.env = type(
            "SyntheticEnvironment",
            (),
            {"cwd": cwd, "_remote_home": remote_home},
        )()
        self.cwd = cwd
        self.writes: list[str] = []
        self.replaces: list[str] = []
        self.v4a_targets: dict[str, ResolvedMutationTarget] | None = None

    def write_file(self, path: str, content: str) -> WriteResult:
        self.writes.append(path)
        return WriteResult(bytes_written=len(content))

    def patch_replace(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> PatchResult:
        self.replaces.append(path)
        return PatchResult(success=True, files_modified=[path])

    def patch_v4a_resolved(
        self,
        patch_content: str,
        resolved_targets: dict[str, ResolvedMutationTarget],
    ) -> PatchResult:
        self.v4a_targets = resolved_targets
        return PatchResult(error="synthetic stop")


@pytest.fixture
def mutation_harness(monkeypatch):
    monkeypatch.setattr(file_tools, "_terminal_env_type_for_task", lambda _task: "local")
    monkeypatch.setattr(file_tools, "_check_cross_profile_path", lambda *args: None)
    monkeypatch.setattr(file_tools, "_path_resolution_warning", lambda *args: None)
    monkeypatch.setattr(file_tools, "_mark_verification_stale", lambda *args, **kwargs: None)
    monkeypatch.setattr(file_tools.file_state, "lock_path", lambda _path: nullcontext())
    monkeypatch.setattr(file_tools.file_state, "check_stale", lambda *args: None)
    monkeypatch.setattr(file_tools.file_state, "note_write", lambda *args: None)


@pytest.mark.parametrize("tool_kind", ["write", "replace"])
def test_policy_and_mutation_share_one_resolver_result(
    monkeypatch, mutation_harness, tool_kind
):
    operations = RecordingOperations()
    resolver_calls: list[str] = []
    policy_calls: list[tuple[str, str]] = []

    def resolve(path: str, task_id: str):
        resolver_calls.append(path)
        return "/backend/captured.txt"

    def policy(path: str, task_id: str, resolved_path: str | None = None):
        policy_calls.append((path, resolved_path or ""))
        return None

    monkeypatch.setattr(file_tools, "_resolve_path_for_task", resolve)
    monkeypatch.setattr(file_tools, "_check_sensitive_path", policy)
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda _task: operations)

    if tool_kind == "write":
        payload = json.loads(file_tools.write_file_tool("display.txt", "new"))
        mutated = operations.writes
    else:
        payload = json.loads(
            file_tools.patch_tool(
                mode="replace",
                path="display.txt",
                old_string="old",
                new_string="new",
            )
        )
        mutated = operations.replaces

    assert "error" not in payload
    assert resolver_calls == ["display.txt"]
    assert policy_calls == [("display.txt", "/backend/captured.txt")]
    assert mutated == ["/backend/captured.txt"]


def test_local_symlink_retarget_after_policy_does_not_switch_mutation(
    tmp_path, monkeypatch, mutation_harness
):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first")
    second.write_text("second")
    link = tmp_path / "target.txt"
    link.symlink_to(first)
    operations = RecordingOperations()
    original_resolver = file_tools._resolve_path_for_task
    resolver_calls = 0

    def resolve(path: str, task_id: str):
        nonlocal resolver_calls
        resolver_calls += 1
        return original_resolver(path, task_id)

    def retarget_after_policy(path: str, task_id: str, resolved_path: str | None = None):
        link.unlink()
        link.symlink_to(second)
        return None

    monkeypatch.setattr(file_tools, "_resolve_path_for_task", resolve)
    monkeypatch.setattr(file_tools, "_check_sensitive_path", retarget_after_policy)
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda _task: operations)

    payload = json.loads(file_tools.write_file_tool(str(link), "changed"))

    assert "error" not in payload
    assert resolver_calls == 1
    assert operations.writes == [str(first)]


def test_original_absolute_sensitive_namespace_remains_protected(monkeypatch):
    monkeypatch.setattr(file_tools, "_get_hermes_config_resolved", lambda: None)

    error = file_tools._check_sensitive_path(
        "/etc/synthetic-target",
        resolved_path="/run/synthetic-target",
    )

    assert error is not None
    assert "sensitive system path" in error


def test_ssh_target_uses_remote_home_and_is_resolved_once(
    monkeypatch, mutation_harness
):
    operations = RecordingOperations(remote_home="/home/remote-user")
    resolver_calls = 0
    policy_targets: list[str] = []
    original_resolver = file_tools._resolve_path_for_task

    def resolve(path: str, task_id: str, backend_cwd: str | None = None):
        nonlocal resolver_calls
        resolver_calls += 1
        return original_resolver(path, task_id, backend_cwd)

    monkeypatch.setattr(file_tools, "_terminal_env_type_for_task", lambda _task: "ssh")
    monkeypatch.setattr(file_tools, "_resolve_path_for_task", resolve)
    monkeypatch.setattr(
        file_tools,
        "_check_sensitive_path",
        lambda _path, _task, resolved: policy_targets.append(resolved),
    )
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda _task: operations)

    payload = json.loads(file_tools.write_file_tool("~/note.txt", "remote"))

    assert "error" not in payload
    assert resolver_calls == 1
    assert policy_targets == ["/home/remote-user/note.txt"]
    assert operations.writes == ["/home/remote-user/note.txt"]


def test_docker_relative_target_initializes_cwd_before_resolution(
    monkeypatch, mutation_harness
):
    operations = RecordingOperations(cwd="/workspace/project")
    initialized = False
    resolver_calls = 0
    policy_targets: list[str] = []
    original_resolver = file_tools._resolve_path_for_task

    def get_file_ops(_task: str):
        nonlocal initialized
        initialized = True
        return operations

    def resolve(path: str, task_id: str, backend_cwd: str | None = None):
        nonlocal resolver_calls
        assert initialized
        resolver_calls += 1
        return original_resolver(path, task_id, backend_cwd)

    monkeypatch.setattr(file_tools, "_terminal_env_type_for_task", lambda _task: "docker")
    monkeypatch.setattr(file_tools, "_get_file_ops", get_file_ops)
    monkeypatch.setattr(file_tools, "_resolve_path_for_task", resolve)
    monkeypatch.setattr(
        file_tools,
        "_check_sensitive_path",
        lambda _path, _task, resolved: policy_targets.append(resolved),
    )

    payload = json.loads(file_tools.write_file_tool("src/app.py", "content"))

    assert "error" not in payload
    assert resolver_calls == 1
    assert policy_targets == ["/workspace/project/src/app.py"]
    assert operations.writes == ["/workspace/project/src/app.py"]


def test_v4a_captures_each_distinct_target_once(monkeypatch, mutation_harness):
    operations = RecordingOperations()
    resolver_calls: list[str] = []
    policy_targets: list[str] = []

    def resolve(path: str, task_id: str):
        resolver_calls.append(path)
        return f"/backend/{path}"

    monkeypatch.setattr(file_tools, "_resolve_path_for_task", resolve)
    monkeypatch.setattr(
        file_tools,
        "_check_sensitive_path",
        lambda _path, _task, resolved: policy_targets.append(resolved),
    )
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda _task: operations)
    patch = """*** Begin Patch
*** Update File: update.txt
@@
-old
+new
*** Add File: add.txt
+new
*** Delete File: delete.txt
*** Move File: old.txt -> moved.txt
*** End Patch"""

    payload = json.loads(file_tools.patch_tool(mode="patch", patch=patch))

    assert payload["error"] == "synthetic stop"
    assert resolver_calls == [
        "update.txt",
        "add.txt",
        "delete.txt",
        "old.txt",
        "moved.txt",
    ]
    assert operations.v4a_targets == {
        path: ResolvedMutationTarget(path, f"/backend/{path}")
        for path in resolver_calls
    }
    assert policy_targets == [f"/backend/{path}" for path in resolver_calls]


class RecordingShellOperations(ShellFileOperations):
    def __init__(self):
        self.reads: list[str] = []
        self.writes: list[str] = []

    def read_file_raw(self, path: str) -> ReadResult:
        self.reads.append(path)
        return ReadResult(content="old\n")

    def write_file(self, path: str, content: str) -> WriteResult:
        self.writes.append(path)
        return WriteResult(bytes_written=len(content))

    def _check_lint(self, path: str, content: str | None = None) -> LintResult:
        return LintResult(skipped=True)


def test_v4a_backend_calls_use_identity_but_diff_keeps_display_path():
    operations = RecordingShellOperations()
    patch = """*** Begin Patch
*** Update File: display.txt
@@
-old
+new
*** End Patch"""
    targets = {
        "display.txt": ResolvedMutationTarget(
            "display.txt", "/backend/captured.txt"
        )
    }

    result = operations.patch_v4a_resolved(patch, targets)

    assert result.success
    assert operations.reads == ["/backend/captured.txt", "/backend/captured.txt"]
    assert operations.writes == ["/backend/captured.txt"]
    assert "display.txt" in result.diff
    assert "/backend/captured.txt" not in result.diff


def test_v4a_missing_identity_mapping_fails_before_backend_access():
    operations = RecordingShellOperations()
    patch = """*** Begin Patch
*** Add File: display.txt
+new
*** End Patch"""

    result = operations.patch_v4a_resolved(patch, {})

    assert not result.success
    assert result.error == "No captured resolved identity for a V4A target"
    assert operations.reads == []
    assert operations.writes == []


def test_historical_patch_v4a_one_argument_interface_still_works():
    operations = RecordingShellOperations()
    patch = """*** Begin Patch
*** Update File: display.txt
@@
-old
+new
*** End Patch"""

    result = operations.patch_v4a(patch)

    assert result.success
    assert operations.reads == ["display.txt", "display.txt"]
    assert operations.writes == ["display.txt"]
