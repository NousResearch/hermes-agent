"""Resolved mutation targets are captured once and reused end to end."""

from __future__ import annotations

import json
from contextlib import nullcontext

import pytest

from tools import file_tools
from tools import file_tools_paths, file_tools_write_guards
from tools.file_operations import (
    ResolvedMutationTarget,
    ShellFileOperations,
)
from tools.file_operations_common import LintResult, PatchResult, ReadResult, WriteResult


class RecordingOperations:
    def __init__(self, *, cwd: str = "/backend/work", remote_home: str | None = None):
        self.env = type(
            "SyntheticEnvironment",
            (),
            {"cwd": cwd, "_remote_home": remote_home,
             "_remote_home_detected": remote_home is not None},
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


class MetadataPatchOperations(RecordingOperations):
    def __init__(self, result: PatchResult):
        super().__init__()
        self.result = result

    def patch_v4a_resolved(
        self,
        patch_content: str,
        resolved_targets: dict[str, ResolvedMutationTarget],
    ) -> PatchResult:
        self.v4a_targets = resolved_targets
        return self.result


@pytest.fixture
def mutation_harness(monkeypatch):
    monkeypatch.setattr(file_tools, "_terminal_env_type_for_task", lambda _task: "local")
    monkeypatch.setattr(
        file_tools, "_resolve_entry_for_task",
        lambda path, _task: f"/backend/{path}",
    )
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


def test_new_write_guards_receive_captured_backend_identity(
    monkeypatch, mutation_harness
):
    operations = RecordingOperations()
    resolver_calls: list[str] = []
    guard_calls: list[tuple[str, object]] = []

    def resolve(path: str, task_id: str):
        resolver_calls.append(path)
        return "/backend/captured.txt"

    monkeypatch.setattr(file_tools, "_resolve_path_for_task", resolve)
    monkeypatch.setattr(file_tools, "_check_sensitive_path", lambda *args: None)
    monkeypatch.setattr(
        file_tools,
        "_check_binary_document_write",
        lambda path, _task, resolved: guard_calls.append(("binary", (path, resolved))),
    )
    monkeypatch.setattr(
        file_tools,
        "_check_protected_instruction_write",
        lambda paths, _task, resolved: guard_calls.append(
            ("protected", (paths, resolved))
        ),
    )
    monkeypatch.setattr(
        file_tools,
        "_check_approval_required_write",
        lambda paths, _task, resolved: guard_calls.append(
            ("approval", (paths, resolved))
        ),
    )
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda _task: operations)

    payload = json.loads(file_tools.write_file_tool("display.txt", "new"))

    assert "error" not in payload
    assert resolver_calls == ["display.txt"]
    assert guard_calls == [
        ("binary", ("display.txt", "/backend/captured.txt")),
        ("protected", (["display.txt"], {"display.txt": "/backend/captured.txt"})),
        ("approval", (["display.txt"], {"display.txt": "/backend/captured.txt"})),
    ]
    assert operations.writes == ["/backend/captured.txt"]


def test_binary_document_guard_checks_captured_backend_extension():
    error = file_tools._check_binary_document_write(
        "display-name",
        resolved_path="/backend/document.docx",
    )

    assert error is not None
    assert ".docx" in error


def test_local_symlink_retarget_after_policy_does_not_switch_mutation(
    tmp_path, monkeypatch, mutation_harness
):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    link = tmp_path / "target.txt"
    link.symlink_to(first)
    operations = RecordingOperations()
    original_resolver = file_tools._resolve_path_for_task
    resolver_calls = 0
    # This tests the resolver-capture invariant, not staleness: the read-before-write
    # guard would refuse overwriting first.txt (never read this task). Switch it off
    # the same way tests/tools/test_file_state_registry.py isolates orthogonal behavior.
    monkeypatch.setenv("HERMES_DISABLE_FILE_STATE_GUARD", "1")

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
    monkeypatch.setattr(
        file_tools_write_guards, "_get_hermes_config_resolved", lambda: None
    )

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
    monkeypatch.setattr(file_tools_paths, "_terminal_env_type_for_task", lambda _task: "ssh")
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


def test_shared_backend_env_cwd_cannot_override_task_session_cwd(
    monkeypatch, mutation_harness
):
    operations = RecordingOperations(cwd="/workspace/session-b")
    monkeypatch.setattr(file_tools, "_terminal_env_type_for_task", lambda _task: "docker")
    monkeypatch.setattr(
        file_tools,
        "_authoritative_workspace_root",
        lambda task_id: "/workspace/session-a" if task_id == "session-a" else None,
    )
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda _task: operations)
    monkeypatch.setattr(file_tools, "_check_sensitive_path", lambda *args: None)

    payload = json.loads(
        file_tools.write_file_tool("target.py", "content", task_id="session-a")
    )

    assert "error" not in payload
    assert operations.writes == ["/workspace/session-a/target.py"]


def test_v4a_aliases_to_same_backend_fail_before_apply(
    monkeypatch, mutation_harness
):
    operations = RecordingOperations()
    monkeypatch.setattr(
        file_tools,
        "_resolve_path_for_task",
        lambda _path, _task: "/backend/shared.txt",
    )
    monkeypatch.setattr(file_tools, "_check_sensitive_path", lambda *args: None)
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda _task: operations)
    patch = """*** Begin Patch
*** Update File: alias-one.txt
@@
-old
+first
*** Update File: alias-two.txt
@@
-old
+second
*** End Patch"""

    payload = json.loads(file_tools.patch_tool(mode="patch", patch=patch))

    assert "multiple display paths" in payload["error"]
    assert operations.v4a_targets is None


@pytest.mark.parametrize("partial_failure", [False, True])
def test_v4a_result_metadata_uses_backend_identities(
    monkeypatch, mutation_harness, partial_failure
):
    result = PatchResult(
        success=not partial_failure,
        files_modified=["old.txt -> moved.txt"],
        files_created=["add.txt"],
        files_deleted=["delete.txt"],
        lint={"add.txt": {"status": "passed"}},
        error="synthetic partial failure" if partial_failure else None,
    )
    operations = MetadataPatchOperations(result)
    monkeypatch.setattr(
        file_tools,
        "_resolve_path_for_task",
        lambda path, _task: f"/backend/{path}",
    )
    monkeypatch.setattr(file_tools, "_check_sensitive_path", lambda *args: None)
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda _task: operations)
    patch = """*** Begin Patch
*** Add File: add.txt
+new
*** Delete File: delete.txt
*** Move File: old.txt->moved.txt
*** End Patch"""

    payload = json.loads(file_tools.patch_tool(mode="patch", patch=patch))

    assert payload["files_modified"] == (
        ["/backend/old.txt -> /backend/moved.txt"] if partial_failure else
        ["/backend/add.txt", "/backend/delete.txt", "/backend/old.txt", "/backend/moved.txt"]
    )
    assert payload["files_created"] == ["/backend/add.txt"]
    assert payload["files_deleted"] == ["/backend/delete.txt"]
    assert list(payload["lint"]) == ["/backend/add.txt"]
    assert bool(payload.get("error")) is partial_failure


@pytest.mark.parametrize("separator", ["->", " -> ", "\t->\t"])
def test_legacy_v4a_move_rewrite_accepts_parser_whitespace(separator):
    patch = (
        "*** Begin Patch\r\n"
        f"*** Move File: old.txt{separator}moved.txt\r\n"
        "*** End Patch\r\n"
    )
    targets = {
        "old.txt": ResolvedMutationTarget("old.txt", "/backend/old.txt", "/backend/old.txt"),
        "moved.txt": ResolvedMutationTarget("moved.txt", "/backend/moved.txt", "/backend/moved.txt"),
    }

    rewritten = file_tools._rewrite_v4a_patch_with_resolved_targets(
        patch, targets
    )

    assert "*** Move File: /backend/old.txt -> /backend/moved.txt\r\n" in rewritten


def test_v4a_captures_each_distinct_target_once(monkeypatch, mutation_harness):
    operations = RecordingOperations()
    resolver_calls: list[str] = []
    policy_targets: list[str] = []
    entry_calls: list[str] = []

    def entry(path: str, task_id: str):
        entry_calls.append(path)
        return f"/backend/{path}"

    monkeypatch.setattr(file_tools, "_resolve_entry_for_task", entry)

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
        path: ResolvedMutationTarget(
            path, f"/backend/{path}", f"/backend/{path}" if path in entry_calls else None
        )
        for path in resolver_calls
    }
    assert entry_calls == ["delete.txt", "old.txt", "moved.txt"]
    assert policy_targets == [f"/backend/{path}" for path in resolver_calls]


class RecordingShellOperations(ShellFileOperations):
    def __init__(self):
        self.reads: list[str] = []
        self.writes: list[str] = []
        self.pre_contents: list[str | None] = []

    def read_file_raw(self, path: str) -> ReadResult:
        self.reads.append(path)
        return ReadResult(content="old\n")

    def write_file(
        self,
        path: str,
        content: str,
        pre_content: str | None = None,
    ) -> WriteResult:
        self.writes.append(path)
        self.pre_contents.append(pre_content)
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
    assert operations.pre_contents == ["old\n"]
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
    assert operations.pre_contents == ["old\n"]


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("operation", ["delete", "move"])
def test_v4a_captured_entry_survives_parent_symlink_retarget(
    tmp_path, monkeypatch, mutation_harness, operation
):
    from tools.environments.local import LocalEnvironment

    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "note.txt").write_text("first", encoding="utf-8")
    (second / "note.txt").write_text("second", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(first, target_is_directory=True)
    monkeypatch.setattr(file_tools, "_resolve_entry_for_task", file_tools_paths._resolve_entry_for_task)
    monkeypatch.setattr(file_tools, "_authoritative_workspace_root", lambda _task: str(tmp_path))
    operations = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda _task: operations)
    changed = False

    def policy(*args):
        nonlocal changed
        if not changed:
            alias.unlink()
            alias.symlink_to(second, target_is_directory=True)
            changed = True
        return None

    monkeypatch.setattr(file_tools, "_check_sensitive_path", policy)
    header = ("*** Delete File: alias/note.txt" if operation == "delete" else
              "*** Move File: alias/note.txt -> alias/moved.txt")
    payload = json.loads(file_tools.patch_tool(
        mode="patch", patch=f"*** Begin Patch\n{header}\n*** End Patch\n"
    ))

    assert payload.get("success"), payload
    assert not (first / "note.txt").exists()
    assert (second / "note.txt").read_text(encoding="utf-8") == "second"
    assert str(first / "note.txt") in payload["files_modified"]
    if operation == "move":
        assert (first / "moved.txt").read_text(encoding="utf-8") == "first"
        assert not (second / "moved.txt").exists()


@pytest.mark.parametrize("path", ["~/note.txt", "~other/note.txt"])
def test_unresolved_ssh_home_refuses_mutation(monkeypatch, mutation_harness, path):
    operations = RecordingOperations()
    monkeypatch.setattr(file_tools, "_terminal_env_type_for_task", lambda _task: "ssh")
    monkeypatch.setattr(file_tools_paths, "_terminal_env_type_for_task", lambda _task: "ssh")
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda _task: operations)
    monkeypatch.setattr(file_tools_paths, "_ssh_remote_home", lambda _task: None)

    payload = json.loads(file_tools.write_file_tool(path, "must not write"))

    assert payload["error"] == "Unable to resolve file mutation target"
    assert operations.writes == []


def test_partial_update_and_move_reports_content_and_entry_identities(
    monkeypatch, mutation_harness
):
    operations = MetadataPatchOperations(PatchResult(
        error="synthetic partial failure",
        files_modified=["same.txt", "same.txt -> moved.txt"],
        lint={"same.txt": {"status": "passed"}},
    ))
    monkeypatch.setattr(file_tools, "_resolve_path_for_task", lambda path, _task: f"/content/{path}")
    monkeypatch.setattr(file_tools, "_resolve_entry_for_task", lambda path, _task: f"/entries/{path}")
    monkeypatch.setattr(file_tools, "_check_sensitive_path", lambda *args: None)
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda _task: operations)
    patch = """*** Begin Patch
*** Update File: same.txt
@@
-old
+new
*** Move File: same.txt -> moved.txt
*** End Patch"""

    payload = json.loads(file_tools.patch_tool(mode="patch", patch=patch))

    assert payload["error"] == "synthetic partial failure"
    assert payload["files_modified"] == [
        "/content/same.txt", "/entries/same.txt -> /entries/moved.txt"
    ]
    assert list(payload["lint"]) == ["/content/same.txt"]


@pytest.mark.parametrize("name,expected", [("AGENTS.md", "AGENTS.md"), ("ordinary.txt", None)])
def test_protected_guard_never_resolves_an_explicit_identity(monkeypatch, name, expected):
    monkeypatch.setattr(file_tools_write_guards, "_hermes_exempt_homes", lambda: ())

    def resolve_again(*args, **kwargs):
        pytest.fail("an explicit backend identity must not be resolved again")

    with monkeypatch.context() as scoped:
        scoped.setattr(file_tools_write_guards.os.path, "realpath", resolve_again)
        scoped.setattr(file_tools_write_guards, "_resolve_path_for_task", resolve_again)
        actual = file_tools_write_guards._protected_instruction_reason(
            "display.txt", task_id="remote-task", enabled=True, extra_patterns=[],
            resolved_path=f"/remote/work/{name}",
        )

    assert actual == expected


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("name,expected", [("AGENTS.md", "AGENTS.md"), ("ordinary.txt", None)])
def test_remote_protected_identity_ignores_a_controller_symlink(
    tmp_path, monkeypatch, name, expected
):
    # A remote path's spelling can happen to exist on the controller with a
    # symlink whose target has the opposite protection classification.
    target_name = "ordinary.txt" if name == "AGENTS.md" else "AGENTS.md"
    target = tmp_path / "controller" / target_name
    target.parent.mkdir()
    target.write_text("controller-only content", encoding="utf-8")
    captured = tmp_path / "remote" / name
    captured.parent.mkdir()
    captured.symlink_to(target)
    monkeypatch.setattr(file_tools_write_guards, "_hermes_exempt_homes", lambda: ())

    actual = file_tools_write_guards._protected_instruction_reason(
        "display.txt", task_id="remote-task", enabled=True, extra_patterns=[],
        resolved_path=str(captured),
    )

    assert actual == expected
    assert target.read_text(encoding="utf-8") == "controller-only content"
