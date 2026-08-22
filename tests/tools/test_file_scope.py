"""Behavior tests for profile-local file tool path scopes."""

from __future__ import annotations

import json
from pathlib import PurePosixPath

import pytest


def _write_scope_config(home, *, read_dirs=None, write_dirs=None, deny_dirs=None):
    home.mkdir(exist_ok=True)
    scope = {}
    if read_dirs is not None:
        scope["read_dirs"] = [str(path) for path in read_dirs]
    if write_dirs is not None:
        scope["write_dirs"] = [str(path) for path in write_dirs]
    if deny_dirs is not None:
        scope["deny_dirs"] = [str(path) for path in deny_dirs]
    (home / "config.yaml").write_text(
        json.dumps({"security": {"file_scope": scope}}),
        encoding="utf-8",
    )


def test_read_file_rejects_path_outside_profile_read_dirs(tmp_path, monkeypatch):
    from tools.file_tools import read_file_tool

    home = tmp_path / "profile-home"
    allowed = tmp_path / "vault"
    outside = tmp_path / "private"
    allowed.mkdir()
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("must not leak\n", encoding="utf-8")
    _write_scope_config(home, read_dirs=[allowed])
    monkeypatch.setenv("HERMES_HOME", str(home))

    result = json.loads(read_file_tool(str(secret), task_id="file-scope-read-deny"))

    assert "error" in result
    assert "outside security.file_scope.read_dirs" in result["error"]
    assert "must not leak" not in json.dumps(result)


def test_write_file_enforces_profile_write_dirs(tmp_path, monkeypatch):
    from tools.file_tools import write_file_tool

    home = tmp_path / "profile-home"
    queue = tmp_path / "queue"
    outside = tmp_path / "outside"
    queue.mkdir()
    outside.mkdir()
    _write_scope_config(home, write_dirs=[queue])
    monkeypatch.setenv("HERMES_HOME", str(home))

    denied = json.loads(write_file_tool(
        str(outside / "message.txt"),
        "blocked\n",
        task_id="file-scope-write-deny",
    ))
    allowed = json.loads(write_file_tool(
        str(queue / "message.txt"),
        "queued\n",
        task_id="file-scope-write-allow",
    ))

    assert "outside security.file_scope.write_dirs" in denied["error"]
    assert not (outside / "message.txt").exists()
    assert allowed.get("error") is None
    assert (queue / "message.txt").read_text(encoding="utf-8") == "queued\n"


def test_search_files_rejects_root_outside_profile_read_dirs(tmp_path, monkeypatch):
    from tools.file_tools import search_tool

    home = tmp_path / "profile-home"
    allowed = tmp_path / "vault"
    outside = tmp_path / "private"
    allowed.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("needle\n", encoding="utf-8")
    _write_scope_config(home, read_dirs=[allowed])
    monkeypatch.setenv("HERMES_HOME", str(home))

    result = json.loads(search_tool(
        "needle",
        path=str(outside),
        task_id="file-scope-search-deny",
    ))

    assert "outside security.file_scope.read_dirs" in result["error"]
    assert "needle" not in json.dumps(result)


def test_patch_rejects_target_outside_profile_write_dirs(tmp_path, monkeypatch):
    from tools.file_tools import patch_tool

    home = tmp_path / "profile-home"
    queue = tmp_path / "queue"
    outside = tmp_path / "outside"
    queue.mkdir()
    outside.mkdir()
    target = outside / "message.txt"
    target.write_text("before\n", encoding="utf-8")
    _write_scope_config(home, write_dirs=[queue])
    monkeypatch.setenv("HERMES_HOME", str(home))

    result = json.loads(patch_tool(
        mode="replace",
        path=str(target),
        old_string="before",
        new_string="after",
        task_id="file-scope-patch-deny",
    ))

    assert "outside security.file_scope.write_dirs" in result["error"]
    assert target.read_text(encoding="utf-8") == "before\n"


def test_search_omits_results_from_denied_subdirectory(tmp_path, monkeypatch):
    from tools.file_tools import search_tool

    home = tmp_path / "profile-home"
    vault = tmp_path / "vault"
    denied_dir = vault / "private"
    allowed_dir = vault / "public"
    denied_dir.mkdir(parents=True)
    allowed_dir.mkdir()
    (denied_dir / "secret.txt").write_text("needle secret\n", encoding="utf-8")
    (allowed_dir / "note.txt").write_text("needle public\n", encoding="utf-8")
    _write_scope_config(home, read_dirs=[vault], deny_dirs=[denied_dir])
    monkeypatch.setenv("HERMES_HOME", str(home))

    result = json.loads(search_tool(
        "needle",
        path=str(vault),
        task_id="file-scope-search-filter",
    ))

    rendered = json.dumps(result)
    assert "needle public" in rendered
    assert "needle secret" not in rendered
    assert "file scope" in result["_omitted"]


def test_read_scope_resolves_traversal_and_symlink_targets(tmp_path, monkeypatch):
    from tools.file_tools import read_file_tool

    home = tmp_path / "profile-home"
    vault = tmp_path / "vault"
    outside = tmp_path / "outside"
    vault.mkdir()
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("must not leak\n", encoding="utf-8")
    link = vault / "linked-secret.txt"
    try:
        link.symlink_to(secret)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    _write_scope_config(home, read_dirs=[vault])
    monkeypatch.setenv("HERMES_HOME", str(home))

    traversal = json.loads(read_file_tool(
        str(vault / ".." / "outside" / "secret.txt"),
        task_id="file-scope-traversal-deny",
    ))
    symlink = json.loads(read_file_tool(
        str(link),
        task_id="file-scope-symlink-deny",
    ))

    assert "outside security.file_scope.read_dirs" in traversal["error"]
    assert "outside security.file_scope.read_dirs" in symlink["error"]


def test_deny_dirs_override_read_and_write_allowlists(tmp_path, monkeypatch):
    from tools.file_tools import read_file_tool, write_file_tool

    home = tmp_path / "profile-home"
    vault = tmp_path / "vault"
    denied_dir = vault / "private"
    denied_dir.mkdir(parents=True)
    target = denied_dir / "note.txt"
    target.write_text("private\n", encoding="utf-8")
    _write_scope_config(
        home,
        read_dirs=[vault],
        write_dirs=[vault],
        deny_dirs=[denied_dir],
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    read_result = json.loads(read_file_tool(
        str(target),
        task_id="file-scope-deny-read",
    ))
    write_result = json.loads(write_file_tool(
        str(target),
        "changed\n",
        task_id="file-scope-deny-write",
    ))

    assert "security.file_scope.deny_dirs" in read_result["error"]
    assert "security.file_scope.deny_dirs" in write_result["error"]
    assert target.read_text(encoding="utf-8") == "private\n"


def test_empty_scope_lists_preserve_unrestricted_behavior(tmp_path, monkeypatch):
    from tools.file_tools import read_file_tool, write_file_tool

    home = tmp_path / "profile-home"
    workspace = tmp_path / "任意目录"
    workspace.mkdir()
    source = workspace / "资料.txt"
    source.write_text("unicode content\n", encoding="utf-8")
    output = workspace / "输出.txt"
    _write_scope_config(home, read_dirs=[], write_dirs=[], deny_dirs=[])
    monkeypatch.setenv("HERMES_HOME", str(home))

    read_result = json.loads(read_file_tool(
        str(source),
        task_id="file-scope-empty-read",
    ))
    write_result = json.loads(write_file_tool(
        str(output),
        "unicode output\n",
        task_id="file-scope-empty-write",
    ))

    assert "unicode content" in read_result["content"]
    assert write_result.get("error") is None
    assert output.read_text(encoding="utf-8") == "unicode output\n"


def test_remote_scope_uses_backend_realpath_for_symlink_targets(monkeypatch):
    import tools.file_tools as file_tools

    class Result:
        exit_code = 0

        def __init__(self, stdout):
            self.stdout = stdout

    class RemoteOps:
        def _exec(self, command):
            if "/vault/link" in command:
                return Result("/outside/secret.txt\n/vault\n")
            return Result("/vault\n")

        @staticmethod
        def _escape_shell_arg(value):
            return f"'{value}'"

    monkeypatch.setattr(
        file_tools,
        "_terminal_env_type_for_task",
        lambda task_id="default": "ssh",
    )
    monkeypatch.setattr(
        file_tools,
        "_resolve_path_for_task",
        lambda path, task_id="default": PurePosixPath(path),
    )
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda task_id="default": RemoteOps())
    monkeypatch.setattr(
        file_tools,
        "_file_scope_config",
        lambda: {"read_dirs": ["/vault"]},
    )

    error = file_tools._check_file_scope("/vault/link", "read", "remote-task")
    assert "outside security.file_scope.read_dirs" in error


def test_v4a_patch_rejects_entire_batch_when_one_target_is_outside(tmp_path, monkeypatch):
    from tools.file_tools import patch_tool

    home = tmp_path / "profile-home"
    queue = tmp_path / "queue"
    outside = tmp_path / "outside"
    queue.mkdir()
    outside.mkdir()
    allowed_file = queue / "allowed.txt"
    outside_file = outside / "outside.txt"
    allowed_file.write_text("before allowed\n", encoding="utf-8")
    outside_file.write_text("before outside\n", encoding="utf-8")
    _write_scope_config(home, write_dirs=[queue])
    monkeypatch.setenv("HERMES_HOME", str(home))

    result = json.loads(patch_tool(
        mode="patch",
        patch=(
            "*** Begin Patch\n"
            f"*** Update File: {allowed_file}\n"
            "@@\n"
            "-before allowed\n"
            "+after allowed\n"
            f"*** Update File: {outside_file}\n"
            "@@\n"
            "-before outside\n"
            "+after outside\n"
            "*** End Patch"
        ),
        task_id="file-scope-v4a-atomic-deny",
    ))

    assert "outside security.file_scope.write_dirs" in result["error"]
    assert allowed_file.read_text(encoding="utf-8") == "before allowed\n"
    assert outside_file.read_text(encoding="utf-8") == "before outside\n"


def test_malformed_allowlist_fails_closed(tmp_path, monkeypatch):
    from tools.file_tools import read_file_tool

    home = tmp_path / "profile-home"
    home.mkdir()
    target = tmp_path / "note.txt"
    target.write_text("content\n", encoding="utf-8")
    (home / "config.yaml").write_text(
        json.dumps({"security": {"file_scope": {"read_dirs": str(tmp_path)}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    result = json.loads(read_file_tool(
        str(target),
        task_id="file-scope-invalid-config",
    ))

    assert "could not safely apply security.file_scope" in result["error"]
    assert "must be a list of paths" in result["error"]
