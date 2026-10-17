import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.file_tools import patch_tool, read_file_tool, search_tool, write_file_tool


@pytest.fixture
def confined_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(
        "tools.file_tools_paths._authoritative_workspace_root",
        lambda task_id="default": str(workspace),
    )
    config = {"security": {"file_tools_workspace_only": True}}
    with patch("hermes_cli.config.load_config_readonly", return_value=config):
        yield workspace, outside


def _error(result):
    return json.loads(result)["error"]


@pytest.mark.parametrize(
    ("invoke", "operation"),
    [
        (lambda path: read_file_tool(str(path)), "read"),
        (lambda path: search_tool("needle", path=str(path)), "search"),
        (lambda path: write_file_tool(str(path), "changed"), "write"),
        (
            lambda path: patch_tool(
                path=str(path), old_string="before", new_string="after"
            ),
            "patch",
        ),
    ],
)
def test_absolute_outside_path_is_refused_before_io(confined_workspace, invoke, operation):
    _, outside = confined_workspace
    target = outside / "target.txt"
    target.write_text("before", encoding="utf-8")

    error = _error(invoke(target))

    assert "WORKSPACE_CONFINEMENT_REFUSAL" in error
    assert f"filesystem {operation}" in error
    assert target.read_text(encoding="utf-8") == "before"


def test_relative_inside_paths_remain_usable(confined_workspace):
    workspace, _ = confined_workspace
    target = workspace / "inside.txt"
    target.write_text("needle before", encoding="utf-8")

    read_result = json.loads(read_file_tool("inside.txt"))
    search_result = json.loads(search_tool("needle", path="."))

    assert "error" not in read_result
    assert "needle before" in read_result["content"]
    assert "error" not in search_result


def test_host_symlink_escape_is_refused(confined_workspace):
    workspace, outside = confined_workspace
    secret = outside / "secret.txt"
    secret.write_text("not visible", encoding="utf-8")
    (workspace / "escape").symlink_to(outside, target_is_directory=True)

    error = _error(read_file_tool("escape/secret.txt"))

    assert "WORKSPACE_CONFINEMENT_REFUSAL" in error
    assert str(secret) in error


def test_v4a_patch_refuses_any_outside_header(confined_workspace):
    workspace, outside = confined_workspace
    inside = workspace / "inside.txt"
    inside.write_text("before\n", encoding="utf-8")
    outside_target = outside / "outside.txt"
    patch_text = (
        "*** Begin Patch\n"
        "*** Update File: inside.txt\n"
        "@@\n-before\n+after\n"
        f"*** Add File: {outside_target}\n"
        "+escaped\n"
        "*** End Patch"
    )

    error = _error(patch_tool(mode="patch", patch=patch_text))

    assert "WORKSPACE_CONFINEMENT_REFUSAL" in error
    assert inside.read_text(encoding="utf-8") == "before\n"
    assert not outside_target.exists()


def test_missing_authoritative_root_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tools.file_tools_paths._authoritative_workspace_root",
        lambda task_id="default": None,
    )
    config = {"security": {"file_tools_workspace_only": True}}
    with patch("hermes_cli.config.load_config_readonly", return_value=config):
        error = _error(read_file_tool(str(tmp_path / "anything.txt")))

    assert "no authoritative workspace root" in error


def test_default_profile_behavior_is_unchanged(tmp_path):
    target = tmp_path / "outside.txt"
    target.write_text("still readable", encoding="utf-8")
    with patch("hermes_cli.config.load_config_readonly", return_value={}):
        result = json.loads(read_file_tool(str(target)))

    assert "error" not in result
    assert "still readable" in result["content"]
