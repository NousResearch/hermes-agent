"""Regression tests for case-sensitive filesystem path conflicts."""

from pathlib import Path

from tools.path_case import (
    case_conflict_for_path,
    check_case_conflicts,
    extract_write_targets,
    filesystem_is_case_sensitive,
    rewrite_target_path,
)


def test_probe_reports_case_sensitive_filesystem(tmp_path):
    assert filesystem_is_case_sensitive(tmp_path) is True


def test_case_conflict_detects_existing_parent_with_different_case(tmp_path):
    existing = tmp_path / "Desktop"
    existing.mkdir()

    conflict = case_conflict_for_path(existing.parent / "desktop" / "plan.md")

    assert conflict == existing / "plan.md"


def test_case_conflict_is_not_reported_when_target_filesystem_is_case_insensitive(
    tmp_path, monkeypatch
):
    existing = tmp_path / "Desktop"
    existing.mkdir()
    requested = existing.parent / "desktop" / "plan.md"
    monkeypatch.setattr("tools.path_case.filesystem_is_case_sensitive", lambda _: False)

    assert case_conflict_for_path(requested) is None


def test_extracts_move_destination_for_case_check():
    targets = extract_write_targets("mv '/tmp/source.md' '/home/r/Desktop/'", "/")

    assert targets == [Path("/home/r/Desktop/")]


def test_rewrites_case_variant_destination_with_trailing_slash(tmp_path):
    command = "mv '/tmp/source.md' '/home/r/desktop/'"
    requested = Path("/home/r/desktop")
    existing = Path("/home/r/Desktop")

    assert rewrite_target_path(command, requested, existing) == (
        "mv '/tmp/source.md' '/home/r/Desktop/'"
    )


def test_terminal_schema_exposes_case_resolution_choice():
    from tools.terminal_tool import TERMINAL_SCHEMA

    property_schema = TERMINAL_SCHEMA["parameters"]["properties"]["case_resolution"]

    assert property_schema["enum"] == ["use_existing", "create_variant"]
    assert "Omit to ask the user" in property_schema["description"]


def test_remote_check_runs_probe_in_target_environment(tmp_path):
    existing = tmp_path / "Desktop"
    existing.mkdir()
    calls = []

    class FakeEnvironment:
        def execute(self, command, cwd=""):
            calls.append((command, cwd))
            return {"returncode": 0, "output": repr([{
                "requested_path": str(tmp_path / "desktop" / "plan.md"),
                "existing_path": str(existing / "plan.md"),
            }])}

    conflicts = check_case_conflicts(
        FakeEnvironment(), "mkdir desktop", str(tmp_path)
    )

    assert conflicts[0]["existing_path"].endswith("Desktop/plan.md")
    assert calls and calls[0][1] == str(tmp_path)
