"""Terminal traversal regression for Windows Cloud Files hydration (#97898)."""

import json

import pytest

from tools.terminal_tool_guards import windows_cloud_traversal_block
from tools.terminal_tool_windows_cloud import cloud_placeholder_traversal_reason


def test_cloud_traversal_policy_blocks_broad_roots_but_allows_explicit_cloud_access():
    environ = {
        "USERPROFILE": r"C:\Users\jeff",
        "OneDrive": r"C:\Users\jeff\OneDrive",
        "iCloudDrive": r"C:\Users\jeff\iCloudDrive",
        "OneDriveBusiness": "relative-malformed-root",
    }
    blocked = (
        ("find . -name .git -type d", r"C:\Users\jeff"),
        ("du -sh ./*", r"C:\Users\jeff"),
        ("rg --files .", r"C:\Users\jeff"),
        ("rg needle .", r"C:\Users\jeff"),
        ("rg needle", r"C:\Users\jeff"),
        ("grep -Rn needle .", r"C:\Users\jeff"),
        ("cd /c/Users/jeff && find . -name .git", r"C:\repos\app"),
        ('find "$HOME" -name .git', r"C:\repos\app"),
        ("find C:/Users -name pyproject.toml", r"C:\repos\app"),
        (r"find C:\Users\jeff -name .git", r"C:\repos\app"),
        (r"rg needle C:\Users\jeff", r"C:\repos\app"),
        (r"cd C:\Users\jeff && find . -name .git", r"C:\repos\app"),
        ("find Pictures -type f", r"C:\Users\jeff"),
    )
    for command, cwd in blocked:
        assert cloud_placeholder_traversal_reason(
            command, cwd=cwd, environ=environ
        ), command

    allowed = (
        ("find src -name .git", r"C:\Users\jeff"),
        ("rg needle file.txt", r"C:\Users\jeff"),
        ("find OneDrive -type f", r"C:\Users\jeff"),
        ("find /c/Users/jeff/OneDrive/work -type f", r"C:\repos\app"),
        ("find relative-malformed-root -type f", r"C:\Users\jeff"),
        (r"find C:\repos\app\src -name .git", r"C:\repos\app"),
        (r"echo noop\; find C:\Users\jeff -name .git", r"C:\repos\app"),
        ("rg needle", r"C:\repos\app"),
    )
    for command, cwd in allowed:
        assert cloud_placeholder_traversal_reason(
            command, cwd=cwd, environ=environ
        ) is None, command


@pytest.mark.platforms("windows")
def test_terminal_guard_blocks_profile_scan_before_execution(tmp_path, monkeypatch):
    profile = tmp_path / "profile"
    profile.mkdir()
    monkeypatch.setenv("USERPROFILE", str(profile))
    monkeypatch.setenv("OneDrive", str(profile / "OneDrive"))

    class LocalEnv:
        env = {}

    blocked = windows_cloud_traversal_block(
        command="find . -name .git -type d",
        env=LocalEnv(),
        env_type="local",
        cwd=str(profile),
        workdir=str(profile),
        session_key="",
    )

    assert blocked is not None
    payload = json.loads(blocked)
    assert payload["status"] == "blocked"
    assert "hydrate OneDrive/iCloud" in payload["error"]
