"""Tests for the user-configured permissions.deny policy namespace."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from agent import deny_policy
from tools import approval as approval_mod
from tools import file_tools


def _install_permissions_config(monkeypatch, *, paths=None, commands=None):
    config = {
        "permissions": {
            "deny": {
                "paths": list(paths or []),
                "commands": list(commands or []),
            }
        }
    }
    monkeypatch.setattr(deny_policy, "load_user_config", lambda: config)
    return config


class TestPermissionsDenyPathMatching:
    def test_glob_matches_path_case_insensitively(self):
        match = deny_policy.match_permissions_deny_path(
            "/tmp/Secret/child.txt",
            patterns=["*/tmp/secret/**"],
        )
        assert match is not None
        assert match.pattern == "*/tmp/secret/**"

    def test_plain_directory_pattern_matches_children(self, tmp_path):
        secret_dir = tmp_path / "secret"
        target = secret_dir / "nested" / "file.txt"

        match = deny_policy.match_permissions_deny_path(
            str(target),
            patterns=[str(secret_dir)],
        )

        assert match is not None
        assert match.pattern == str(secret_dir)

    def test_windows_style_glob_normalizes_separators(self):
        match = deny_policy.match_permissions_deny_path(
            r"C:\Users\alice\obsidian\Daedalus\capsule.md",
            patterns=["*/Users/*/obsidian/Daedalus/**"],
        )
        assert match is not None

    def test_non_matching_path_is_allowed(self, tmp_path):
        match = deny_policy.match_permissions_deny_path(
            str(tmp_path / "public" / "file.txt"),
            patterns=[str(tmp_path / "secret" / "**")],
        )
        assert match is None


class TestPermissionsDenyFileTools:
    def test_read_file_denied_before_file_ops(self, monkeypatch, tmp_path):
        secret = tmp_path / "secret.txt"
        secret.write_text("do not read\n", encoding="utf-8")
        _install_permissions_config(monkeypatch, paths=[str(secret)])

        with patch("tools.file_tools._get_file_ops") as mock_get:
            result = json.loads(file_tools.read_file_tool(str(secret), task_id="deny-read"))

        assert "error" in result
        assert "permissions.deny.paths" in result["error"]
        assert "do not read" not in result["error"]
        mock_get.assert_not_called()

    def test_write_file_denied_before_file_ops(self, monkeypatch, tmp_path):
        secret = tmp_path / "secret.txt"
        _install_permissions_config(monkeypatch, paths=[str(secret)])

        with patch("tools.file_tools._get_file_ops") as mock_get:
            result = json.loads(file_tools.write_file_tool(str(secret), "new data", task_id="deny-write"))

        assert "error" in result
        assert "permissions.deny.paths" in result["error"]
        mock_get.assert_not_called()
        assert not secret.exists()

    def test_patch_replace_denied_before_file_ops(self, monkeypatch, tmp_path):
        secret = tmp_path / "secret.txt"
        secret.write_text("old\n", encoding="utf-8")
        _install_permissions_config(monkeypatch, paths=[str(secret)])

        with patch("tools.file_tools._get_file_ops") as mock_get:
            result = json.loads(file_tools.patch_tool(
                mode="replace",
                path=str(secret),
                old_string="old",
                new_string="new",
                task_id="deny-patch",
            ))

        assert "error" in result
        assert "permissions.deny.paths" in result["error"]
        mock_get.assert_not_called()
        assert secret.read_text(encoding="utf-8") == "old\n"

    def test_search_denied_root_before_file_ops(self, monkeypatch, tmp_path):
        secret_dir = tmp_path / "secret"
        secret_dir.mkdir()
        (secret_dir / "x.txt").write_text("needle\n", encoding="utf-8")
        _install_permissions_config(monkeypatch, paths=[str(secret_dir / "**")])

        with patch("tools.file_tools._get_file_ops") as mock_get:
            result = json.loads(file_tools.search_tool("needle", path=str(secret_dir), task_id="deny-search"))

        assert "error" in result
        assert "permissions.deny.paths" in result["error"]
        mock_get.assert_not_called()

    def test_search_omits_denied_results_from_allowed_root(self, monkeypatch, tmp_path):
        public_dir = tmp_path / "public"
        secret_dir = public_dir / "secret"
        public_dir.mkdir()
        secret_dir.mkdir()
        allowed = public_dir / "allowed.txt"
        denied = secret_dir / "denied.txt"
        allowed.write_text("needle public\n", encoding="utf-8")
        denied.write_text("needle secret\n", encoding="utf-8")
        _install_permissions_config(monkeypatch, paths=[str(secret_dir / "**")])

        fake_result = MagicMock()
        fake_result.matches = [
            MagicMock(path=str(allowed), content="needle public"),
            MagicMock(path=str(denied), content="needle secret"),
        ]
        fake_result.files = []
        fake_result.counts = {}
        fake_result.to_dict.return_value = {
            "matches": [{"path": str(allowed), "content": "needle public"}],
            "total_count": 1,
        }
        fake_ops = MagicMock()
        fake_ops.search.return_value = fake_result

        with patch("tools.file_tools._get_file_ops", return_value=fake_ops):
            result = json.loads(file_tools.search_tool("needle", path=str(public_dir), task_id="deny-filter"))

        assert "_omitted" in result
        assert "permissions.deny.paths" in result["_omitted"]
        remaining_paths = [match.path for match in fake_result.matches]
        assert str(allowed) in remaining_paths
        assert str(denied) not in remaining_paths


class TestPermissionsDenyCommandsAlias:
    def test_permissions_deny_commands_alias_blocks_before_yolo(self, monkeypatch):
        monkeypatch.setattr(approval_mod, "_get_approval_config", lambda: {"mode": "off", "deny": []})
        _install_permissions_config(monkeypatch, commands=["git push --force*"])

        result = approval_mod.check_all_command_guards("git push --force origin main", "local")

        assert result["approved"] is False
        assert result.get("user_deny") is True
        assert "permissions.deny.commands" in result["message"]

    def test_approvals_deny_still_works_without_permissions_config(self, monkeypatch):
        monkeypatch.setattr(approval_mod, "_get_approval_config", lambda: {
            "mode": "manual",
            "deny": ["git push --force*"],
        })
        monkeypatch.setattr(deny_policy, "load_user_config", lambda: {})

        result = approval_mod.check_all_command_guards("git push --force origin main", "local")

        assert result["approved"] is False
        assert result.get("user_deny") is True
