"""Tests for the user-configured permissions.deny policy namespace."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from unittest.mock import MagicMock, patch

import pytest

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

    def test_tilde_rule_uses_effective_profile_home(self):
        with patch(
            "hermes_constants.get_subprocess_home",
            return_value=r"C:\Profiles\coder",
        ):
            match = deny_policy.match_permissions_deny_path(
                r"C:\Profiles\coder\secret\file.txt",
                patterns=["~/secret/**"],
            )

        assert match is not None
        assert match.pattern == "~/secret/**"

    def test_lexical_alias_matches_glob_before_canonical_rewrite(self):
        def fake_realpath(value):
            normalized = str(value).replace("\\", "/")
            if normalized == "/workspace/link/secret/file.txt":
                return "/outside/secret/file.txt"
            return value

        with patch("agent.deny_policy.os.path.realpath", side_effect=fake_realpath):
            match = deny_policy.match_permissions_deny_path(
                "/workspace/link/secret/file.txt",
                patterns=["/workspace/*/secret/**"],
            )

        assert match is not None
        assert match.pattern == "/workspace/*/secret/**"

    def test_rule_alias_matches_canonical_target(self):
        def fake_realpath(value):
            normalized = str(value).replace("\\", "/").rstrip("/")
            if normalized == "/workspace/link/secret":
                return "/outside/secret"
            return value

        with patch("agent.deny_policy.os.path.realpath", side_effect=fake_realpath):
            match = deny_policy.match_permissions_deny_path(
                "/outside/secret/file.txt",
                patterns=["/workspace/link/secret/**"],
            )

        assert match is not None
        assert match.pattern == "/workspace/link/secret/**"

    def test_rule_alias_preserves_case_until_after_realpath(self):
        def fake_realpath(value):
            if str(value).replace("\\", "/").rstrip("/") == "/Workspace/Link/Secret":
                return "/outside/secret"
            return value

        with patch("agent.deny_policy.os.path.realpath", side_effect=fake_realpath):
            match = deny_policy.match_permissions_deny_path(
                "/outside/secret/file.txt",
                patterns=["/Workspace/Link/Secret/**"],
            )

        assert match is not None
        assert match.pattern == "/Workspace/Link/Secret/**"

    def test_real_symlink_aliases_are_denied_in_both_directions(self, tmp_path):
        target = tmp_path / "secret-target"
        target.mkdir()
        candidate = target / "file.txt"
        candidate.write_text("secret\n", encoding="utf-8")
        alias = tmp_path / "secret-link"
        try:
            alias.symlink_to(target, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")

        through_alias = deny_policy.match_permissions_deny_path(
            str(alias / "file.txt"),
            patterns=[str(target / "**")],
        )
        against_alias_rule = deny_policy.match_permissions_deny_path(
            str(candidate),
            patterns=[str(alias / "**")],
        )

        assert through_alias is not None
        assert against_alias_rule is not None

    def test_remote_matching_never_uses_host_realpath(self):
        with patch(
            "agent.deny_policy.os.path.realpath",
            side_effect=AssertionError("host realpath must not be called"),
        ) as mock_realpath:
            match = deny_policy.match_permissions_deny_path(
                "/remote/secret/file.txt",
                patterns=["/remote/secret/**"],
                canonicalize=False,
            )

        assert match is not None
        mock_realpath.assert_not_called()

    def test_disjoint_wildcard_sibling_does_not_overlap_search_root(self):
        pattern = "/workspace/foo*/secret/**"

        disjoint = deny_policy.match_permissions_deny_search_root(
            "/workspace/bar",
            patterns=[pattern],
            canonicalize=False,
        )
        matching = deny_policy.match_permissions_deny_search_root(
            "/workspace/foobar",
            patterns=[pattern],
            canonicalize=False,
        )

        assert disjoint is None
        assert matching is not None


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

    def test_read_preserves_lexical_alias_before_task_resolution(self, monkeypatch):
        lexical = "C:/workspace/link/secret/file.txt"
        resolved = Path("C:/outside/secret/file.txt")
        _install_permissions_config(
            monkeypatch,
            paths=["C:/workspace/*/secret/**"],
        )

        with (
            patch("tools.file_tools._resolve_path_for_task", return_value=resolved),
            patch("tools.file_tools._get_file_ops") as mock_get,
        ):
            result = json.loads(file_tools.read_file_tool(
                lexical,
                task_id="deny-read-alias",
            ))

        assert "error" in result
        assert "permissions.deny.paths" in result["error"]
        mock_get.assert_not_called()

    def test_ssh_path_deny_disables_local_canonicalization(self, monkeypatch):
        remote = PurePosixPath("/remote/secret/file.txt")
        _install_permissions_config(monkeypatch, paths=["/remote/secret/**"])

        with (
            patch("tools.file_tools._terminal_env_type_for_task", return_value="ssh"),
            patch("tools.file_tools._resolve_path_for_task", return_value=remote),
            patch(
                "agent.deny_policy.match_permissions_deny_path",
                wraps=deny_policy.match_permissions_deny_path,
            ) as mock_match,
            patch("tools.file_tools._get_file_ops") as mock_get,
        ):
            result = json.loads(file_tools.read_file_tool(
                str(remote),
                task_id="deny-ssh-path",
            ))

        assert "error" in result
        assert "permissions.deny.paths" in result["error"]
        assert mock_match.call_count >= 1
        assert all(call.kwargs["canonicalize"] is False for call in mock_match.call_args_list)
        mock_get.assert_not_called()

    def test_ssh_search_does_not_probe_host_file_type(self, monkeypatch):
        remote = PurePosixPath("/remote/secret")
        _install_permissions_config(monkeypatch, paths=["/remote/secret/**"])

        with (
            patch("tools.file_tools._terminal_env_type_for_task", return_value="ssh"),
            patch("tools.file_tools._resolve_path_for_task", return_value=remote),
            patch(
                "tools.file_tools.os.path.isfile",
                side_effect=AssertionError("host isfile must not be called"),
            ) as mock_isfile,
            patch("tools.file_tools._get_file_ops") as mock_get,
        ):
            result = json.loads(file_tools.search_tool(
                "needle",
                path=str(remote),
                task_id="deny-ssh-search",
            ))

        assert "error" in result
        assert "permissions.deny.paths" in result["error"]
        mock_isfile.assert_not_called()
        mock_get.assert_not_called()

    def test_write_preserves_relative_lexical_path_before_task_resolution(self, monkeypatch, tmp_path):
        resolved = tmp_path / "secret" / "file.txt"
        _install_permissions_config(monkeypatch, paths=["secret/**"])

        with (
            patch("tools.file_tools._resolve_path_for_task", return_value=resolved),
            patch("tools.file_tools._get_file_ops") as mock_get,
        ):
            result = json.loads(file_tools.write_file_tool(
                "secret/file.txt",
                "new data",
                task_id="deny-write-relative",
            ))

        assert "error" in result
        assert "permissions.deny.paths" in result["error"]
        mock_get.assert_not_called()
        assert not resolved.exists()

    def test_read_file_denied_extractable_document_before_extraction(self, monkeypatch, tmp_path):
        secret = tmp_path / "secret.ipynb"
        secret.write_text(
            json.dumps({"cells": [{"cell_type": "markdown", "source": ["do not read"]}]}),
            encoding="utf-8",
        )
        _install_permissions_config(monkeypatch, paths=[str(secret)])

        with (
            patch("tools.read_extract.extract_document_text") as mock_extract,
            patch("tools.file_tools._get_file_ops") as mock_get,
        ):
            result = json.loads(file_tools.read_file_tool(str(secret), task_id="deny-ipynb"))

        assert "error" in result
        assert "permissions.deny.paths" in result["error"]
        assert "do not read" not in result["error"]
        mock_extract.assert_not_called()
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

    def test_v4a_patch_denied_header_blocks_before_file_ops(self, monkeypatch, tmp_path):
        secret = tmp_path / "secret.txt"
        secret.write_text("old\n", encoding="utf-8")
        _install_permissions_config(monkeypatch, paths=[str(secret)])
        patch_text = (
            "*** Begin Patch\n"
            f"*** Update File: {secret}\n"
            "@@\n"
            "-old\n"
            "+new\n"
            "*** End Patch\n"
        )

        with patch("tools.file_tools._get_file_ops") as mock_get:
            result = json.loads(file_tools.patch_tool(
                mode="patch",
                patch=patch_text,
                task_id="deny-v4a-patch",
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

    def test_search_blocks_parent_with_denied_descendant_before_file_ops(self, monkeypatch, tmp_path):
        public_dir = tmp_path / "public"
        secret_dir = public_dir / "secret"
        public_dir.mkdir()
        secret_dir.mkdir()
        (public_dir / "allowed.txt").write_text("needle public\n", encoding="utf-8")
        (secret_dir / "denied.txt").write_text("needle secret\n", encoding="utf-8")
        _install_permissions_config(monkeypatch, paths=[str(secret_dir / "**")])

        with patch("tools.file_tools._get_file_ops") as mock_get:
            result = json.loads(file_tools.search_tool(
                "needle",
                path=str(public_dir),
                task_id="deny-search-descendant",
            ))

        assert "error" in result
        assert "permissions.deny.paths" in result["error"]
        mock_get.assert_not_called()

    def test_search_blocks_wildcard_segment_overlap_before_file_ops(self, monkeypatch):
        root = "/workspace/private-team"
        _install_permissions_config(
            monkeypatch,
            paths=["/workspace/private*/secret/**"],
        )

        with (
            patch("tools.file_tools._resolve_path_for_task", return_value=PurePosixPath(root)),
            patch("tools.file_tools._get_file_ops") as mock_get,
        ):
            result = json.loads(file_tools.search_tool(
                "needle",
                path=root,
                task_id="deny-search-wildcard-segment",
            ))

        assert "error" in result
        assert "permissions.deny.paths" in result["error"]
        mock_get.assert_not_called()

    def test_search_blocks_pattern_without_literal_prefix_before_file_ops(self, monkeypatch):
        root = "/workspace/public"
        _install_permissions_config(
            monkeypatch,
            paths=["**/secret/**"],
        )

        with (
            patch("tools.file_tools._resolve_path_for_task", return_value=PurePosixPath(root)),
            patch("tools.file_tools._get_file_ops") as mock_get,
        ):
            result = json.loads(file_tools.search_tool(
                "needle",
                path=root,
                task_id="deny-search-universal-overlap",
            ))

        assert "error" in result
        assert "permissions.deny.paths" in result["error"]
        mock_get.assert_not_called()

    def test_search_file_probe_uses_task_resolved_root_only(self, monkeypatch):
        lexical = "shared"
        resolved = PurePosixPath("/task/shared")
        _install_permissions_config(
            monkeypatch,
            paths=["/task/shared/secret/**"],
        )

        def fake_isfile(candidate):
            return str(candidate) == lexical

        with (
            patch("tools.file_tools._terminal_env_type_for_task", return_value="local"),
            patch("tools.file_tools._resolve_path_for_task", return_value=resolved),
            patch("tools.file_tools.os.path.isfile", side_effect=fake_isfile) as mock_isfile,
            patch("tools.file_tools._get_file_ops") as mock_get,
        ):
            result = json.loads(file_tools.search_tool(
                "needle",
                path=lexical,
                task_id="deny-search-authoritative-root",
            ))

        assert "error" in result
        assert "permissions.deny.paths" in result["error"]
        mock_isfile.assert_called_once_with(str(resolved))
        mock_get.assert_not_called()

    def test_search_preserves_lexical_root_before_task_resolution(self, monkeypatch):
        lexical = "C:/workspace/link"
        resolved = Path("C:/outside")
        _install_permissions_config(
            monkeypatch,
            paths=["C:/workspace/*/secret/**"],
        )

        with (
            patch("tools.file_tools._resolve_path_for_task", return_value=resolved),
            patch("tools.file_tools._get_file_ops") as mock_get,
        ):
            result = json.loads(file_tools.search_tool(
                "needle",
                path=lexical,
                task_id="deny-search-alias",
            ))

        assert "error" in result
        assert "permissions.deny.paths" in result["error"]
        mock_get.assert_not_called()

    def test_search_allows_local_file_with_textual_denied_descendant(self, monkeypatch, tmp_path):
        candidate = tmp_path / "public.txt"
        candidate.write_text("needle public\n", encoding="utf-8")
        _install_permissions_config(
            monkeypatch,
            paths=[str(candidate / "secret" / "**")],
        )

        fake_result = MagicMock()
        fake_result.matches = [MagicMock(path=str(candidate), content="needle public")]
        fake_result.files = []
        fake_result.counts = {}
        fake_result.to_dict.return_value = {
            "matches": [{"path": str(candidate), "content": "needle public"}],
            "total_count": 1,
        }
        fake_ops = MagicMock()
        fake_ops.search.return_value = fake_result

        with patch("tools.file_tools._get_file_ops", return_value=fake_ops) as mock_get:
            result = json.loads(file_tools.search_tool(
                "needle",
                path=str(candidate),
                task_id="deny-search-file",
            ))

        assert "error" not in result
        mock_get.assert_called_once()

    def test_policy_config_load_failure_blocks_before_file_ops(self, monkeypatch, tmp_path):
        candidate = tmp_path / "candidate.txt"
        candidate.write_text("must not be read\n", encoding="utf-8")
        monkeypatch.setattr(
            deny_policy,
            "load_user_config",
            MagicMock(side_effect=RuntimeError("broken config")),
        )

        with patch("tools.file_tools._get_file_ops") as mock_get:
            result = json.loads(file_tools.read_file_tool(
                str(candidate),
                task_id="deny-config-failure",
            ))

        assert "error" in result
        assert "permissions.deny.paths" in result["error"]
        assert "No backend content operation was attempted" in result["error"]
        mock_get.assert_not_called()

    def test_malformed_path_policy_blocks_before_file_ops(self, monkeypatch, tmp_path):
        candidate = tmp_path / "candidate.txt"
        candidate.write_text("must not be read\n", encoding="utf-8")
        _install_permissions_config(monkeypatch, paths=[str(candidate), 42])

        with patch("tools.file_tools._get_file_ops") as mock_get:
            result = json.loads(file_tools.read_file_tool(
                str(candidate),
                task_id="deny-malformed-path-config",
            ))

        assert "error" in result
        assert "permissions.deny.paths" in result["error"]
        assert "could not be evaluated" in result["error"]
        mock_get.assert_not_called()

    def test_effective_home_resolution_failure_blocks_before_file_ops(self, monkeypatch, tmp_path):
        candidate = tmp_path / "candidate.txt"
        candidate.write_text("must not be read\n", encoding="utf-8")
        _install_permissions_config(monkeypatch, paths=["~/secret/**"])

        with (
            patch(
                "hermes_constants.get_subprocess_home",
                side_effect=RuntimeError("home unavailable"),
            ),
            patch("tools.file_tools._get_file_ops") as mock_get,
        ):
            result = json.loads(file_tools.read_file_tool(
                str(candidate),
                task_id="deny-home-failure",
            ))

        assert "error" in result
        assert "permissions.deny.paths" in result["error"]
        assert "could not be evaluated" in result["error"]
        mock_get.assert_not_called()


class TestPermissionsDenyCommandsAlias:
    def test_permissions_deny_commands_alias_blocks_before_yolo(self, monkeypatch):
        monkeypatch.setattr(approval_mod, "_get_approval_config", lambda: {"mode": "off", "deny": []})
        _install_permissions_config(monkeypatch, commands=["git push --force*"])

        result = approval_mod.check_all_command_guards("git push --force origin main", "local")

        assert result["approved"] is False
        assert result.get("user_deny") is True
        assert "permissions.deny.commands" in result["message"]

    def test_permissions_deny_commands_load_failure_blocks_both_guards_before_yolo(self, monkeypatch):
        monkeypatch.setattr(
            approval_mod,
            "_get_approval_config",
            lambda: {"mode": "off", "deny": []},
        )
        monkeypatch.setattr(
            deny_policy,
            "load_user_config",
            MagicMock(side_effect=RuntimeError("broken config")),
        )

        for guard in (
            approval_mod.check_dangerous_command,
            approval_mod.check_all_command_guards,
        ):
            result = guard("git status", "local")
            assert result["approved"] is False
            assert result.get("user_deny") is True
            assert "permissions.deny.commands" in result["message"]
            assert "could not be evaluated" in result["message"]

    def test_malformed_command_policy_blocks_in_both_guard_paths(self, monkeypatch):
        monkeypatch.setattr(
            approval_mod,
            "_get_approval_config",
            lambda: {"mode": "off", "deny": []},
        )
        _install_permissions_config(monkeypatch, commands=[42])

        for guard in (
            approval_mod.check_dangerous_command,
            approval_mod.check_all_command_guards,
        ):
            result = guard("git status", "local")
            assert result["approved"] is False
            assert result.get("user_deny") is True
            assert "permissions.deny.commands" in result["message"]
            assert "could not be evaluated" in result["message"]

    def test_approvals_deny_still_works_without_permissions_config(self, monkeypatch):
        monkeypatch.setattr(approval_mod, "_get_approval_config", lambda: {
            "mode": "manual",
            "deny": ["git push --force*"],
        })
        monkeypatch.setattr(deny_policy, "load_user_config", lambda: {})

        result = approval_mod.check_all_command_guards("git push --force origin main", "local")

        assert result["approved"] is False
        assert result.get("user_deny") is True
