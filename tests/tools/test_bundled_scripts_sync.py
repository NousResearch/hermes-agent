from pathlib import Path
from unittest.mock import patch

from tools import bundled_scripts_sync as bss


class TestSyncBundledScripts:
    def test_links_bundled_script_into_local_bin(self, tmp_path):
        repo = tmp_path / "repo"
        src = repo / "scripts" / "telegram-healthcheck-stateful.sh"
        src.parent.mkdir(parents=True)
        src.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")

        home = tmp_path / "home"
        local_bin = home / ".local" / "bin"
        manifest = home / ".hermes" / ".bundled_scripts_manifest"

        with patch.object(bss, "_project_root", return_value=repo), patch.object(
            bss, "SCRIPTS_MANIFEST", manifest
        ), patch.object(bss.Path, "home", return_value=home), patch.object(
            bss,
            "BUNDLED_SCRIPTS",
            [("scripts/telegram-healthcheck-stateful.sh", "telegram-healthcheck-stateful")],
        ):
            result = bss.sync_bundled_scripts(quiet=True)

        dest = local_bin / "telegram-healthcheck-stateful"
        assert result["linked"] == ["telegram-healthcheck-stateful"]
        assert dest.is_symlink()
        assert dest.resolve() == src.resolve()
        assert "telegram-healthcheck-stateful:" in manifest.read_text(encoding="utf-8")

    def test_repoints_existing_symlink_when_source_changes(self, tmp_path):
        repo = tmp_path / "repo"
        src = repo / "scripts" / "telegram-healthcheck-stateful.sh"
        src.parent.mkdir(parents=True)
        src.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")

        old_repo = tmp_path / "old-repo"
        old_src = old_repo / "scripts" / "telegram-healthcheck-stateful.sh"
        old_src.parent.mkdir(parents=True)
        old_src.write_text("#!/bin/sh\necho old\n", encoding="utf-8")

        home = tmp_path / "home"
        local_bin = home / ".local" / "bin"
        local_bin.mkdir(parents=True)
        dest = local_bin / "telegram-healthcheck-stateful"
        dest.symlink_to(old_src)
        manifest = home / ".hermes" / ".bundled_scripts_manifest"

        with patch.object(bss, "_project_root", return_value=repo), patch.object(
            bss, "SCRIPTS_MANIFEST", manifest
        ), patch.object(bss.Path, "home", return_value=home), patch.object(
            bss,
            "BUNDLED_SCRIPTS",
            [("scripts/telegram-healthcheck-stateful.sh", "telegram-healthcheck-stateful")],
        ):
            result = bss.sync_bundled_scripts(quiet=True)

        assert result["updated"] == ["telegram-healthcheck-stateful"]
        assert dest.is_symlink()
        assert dest.resolve() == src.resolve()

    def test_prefix_overrides_link_destination(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        src = repo / "scripts" / "telegram-healthcheck-stateful.sh"
        src.parent.mkdir(parents=True)
        src.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
        prefix = tmp_path / "prefix"
        manifest = tmp_path / "home" / ".hermes" / ".bundled_scripts_manifest"
        monkeypatch.setenv("PREFIX", str(prefix))

        with patch.object(bss, "_project_root", return_value=repo), patch.object(
            bss, "SCRIPTS_MANIFEST", manifest
        ), patch.object(
            bss,
            "BUNDLED_SCRIPTS",
            [("scripts/telegram-healthcheck-stateful.sh", "telegram-healthcheck-stateful")],
        ):
            result = bss.sync_bundled_scripts(quiet=True)

        dest = prefix / "bin" / "telegram-healthcheck-stateful"
        assert result["linked"] == ["telegram-healthcheck-stateful"]
        assert dest.is_symlink()
        assert dest.resolve() == src.resolve()

    def test_reports_missing_sources(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        home = tmp_path / "home"
        manifest = home / ".hermes" / ".bundled_scripts_manifest"

        with patch.object(bss, "_project_root", return_value=repo), patch.object(
            bss, "SCRIPTS_MANIFEST", manifest
        ), patch.object(bss.Path, "home", return_value=home), patch.object(
            bss,
            "BUNDLED_SCRIPTS",
            [("scripts/missing.sh", "missing-helper")],
        ):
            result = bss.sync_bundled_scripts(quiet=True)

        assert result["missing"] == ["missing-helper"]
        assert result["linked"] == []
        assert result["updated"] == []
