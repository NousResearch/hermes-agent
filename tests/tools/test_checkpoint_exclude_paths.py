"""Regression tests for ``checkpoints.exclude_paths`` (#125623).

Working directories that equal or live under a configured path glob are
never snapshotted — one pre-write checkpoint inside an installed game
(wine ``drive_c`` tree) staged 811 MB of assets, each file under
``max_file_size_mb``, and the store cap could not reclaim it because
prune always keeps one snapshot per project.
"""

import os
from pathlib import Path

from tools.checkpoint_manager import CheckpointManager, _excluded_by_config


PATTERNS = ["**/drive_c/**", "~/Games/**", "/media/*/4tb/Games/**"]


class TestExcludedByConfig:
    def test_wine_prefix_subtree_is_excluded(self):
        # The ballooning case from the issue: the working dir IS inside a
        # wine prefix's drive_c tree.
        assert _excluded_by_config(
            "/Users/foo/.wine/drive_c/Games/EA", PATTERNS) is not None

    def test_drive_c_base_dir_itself_is_excluded(self):
        assert _excluded_by_config("/Users/foo/.wine/drive_c", PATTERNS) is not None

    def test_tilde_glob_expands(self):
        # ~ must expand to the real home (regardless of its basename).
        home_games = os.path.join(Path.home(), "Games", "Chess")
        assert _excluded_by_config(home_games, PATTERNS) is not None

    def test_absolute_glob_covers_base_and_subdirs(self):
        # Trailing /** covers the base directory itself...
        assert _excluded_by_config("/media/x/4tb/Games", PATTERNS) is not None
        # ...and everything under it.
        assert _excluded_by_config("/media/x/4tb/Games/Subdir", PATTERNS) is not None

    def test_component_boundary_not_swallowed(self):
        # /Games must never swallow /Games2 — matching is component-wise.
        assert _excluded_by_config("/media/x/4tb/Games2", PATTERNS) is None

    def test_normal_project_is_untouched(self):
        assert _excluded_by_config("/Users/foo/dev/project", PATTERNS) is None

    def test_empty_patterns_never_exclude(self):
        assert _excluded_by_config("/Users/foo/.wine/drive_c/Games", []) is None

    def test_blank_entries_are_ignored(self):
        assert _excluded_by_config("/Users/foo/.wine/drive_c/Games", ["", "  "]) is None

    def test_scalar_pattern_is_supported(self):
        assert _excluded_by_config("/Users/foo/.wine/drive_c/Games", "**/drive_c/**") is not None

    def test_relative_pattern_matches_path_suffix(self):
        assert _excluded_by_config("/home/dog/dev/relproj/node_modules/pkg", "node_modules/**") is not None

    def test_component_glob_does_not_overmatch_nested_paths(self):
        assert _excluded_by_config("/srv/a/b/data", "/srv/*/data") is None


class TestEnsureCheckpointSkip:
    def test_excluded_dir_never_snapshotted(self, tmp_path, monkeypatch):
        base = tmp_path / "checkpoints"
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", base)
        game = tmp_path / "prefix" / "drive_c" / "Games" / "game"
        game.mkdir(parents=True)
        (game / "config.ini").write_text("vsync=1\n")

        mgr = CheckpointManager(enabled=True, exclude_paths=[f"{tmp_path}/prefix/drive_c/**"])
        assert mgr.ensure_checkpoint(str(game), "auto") is False
        # Skipped directories must not enter the per-turn dedup set: a skip
        # is a policy decision, not a completed checkpoint.
        assert mgr._checkpointed_dirs == set()

    def test_non_excluded_dir_still_snapshots(self, tmp_path, monkeypatch):
        base = tmp_path / "checkpoints"
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", base)
        project = tmp_path / "project"
        project.mkdir()
        (project / "main.py").write_text("print('hello')\n")

        mgr = CheckpointManager(enabled=True, exclude_paths=[f"{tmp_path}/prefix/drive_c/**"])
        assert mgr.ensure_checkpoint(str(project), "auto") is True
        normalized = str(Path(str(project)).expanduser().resolve())
        assert normalized in mgr._checkpointed_dirs

    def test_default_empty_excludes_preserves_behavior(self, tmp_path, monkeypatch):
        base = tmp_path / "checkpoints"
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", base)
        project = tmp_path / "project"
        project.mkdir()
        (project / "main.py").write_text("print('hello')\n")

        mgr = CheckpointManager(enabled=True)  # no exclude_paths
        assert mgr.exclude_paths == []
        assert mgr.ensure_checkpoint(str(project), "auto") is True


class TestWiring:
    def test_gateway_kwargs_carry_exclude_paths(self):
        from gateway.run import _checkpoint_agent_kwargs
        kwargs = _checkpoint_agent_kwargs({
            "checkpoints": {"enabled": True, "exclude_paths": ["**/drive_c/**"]}})
        assert kwargs["checkpoint_exclude_paths"] == ["**/drive_c/**"]

    def test_gateway_kwargs_default_to_empty_list(self):
        from gateway.run import _checkpoint_agent_kwargs
        kwargs = _checkpoint_agent_kwargs({"checkpoints": {"enabled": True}})
        assert kwargs["checkpoint_exclude_paths"] == []

    def test_config_defaults_declare_exclude_paths(self):
        from hermes_cli.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["checkpoints"]["exclude_paths"] == []
