"""Tests for atomic_roundtrip_yaml_save() — comment-preserving full-state writes.

This helper backs tui_gateway/server.py:_save_cfg(), which used to call
yaml.safe_dump and silently clobber user-edited config files on every
TUI/gateway setting change (e.g. /personality, /reasoning, /details_mode).
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import hermes_yaml as yaml


class TestAtomicRoundtripYamlSave:
    @pytest.fixture
    def config_path(self, tmp_path):
        return tmp_path / "config.yaml"

    def test_creates_file_when_missing(self, config_path):
        from utils import atomic_roundtrip_yaml_save

        atomic_roundtrip_yaml_save(config_path, {"model": {"default": "test-model"}})

        assert config_path.exists()
        assert yaml.safe_load(config_path.read_text())["model"]["default"] == "test-model"

    def test_preserves_top_level_key_order(self, config_path):
        """Existing top-level keys keep their author-intended ordering."""
        config_path.write_text(
            "model:\n"
            "  default: claude-opus-4-7\n"
            "providers: {}\n"
            "agent:\n"
            "  max_turns: 90\n"
            "display:\n"
            "  skin: default\n",
            encoding="utf-8",
        )

        from utils import atomic_roundtrip_yaml_save

        # Pass keys in alphabetical order to make sure dict iteration order
        # in the caller doesn't accidentally rewrite the file alphabetically
        # (the old yaml.safe_dump bug).
        atomic_roundtrip_yaml_save(
            config_path,
            {
                "agent": {"max_turns": 100},
                "display": {"skin": "mono"},
                "model": {"default": "claude-opus-4-7"},
                "providers": {},
            },
        )

        text = config_path.read_text(encoding="utf-8")
        top_keys = [
            line.split(":", 1)[0]
            for line in text.splitlines()
            if line and not line.startswith(" ") and not line.startswith("#")
        ]
        # Comments are stripped from `top_keys` above, so the surviving
        # order should match the original file, NOT alphabetical.
        assert top_keys == ["model", "providers", "agent", "display"]

    def test_preserves_comments(self, config_path):
        config_path.write_text(
            "# header comment\n"
            "model:\n"
            "  # inline note\n"
            "  default: claude-opus-4-7\n"
            "display:\n"
            "  skin: default  # trailing note\n",
            encoding="utf-8",
        )

        from utils import atomic_roundtrip_yaml_save

        atomic_roundtrip_yaml_save(
            config_path,
            {
                "model": {"default": "claude-opus-4-7"},
                "display": {"skin": "mono"},
            },
        )

        text = config_path.read_text(encoding="utf-8")
        assert "# header comment" in text
        assert "# inline note" in text
        assert "# trailing note" in text
        assert yaml.safe_load(text)["display"]["skin"] == "mono"

    def test_preserves_readable_unicode(self, config_path):
        """Personalities with kaomoji/Chinese characters stay readable on disk
        instead of getting mangled to \\uXXXX escapes (the headline bug:
        kawaii/catgirl personality emoji turning into \\u30CE\\uFF65)."""
        config_path.write_text(
            "agent:\n"
            "  personalities:\n"
            "    catgirl: \"nya (=^･ω･^=) 你好\"\n"
            "display:\n"
            "  skin: default\n",
            encoding="utf-8",
        )

        from utils import atomic_roundtrip_yaml_save

        atomic_roundtrip_yaml_save(
            config_path,
            {
                "agent": {"personalities": {"catgirl": "nya (=^･ω･^=) 你好"}},
                "display": {"skin": "mono"},
            },
        )

        text = config_path.read_text(encoding="utf-8")
        assert "你好" in text
        assert "(=^･ω･^=)" in text
        assert "\\u4f60" not in text
        assert "\\u30CE" not in text

    def test_preserves_long_double_quoted_scalar_with_backslash(self, config_path):
        """A no-op save must not turn fold indentation after a backslash into data."""
        value = "A" * 74 + r"D:\CentBrowserPortable " + "B" * 40
        config_path.write_text(
            'policy: "' + value.replace("\\", "\\\\") + '"\n',
            encoding="utf-8",
        )

        from utils import atomic_roundtrip_yaml_save

        atomic_roundtrip_yaml_save(config_path, {"policy": value})

        assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["policy"] == value

    def test_appends_new_keys(self, config_path):
        config_path.write_text(
            "model:\n"
            "  default: test-model\n",
            encoding="utf-8",
        )

        from utils import atomic_roundtrip_yaml_save

        atomic_roundtrip_yaml_save(
            config_path,
            {
                "model": {"default": "test-model"},
                "display": {"personality": "noir"},
                "agent": {"system_prompt": "you are noir"},
            },
        )

        result = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert result["model"]["default"] == "test-model"
        assert result["display"]["personality"] == "noir"
        assert result["agent"]["system_prompt"] == "you are noir"

    def test_deletes_keys_missing_from_new_state(self, config_path):
        """Mirrors the cfg.pop()-then-_save_cfg(cfg) pattern in tui_gateway:
        e.g. /prompt clear removes custom_prompt entirely."""
        config_path.write_text(
            "model:\n"
            "  default: test-model\n"
            "custom_prompt: 'old prompt'\n",
            encoding="utf-8",
        )

        from utils import atomic_roundtrip_yaml_save

        atomic_roundtrip_yaml_save(
            config_path,
            {"model": {"default": "test-model"}},
        )

        result = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert "custom_prompt" not in result
        assert result["model"]["default"] == "test-model"

    def test_overwrites_scalar_value(self, config_path):
        config_path.write_text(
            "display:\n"
            "  personality: noir\n",
            encoding="utf-8",
        )

        from utils import atomic_roundtrip_yaml_save

        atomic_roundtrip_yaml_save(
            config_path,
            {"display": {"personality": "kawaii"}},
        )

        result = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert result["display"]["personality"] == "kawaii"

    def test_overwrites_list_wholesale(self, config_path):
        config_path.write_text(
            "toolsets:\n"
            "  - one\n"
            "  - two\n",
            encoding="utf-8",
        )

        from utils import atomic_roundtrip_yaml_save

        atomic_roundtrip_yaml_save(config_path, {"toolsets": ["three"]})

        result = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert result["toolsets"] == ["three"]

    def test_recurses_into_nested_dicts(self, config_path):
        """Deep mutations target the matching subtree, not the whole parent.

        Without this, writing display.personality would drop sibling display.skin.
        """
        config_path.write_text(
            "display:\n"
            "  skin: default\n"
            "  personality: noir\n"
            "  compact: false\n",
            encoding="utf-8",
        )

        from utils import atomic_roundtrip_yaml_save

        atomic_roundtrip_yaml_save(
            config_path,
            {
                "display": {
                    "skin": "default",
                    "personality": "kawaii",
                    "compact": False,
                }
            },
        )

        result = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert result["display"]["skin"] == "default"
        assert result["display"]["personality"] == "kawaii"
        assert result["display"]["compact"] is False

    @staticmethod
    def _deny_config_reads(config_path):
        real_open = open

        def fake_open(file, mode="r", *args, **kwargs):
            if Path(file) == config_path and "r" in mode:
                raise PermissionError("denied")
            return real_open(file, mode, *args, **kwargs)

        return fake_open

    def test_refuses_to_overwrite_unreadable_existing_config(self, config_path):
        """Shares the fail-closed contract with hermes_cli.config.atomic_config_write:
        an existing-but-unreadable config.yaml (permission error, broken mount)
        must raise rather than being silently replaced with only new_state."""
        original = "model:\n  default: test-model\n"
        config_path.write_text(original, encoding="utf-8")

        from utils import atomic_roundtrip_yaml_save

        with patch("builtins.open", side_effect=self._deny_config_reads(config_path)):
            with pytest.raises(RuntimeError, match="this change was not saved"):
                atomic_roundtrip_yaml_save(config_path, {"model": {"default": "replacement"}})

        assert config_path.read_text(encoding="utf-8") == original

    def test_restores_owner(self, config_path, monkeypatch):
        """Mirrors atomic_roundtrip_yaml_update_restores_owner — the write path
        must preserve the config file's original owner across the temp-file +
        atomic-replace swap, not just its mode."""
        if os.name != "posix":
            pytest.skip("POSIX-only")

        config_path.write_text("model:\n  default: test-model\n", encoding="utf-8")

        chown_calls: list[tuple[Path, int, int]] = []
        monkeypatch.setattr("utils._preserve_file_owner", lambda _path: (345, 678))
        monkeypatch.setattr(
            "utils.os.chown",
            lambda path, uid, gid: chown_calls.append((Path(path), uid, gid)),
        )

        from utils import atomic_roundtrip_yaml_save

        atomic_roundtrip_yaml_save(config_path, {"model": {"default": "updated-model"}})

        assert chown_calls == [(config_path, 345, 678)]


class TestSequenceEntriesKeepTheirOwnFormatting:
    """An entry that survives a list edit keeps its own quoting and comment (#147594)."""

    ORIGINAL = (
        "skills:\n"
        "  external_dirs:\n"
        '    - "skills/design"  # design\n'
        '    - "skills/pilot"  # pilot\n'
        "    - 'skills/third'  # third\n"
    )
    DESIGN, PILOT, THIRD = "skills/design", "skills/pilot", "skills/third"

    @pytest.fixture
    def config_path(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(self.ORIGINAL, encoding="utf-8")
        return path

    def _save_dirs(self, config_path, dirs):
        from utils import atomic_roundtrip_yaml_save

        atomic_roundtrip_yaml_save(config_path, {"skills": {"external_dirs": dirs}})
        written = config_path.read_text(encoding="utf-8-sig")
        assert yaml.safe_load(written)["skills"]["external_dirs"] == dirs
        return [" ".join(line.split()) for line in written.splitlines()[2:]]

    def test_removing_an_entry_leaves_the_others_untouched(self, config_path):
        assert self._save_dirs(config_path, [self.PILOT, self.THIRD]) == [
            '- "skills/pilot" # pilot',
            "- 'skills/third' # third",
        ]

    def test_removing_a_middle_entry_leaves_the_others_untouched(self, config_path):
        assert self._save_dirs(config_path, [self.DESIGN, self.THIRD]) == [
            '- "skills/design" # design',
            "- 'skills/third' # third",
        ]

    def test_reinserting_an_entry_leaves_the_survivors_untouched(self, config_path):
        self._save_dirs(config_path, [self.PILOT, self.THIRD])

        assert self._save_dirs(config_path, [self.DESIGN, self.PILOT, self.THIRD]) == [
            "- skills/design",
            '- "skills/pilot" # pilot',
            "- 'skills/third' # third",
        ]

    def test_editing_an_entry_in_place_keeps_its_siblings(self, config_path):
        lines = self._save_dirs(config_path, [self.DESIGN, "skills/renamed", self.THIRD])

        assert lines[0] == '- "skills/design" # design'
        assert lines[2] == "- 'skills/third' # third"

    def test_editing_a_mapping_entry_keeps_the_comments_inside_it(self, config_path):
        from utils import atomic_roundtrip_yaml_save

        config_path.write_text(
            "servers:\n"
            "  - name: first  # keep me\n"
            "    port: 1\n"
            "  - name: second\n"
            "    port: 2\n",
            encoding="utf-8",
        )
        servers = [{"name": "first", "port": 10}, {"name": "second", "port": 2}]

        atomic_roundtrip_yaml_save(config_path, {"servers": servers})

        written = config_path.read_text(encoding="utf-8-sig")
        assert "# keep me" in written
        assert yaml.safe_load(written)["servers"] == servers

    @pytest.mark.parametrize(
        ("before", "after"),
        [
            (["a", "b", "c"], ["c", "b", "a"]),
            (["a", "b", "b", "c"], ["b", "x", "b"]),
            (["a", "b"], []),
            ([1, True, "1"], [True, 1, "1"]),
            ([0, 1, 1.0], [1.0]),
            ([0, 1, 1.0], [1]),
            (["a"], ["x", "a", "y", "z"]),
        ],
    )
    def test_any_reshuffle_writes_exactly_the_requested_list(self, config_path, before, after):
        from utils import atomic_roundtrip_yaml_save

        atomic_roundtrip_yaml_save(config_path, {"items": before})
        atomic_roundtrip_yaml_save(config_path, {"items": after})

        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))["items"]
        assert loaded == after
        assert [type(value) for value in loaded] == [type(value) for value in after]
