"""Tests for the HOME policy of the bubblewrap backend.

Everything here is pure: the resolvers and the layout builder take explicit
inputs and never spawn bwrap. The sandbox itself is exercised in
test_bubblewrap_secrets.py and test_bubblewrap_integration.py.
"""

import os
from unittest.mock import patch

import pytest

from tools.environments import bubblewrap_home
from tools.environments.bubblewrap_home import (
    ALLOWED_HOME_ENTRIES,
    DENIED_HOME_PATHS,
    allow_unit,
    denied_home_names,
    denied_home_paths,
    resolve_allowlist,
)


@pytest.fixture
def home(tmp_path):
    path = tmp_path / "home"
    path.mkdir()
    return str(path)


class TestDenylist:
    def test_denylist_names_the_listed_paths_under_home(self, home):
        names = denied_home_names(home)
        for rel in (".ssh", ".pgpass", ".git-credentials", ".config/gh", ".cargo/credentials.toml",
                    ".local/share/keyrings", ".cache/huggingface/token"):
            assert os.path.join(home, rel) in names

    def test_denylist_holds_every_home_path_of_the_file_safety_policy(self, home):
        from agent.file_safety import build_write_denied_paths, build_write_denied_prefixes

        policy = set(build_write_denied_paths(home))
        policy |= {prefix.rstrip(os.sep) for prefix in build_write_denied_prefixes(home)}
        under_home = {p for p in policy if p.startswith(home + os.sep)}
        assert under_home, "the policy names paths under HOME"
        hidden = denied_home_paths(home)
        for path in under_home:
            assert any(path == h or path.startswith(h + os.sep) for h in hidden), path

    def test_denylist_follows_a_path_added_to_the_policy(self, home):
        invented = os.path.join(home, ".zz-invented-store")
        with patch("agent.file_safety.build_write_denied_paths", return_value={invented, "/etc/shadow"}):
            hidden = denied_home_paths(home)
        assert invented in hidden
        assert ".zz-invented-store" not in DENIED_HOME_PATHS
        assert "/etc/shadow" not in hidden

    def test_denylist_follows_a_prefix_added_to_the_policy(self, home):
        invented = os.path.join(home, ".zz-invented-dir")
        with patch("agent.file_safety.build_write_denied_prefixes", return_value=[invented + os.sep]):
            assert invented in denied_home_paths(home)

    def test_denylist_resolves_a_symlinked_entry_to_the_directory_that_holds_the_secret(self, home):
        os.makedirs(os.path.join(home, "dotfiles", "ssh"))
        os.symlink(os.path.join(home, "dotfiles", "ssh"), os.path.join(home, ".ssh"))
        hidden = denied_home_paths(home)
        assert os.path.join(home, "dotfiles", "ssh") in hidden
        assert os.path.join(home, ".ssh") in denied_home_names(home)

    def test_denylist_drops_an_entry_that_lies_under_another(self, home):
        hidden = denied_home_paths(home)
        assert os.path.join(home, ".ssh") in hidden
        assert not any(h.startswith(os.path.join(home, ".ssh") + os.sep) for h in hidden)

    def test_denylist_has_no_duplicates_and_stays_under_home(self, home):
        hidden = denied_home_paths(home)
        assert len(hidden) == len(set(hidden))
        assert all(h.startswith(home + os.sep) for h in hidden)

    def test_denylist_survives_a_policy_that_raises(self, home):
        with patch("agent.file_safety.build_write_denied_paths", side_effect=RuntimeError("boom")):
            hidden = denied_home_paths(home)
        assert os.path.join(home, ".ssh") in hidden

    def test_module_denylist_entries_are_relative(self):
        assert all(not os.path.isabs(rel) and not rel.startswith("~") for rel in DENIED_HOME_PATHS)
        assert bubblewrap_home.DENIED_HOME_PATHS is DENIED_HOME_PATHS


def _allowlist(home, path_env="", allow_items=()):
    return resolve_allowlist(
        home, path_env, allow_items,
        denied_names=denied_home_names(home), denied_paths=denied_home_paths(home),
    )


class TestAllowlist:
    def test_allowlist_shipped_set(self):
        assert set(ALLOWED_HOME_ENTRIES) == {
            ".bashrc", ".bash_profile", ".bash_login", ".bash_logout", ".bash_aliases", ".profile",
            ".zshrc", ".zshenv", ".zprofile", ".inputrc", ".gitconfig", ".editorconfig",
            ".tool-versions", ".terminfo", ".cache", ".cargo", ".rustup", ".nvm", ".bun", ".deno",
            ".gem", ".npm", ".pyenv", ".rbenv", ".sdkman", ".volta", ".asdf", ".m2", ".gradle",
            ".dotnet", ".pub-cache", ".conda", ".nix-profile",
            ".config/git", ".config/pip", ".config/uv", ".config/npm", ".config/pnpm",
            ".config/yarn", ".config/go", ".config/fontconfig",
            ".local/bin", ".local/lib", ".local/include",
            ".local/share/uv", ".local/share/pipx", ".local/share/pnpm", ".local/share/virtualenvs",
            ".local/share/man", ".local/share/bash-completion", ".local/share/fonts", ".local/share/mime",
        }
        assert len(ALLOWED_HOME_ENTRIES) == len(set(ALLOWED_HOME_ENTRIES))

    def test_allowlist_default_holds_the_shipped_set(self, home):
        assert set(_allowlist(home)) == set(ALLOWED_HOME_ENTRIES)

    @pytest.mark.parametrize("rel, unit", [
        (".zz-tool/bin", ".zz-tool"),
        (".zz-tool", ".zz-tool"),
        (".local/bin", ".local/bin"),
        (".local/share/zz/bin", ".local/share/zz"),
        (".config/zz-app/deep/er", ".config/zz-app"),
        (".local", None),
        (".local/share", None),
        (".config", None),
        ("bin", None),
        ("Documents/.hidden", None),
    ])
    def test_allowlist_unit_is_the_smallest_holder(self, home, rel, unit):
        assert allow_unit(home, os.path.join(home, rel)) == unit

    def test_allowlist_unit_is_none_outside_home(self, home):
        assert allow_unit(home, "/usr/bin") is None
        assert allow_unit(home, home) is None

    def test_allowlist_path_rule_adds_the_smallest_units(self, home):
        path_env = os.pathsep.join([
            "/usr/bin", os.path.join(home, ".zz-tool", "bin"), os.path.join(home, ".local", "bin"),
            os.path.join(home, ".local", "share", "zz", "bin"), os.path.join(home, "bin"), "",
        ])
        allowed = _allowlist(home, path_env)
        assert ".zz-tool" in allowed
        assert ".local/bin" in allowed
        assert ".local/share/zz" in allowed
        assert ".local" not in allowed
        assert ".local/share" not in allowed
        assert "bin" not in allowed

    def test_allowlist_path_rule_follows_a_symlinked_home(self, home, tmp_path):
        link = tmp_path / "home-link"
        link.symlink_to(home)
        allowed = _allowlist(str(link), os.path.join(home, ".zz-tool", "bin"))
        assert ".zz-tool" in allowed

    def test_allowlist_path_rule_never_shows_a_denied_path(self, home, caplog):
        allowed = _allowlist(home, os.path.join(home, ".ssh", "bin"))
        assert ".ssh" not in allowed

    def test_allowlist_accepts_operator_units(self, home):
        allowed = _allowlist(home, allow_items=(".zz-extra", ".config/zz-app", "~/.zz-tilde", " .zz-space "))
        assert {".zz-extra", ".config/zz-app", ".zz-tilde", ".zz-space"} <= set(allowed)

    @pytest.mark.parametrize("item", [
        "/etc/passwd", ".config", ".local/share", ".config/a/b/c", ".config/a/b", "Documents", "..", ".", "",
        ".zz/../.ssh",
    ])
    def test_allowlist_drops_an_item_that_is_not_a_unit(self, home, caplog, item):
        with caplog.at_level("WARNING"):
            allowed = _allowlist(home, allow_items=(item,))
        assert set(allowed) == set(ALLOWED_HOME_ENTRIES)
        warnings = [r for r in caplog.records if "bubblewrap_home_allow" in r.getMessage()]
        assert len(warnings) == (1 if item.strip() else 0)

    @pytest.mark.parametrize("item", [".aws", ".config/gh", ".ssh"])
    def test_allowlist_denylist_wins_over_the_allow_key(self, home, caplog, item):
        with caplog.at_level("WARNING"):
            allowed = _allowlist(home, allow_items=(item,))
        assert item not in allowed
        assert len([r for r in caplog.records if "bubblewrap_home_allow" in r.getMessage()]) == 1

    def test_allowlist_keeps_a_unit_that_only_contains_a_denied_path(self, home):
        assert ".cargo" in _allowlist(home)

    def test_allowlist_reads_no_process_environment(self, home, monkeypatch):
        monkeypatch.setenv("PATH", os.path.join(home, ".zz-env", "bin"))
        monkeypatch.setenv("HOME", "/nonexistent")
        assert ".zz-env" not in _allowlist(home, "/usr/bin")

    def test_allowlist_is_sorted_and_unique(self, home):
        allowed = _allowlist(home, os.path.join(home, ".cargo", "bin"), (".cargo",))
        assert list(allowed) == sorted(set(allowed))
