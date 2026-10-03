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
    DENIED_HOME_PATHS,
    denied_home_names,
    denied_home_paths,
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
