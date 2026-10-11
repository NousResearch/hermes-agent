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
    home_layout_args,
    resolve_allowlist,
    resolve_home_root,
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


def _mounts(argv):
    """(flag, dest) for every mount directive in *argv*, in order."""
    arity = {"--tmpfs": 1, "--remount-ro": 1, "--bind": 2, "--ro-bind": 2, "--bind-try": 2,
             "--ro-bind-try": 2, "--symlink": 2}
    found, i = [], 0
    while i < len(argv):
        flag = argv[i]
        assert flag in arity, f"unexpected token {flag!r} at {i}"
        found.append((flag, argv[i + arity[flag]]))
        i += 1 + arity[flag]
    return found


def _layout(home, allowlist=None, hidden=(), **kwargs):
    allowlist = _allowlist(home) if allowlist is None else allowlist
    return home_layout_args(home, allowlist, hidden, empty_file=os.path.join(os.path.dirname(home), "empty"), **kwargs)


def _touch(home, rel, text="x"):
    path = os.path.join(home, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


class TestLayout:
    def test_layout_starts_with_a_tmpfs_on_home_and_ends_with_its_remount(self, home):
        argv = _layout(home)
        assert argv[:2] == ["--tmpfs", home]
        assert argv[-2:] == ["--remount-ro", home]

    def test_layout_shows_non_dot_entries_read_only(self, home):
        os.makedirs(os.path.join(home, "proj"))
        _touch(home, "notes.txt")
        mounts = _mounts(_layout(home))
        assert ("--ro-bind-try", os.path.join(home, "proj")) in mounts
        assert ("--ro-bind-try", os.path.join(home, "notes.txt")) in mounts

    def test_layout_shows_non_dot_entries_read_write_under_a_writable_home(self, home):
        os.makedirs(os.path.join(home, "proj"))
        _touch(home, ".bashrc")
        for root in (home, os.path.dirname(home)):
            mounts = _mounts(_layout(home, writable_roots=(root,)))
            assert ("--bind-try", os.path.join(home, "proj")) in mounts
            assert ("--ro-bind-try", os.path.join(home, ".bashrc")) in mounts

    def test_layout_keeps_entries_read_only_when_only_a_child_is_writable(self, home):
        os.makedirs(os.path.join(home, "proj", "sub"))
        mounts = _mounts(_layout(home, writable_roots=(os.path.join(home, "proj", "sub"),)))
        assert ("--ro-bind-try", os.path.join(home, "proj")) in mounts

    def test_layout_hides_a_dot_entry_that_is_not_allowed(self, home):
        _touch(home, ".zz-unlisted/secret")
        _touch(home, ".zz-unlisted-file")
        _touch(home, ".pgpass")
        dests = [dest for _flag, dest in _mounts(_layout(home))]
        assert not any(".zz-unlisted" in dest or ".pgpass" in dest for dest in dests)

    def test_layout_shows_an_allowed_dot_entry_read_only_even_under_a_writable_home(self, home):
        _touch(home, ".gitconfig")
        os.makedirs(os.path.join(home, ".cargo", "bin"))
        mounts = _mounts(_layout(home, writable_roots=(home,)))
        assert ("--ro-bind-try", os.path.join(home, ".gitconfig")) in mounts
        assert ("--ro-bind-try", os.path.join(home, ".cargo")) in mounts

    def test_layout_gives_each_default_deny_directory_a_tmpfs_with_its_allowed_children(self, home):
        _touch(home, ".config/git/config")
        _touch(home, ".config/zz-unlisted/token")
        _touch(home, ".local/bin/tool")
        _touch(home, ".local/state/history")
        _touch(home, ".local/share/pnpm/x")
        _touch(home, ".local/share/zz-unlisted/db")
        mounts = _mounts(_layout(home))
        for rel in (".config", ".local", ".local/share"):
            assert ("--tmpfs", os.path.join(home, rel)) in mounts
            assert ("--remount-ro", os.path.join(home, rel)) in mounts
        assert ("--ro-bind-try", os.path.join(home, ".config/git")) in mounts
        assert ("--ro-bind-try", os.path.join(home, ".local/bin")) in mounts
        assert ("--ro-bind-try", os.path.join(home, ".local/share/pnpm")) in mounts
        dests = [dest for _flag, dest in mounts]
        assert not any("zz-unlisted" in dest or dest.endswith(".local/state") for dest in dests)

    def test_layout_seals_inner_directories_before_outer_ones(self, home):
        _touch(home, ".local/share/pnpm/x")
        remounts = [dest for flag, dest in _mounts(_layout(home)) if flag == "--remount-ro"]
        assert remounts.index(os.path.join(home, ".local/share")) < remounts.index(os.path.join(home, ".local"))
        assert remounts[-1] == home

    def test_layout_orders_tmpfs_before_children_before_remount(self, home):
        _touch(home, ".config/git/config")
        mounts = _mounts(_layout(home))
        config = os.path.join(home, ".config")
        assert (mounts.index(("--tmpfs", config))
                < mounts.index(("--ro-bind-try", os.path.join(config, "git")))
                < mounts.index(("--remount-ro", config)))

    def test_layout_skips_a_default_deny_directory_missing_on_the_host(self, home):
        dests = [dest for _flag, dest in _mounts(_layout(home))]
        assert dests == [home, home]

    def test_layout_makes_a_visible_symlink_again_and_never_binds_through_it(self, home):
        _touch(home, "dotfiles/bashrc")
        os.symlink("dotfiles/bashrc", os.path.join(home, ".bashrc"))
        os.symlink("dotfiles", os.path.join(home, "df"))
        argv = _layout(home)
        at = argv.index(os.path.join(home, ".bashrc"))
        assert argv[at - 2:at + 1] == ["--symlink", "dotfiles/bashrc", os.path.join(home, ".bashrc")]
        mounts = _mounts(argv)
        assert ("--symlink", os.path.join(home, ".bashrc")) in mounts
        assert ("--symlink", os.path.join(home, "df")) in mounts
        binds = [dest for flag, dest in mounts if "bind" in flag]
        assert os.path.join(home, ".bashrc") not in binds and os.path.join(home, "df") not in binds

    def test_layout_allowed_entry_linked_to_a_hidden_directory_yields_only_a_symlink(self, home):
        _touch(home, ".zz-secret/key")
        os.symlink(".zz-secret", os.path.join(home, ".cargo"))
        mounts = _mounts(_layout(home))
        assert ("--symlink", os.path.join(home, ".cargo")) in mounts
        assert not any(".zz-secret" in dest for _flag, dest in mounts)

    def test_layout_skips_a_default_deny_directory_that_is_a_symlink(self, home):
        _touch(home, "dotfiles/config/git/config")
        os.symlink("dotfiles/config", os.path.join(home, ".config"))
        dests = [dest for _flag, dest in _mounts(_layout(home))]
        assert not any(dest.startswith(os.path.join(home, ".config")) for dest in dests)

    def test_layout_has_no_symlink_as_a_mount_destination(self, home):
        _touch(home, "dotfiles/bashrc")
        os.symlink("dotfiles/bashrc", os.path.join(home, ".bashrc"))
        os.makedirs(os.path.join(home, "real"))
        os.symlink("real", os.path.join(home, "link"))
        _touch(home, ".config/git/config")
        os.symlink("git", os.path.join(home, ".config", "pip"))
        for flag, dest in _mounts(_layout(home)):
            if flag != "--symlink":
                assert not os.path.islink(dest), dest
                assert os.path.realpath(dest) == dest

    def test_layout_overlays_a_hidden_path_below_a_visible_entry(self, home):
        creds = _touch(home, ".cargo/credentials.toml")
        keys = os.path.join(home, "Documents", "keys")
        os.makedirs(keys)
        empty = os.path.join(os.path.dirname(home), "empty")
        argv = _layout(home, hidden=(creds, keys))
        mounts = _mounts(argv)
        assert ("--ro-bind", creds) in mounts
        assert argv[argv.index(creds) - 1] == empty
        assert ("--tmpfs", keys) in mounts
        assert mounts.index(("--ro-bind-try", os.path.join(home, ".cargo"))) < mounts.index(("--ro-bind", creds))

    def test_layout_emits_no_overlay_for_a_hidden_path_that_is_not_visible(self, home):
        ssh = os.path.join(home, ".ssh")
        os.makedirs(ssh)
        dests = [dest for _flag, dest in _mounts(_layout(home, hidden=(ssh, os.path.join(home, ".netrc"))))]
        assert ssh not in dests

    def test_layout_leaves_out_a_top_level_entry_that_is_hidden(self, home):
        secrets = os.path.join(home, "Secrets")
        os.makedirs(secrets)
        dests = [dest for _flag, dest in _mounts(_layout(home, hidden=(secrets,)))]
        assert secrets not in dests

    def test_layout_places_binds_after_entries_and_late_args_after_overlays(self, home):
        os.makedirs(os.path.join(home, "proj", "sub"))
        gh = os.path.join(home, "proj", "sub", "gh")
        os.makedirs(gh)
        sub = os.path.join(home, "proj", "sub")
        state = os.path.join(home, ".zz-state")
        argv = _layout(home, hidden=(gh,), binds=(("--bind-try", sub, sub),), late_args=["--bind", state, state])
        mounts = _mounts(argv)
        order = [mounts.index(m) for m in (
            ("--ro-bind-try", os.path.join(home, "proj")), ("--bind-try", sub), ("--tmpfs", gh),
            ("--bind", state), ("--remount-ro", home),
        )]
        assert order == sorted(order)

    def test_layout_overlays_a_hidden_path_inside_a_bind(self, home):
        data = os.path.join(home, ".zz-data")
        token = _touch(home, ".zz-data/token")
        mounts = _mounts(_layout(home, hidden=(token,), binds=(("--bind", data, data),)))
        assert ("--ro-bind", token) in mounts

    def test_layout_uses_the_listing_it_is_given(self, home):
        os.makedirs(os.path.join(home, "proj"))
        os.makedirs(os.path.join(home, "other"))
        dests = [dest for _flag, dest in _mounts(_layout(home, listing=("proj",)))]
        assert os.path.join(home, "proj") in dests
        assert os.path.join(home, "other") not in dests

    def test_layout_is_empty_without_a_home_root(self):
        assert home_layout_args(None, (), (), empty_file="/nonexistent") == []

    def test_home_root_is_the_real_path(self, home, tmp_path):
        link = tmp_path / "parent-link"
        link.symlink_to(os.path.dirname(home))
        assert resolve_home_root(os.path.join(str(link), "home")) == os.path.realpath(home)

    @pytest.mark.parametrize("value", ["/", "", "/nonexistent/zz-home"])
    def test_home_root_is_none_for_an_unusable_home(self, value, caplog):
        with caplog.at_level("WARNING"):
            assert resolve_home_root(value) is None
        assert len(caplog.records) == 1

    def test_layout_does_not_seal_a_default_deny_directory_a_bind_replaces(self, home):
        _touch(home, ".config/git/config")
        config = os.path.join(home, ".config")
        mounts = _mounts(_layout(home, binds=(("--bind", config, config),)))
        assert ("--bind", config) in mounts
        assert ("--remount-ro", config) not in mounts
        assert ("--remount-ro", home) in mounts

    def test_layout_does_not_seal_a_directory_covered_by_a_bind_above_it(self, home):
        _touch(home, ".local/share/pnpm/x")
        local = os.path.join(home, ".local")
        for flag in ("--ro-bind", "--bind", "--ro-bind-try"):
            mounts = _mounts(_layout(home, binds=((flag, local, local),)))
            sealed = [dest for f, dest in mounts if f == "--remount-ro"]
            assert sealed == [home], (flag, sealed)
