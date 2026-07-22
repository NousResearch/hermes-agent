"""Tests for shared filesystem utilities."""

from utils import copy_file_writable, copytree_writable, ensure_owner_writable, make_tree_owner_writable


class TestEnsureOwnerWritable:
    """Unit tests for the writable-mode helpers."""

    def test_handles_missing_path(self, tmp_path):
        # Should not raise on missing path
        ensure_owner_writable(tmp_path / "does-not-exist")

    def test_grants_user_write_to_readonly_file(self, tmp_path):
        import os
        import stat as stat_mod

        f = tmp_path / "readonly.md"
        f.write_text("x")
        os.chmod(f, 0o444)  # mimic Nix store mode

        ensure_owner_writable(f)

        assert os.stat(f).st_mode & stat_mod.S_IWUSR

    def test_preserves_executable_bit(self, tmp_path):
        import os
        import stat as stat_mod

        script = tmp_path / "skill.sh"
        script.write_text("#!/bin/sh\n")
        os.chmod(script, 0o555)  # executable, read-only — Nix-store-style

        ensure_owner_writable(script)

        m = stat_mod.S_IMODE(os.stat(script).st_mode)
        assert m & stat_mod.S_IWUSR
        assert m & stat_mod.S_IXUSR  # executable still set

    def test_idempotent_on_writable_target(self, tmp_path):
        import os
        import stat as stat_mod

        f = tmp_path / "ok.txt"
        f.write_text("x")
        before = os.stat(f).st_mode

        ensure_owner_writable(f)

        # Same or only the (already-set) user-write bit changed
        after = os.stat(f).st_mode
        assert after | stat_mod.S_IWUSR == before | stat_mod.S_IWUSR


class TestCopyFileWritable:
    """``copy_file_writable`` is the ``copy_function`` for copytree, plus a
    drop-in replacement for ``shutil.copy2``."""

    def test_readonly_source_yields_writable_destination(self, tmp_path):
        import os
        import stat as stat_mod

        src = tmp_path / "src.md"
        dst = tmp_path / "dst.md"
        src.write_text("hello")
        os.chmod(src, 0o444)

        copy_file_writable(src, dst)

        assert dst.read_text() == "hello"
        assert os.stat(dst).st_mode & stat_mod.S_IWUSR

    def test_executable_source_keeps_executable_bit(self, tmp_path):
        import os
        import stat as stat_mod

        src = tmp_path / "script.sh"
        dst = tmp_path / "script-copy.sh"
        src.write_text("#!/bin/sh\n")
        os.chmod(src, 0o555)

        copy_file_writable(src, dst)

        m = stat_mod.S_IMODE(os.stat(dst).st_mode)
        assert m & stat_mod.S_IWUSR
        assert m & stat_mod.S_IXUSR


class TestMakeTreeOwnerWritable:
    """``make_tree_owner_writable`` walks an existing tree and grants the
    owner-write bit to every entry."""

    def test_grants_user_write_to_all_descendants(self, tmp_path):
        import os
        import stat as stat_mod

        sub = tmp_path / "sub"
        nested = sub / "nested"
        nested.mkdir(parents=True)
        f = nested / "file.txt"
        f.write_text("x")
        # Lock everything down depth-first like the Nix store would.
        os.chmod(f, 0o444)
        os.chmod(nested, 0o555)
        os.chmod(sub, 0o555)
        os.chmod(tmp_path, 0o555)

        try:
            make_tree_owner_writable(tmp_path)

            assert os.stat(tmp_path).st_mode & stat_mod.S_IWUSR
            assert os.stat(sub).st_mode & stat_mod.S_IWUSR
            assert os.stat(nested).st_mode & stat_mod.S_IWUSR
            assert os.stat(f).st_mode & stat_mod.S_IWUSR
        finally:
            # Restore writable mode on the whole tree so pytest tmp_path
            # teardown does not error if an assertion above failed midway.
            os.chmod(tmp_path, 0o755)
            os.chmod(sub, 0o755)
            os.chmod(nested, 0o755)
            os.chmod(f, 0o644)

    def test_handles_missing_root(self, tmp_path):
        # No-op on missing path
        make_tree_owner_writable(tmp_path / "ghost")


class TestCopytreeWritable:
    """End-to-end: ``copytree_writable`` produces a fully editable copy of
    a read-only source tree."""

    def test_readonly_source_tree_yields_editable_destination(self, tmp_path):
        import os
        import stat as stat_mod

        src = tmp_path / "src"
        nested = src / "category" / "skill-x"
        nested.mkdir(parents=True)
        (nested / "SKILL.md").write_text("# X\n")
        (nested / "main.py").write_text("print(1)\n")
        # Apply Nix-store-style modes depth-first.
        for path in sorted(src.rglob("*"), reverse=True):
            os.chmod(path, 0o444 if path.is_file() else 0o555)
        os.chmod(src, 0o555)

        dst = tmp_path / "dst"
        try:
            copytree_writable(src, dst)

            sk = dst / "category" / "skill-x" / "SKILL.md"
            assert sk.exists()
            assert os.stat(sk).st_mode & stat_mod.S_IWUSR
            assert os.stat(dst / "category" / "skill-x").st_mode & stat_mod.S_IWUSR
            # Append-edit must succeed (this is the regression: previously
            # raised PermissionError because mode 0444 was preserved).
            with sk.open("a") as fh:
                fh.write("\nappended\n")
        finally:
            for path in sorted(src.rglob("*"), reverse=True):
                os.chmod(path, 0o755 if path.is_dir() else 0o644)
            os.chmod(src, 0o755)
