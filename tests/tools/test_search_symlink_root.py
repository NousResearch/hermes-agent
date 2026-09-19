"""Regression for #116270: a symlinked search root made the grep fallback
return a confident zero.

On BSD/macOS grep, ``grep -r`` skips a symlink handed to it as the path
argument (exit 1, nothing on stderr — byte-identical to "no match"; ``-R``
only follows links met during traversal, not the argument itself), while GNU
grep >= 3.x follows it; ``find``'s ``-type f`` tests the link rather than its
target on every platform, so the pruned path dropped the root everywhere and
the plain path did on BSD. The fix resolves the root through its link before
either engine sees it. ripgrep always followed argument symlinks; one test
here pins that so the two engines cannot drift apart again.
"""

import os

import pytest

from tools.file_operations import ShellFileOperations
from tools.environments.local import LocalEnvironment


@pytest.fixture
def grep_ops(tmp_path, monkeypatch):
    ops = ShellFileOperations(
        LocalEnvironment(cwd=str(tmp_path)),
        cwd=str(tmp_path),
    )
    monkeypatch.setattr(ops, "_has_command", lambda command: command == "grep")
    return ops


def _symlink_or_skip(target, link):
    try:
        os.symlink(str(target), str(link))
    except OSError as exc:
        pytest.skip(f"symlink unavailable: {exc}")


def test_symlinked_file_root_returns_matches(tmp_path, grep_ops):
    """Absolute symlink root: the plain grep path must search the target."""
    real = tmp_path / "real.md"
    real.write_text("NEEDLE in a real file\n")
    _symlink_or_skip(real, tmp_path / "link.md")

    result = grep_ops.search("NEEDLE", path=str(tmp_path / "link.md"), target="content")

    assert result.error is None
    assert result.total_count == 1


def test_relative_symlinked_root_resolves_against_cwd(tmp_path, grep_ops):
    """``./link.md`` must resolve against the shell's live $PWD, not stay a link."""
    real = tmp_path / "real.md"
    real.write_text("NEEDLE in a real file\n")
    _symlink_or_skip(real, tmp_path / "link.md")

    result = grep_ops.search("NEEDLE", path="./link.md", target="content")

    assert result.error is None
    assert result.total_count == 1


def test_symlinked_root_into_hidden_dir_takes_pruned_path(tmp_path, grep_ops):
    """A link whose target sits under a dot-directory must keep the #18473
    pruned-path win: the target is searched while hidden siblings are not."""
    hidden = tmp_path / ".hermes"
    (hidden / "skills").mkdir(parents=True)
    (hidden / "skills" / "SKILL.md").write_text("NEEDLE under a hidden home\n")
    (hidden / ".hub").mkdir()
    (hidden / ".hub" / "catalog.json").write_text("NEEDLE cached from the hub\n")
    _symlink_or_skip(hidden, tmp_path / "hermes-link")

    result = grep_ops.search("NEEDLE", path=str(tmp_path / "hermes-link"), target="content")

    assert result.error is None
    assert [m.path.rsplit("/", 1)[-1] for m in result.matches] == ["SKILL.md"]


def test_symlink_root_inside_hidden_dir_still_resolves(tmp_path, grep_ops):
    """A symlink under a dot-directory must resolve on every grep family.

    The root string carries the dot component, so the pruned ``find … -type f``
    route runs even without the fix — unlike the plain-lane cases above, which
    GNU grep >= 3.x survives by following an argument symlink. There ``-type f``
    tests the link itself on every platform, so this is the half of #116270
    that reproduces on Linux too, and this test is the one that stays red on a
    Linux revert of the fix.
    """
    hidden = tmp_path / ".hermes"
    hidden.mkdir()
    real = hidden / "real.md"
    real.write_text("NEEDLE under a hidden home\n")
    _symlink_or_skip(real, hidden / "link.md")

    result = grep_ops.search("NEEDLE", path=str(hidden / "link.md"), target="content")

    assert result.error is None
    assert result.total_count == 1


def test_symlinked_directory_root_is_traversed(tmp_path, grep_ops):
    """A link to a directory must search the directory's files, not skip them."""
    real_dir = tmp_path / "docs"
    real_dir.mkdir()
    (real_dir / "a.md").write_text("NEEDLE one\n")
    (real_dir / "b.md").write_text("NEEDLE two\n")
    _symlink_or_skip(real_dir, tmp_path / "docs-link")

    result = grep_ops.search("NEEDLE", path=str(tmp_path / "docs-link"), target="content")

    assert result.error is None
    assert result.total_count == 2


def test_plain_root_unchanged_without_symlink(tmp_path, grep_ops):
    """Non-symlink roots keep the established fallback behavior."""
    real = tmp_path / "real.md"
    real.write_text("NEEDLE in a real file\n")

    result = grep_ops.search("NEEDLE", path=str(real), target="content")

    assert result.error is None
    assert result.total_count == 1


def test_true_zero_stays_zero_for_plain_root(tmp_path, grep_ops):
    """A plain file with no match must keep answering a confident zero."""
    real = tmp_path / "real.md"
    real.write_text("nothing relevant here\n")

    result = grep_ops.search("NEEDLE", path=str(real), target="content")

    assert result.error is None
    assert result.total_count == 0


def test_resolve_leaves_non_symlink_root_unwritten(tmp_path, grep_ops):
    """Non-symlink roots (absolute, relative, empty) pass through unchanged."""
    plain = tmp_path / "real.md"
    plain.write_text("NEEDLE in a real file\n")

    assert grep_ops._resolve_grep_root_symlink(str(plain)) == str(plain)
    assert grep_ops._resolve_grep_root_symlink("real.md") == "real.md"
    assert grep_ops._resolve_grep_root_symlink("") == ""


def test_rg_follows_symlinked_root(tmp_path, monkeypatch):
    """Pin rg's native behavior so both engines keep agreeing on link roots."""
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)), cwd=str(tmp_path))
    if not ops._resolve_command("rg"):
        pytest.skip("ripgrep not available")
    real = tmp_path / "real.md"
    real.write_text("NEEDLE in a real file\n")
    _symlink_or_skip(real, tmp_path / "link.md")

    result = ops.search("NEEDLE", path=str(tmp_path / "link.md"), target="content")

    assert result.error is None
    assert result.total_count == 1
