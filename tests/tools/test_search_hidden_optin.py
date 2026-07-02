"""Hidden-dir search: consistent default exclusion + explicit ``include_hidden`` opt-in.

Companion to ``test_search_hidden_dirs.py`` (#1558 — hidden caches can carry
adversarial text, so search excludes dotdirs by default; that contract is kept).
This file pins the two halves that were missing and sent a production agent into
a 90-iteration retry loop hunting files that existed the whole time:

1. **Consistency.** rg's own hidden pruning is glob-dependent — a ``-g`` glob is
   also tested against directory *names* during traversal, so ``-g '*'``
   descended into hidden trees (leaking exactly what #1558 excludes) while any
   specific glob (``'*config*'``) pruned them and silently returned
   ``{"total_count": 0}`` for files a ``'*'`` listing had just shown. The model
   was told two contradicting answers about the same tree. Now one post-hoc
   filter gives every pattern and backend the same answer.
2. **Discoverability.** A caller that legitimately needs a dotdir (on agent
   installs the whole working tree may live under one) has explicit paths:
   ``include_hidden=true``, or passing the hidden directory as ``path=`` —
   and a glob parse failure surfaces as an error instead of a silent 0.
"""

import shutil

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations


requires_rg = pytest.mark.skipif(
    shutil.which("rg") is None, reason="ripgrep not installed")


@pytest.fixture
def hidden_tree(tmp_path):
    """A home-like tree whose real content lives under a hidden root."""
    scripts = tmp_path / ".agenthome" / "skills" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "preflight.py").write_text("print('needle_content')\n")
    (tmp_path / ".agenthome" / "config.yaml").write_text("timezone: UTC\n")
    (tmp_path / "VISIBLE.md").write_text("plain visible file\n")
    return tmp_path


def _ops(root):
    return ShellFileOperations(LocalEnvironment(cwd=str(root)), cwd=str(root))


# ---------------------------------------------------------------- consistency
@requires_rg
def test_bare_star_no_longer_leaks_hidden_files(hidden_tree):
    """'-g *' matched the hidden dir NAME during traversal and listed its
    files — contradicting every specific glob AND the #1558 exclusion."""
    res = _ops(hidden_tree).search("*", path=str(hidden_tree), target="files")
    assert not res.error
    assert not any(".agenthome" in f for f in res.files)
    assert any("VISIBLE.md" in f for f in res.files)


@requires_rg
def test_star_and_specific_glob_agree_about_hidden_files(hidden_tree):
    """The gaslighting bug: the same tree gave two contradicting answers."""
    ops = _ops(hidden_tree)
    star = ops.search("*", path=str(hidden_tree), target="files")
    specific = ops.search("*preflight*", path=str(hidden_tree), target="files")
    star_hit = any("preflight.py" in f for f in star.files)
    specific_hit = any("preflight.py" in f for f in specific.files)
    assert star_hit == specific_hit == False  # noqa: E712 — agreement is the point


# ---------------------------------------------------------------- the opt-in
@requires_rg
def test_include_hidden_finds_files_under_dotdirs(hidden_tree):
    res = _ops(hidden_tree).search("*preflight*", path=str(hidden_tree),
                                   target="files", include_hidden=True)
    assert not res.error
    assert any("preflight.py" in f for f in res.files)


@requires_rg
def test_include_hidden_content_search_reaches_dotdirs(hidden_tree):
    ops = _ops(hidden_tree)
    default = ops.search("needle_content", path=str(hidden_tree), target="content")
    opted = ops.search("needle_content", path=str(hidden_tree), target="content",
                       include_hidden=True)
    assert default.total_count == 0        # #1558 default kept
    assert opted.total_count >= 1


@requires_rg
def test_explicit_hidden_root_is_searched_without_the_flag(hidden_tree):
    """Passing the dotdir as path= is the other sanctioned escape hatch."""
    res = _ops(hidden_tree).search(
        "*preflight*", path=str(hidden_tree / ".agenthome"), target="files")
    assert not res.error
    assert any("preflight.py" in f for f in res.files)


# ------------------------------------------------------------- honest errors
@requires_rg
def test_files_glob_parse_error_is_surfaced_not_silent_zero(hidden_tree):
    """A broken glob must be an error the model can react to — a silent
    {"total_count": 0} reads as 'the file does not exist' and misleads."""
    res = _ops(hidden_tree).search("[", path=str(hidden_tree), target="files")
    assert res.error
    assert not res.files


# ------------------------------------------------------------- find fallback
def test_find_fallback_matches_rg_semantics(hidden_tree, monkeypatch):
    ops = _ops(hidden_tree)
    monkeypatch.setattr(ops, "_has_command", lambda cmd: cmd == "find")
    default = ops.search("*preflight*", path=str(hidden_tree), target="files")
    opted = ops.search("*preflight*", path=str(hidden_tree), target="files",
                       include_hidden=True)
    assert not any("preflight.py" in f for f in default.files)
    assert any("preflight.py" in f for f in opted.files)


def test_find_fallback_explicit_hidden_root(hidden_tree, monkeypatch):
    ops = _ops(hidden_tree)
    monkeypatch.setattr(ops, "_has_command", lambda cmd: cmd == "find")
    res = ops.search("*preflight*", path=str(hidden_tree / ".agenthome"),
                     target="files")
    assert any("preflight.py" in f for f in res.files)
