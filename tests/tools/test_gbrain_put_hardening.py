"""GBrain put() writer hardening — atomic owner-only writes.

Behavioural tests for the W3 direct-writer hardening of ``gbrain_put()``:
new files mode 600, new directories mode 700, atomic replacement of
existing files, failure preserves the original, symlink/path-traversal
rejection, no temp-file leakage, and no process-umask change.

Run under process umask 0002.
"""

import json
import os
import stat
from pathlib import Path

import pytest

from tools import gbrain


def _make_brain(tmp_path, pages):
    """Write a tiny markdown brain repo (mode 700 root) and return the path."""
    repo = tmp_path / "brain"
    repo.mkdir(mode=0o700)
    for slug, content in pages.items():
        target = repo / f"{slug}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return repo


def _file_mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def _read_umask() -> int:
    """Read current process umask without permanently changing it."""
    old = os.umask(0o022)
    os.umask(old)
    return old


def _temp_leftovers(repo: Path) -> list:
    return [p for p in repo.rglob("*") if p.name.startswith(".gbrain-")]


# ── Permission tests ───────────────────────────────────────────────────────


def test_gbrain_put_new_file_mode_600(monkeypatch, tmp_path):
    """Newly created page file must be owner-only (mode 600)."""
    repo = _make_brain(tmp_path, {})
    monkeypatch.setattr(gbrain, "GBRAIN_REPO", repo)

    gbrain.gbrain_put({"slug": "page", "content": "# Page\n"})

    assert _file_mode(repo / "page.md") == 0o600


def test_gbrain_put_new_dirs_mode_700(monkeypatch, tmp_path):
    """Newly created directories must be owner-only (mode 700)."""
    repo = _make_brain(tmp_path, {})
    monkeypatch.setattr(gbrain, "GBRAIN_REPO", repo)

    gbrain.gbrain_put({"slug": "deep/nested/page", "content": "# Deep\n"})

    assert _file_mode(repo / "deep") == 0o700
    assert _file_mode(repo / "deep" / "nested") == 0o700
    assert _file_mode(repo / "deep" / "nested" / "page.md") == 0o600


def test_gbrain_put_existing_file_atomic_replace(monkeypatch, tmp_path):
    """Existing file is atomically replaced; final mode is owner-only 600."""
    repo = _make_brain(tmp_path, {"page": "OLD\n"})
    monkeypatch.setattr(gbrain, "GBRAIN_REPO", repo)

    gbrain.gbrain_put({"slug": "page", "content": "NEW\n"})

    assert (repo / "page.md").read_text() == "NEW\n"
    assert _file_mode(repo / "page.md") == 0o600


# ── Failure / atomicity tests ──────────────────────────────────────────────


def _raise_oserror(*_a, **_k):
    raise OSError("simulated failure")


def test_gbrain_put_failure_preserves_original(monkeypatch, tmp_path):
    """If the atomic replace fails, the original file is untouched and no
    temp file leaks."""
    repo = _make_brain(tmp_path, {"page": "ORIGINAL\n"})
    monkeypatch.setattr(gbrain, "GBRAIN_REPO", repo)
    monkeypatch.setattr(os, "replace", _raise_oserror)

    with pytest.raises(OSError):
        gbrain.gbrain_put({"slug": "page", "content": "NEW\n"})

    assert (repo / "page.md").read_text() == "ORIGINAL\n"
    assert _temp_leftovers(repo) == []


# ── Containment / symlink rejection ────────────────────────────────────────


def test_gbrain_put_rejects_path_traversal(monkeypatch, tmp_path):
    """Slug escaping the repo root is rejected."""
    repo = _make_brain(tmp_path, {"page": "OK\n"})
    monkeypatch.setattr(gbrain, "GBRAIN_REPO", repo)

    data = json.loads(gbrain.gbrain_put({"slug": "../../etc/passwd", "content": "evil\n"}))

    assert "error" in data


def test_gbrain_put_rejects_dot_slug(monkeypatch, tmp_path):
    """A dot slug cannot create the surprising root-level ``..md`` page."""
    repo = _make_brain(tmp_path, {})
    monkeypatch.setattr(gbrain, "GBRAIN_REPO", repo)

    data = json.loads(gbrain.gbrain_put({"slug": ".", "content": "NO\n"}))

    assert "error" in data
    assert not (repo / "..md").exists()


def test_gbrain_put_rejects_symlink(monkeypatch, tmp_path):
    """A symlink slug (aliasing an in-repo file) is rejected; the target is
    not written through."""
    repo = _make_brain(tmp_path, {"real": "REAL\n"})
    (repo / "link.md").symlink_to(repo / "real.md")
    monkeypatch.setattr(gbrain, "GBRAIN_REPO", repo)

    data = json.loads(gbrain.gbrain_put({"slug": "link", "content": "HIJACK\n"}))

    assert "error" in data
    assert (repo / "real.md").read_text() == "REAL\n"


def test_gbrain_put_rejects_intermediate_symlink(monkeypatch, tmp_path):
    """A symlinked directory component cannot redirect writes."""
    repo = _make_brain(tmp_path, {})
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo / "alias").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(gbrain, "GBRAIN_REPO", repo)

    data = json.loads(gbrain.gbrain_put({"slug": "alias/page", "content": "NO\n"}))

    assert "error" in data
    assert not (outside / "page.md").exists()


def test_gbrain_put_rejects_non_regular_target(monkeypatch, tmp_path):
    """A directory at the page path is rejected rather than replaced."""
    repo = _make_brain(tmp_path, {})
    (repo / "page.md").mkdir()
    monkeypatch.setattr(gbrain, "GBRAIN_REPO", repo)

    data = json.loads(gbrain.gbrain_put({"slug": "page", "content": "NO\n"}))

    assert "error" in data
    assert (repo / "page.md").is_dir()


# ── Cleanup / no side effects ──────────────────────────────────────────────


def test_gbrain_put_no_temp_leakage(monkeypatch, tmp_path):
    """No temporary files remain after a successful write."""
    repo = _make_brain(tmp_path, {})
    monkeypatch.setattr(gbrain, "GBRAIN_REPO", repo)

    gbrain.gbrain_put({"slug": "page", "content": "# Page\n"})

    assert _temp_leftovers(repo) == []


def test_gbrain_put_no_umask_change(monkeypatch, tmp_path):
    """gbrain_put must not alter the process umask."""
    repo = _make_brain(tmp_path, {})
    monkeypatch.setattr(gbrain, "GBRAIN_REPO", repo)

    before = _read_umask()
    gbrain.gbrain_put({"slug": "page", "content": "# Page\n"})
    after = _read_umask()

    assert before == after
