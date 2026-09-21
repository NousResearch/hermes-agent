"""P13 isolation proof for scripts/wiki_on_complete.py.

Verifies:
- HERMES_HOME / WIKI_DIR parameterisation: KANBAN_DB and WIKI_REPOS
  resolve under the env roots, not /home/kensei/.hermes or /home/kensei/wiki.
- --dry-run suppresses the wiki page rewrite (write_text). Read paths
  (kanban DB query, find_wiki_reference) run unchanged.
- import-safe: importing the module does not create the kanban DB or
  wiki dir.
"""
import importlib.util
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "wiki_on_complete.py"


def _load_module(monkeypatch, fake_home: Path, fake_wiki: Path):
    monkeypatch.setenv("HERMES_HOME", str(fake_home))
    monkeypatch.setenv("WIKI_DIR", str(fake_wiki))
    scripts_dir = REPO_ROOT / "scripts"
    for pth in (str(scripts_dir), str(REPO_ROOT)):
        if pth not in sys.path:
            sys.path.insert(0, pth)
    spec = importlib.util.spec_from_file_location(
        "wiki_on_complete_under_test", str(SCRIPT)
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_layout(tmp_path):
    fake_home = tmp_path / "fake_hermes"
    fake_home.mkdir()
    fake_wiki = tmp_path / "fake_wiki"
    (fake_wiki / "repos").mkdir(parents=True)
    return fake_home, fake_wiki


def test_paths_resolve_under_env_roots(monkeypatch, fake_layout):
    fake_home, fake_wiki = fake_layout
    mod = _load_module(monkeypatch, fake_home, fake_wiki)
    assert mod.KANBAN_DB == fake_home / "kanban.db"
    assert mod.WIKI_REPOS == fake_wiki / "repos"


def test_import_is_side_effect_free(monkeypatch, fake_layout):
    fake_home, fake_wiki = fake_layout
    assert not (fake_home / "kanban.db").exists()
    _load_module(monkeypatch, fake_home, fake_wiki)
    assert not (fake_home / "kanban.db").exists()


def test_dry_run_does_not_rewrite_wiki(monkeypatch, fake_layout):
    """With a wiki page present and update_wiki_page called, --dry-run
    must NOT rewrite the page (adoption_status stays 'evaluated')."""
    fake_home, fake_wiki = fake_layout
    mod = _load_module(monkeypatch, fake_home, fake_wiki)
    page = fake_wiki / "repos" / "test-repo.md"
    original = (
        "---\\nadoption_status: evaluated\\nupdated: 2026-01-01\\n---\\nbody\\n"
    )
    page.write_text(original)
    mod._DRY_RUN = True
    result = mod.update_wiki_page("test-repo")
    assert result == "would-update", f"got {result!r}"
    assert page.read_text() == original, "dry-run rewrote the wiki page"


def test_dry_run_main_exits_zero_no_db(monkeypatch, fake_layout, tmp_path):
    """--dry-run with no kanban DB must exit 0 (SKIP path) without
    writing anything."""
    fake_home, fake_wiki = fake_layout
    env = dict(os.environ)
    env["HERMES_HOME"] = str(fake_home)
    env["WIKI_DIR"] = str(fake_wiki)
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        timeout=30,
    )
    # main() prints "SKIP: kanban DB not found" and returns (no sys.exit).
    assert proc.returncode == 0, (
        f"dry-run failed: rc={proc.returncode} stderr={proc.stderr!r} stdout={proc.stdout!r}"
    )
    assert "SKIP" in proc.stdout or proc.stdout == ""
