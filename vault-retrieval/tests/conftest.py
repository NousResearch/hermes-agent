"""Shared test fixtures and per-test plugin module imports.

Each test file imports only the modules it needs. We deliberately do NOT
pre-import all vault_retrieval.* modules here — that would couple every
test to every module's existence and make incremental TDD impossible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


# Make the plugin importable. The plugin lives at <repo>/vault-retrieval/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLUGIN_DIR = _REPO_ROOT / "vault-retrieval"
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))


@pytest.fixture
def vault_root(tmp_path: Path) -> Path:
    """A minimal Obsidian-Vault-shaped directory."""
    root = tmp_path / "vault"
    root.mkdir()
    (root / "99 System").mkdir()
    (root / "02 Projects").mkdir()
    return root


@pytest.fixture
def small_note(vault_root: Path) -> Path:
    p = vault_root / "99 System" / "note.md"
    p.write_text("hello world\nsecond line\n", encoding="utf-8")
    return p


@pytest.fixture
def large_note(vault_root: Path) -> Path:
    """A note > 20,000 characters (large-file threshold)."""
    p = vault_root / "99 System" / "large.md"
    body = "# Large\n\n" + ("x" * 60_000) + "\n\n## End\n"
    p.write_text(body, encoding="utf-8")
    return p


@pytest.fixture
def frontmatter_note(vault_root: Path) -> Path:
    """A note with full frontmatter (canonical_for, owner, last_updated, review_by)."""
    p = vault_root / "99 System" / "fm.md"
    p.write_text(
        "---\n"
        "title: Sample\n"
        "canonical_for: test-fm\n"
        "owner: tech-cto\n"
        "last_updated: 2026-10-01T10:00:00+07:00\n"
        "review_by: 2026-12-01\n"
        "---\n\n"
        "Body line 1\n"
        "Body line 2\n"
        "Body line 3\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture(autouse=True)
def _reset_turn_registry():
    """Reset the active process-wide per-turn registry between tests."""
    from importlib import import_module

    import_module("vault_retrieval.budgets")._turn_registry.clear()
    yield
    import_module("vault_retrieval.budgets")._turn_registry.clear()
