"""Tests for website/scripts/check_doc_links.py.

Cross-page links in hand-authored docs must be relative Markdown paths so they
resolve on GitHub's file viewer as well as on the rendered site (#114428).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "website" / "scripts" / "check_doc_links.py"


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location("check_doc_links", CHECKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module




def test_fix_rewrites_route_to_relative_path_keeping_anchor(checker, tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    (docs / "getting-started").mkdir(parents=True)
    (docs / "user-guide" / "features").mkdir(parents=True)
    (docs / "user-guide" / "features" / "skills.md").write_text("# Skills\n", encoding="utf-8")
    (docs / "user-guide" / "features" / "index.md").write_text("# Features\n", encoding="utf-8")
    src = docs / "getting-started" / "learning-path.md"
    src.write_text(
        "[Skills](/user-guide/features/skills#bundles) [Docs](/docs/user-guide/features)\n"
        "[Site page](/skills) [Same](./quickstart.md)\n"
        "```\n[example](/user-guide/features/skills)\n```\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(checker, "EN_DOCS", docs)
    monkeypatch.setattr(checker, "ZH_DOCS", tmp_path / "missing")

    assert checker.main([]) == 1
    assert checker.main(["--fix"]) == 0
    text = src.read_text(encoding="utf-8")
    assert "[Skills](../user-guide/features/skills.md#bundles)" in text
    assert "[Docs](../user-guide/features/index.md)" in text
    # Non-doc routes and fenced examples are left alone.
    assert "[Site page](/skills)" in text
    assert "```\n[example](/user-guide/features/skills)\n```" in text
    assert checker.main([]) == 0

    # A route that maps to no file is an error, not a silent rewrite.
    src.write_text("[Gone](/user-guide/nope)\n", encoding="utf-8")
    assert checker.main(["--fix"]) == 1
    assert src.read_text(encoding="utf-8") == "[Gone](/user-guide/nope)\n"
