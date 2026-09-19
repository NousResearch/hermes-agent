"""`llms.txt` is how an LLM learns what Hermes can do.

It is the index every model reads when pointed at our docs — including Hermes
itself, whose `hermes-agent` skill routes unknown-feature questions there.
`website/` is never packaged, so there is no shipped copy to fall back on.

The index used to be a hand-written list of page paths, and it rotted to 53%
coverage: Bot Mode, the desktop app, computer use, web search, and 22 messaging
platforms were all absent, which is why an agent asked how to make bots talk to
each other answered that it couldn't. These tests hold the two directions of
that contract — every page reachable, every link real — so the index tracks the
docs tree instead of someone's memory of it.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "website" / "scripts" / "generate-llms-txt.py"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("generate_llms_txt", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def index(gen) -> str:
    return gen.emit_llms_index()


def _linked(gen, index: str) -> set[str]:
    return set(re.findall(rf"\]\({re.escape(gen.SITE_BASE)}/([^)]+)\)", index))


def _pages_on_disk(gen) -> set[str]:
    """Walk the docs tree directly, duplicating only the two documented
    exclusions.

    Deliberately does not call `iter_docs()`: the index is built from that
    enumeration, so checking one against the other would pass even if the
    enumerator went blind to a whole tree — which is the failure being guarded.
    """
    pages = set()
    for path in (*gen.DOCS.rglob("*.md"), *gen.DOCS.rglob("*.mdx")):
        rel = path.relative_to(gen.DOCS).with_suffix("")
        slug = str(rel.parent) if rel.name == "index" else str(rel)
        # The docs landing page is the index's subject; per-skill pages are
        # summarized by the two catalog reference pages.
        if slug == "." or slug.startswith(("user-guide/skills/bundled", "user-guide/skills/optional")):
            continue
        pages.add(slug)
    return pages


def test_every_docs_page_is_indexed(gen, index):
    """The regression: a page the index omits is a feature the agent denies."""
    pages = _pages_on_disk(gen)
    assert len(pages) > 100, "docs root resolved wrong — the rest of this file proves nothing"

    missing = pages - _linked(gen, index)
    assert not missing, (
        f"{len(missing)} docs pages missing from llms.txt: {sorted(missing)} — "
        "they should have been absorbed into a section automatically"
    )


def test_the_enumerator_sees_the_whole_docs_tree(gen):
    """Everything downstream trusts `iter_docs()`, so pin it to the filesystem."""
    assert set(gen.iter_docs()) == _pages_on_disk(gen)


def test_every_indexed_page_exists(gen, index):
    """The other direction: a renamed page leaves the index pointing at a 404."""
    for slug in sorted(_linked(gen, index)):
        assert gen.doc_path(slug) is not None, (
            f"llms.txt links {slug}, which is not in the docs tree — "
            "drop the SECTIONS row and let the page be absorbed under its new path"
        )


def test_pages_are_listed_once(gen, index):
    """Curating a page must promote it, not duplicate it."""
    entries = re.findall(rf"^- \[.*?\]\({re.escape(gen.SITE_BASE)}/([^)]+)\)", index, re.MULTILINE)
    duplicated = {slug for slug in entries if entries.count(slug) > 1}
    assert not duplicated, f"listed more than once in llms.txt: {sorted(duplicated)}"


def test_curation_orders_pages_without_gatekeeping_them(gen):
    """SECTIONS decides what leads a section, never what the index contains."""
    curated = {slug for _section, items in gen.SECTIONS for slug, _t, _d in items}
    pages = set(gen.iter_docs())

    assert curated < pages, "every page is curated — absorption is no longer exercised"
    assert gen.section_for("user-guide/features/some-feature-shipped-tomorrow") in dict(gen.ABSORB)
    assert gen.section_for("a-tree-nobody-anticipated/page") == gen.MISC_SECTION


def test_section_landing_pages_resolve_to_their_directory(gen):
    """`messaging/index.md` is served at `/messaging`; `/messaging/index` 404s."""
    assert gen.slug_for(gen.DOCS / "user-guide" / "messaging" / "index.md") == "user-guide/messaging"
    assert "user-guide/messaging/index" not in _linked(gen, gen.emit_llms_index())


def test_mdx_pages_are_indexed_without_their_imports(gen):
    """MDX docs are real pages; their component imports are not prose."""
    mdx = [p for p in gen.DOCS.rglob("*.mdx") if gen.slug_for(p)]
    assert mdx, "no .mdx docs — this test no longer guards anything"
    assert {gen.slug_for(p) for p in mdx} <= set(gen.iter_docs())

    _meta, body = gen.read_frontmatter(mdx[0])
    assert not re.search(r"^import\s", body, re.MULTILINE)


def test_per_skill_catalog_pages_stay_out(gen):
    """~195 generated skill pages would bury the product docs in the index."""
    assert not [slug for slug in gen.iter_docs() if slug.startswith(gen.SKILL_CATALOG)]
    assert "reference/skills-catalog" in gen.iter_docs(), "the summary page must remain"


def test_bot_mode_is_reachable(gen, index):
    """The page behind the original complaint, and the answer it has to carry."""
    assert "user-guide/bot-mode" in _linked(gen, index)
    assert "hermes peer dm" in (gen.DOCS / "user-guide" / "bot-mode.md").read_text(encoding="utf-8")
