"""Tests for website/scripts/extract-skills.py helpers.

Covers the two behavioral contracts added when the Skills Hub page gained
per-skill source links and a cleaned-up category sidebar:

1. ``_source_url`` — every community skill must resolve to a clickable
   origin URL (explicit ``extra`` URL preferred, else synthesized from the
   identifier shape). Built-in/optional skills intentionally return "" —
   they have a generated docs page (docsPath) instead.

2. ``_guess_category`` — tags only map to a curated category bucket;
   unknown tags fall to ``uncategorized`` (folded into "Other" later) so the
   sidebar doesn't fill with one-off junk like version strings or brand
   names.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRACT = REPO_ROOT / "website" / "scripts" / "extract-skills.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("extract_skills", EXTRACT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("directory,source,prefix", [
    ("skills", "built-in", "NousResearch/hermes-agent/skills"),
    ("optional-skills", "optional", "official"),
])
def test_local_skills_publish_exact_install_target(mod, tmp_path, monkeypatch, directory, source, prefix):
    skill = tmp_path / directory / "creative" / "nested" / "example"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: Different Display Name\n---\nExample.")
    monkeypatch.setattr(mod, "REPO_ROOT", str(tmp_path))
    [entry] = mod.extract_local_skills()
    expected = f"{prefix}/creative/nested/example"
    assert entry["installIdentifier"] == expected
    assert entry["installCmd"] == f"hermes skills install {expected}"


def test_unified_skills_keep_source_identifiers_and_match_cli_install_targets(mod, tmp_path, monkeypatch):
    entries = [
        {"source": "clawhub", "identifier": "apple-design", "name": "Display name"},
        {"source": "skills-sh", "identifier": "skills-sh/owner/repo/a skill?mode=one&two#readme"},
        {"source": "github", "identifier": "anthropics/skills/skills/pdf"},
        {"source": "well-known", "identifier": "https://example.com/skills/example"},
    ]
    index = tmp_path / "skills-index.json"
    index.write_text(json.dumps({"skills": entries}))
    monkeypatch.setattr(mod, "UNIFIED_INDEX_PATH", str(index))
    extracted, _ = mod.extract_unified_index_skills()

    assert len(extracted) == len(entries)
    for source, published in zip(entries, extracted):
        identifier = source["identifier"]
        expected = f"clawhub/{identifier}" if source["source"] == "clawhub" else identifier
        assert published["identifier"] == identifier
        assert published["installIdentifier"] == expected
        assert published["installCmd"] == f"hermes skills install {expected}"


# --------------------------------------------------------------------------
# _source_url
# --------------------------------------------------------------------------

def test_source_url_prefers_explicit_detail_url(mod):
    extra = {"detail_url": "https://skills.sh/owner/repo/skill"}
    assert (
        mod._source_url("skills.sh", "skills-sh/owner/repo/skill", extra)
        == "https://skills.sh/owner/repo/skill"
    )


def test_source_url_prefers_browse_sh_source_url(mod):
    # browse.sh adapter carries its origin under extra["source_url"].
    extra = {"source_url": "https://airbnb.com/host"}
    assert (
        mod._source_url("browse-sh", "browse-sh/airbnb.com/login-abc", extra)
        == "https://airbnb.com/host"
    )


def test_source_url_synthesizes_github_tree_url(mod):
    url = mod._source_url("github", "anthropics/skills/skills/algorithmic-art", {})
    assert url == "https://github.com/anthropics/skills/tree/main/skills/algorithmic-art"


def test_source_url_synthesizes_github_root_when_no_subpath(mod):
    assert mod._source_url("github", "owner/repo", {}) == "https://github.com/owner/repo"


def test_source_url_synthesizes_clawhub(mod):
    # ClawHub URLs require the owner handle; without it we cannot build a
    # valid URL, so the result is "" (better than a broken 404 link).
    assert mod._source_url("clawhub", "go-music-skill", {}) == ""


def test_source_url_synthesizes_clawhub_strips_prefix(mod):
    # identifier may arrive already prefixed; we must not double-prefix.
    assert (
        mod._source_url("clawhub", "clawhub/go-music-skill", {})
        == ""
    )


def test_source_url_synthesizes_clawhub_with_owner(mod):
    # When the owner handle is available in extra, the URL includes it.
    assert (
        mod._source_url("clawhub", "go-music-skill", {"owner": "somepublisher"})
        == "https://clawhub.ai/somepublisher/skills/go-music-skill"
    )


def test_source_url_synthesizes_clawhub_with_owner_strips_prefix(mod):
    # Owner + prefixed identifier: prefix is stripped, owner is used.
    assert (
        mod._source_url("clawhub", "clawhub/go-music-skill", {"owner": "somepublisher"})
        == "https://clawhub.ai/somepublisher/skills/go-music-skill"
    )


def test_source_url_synthesizes_lobehub(mod):
    assert mod._source_url("lobehub", "lobehub/chinese-paper", {}) == "https://lobehub.com/agent/chinese-paper"


def test_source_url_empty_for_unknown_source_without_identifier(mod):
    assert mod._source_url("mystery", "", {}) == ""


# --------------------------------------------------------------------------
# _guess_category
# --------------------------------------------------------------------------





def test_guess_category_rejects_junk_tag(mod):
    # This is the whole point: version strings / brand names must NOT become
    # their own sidebar category. They land in "uncategorized" → "Other".
    assert mod._guess_category(["0.10.7 Dev"]) == "uncategorized"
    assert mod._guess_category(["Doramagic Crystal"]) == "uncategorized"
    assert mod._guess_category(["Ap2"]) == "uncategorized"


def test_guess_category_empty_tags(mod):
    assert mod._guess_category([]) == "uncategorized"


def test_guess_category_skips_first_junk_tag_for_later_known_tag(mod):
    # First tag is junk, second is curated — we should still find the curated one.
    assert mod._guess_category(["Some Brand", "security"]) == "security"
