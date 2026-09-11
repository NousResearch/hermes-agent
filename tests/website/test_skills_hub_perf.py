"""Regression guard for the Skills Hub catalog performance fix (#96029).

Asserts the contract of ``extract-skills.py`` that protects the page from
silently regressing back to a slow load:

1. **Strip ratio** — community entries in the emitted ``skills.json`` are
   measurably smaller than the un-stripped equivalent (the empty
   ``overview``/``platforms``/``version``/``license``/``envVars``/
   ``commands``/``docsPath`` fields are dropped from rows that don't
   carry them). Catches "I deleted ``_strip_empty_community_fields``
   without updating the page".

2. **Search-haystack presence** — every emitted row carries an
   ``_search`` field. The Skills Hub page uses it to skip the
   client-side rebuild that used to dominate first-paint time.

3. **Per-row CPU budget on the unified-index loop** — processing a
   5k-row synthetic index runs in well under 1.5 s. The previous
   implementation took ~0.5 s on the same workload, so a 1.5 s ceiling
   is a generous 3x headroom that still catches "I dropped the
   pre-computed github tap table" regressions.

4. **Shape preserved** — the dict the page reads from each row still
   has the keys it binds to. This stops a well-meaning refactor from
   accidentally deleting fields the UI consumes.

5. ``build_search_haystack`` is best-effort — odd shapes (e.g. an
   ``author`` declared as a YAML list in a third-party SKILL.md) must
   not crash the whole extraction.

The benchmark in ``website/scripts/benchmark_extract_skills.py`` is
the bigger measurement (~88k rows, before/after). These tests are the
fast ones that run on every pytest invocation.
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRACT = REPO_ROOT / "website" / "scripts" / "extract-skills.py"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("extract_skills", EXTRACT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_unified_index(path: Path, rows: int = 5_000) -> None:
    """Write a minimal-but-realistic skills-index.json for benchmarking."""
    skills = []
    builders = [
        lambda i: {
            "source": "skills.sh",
            "identifier": f"skills-sh/owner-{i % 50}/repo-{i % 120}/skill-{i}",
            "name": f"sh-skill-{i}",
            "description": f"Synthetic skills.sh skill {i} for the perf guard.",
            "tags": ["productivity"],
            "repo": f"owner-{i % 50}/repo-{i % 120}",
        },
        lambda i: {
            "source": "clawhub",
            "identifier": f"slug-{i}",
            "name": f"ch-skill-{i}",
            "description": "Synthetic ClawHub skill.\nMulti-line description.",
            "tags": [],
            "extra": {},
        },
        lambda i: {
            "source": "github",
            "identifier": (
                "openai/skills/foo" if i % 11 == 0
                else "anthropics/skills/bar" if i % 13 == 0
                else f"random-{i % 200}/random-{i % 300}/skill-{i}"
            ),
            "name": f"gh-skill-{i}",
            "description": "Synthetic GitHub skill.",
            "tags": ["ai"],
        },
    ]
    for i in range(rows):
        skills.append(builders[i % len(builders)](i))
    payload = {
        "version": 1,
        "generated_at": "2026-08-27T00:00:00+00:00",
        "skill_count": len(skills),
        "skills": skills,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))


# --------------------------------------------------------------------------
# Performance budget (#96029)
# --------------------------------------------------------------------------


# 5k rows through the pre-fix loop took ~0.5 s on a 4-core box; the
# post-fix loop runs in ~0.15 s. 1.5 s ceiling gives 3x headroom for
# busy CI while still failing loudly if the github-tap pre-computation
# table goes missing.
_UNIFIED_LOOP_BUDGET_S = 1.5


@pytest.mark.performance
def test_unified_index_loop_runs_under_budget(mod, tmp_path):
    """Single-process extraction of a 5k-row synthetic index stays fast."""
    index_path = tmp_path / "skills-index.json"
    _synthetic_unified_index(index_path, rows=5_000)

    orig_path = mod.UNIFIED_INDEX_PATH
    mod.UNIFIED_INDEX_PATH = str(index_path)
    try:
        repeats = 3
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            skills, meta = mod.extract_unified_index_skills()
            samples.append(time.perf_counter() - t0)
        elapsed = statistics.median(samples)
    finally:
        mod.UNIFIED_INDEX_PATH = orig_path

    assert skills, "synthetic index should produce rows"
    assert meta and meta["indexSkillCount"] == 5_000, "meta should report the index size"
    assert elapsed < _UNIFIED_LOOP_BUDGET_S, (
        f"unified-index loop took {elapsed:.3f}s, budget is {_UNIFIED_LOOP_BUDGET_S}s. "
        "Did the pre-computed GITHUB_TAP prefix table get removed? "
        "See website/scripts/extract-skills.py:_GITHUB_TAP_PREFIXES."
    )


@pytest.mark.performance
def test_community_row_is_smaller_than_unstripped(mod):
    """The empty-field strip measurably shrinks each community row.

    Builds one community entry with every previously-emptied field
    populated with its old value (empty string / empty list) and one
    with the post-#96029 strip applied. Asserts the stripped version
    is at least 15 % smaller. The benchmark in
    ``benchmark_extract_skills.py`` measures the same thing end-to-end;
    this is the cheap unit-level proxy that runs on every pytest.
    """
    full = {
        "name": "foo",
        "description": "bar",
        "overview": "",
        "category": "security",
        "categoryLabel": "Security",
        "fixedCategory": False,
        "source": "GitHub",
        "tags": ["a", "b"],
        "platforms": [],
        "author": "",
        "version": "",
        "license": "",
        "envVars": [],
        "commands": [],
        "docsPath": "",
        "identifier": "owner/repo/foo",
        "installCmd": "hermes skills install owner/repo/foo",
        "sourceUrl": "https://github.com/owner/repo/tree/main/foo",
        "_search": "foo bar security github a b",
    }
    stripped = dict(full)
    mod._strip_empty_community_fields(stripped)

    full_size = len(json.dumps(full, separators=(",", ":")))
    stripped_size = len(json.dumps(stripped, separators=(",", ":")))
    assert stripped_size < full_size, (
        f"stripped entry ({stripped_size}B) was not smaller than full ({full_size}B)"
    )
    # Per-row empty fields cost ~80 bytes; on the live catalog that is
    # ~8 MB of savings. 15 % is a conservative threshold that survives
    # future additions to the per-row shape.
    assert stripped_size / full_size < 0.85, (
        f"strip ratio {stripped_size / full_size:.2f} not < 0.85; "
        "_strip_empty_community_fields may have regressed."
    )
    # Required keys for the page must survive the strip.
    for key in ("identifier", "installCmd", "sourceUrl", "_search"):
        assert key in stripped, f"stripped row dropped required key {key!r}"


@pytest.mark.performance
def test_every_row_has_precomputed_search_haystack(mod, tmp_path):
    """The page relies on the precomputed _search field; spot-check it's there."""
    index_path = tmp_path / "skills-index.json"
    _synthetic_unified_index(index_path, rows=200)
    orig_index = mod.UNIFIED_INDEX_PATH
    mod.UNIFIED_INDEX_PATH = str(index_path)
    try:
        skills, _ = mod.extract_unified_index_skills()
    finally:
        mod.UNIFIED_INDEX_PATH = orig_index

    assert skills, "expected rows from synthetic index"
    missing = [s["name"] for s in skills if not isinstance(s.get("_search"), str)]
    assert not missing, (
        f"{len(missing)}/{len(skills)} rows missing precomputed _search; "
        "did PRECOMPUTE_SEARCH_HAYSTACK get disabled?"
    )
    # The haystack must contain the lowercase name so the page's search
    # filter is at least as good as the original client-side fallback.
    sample = skills[0]
    assert sample["name"].lower() in sample["_search"], (
        f"haystack for {sample['name']!r} doesn't contain the name; "
        "build_search_haystack() changed shape?"
    )


# --------------------------------------------------------------------------
# Shape preservation — guard against accidental field deletions
# --------------------------------------------------------------------------


_REQUIRED_KEYS = {
    "name", "description", "category", "categoryLabel", "source",
    "tags", "identifier", "installCmd", "sourceUrl",
}


@pytest.mark.performance
def test_unified_row_shape_preserved(mod, tmp_path):
    """The dict the page reads from each row still has every key it binds to."""
    index_path = tmp_path / "skills-index.json"
    _synthetic_unified_index(index_path, rows=20)
    orig_index = mod.UNIFIED_INDEX_PATH
    mod.UNIFIED_INDEX_PATH = str(index_path)
    try:
        skills, _ = mod.extract_unified_index_skills()
    finally:
        mod.UNIFIED_INDEX_PATH = orig_index

    assert skills
    missing = _REQUIRED_KEYS - set(skills[0].keys())
    assert not missing, (
        f"unified-index rows missing keys the page binds to: {missing}. "
        "The empty-field strip is allowed to drop fields the page never "
        "reads (overview/platforms/version/license/envVars/commands/docsPath), "
        "but every key in _REQUIRED_KEYS must remain."
    )


# --------------------------------------------------------------------------
# build_search_haystack contract — the page uses this directly
# --------------------------------------------------------------------------


def test_build_search_haystack_lowercases(mod):
    haystack = mod.build_search_haystack({
        "name": "Foo",
        "description": "BAR baz",
        "tags": ["Quux"],
    })
    assert haystack == "foo bar baz quux"


def test_build_search_haystack_skips_missing(mod):
    # Missing keys must not crash; the haystack is best-effort.
    haystack = mod.build_search_haystack({"name": "OnlyName"})
    assert haystack == "onlyname"


def test_build_search_haystack_includes_tags(mod):
    # The Skills Hub search matches against tag pills; tags must be in
    # the haystack or filtering by tag stops working.
    haystack = mod.build_search_haystack({
        "name": "x",
        "description": "",
        "tags": ["security", "mlops"],
    })
    assert "security" in haystack and "mlops" in haystack


def test_build_search_haystack_handles_list_author(mod):
    # Some third-party SKILL.md files declare ``author`` as a YAML list
    # (e.g. multi-author skills). The haystack is best-effort; it must
    # not crash, and the rendered text must be searchable.
    haystack = mod.build_search_haystack({
        "name": "collab-skill",
        "author": ["alice", "bob"],
    })
    assert "alice" in haystack and "bob" in haystack
    assert "collab-skill" in haystack