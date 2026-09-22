"""Behavior tests for the public Bot Marketplace extractor and route planner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRACT = REPO_ROOT / "website" / "scripts" / "extract-bots.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("extract_bots", EXTRACT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_bot(catalog: Path, name: str, **overrides: object) -> Path:
    entry: dict[str, object] = {
        "name": name,
        "version": "1.0.0",
        "maintainer": "Nous Research",
        "tier": "official",
        "category": "research",
        "tags": ["citations", "briefs"],
        "title": "Research Analyst",
        "summary": "Turns open-ended questions into dated, cited research briefs.",
        "profile": {
            "suggested_name": name,
            "description": "Investigates a topic, compares sources, and produces a decision-ready brief.",
            "soul": "# Research Analyst\n\nCite current sources and separate evidence from inference.",
            "starter_prompt": "What topic should I investigate?",
        },
        "capabilities": {
            "skills": ["official/research/grounded-citations"],
            "toolsets": ["web"],
        },
        "presentation": {"emoji": "🔎", "color": "#7c6cff"},
    }
    entry.update(overrides)
    path = catalog / f"{name}.yaml"
    path.write_text(yaml.safe_dump(entry, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def test_valid_bot_is_normalized_for_cards_details_and_author_links(mod, tmp_path):
    catalog = tmp_path / "bot-catalog"
    catalog.mkdir()
    _write_bot(catalog, "research-analyst")

    bots = mod.load_catalog_entries(catalog)

    assert len(bots) == 1
    bot = bots[0]
    assert bot["name"] == "research-analyst"
    assert bot["title"] == "Research Analyst"
    assert bot["profile"]["description"].startswith("Investigates")
    assert bot["capabilities"] == {
        "skills": ["official/research/grounded-citations"],
        "toolsets": ["web"],
    }
    assert bot["presentation"] == {"emoji": "🔎", "color": "#7C6CFF"}
    assert bot["maintainerSlug"] == "nous-research"
    assert bot["authorPath"] == "/bots/by/nous-research"
    assert "detailPath" not in bot
    assert "addLink" not in bot


@pytest.mark.parametrize(
    "overrides",
    [
        {"presentation": {"emoji": "🤖", "color": "url(javascript:alert(1))"}},
        {"unexpected": "field"},
        {"version": ""},
        {"tags": ["Not-Lowercase"]},
    ],
)
def test_invalid_entry_fails_the_whole_catalog(mod, tmp_path, overrides):
    catalog = tmp_path / "bot-catalog"
    catalog.mkdir()
    _write_bot(catalog, "good")
    _write_bot(catalog, "invalid", **overrides)

    with pytest.raises(mod.BotCatalogError):
        mod.load_catalog_entries(catalog)


def test_canonical_defaults_and_limits_are_preserved(mod, tmp_path):
    catalog = tmp_path / "bot-catalog"
    catalog.mkdir()
    _write_bot(
        catalog,
        "defaults",
        setup={"requirements": [{"kind": "command", "id": "python3", "purpose": "Run analysis."}]},
    )
    raw = yaml.safe_load((catalog / "defaults.yaml").read_text(encoding="utf-8"))
    raw.pop("routines", None)
    (catalog / "defaults.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")

    bot = mod.load_catalog_entries(catalog)[0]

    assert bot["setup"]["requirements"][0]["required"] is True
    assert bot["routines"] == []
    assert bot["presentation"]["color"] == "#7C6CFF"


def test_omitted_setup_uses_canonical_empty_default(mod, tmp_path):
    catalog = tmp_path / "bot-catalog"
    catalog.mkdir()
    path = _write_bot(catalog, "defaults")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw.pop("setup", None)
    path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")

    assert mod.load_catalog_entries(catalog)[0]["setup"] == {"requirements": []}


def test_main_writes_page_meta_and_normalized_install_feed(mod, tmp_path):
    catalog = tmp_path / "bot-catalog"
    catalog.mkdir()
    _write_bot(catalog, "research-analyst", category="research", tier="official")
    _write_bot(
        catalog,
        "inbox-triage",
        title="Inbox Triage",
        category="productivity",
        tier="community",
        tags=["email"],
    )
    (catalog / "removed.yaml").write_text(
        yaml.safe_dump({"removed": [{"name": "retired-bot", "reason": "retired"}]}),
        encoding="utf-8",
    )
    output = tmp_path / "api"

    assert mod.main(catalog_dir=catalog, output_dir=output) == 0

    bots = json.loads((output / "bots.json").read_text(encoding="utf-8"))
    meta = json.loads((output / "bots-meta.json").read_text(encoding="utf-8"))
    feed = json.loads((output / "bot-catalog.json").read_text(encoding="utf-8"))
    assert [bot["name"] for bot in bots] == ["inbox-triage", "research-analyst"]
    assert meta["total"] == 2
    assert meta["byTier"] == {"official": 1, "community": 1}
    assert meta["byCategory"] == {"productivity": 1, "research": 1}
    assert meta["removedCount"] == 1
    assert meta["generatedAt"]
    assert [entry["name"] for entry in feed["entries"]] == ["inbox-triage", "research-analyst"]
    assert "maintainerSlug" not in feed["entries"][0]
    assert feed["entries"][0]["profile"]["soul"].startswith("# Research Analyst")
    assert feed["removed"] == [{"name": "retired-bot", "reason": "retired"}]


def test_setup_routines_and_workflow_samples_are_published_without_changing_canonical_payload(mod, tmp_path):
    catalog = tmp_path / "bot-catalog"
    catalog.mkdir()
    workflows = tmp_path / "optional-skills" / "bots"
    workflow = workflows / "research-analyst"
    (workflow / "references").mkdir(parents=True)
    (workflow / "templates").mkdir()
    (workflow / "SKILL.md").write_text(
        "---\nname: research-analyst\ndescription: Test workflow.\n---\n# Workflow\n",
        encoding="utf-8",
    )
    (workflow / "references" / "sample-input.md").write_text(
        "# Synthetic input\n\nA bounded fixture.\n\n**Expected task:** Compare the supplied evidence and name uncertainties.\n",
        encoding="utf-8",
    )
    (workflow / "templates" / "research-brief.md").write_text(
        "# Decision brief\n\n## Evidence\n- {{claim}}\n",
        encoding="utf-8",
    )
    setup = {
        "requirements": [
            {"kind": "toolset", "id": "web", "required": False, "purpose": "Retrieve current public sources."},
            {"kind": "command", "id": "python3", "required": True, "purpose": "Process local evidence."},
        ]
    }
    routines = [
        {
            "id": "weekly-brief",
            "name": "Weekly topic brief",
            "prompt": "Review the approved topic and prepare a cited brief.",
            "schedule": "Weekly at a time and destination chosen after the first task.",
        }
    ]
    _write_bot(catalog, "research-analyst", setup=setup, routines=routines)
    output = tmp_path / "api"

    assert mod.main(catalog_dir=catalog, output_dir=output, workflow_root=workflows) == 0

    page = json.loads((output / "bots.json").read_text(encoding="utf-8"))[0]
    canonical = json.loads((output / "bot-catalog.json").read_text(encoding="utf-8"))["entries"][0]
    assert page["setup"] == setup
    assert page["routines"] == routines
    assert canonical["setup"] == setup
    assert canonical["routines"] == routines
    assert page["workflowPackage"]["skill"] == "official/bots/research-analyst"
    assert page["workflowPackage"]["exampleTasks"] == [
        "Compare the supplied evidence and name uncertainties."
    ]
    assert page["starterExample"] == "Compare the supplied evidence and name uncertainties."
    assert canonical["profile"]["starter_prompt"] == "What topic should I investigate?"
    assert [(sample["kind"], sample["path"]) for sample in page["workflowPackage"]["samples"]] == [
        ("input", "references/sample-input.md"),
        ("template", "templates/research-brief.md"),
    ]
    assert "bounded fixture" in page["workflowPackage"]["samples"][0]["preview"].lower()
    assert "workflowPackage" not in canonical

    sys.path.insert(0, str(REPO_ROOT))
    try:
        from hermes_cli.bot_catalog import BotCatalogEntry

        parsed = BotCatalogEntry.model_validate(canonical)
    finally:
        sys.path.remove(str(REPO_ROOT))
    assert parsed.setup.requirements[0].required is False
    assert parsed.routines[0].id == "weekly-brief"


def test_removed_entries_are_excluded_from_visible_and_install_catalogs(mod, tmp_path):
    catalog = tmp_path / "bot-catalog"
    catalog.mkdir()
    _write_bot(catalog, "active")
    _write_bot(catalog, "retired")
    (catalog / "removed.yaml").write_text(
        yaml.safe_dump({"removed": [{"name": "retired", "reason": "Withdrawn after review."}]}),
        encoding="utf-8",
    )
    output = tmp_path / "api"

    assert mod.main(catalog_dir=catalog, output_dir=output) == 0

    assert [row["name"] for row in json.loads((output / "bots.json").read_text())] == ["active"]
    feed = json.loads((output / "bot-catalog.json").read_text())
    assert [row["name"] for row in feed["entries"]] == ["active"]
    assert feed["removed"] == [{"name": "retired", "reason": "Withdrawn after review."}]


def test_invalid_removed_row_fails_without_overwriting_last_good_feed(mod, tmp_path):
    catalog = tmp_path / "bot-catalog"
    catalog.mkdir()
    _write_bot(catalog, "active")
    output = tmp_path / "api"
    output.mkdir()
    previous = '{"generated_at":"old","entries":[],"removed":[{"name":"blocked","reason":"keep"}]}\n'
    (output / "bot-catalog.json").write_text(previous, encoding="utf-8")
    (catalog / "removed.yaml").write_text(
        yaml.safe_dump({"removed": [{"name": "retired", "reason": "x" * 501}]}),
        encoding="utf-8",
    )

    with pytest.raises(mod.BotCatalogError):
        mod.main(catalog_dir=catalog, output_dir=output)

    assert (output / "bot-catalog.json").read_text(encoding="utf-8") == previous
    assert not (output / "bots.json").exists()
    assert not (output / "bots-meta.json").exists()


def test_removed_reason_uses_canonical_500_character_limit(mod, tmp_path):
    catalog = tmp_path / "bot-catalog"
    catalog.mkdir()
    _write_bot(catalog, "active")
    reason = "x" * 500
    (catalog / "removed.yaml").write_text(
        yaml.safe_dump({"removed": [{"name": "retired", "reason": reason}]}),
        encoding="utf-8",
    )

    assert mod.load_removed(catalog) == [{"name": "retired", "reason": reason}]


def test_missing_catalog_fails_without_publishing_empty_security_feed(mod, tmp_path):
    output = tmp_path / "api"
    output.mkdir()
    previous = '{"generated_at":"old","entries":[],"removed":[{"name":"blocked","reason":"keep"}]}\n'
    (output / "bot-catalog.json").write_text(previous, encoding="utf-8")

    with pytest.raises(mod.BotCatalogError):
        mod.main(catalog_dir=tmp_path / "missing", output_dir=output)

    assert (output / "bot-catalog.json").read_text(encoding="utf-8") == previous
    assert not (output / "bots.json").exists()
    assert not (output / "bots-meta.json").exists()


def test_reserved_author_slugs_do_not_create_dot_routes(mod):
    assert mod.maintainer_slug(".") == "unknown"
    assert mod.maintainer_slug("..") == "unknown"
