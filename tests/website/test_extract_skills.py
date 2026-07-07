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
    assert mod._source_url("clawhub", "go-music-skill", {}) == "https://clawhub.ai/skills/go-music-skill"


def test_source_url_synthesizes_clawhub_strips_prefix(mod):
    # identifier may arrive already prefixed; we must not double-prefix.
    assert (
        mod._source_url("clawhub", "clawhub/go-music-skill", {})
        == "https://clawhub.ai/skills/go-music-skill"
    )


def test_source_url_synthesizes_lobehub(mod):
    assert mod._source_url("lobehub", "lobehub/chinese-paper", {}) == "https://lobehub.com/agent/chinese-paper"


def test_source_url_empty_for_unknown_source_without_identifier(mod):
    assert mod._source_url("mystery", "", {}) == ""


# --------------------------------------------------------------------------
# _guess_category
# --------------------------------------------------------------------------

def test_guess_category_maps_known_tag(mod):
    assert mod._guess_category(["security"]) == "security"
    assert mod._guess_category(["machine-learning"]) == "mlops"
    assert mod._guess_category(["crypto"]) == "blockchain"


def test_guess_category_accepts_literal_curated_key(mod):
    # A skill tagged literally with a curated category key should route there.
    assert mod._guess_category(["devops"]) == "devops"


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


# --------------------------------------------------------------------------
# Workflow docs / shipped skill content
# --------------------------------------------------------------------------


def test_workflows_docs_lead_with_plain_language_flow():
    docs = REPO_ROOT / "website" / "docs" / "user-guide" / "features" / "workflows.md"
    text = docs.read_text(encoding="utf-8")

    assert (
        "Workflows are named, versioned automations that Hermes can draft from a plain-language goal, "
        "validate, deploy, and run durably. You normally start by describing what you want to "
        "automate; YAML/JSON remains available as an advanced import/export/debug format."
    ) in text

    headings = [
        "# Workflows",
        "## Build from plain language",
        "## When to use Workflows vs Kanban",
        "## Create a workflow in the dashboard",
        "## Review and refine the draft",
        "## Validate and deploy",
        "## Run a test execution",
        "## Monitor execution and linked worker tasks",
        "## Advanced: YAML definitions",
        "## CLI reference",
        "## Schema reference",
        "## Limitations / unsupported primitives",
    ]
    positions = [text.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert "Use **Describe workflow**" in text
    assert "graph validation cannot fully prove yet" in text
    assert "The execution detail node-runs drill-down shows `Linked worker task: <id>`" in text
    assert "Use that task id in Kanban views or CLI commands" in text
    assert "Use the linked worker task drill-down from execution detail to inspect" not in text


def test_workflow_builder_skill_prefers_draft_refine_before_yaml():
    skill = REPO_ROOT / "skills" / "autonomous-ai-agents" / "hermes-workflow-builder" / "SKILL.md"
    text = skill.read_text(encoding="utf-8")

    assert "workflow_draft" in text
    assert "workflow_refine" in text
    assert "Only write YAML by hand" in text
    assert "Always validate before deploy" in text

    first_draft = text.index("workflow_draft")
    first_refine = text.index("workflow_refine")
    manual_yaml = text.index("Only write YAML by hand")
    assert first_draft < manual_yaml
    assert first_refine < manual_yaml
