import json
from unittest.mock import patch

import pytest

from tools.skills_tool import SKILLS_LIST_SCHEMA, _SKILLS_CACHE, skills_list


def _make_skill(
    root,
    name,
    *,
    description=None,
    category=None,
    extra="",
):
    directory = root / category / name if category else root / name
    directory.mkdir(parents=True)
    description = description if description is not None else f"Description for {name}."
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n{extra}---\n\n# {name}\n",
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def _clear_discovery_cache():
    _SKILLS_CACHE.clear()
    yield
    _SKILLS_CACHE.clear()


def test_query_ranks_exact_name_first_and_limits_results(tmp_path):
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(tmp_path, "postgres-backup", description="Back up SQL databases")
        _make_skill(tmp_path, "generic-backup", description="Back up local files")
        result = json.loads(skills_list(query="postgres backup", limit=1))

    assert result["query"] == "postgres backup"
    assert result["total_matches"] == 2
    assert [skill["name"] for skill in result["skills"]] == ["postgres-backup"]


def test_query_uses_private_routing_hints_without_leaking_them(tmp_path):
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(
            tmp_path,
            "parcel-status",
            category="logistics",
            description="Track parcel delivery status",
            extra=(
                "metadata:\n"
                "  hermes:\n"
                "    triggers: [courier, tracking number]\n"
            ),
        )
        result = json.loads(skills_list(query="courier tracking number"))

    assert result["skills"][0]["name"] == "parcel-status"
    assert set(result["skills"][0]) == {"name", "description", "category"}


def test_query_supports_legacy_top_level_tags(tmp_path):
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(
            tmp_path,
            "inventory-audit",
            description="Audit inventory records",
            extra="tags: [warehouse, stocktake]\n",
        )
        result = json.loads(skills_list(query="warehouse stocktake"))

    assert result["skills"][0]["name"] == "inventory-audit"


def test_query_matches_unicode_casefolded_metadata(tmp_path):
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(tmp_path, "cafe-tools", description="Gestiona pedidos de CAFÉ")
        result = json.loads(skills_list(query="cafe\u0301"))

    assert result["skills"][0]["name"] == "cafe-tools"


def test_category_filter_is_applied_before_query_ranking(tmp_path):
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(tmp_path, "finance-python", category="finance", description="Python finance")
        _make_skill(
            tmp_path,
            "development-python",
            category="development",
            description="Python application development",
        )
        result = json.loads(skills_list(category="finance", query="python"))

    assert [skill["name"] for skill in result["skills"]] == ["finance-python"]
    assert result["categories"] == ["finance"]


def test_query_abstains_without_lexical_evidence(tmp_path):
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(tmp_path, "python", description="Python development")
        result = json.loads(skills_list(query="quantum entanglement"))

    assert result["skills"] == []
    assert result["total_matches"] == 0


def test_query_limits_are_clamped(tmp_path):
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        for index in range(3):
            _make_skill(tmp_path, f"python-{index}", description="Python automation")
        minimum = json.loads(skills_list(query="python", limit=0))
        maximum = json.loads(skills_list(query="python", limit=1000))

    assert minimum["count"] == 1
    assert maximum["count"] == 3


def test_exact_name_cannot_be_displaced_by_trigger_stuffing(tmp_path):
    triggers = ", ".join(f"python topic {index}" for index in range(40))
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(tmp_path, "python", description="Python development")
        _make_skill(
            tmp_path,
            "keyword-stuffer",
            description="Unrelated metadata test",
            extra=f"triggers: [{triggers}]\n",
        )
        result = json.loads(skills_list(query="python"))

    assert result["skills"][0]["name"] == "python"


def test_exact_stopword_name_still_matches(tmp_path):
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(tmp_path, "the", description="Deliberately unusual name")
        result = json.loads(skills_list(query="the"))

    assert [skill["name"] for skill in result["skills"]] == ["the"]


def test_malformed_routing_metadata_is_ignored(tmp_path):
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(
            tmp_path,
            "safe-skill",
            description="Safe database maintenance",
            extra="triggers: {unexpected: mapping}\nmetadata:\n  hermes: invalid\n",
        )
        result = json.loads(skills_list(query="database"))

    assert result["skills"][0]["name"] == "safe-skill"


def test_query_echo_and_work_are_bounded(tmp_path):
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(tmp_path, "needle", description="Find a bounded needle")
        result = json.loads(skills_list(query="needle " + "x" * 10_000))

    assert len(result["query"]) == 500


def test_large_catalog_is_stable_and_bounded(tmp_path):
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        for index in range(500):
            _make_skill(
                tmp_path,
                f"catalog-{index:03d}",
                description=f"Shared automation target-{index:03d}",
            )
        first = json.loads(skills_list(query="automation target-321", limit=10))
        second = json.loads(skills_list(query="automation target-321", limit=10))

    assert first["count"] == 10
    assert first["total_matches"] == 500
    assert first["skills"][0]["name"] == "catalog-321"
    assert first["skills"] == second["skills"]


def test_schema_exposes_bounded_query_search():
    properties = SKILLS_LIST_SCHEMA["parameters"]["properties"]
    assert properties["query"]["maxLength"] == 500
    assert properties["limit"]["minimum"] == 1
    assert properties["limit"]["maximum"] == 50


def test_no_query_preserves_legacy_hint(tmp_path):
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(tmp_path, "python", description="Python development")
        result = json.loads(skills_list())

    assert result["hint"] == "Use skill_view(name) to see full content, tags, and linked files"
