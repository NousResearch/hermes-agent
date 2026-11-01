"""Tests for skills/protection.py — the declarative skill protection guard.

Covers both review blockers:
1. Provenance is derived from the LOCATED skill dir (frontmatter name), so a caller
   passing the category-path form (``productivity/docx``) cannot bypass the guard.
2. The refusal dict matches the ``_err()`` schema (``{"success": False, "error": ...}``).
"""

from pathlib import Path
from unittest.mock import patch

from skills.protection import check_mutable, check_deletable


def _make_skill(tmp_path: Path, frontmatter_name: str, category: str = "productivity") -> Path:
    """Create a category-nested skill dir whose SKILL.md frontmatter name differs from the path."""
    skill_dir = tmp_path / category / frontmatter_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {frontmatter_name}\ndescription: Test skill.\n---\n\n# {frontmatter_name}\n",
        encoding="utf-8",
    )
    return skill_dir


@patch("tools.skill_usage.is_hub_installed", return_value=False)
def test_bundled_bare_name_blocked(mock_hub, tmp_path):
    """A bundled skill keyed on the frontmatter name is blocked even when no dir is passed."""
    skill_dir = _make_skill(tmp_path, "docx")
    with patch("tools.skill_usage.is_bundled", side_effect=lambda n: n == "docx") as mock_bundled:
        result = check_mutable(skill_dir, "docx", "edit")
    assert result is not None
    # Guard was keyed on the canonical provenance name.
    assert "docx" in mock_bundled.call_args.args


@patch("tools.skill_usage.is_bundled", return_value=True)
@patch("tools.skill_usage.is_hub_installed", return_value=False)
def test_category_path_form_cannot_bypass(mock_hub, mock_bundled, tmp_path):
    """Blocker 1: passing the category-path form ('productivity/docx') must still be blocked
    because provenance is read from the located dir's frontmatter name, not the caller string."""
    skill_dir = _make_skill(tmp_path, "docx", category="productivity")
    result = check_mutable(skill_dir, "productivity/docx", "edit")
    assert result is not None
    # is_bundled must be asked about the frontmatter name 'docx', never the raw 'productivity/docx'.
    assert mock_bundled.call_args.args == ("docx",)
    assert "productivity/docx" not in mock_bundled.call_args.args


@patch("tools.skill_usage.is_bundled", return_value=False)
def test_hub_installed_blocked(mock_bundled, tmp_path):
    """A hub-installed skill is blocked; provenance again keyed on the located dir's name."""
    skill_dir = _make_skill(tmp_path, "some-hub-skill", category="mlops")
    with patch(
        "tools.skill_usage.is_hub_installed", side_effect=lambda n: n == "some-hub-skill"
    ) as mock_hub:
        result = check_mutable(skill_dir, "mlops/some-hub-skill", "edit")
    assert result is not None
    assert mock_hub.call_args.args == ("some-hub-skill",)


@patch("tools.skill_usage.is_bundled", return_value=False)
@patch("tools.skill_usage.is_hub_installed", return_value=False)
def test_user_skill_mutable(mock_hub, mock_bundled, tmp_path):
    """A user/agent-created skill is editable (no dir located => falls back to the raw name)."""
    result = check_mutable(None, "my-custom-skill", "edit")
    assert result is None


@patch("tools.skill_usage.is_bundled", return_value=True)
@patch("tools.skill_usage.is_hub_installed", return_value=False)
def test_refusal_matches_err_schema(mock_hub, mock_bundled, tmp_path):
    """Blocker 2: the refusal dict uses the _err() schema — success False + error key,
    no 'message' key (which would leave success absent and the dashboard showing a bare bool)."""
    skill_dir = _make_skill(tmp_path, "docx")
    result = check_mutable(skill_dir, "docx", "edit")
    assert result is not None
    assert result.get("success") is False
    assert "error" in result and isinstance(result["error"], str)
    assert "message" not in result
    assert "productivity/docx" not in result


@patch("tools.skill_usage.is_bundled", return_value=True)
@patch("tools.skill_usage.is_hub_installed", return_value=False)
def test_check_deletable_delegates(mock_hub, mock_bundled, tmp_path):
    """check_deletable is the delete surface of the same guard."""
    skill_dir = _make_skill(tmp_path, "docx")
    result = check_deletable(skill_dir, "docx")
    assert result is not None
    assert result.get("success") is False
    assert "delete" in result["error"]


@patch("tools.skill_usage.is_bundled", return_value=False)
@patch("tools.skill_usage.is_hub_installed", return_value=False)
def test_missing_skill_md_falls_back_to_dir_name(mock_hub, mock_bundled, tmp_path):
    """No SKILL.md: provenance falls back to the directory name, so a bundled nested dir
    without readable frontmatter is still protected."""
    skill_dir = tmp_path / "dataeng" / "etl"
    skill_dir.mkdir(parents=True, exist_ok=True)  # no SKILL.md
    with patch(
        "tools.skill_usage.is_bundled", side_effect=lambda n: n == "etl"
    ) as mock_bundled:
        result = check_mutable(skill_dir, "dataeng/etl", "edit")
    assert result is not None
    assert mock_bundled.call_args.args == ("etl",)