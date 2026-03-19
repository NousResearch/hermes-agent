from pathlib import Path

from tools.skill_manager_tool import _validate_frontmatter


def test_document_parse_skill_has_valid_frontmatter_and_body():
    skill_path = Path("optional-skills/research/document-parse/SKILL.md")
    content = skill_path.read_text(encoding="utf-8")

    assert _validate_frontmatter(content) is None
    assert "document_parse" in content
    assert "https://developers.llamaindex.ai/liteparse/guides/library-usage/" in content
