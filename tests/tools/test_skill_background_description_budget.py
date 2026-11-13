"""Background staged creates retain the skill index's description limit."""
from unittest.mock import patch

import pytest

from agent.skill_utils import SKILL_PROMPT_DESC_LIMIT
from tests.tools.test_skill_manager_tool import VALID_SKILL_CONTENT, _skill_dir
from tools.skill_manager_tool import _create_skill


@pytest.mark.parametrize("background", [False, True])
def test_create_rejects_long_descriptions_before_publication(tmp_path, background):
    content = VALID_SKILL_CONTENT.replace(
        "description: A test skill for unit testing.",
        "description: " + "x" * (SKILL_PROMPT_DESC_LIMIT + 1),
    )
    assert content != VALID_SKILL_CONTENT
    with _skill_dir(tmp_path), patch(
        "tools.skill_provenance.is_background_review", return_value=background,
    ):
        result = _create_skill("budget-skill", content)
    assert result["success"] is False, result
    assert "skill description must fit" in result["error"]
    assert not (tmp_path / "budget-skill").exists()
