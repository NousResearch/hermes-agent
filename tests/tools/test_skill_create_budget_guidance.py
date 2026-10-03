"""The advertised create budget must match the existing validator."""
import re

from agent.skill_utils import SKILL_PROMPT_DESC_LIMIT
from tools.skill_manager_tool import SKILL_MANAGE_SCHEMA, _validate_frontmatter
from tools.registry import registry


def test_create_schema_advertises_enforced_description_budget():
    for schema in (SKILL_MANAGE_SCHEMA, registry.get_definitions({"skill_manage"}, quiet=True)[0]["function"]):
        branches = schema["parameters"]["properties"]["operations"]["items"]["anyOf"]
        create = next(b for b in branches if b["properties"]["action"]["enum"] == ["create"])
        guidance = create["properties"]["content"]["description"]
        match = re.search(r"description.*?at most (\d+) characters", guidance)
        assert match, "Create schema must disclose the enforced frontmatter description limit before writing"
        limit = int(match.group(1))
        assert limit == SKILL_PROMPT_DESC_LIMIT
        content = "---\nname: budget-check\ndescription: {}\n---\n\nInstructions.\n"
        assert _validate_frontmatter(content.format("x" * limit), new_skill=True) is None
        assert _validate_frontmatter(content.format("x" * (limit + 1)), new_skill=True) is not None
        assert _validate_frontmatter(content.format("x" * (limit + 1)), new_skill=False) is None
