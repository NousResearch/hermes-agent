"""Regression: the TDD skill must exempt trivial one-off tasks from the iron law.

Issue #83532: the TDD skill was so dogmatic that trivial programs triggered a
full red-green-refactor release process, wasting tokens. The skill is a runtime
instruction artifact; this test guards that the exemption appears in the
"When to Use" exception list so the model sees it.
"""

import re
from pathlib import Path

import pytest


SKILL_PATH = Path(__file__).parents[2] / "skills" / "software-development" / "test-driven-development" / "SKILL.md"


@pytest.fixture
def skill_text():
    return SKILL_PATH.read_text(encoding="utf-8")


def test_tdd_skill_has_when_to_use_exceptions(skill_text):
    """The skill must have an Exceptions list under When to Use."""
    match = re.search(r"## When to Use\n(.+?)(?:\n## |\Z)", skill_text, re.S)
    assert match, "When to Use section not found"
    section = match.group(1)
    assert re.search(r"^\*\*Exceptions", section, re.M), "Exceptions list not found under When to Use"


def test_tdd_skill_exempts_trivial_one_off_tasks(skill_text):
    """The exceptions must mention trivial / one-off / small scripts that don't need a release pipeline."""
    match = re.search(r"## When to Use\n(.+?)(?:\n## |\Z)", skill_text, re.S)
    assert match
    section = match.group(1)
    assert re.search(
        r"\b(trivial|one-off|one off|small script|no release|single file|single-file)\b",
        section,
        re.I,
    ), "Trivial one-off task exemption is missing from When to Use exceptions"
