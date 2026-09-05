"""Regression: the TDD skill must exempt trivial one-off tasks from the iron law.

Issue #83532: the TDD skill was so dogmatic that trivial programs triggered a
full red-green-refactor release process. Guard the exemption through the same
frontmatter parser the prompt builder uses, not a raw source-file scan.
"""

from pathlib import Path

from agent.skill_utils import parse_frontmatter


SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "software-development"
    / "test-driven-development"
    / "SKILL.md"
)


def test_tdd_skill_loads_trivial_one_off_exception():
    """The shipped skill must parse and expose the When to Use exemption."""
    frontmatter, body = parse_frontmatter(SKILL_PATH.read_text(encoding="utf-8"))
    assert frontmatter.get("name"), "TDD skill frontmatter must include a name"
    assert "When to Use" in body
    assert "Exceptions" in body
    lowered = body.lower()
    assert any(
        token in lowered
        for token in ("trivial", "one-off", "one off", "small script", "single-file")
    ), "Trivial one-off exemption missing from the loaded TDD skill body"
