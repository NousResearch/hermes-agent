"""skill_view paging primitive (issue #135980): over-spill skills are unreadable.

The skill write cap (``MAX_SKILL_CONTENT_CHARS`` = 100k) and the tool-result
spill threshold (``budget_for_context_window``) can collide: a skill legally
under the write cap still produces a skill_view result over the spill
threshold, and the dispatch layer spills that result to disk — the curator
can neither read nor remediate the skill. ``offset``/``limit`` paging keeps
each served page under the threshold while a plain (un-paginated) view stays
byte-identical to the historical full-content response.
"""

import json
from unittest.mock import patch

import pytest

from tools.skills_tool import skill_view


def _spill_threshold() -> int:
    """The spill threshold dispatch actually enforces for skill_view, resolved
    through the production budget path. CHARS_PER_TOKEN is 4, so a 200K window
    scales to exactly the 100k write cap and the two constants only coincide
    there; on smaller windows the scaled threshold is strictly LOWER
    (65K -> 39k, 128K -> 76.8k), so a skill legally under the write cap still
    spills. 65K is the window where the sick class is widest while the write
    cap stays unchanged."""
    from tools.budget_config import budget_for_context_window

    return int(budget_for_context_window(65_000).resolve_threshold("skill_view"))


def _oversized_skill_lines() -> list[str]:
    """Lines for a skill that exceeds the spill threshold but stays under the
    100k write cap: the exact sick class the issue describes. Lines are unique
    so each page's content is verifiable."""
    lines = []
    for i in range(1, 2601):
        lines.append(f"line {i:05d}: " + "x" * 24)
    return lines


@pytest.fixture()
def big_skill(tmp_path):
    """A fake skills dir holding one oversized skill (~93k chars, 3000 lines)."""
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "big-skill"
    skill_dir.mkdir(parents=True)
    lines = _oversized_skill_lines()
    body = "\n".join(
        ["---", "name: big-skill", "description: oversized skill fixture", "---", ""]
        + lines
    )
    (skill_dir / "SKILL.md").write_text(body)
    with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
        yield {"skill_dir": skill_dir, "lines": lines, "body": body}


class TestSkillViewPaging:
    def test_oversize_skill_full_view_exceeds_spill_threshold(self, big_skill):
        """The fixture reproduces the sick class: a plain skill_view body is
        over the spill threshold (so dispatch spills it and the curator sees
        only a stub)."""
        result = json.loads(skill_view("big-skill"))
        assert result["success"] is True
        assert len(result["content"]) > _spill_threshold()
        assert len(result["content"]) < 100_000  # was legal to write

    def test_paged_slices_stay_under_spill_threshold(self, big_skill):
        """Paging keeps every served slice under the spill threshold, pages
        tile the full content exactly, and the last page has no next_offset."""
        threshold = _spill_threshold()
        lines = big_skill["lines"]
        full_body = big_skill["body"]

        seen = []
        offset = 1
        for _ in range(10):  # bounded walk; guard against a next_offset loop
            page = json.loads(skill_view("big-skill", offset=offset, limit=1000))
            assert page["success"] is True
            assert len(page["content"]) <= threshold, (
                f"page at offset={offset} exceeds the spill threshold")
            seen.append((offset, page))
            next_offset = page.get("next_offset")
            if next_offset is None:
                break
            assert next_offset > offset
            offset = next_offset
        else:
            pytest.fail("paging never terminated: next_offset loop")

        # Pages tile the content exactly: page 1 holds the first lines in order.
        first = seen[0][1]["content"]
        assert first == "\n".join(
            ["---", "name: big-skill", "description: oversized skill fixture", "---", ""]
            + lines[: 1000 - 5]
        )
        # The full body reassembles from the walked pages (order-preserving,
        # lossless, no duplication).
        reassembled = []
        expected = full_body.split("\n")
        pos = 0
        for offset, page in seen:
            page_lines = page["content"].split("\n")
            assert page_lines == expected[pos: pos + len(page_lines)]
            pos += len(page_lines)
        reassembled = "\n".join(expected[:pos])
        assert reassembled == full_body
        assert pos == len(expected)


    def test_default_view_stays_backward_compatible(self, big_skill):
        """No offset/limit -> the historical response: full content, and no
        paging metadata on the result."""
        result = json.loads(skill_view("big-skill"))
        assert result["success"] is True
        assert result["content"] == big_skill["body"]
        assert "next_offset" not in result
        assert "content_total_lines" not in result
        assert "content_note" not in result


class TestSkillViewPagingDispatchWiring:
    """Dispatch calls the registry handler, not skill_view directly: the params
    must survive the handler boundary (mutation self-proof target)."""

    def test_handler_passes_paging_params_through(self, big_skill):
        from tools.skills_tool import _skill_view_with_bump, reset_skill_view_dedup

        reset_skill_view_dedup()
        page = json.loads(_skill_view_with_bump(
            {"name": "big-skill", "offset": 1, "limit": 1000}, task_id="paging-wiring"))
        assert page["success"] is True
        assert page.get("next_offset") == 1001
        assert len(page["content"]) < len(big_skill["body"])

    def test_handler_limit_only_counts_as_paged(self, big_skill):
        from tools.skills_tool import _skill_view_with_bump, reset_skill_view_dedup

        reset_skill_view_dedup()
        page = json.loads(_skill_view_with_bump(
            {"name": "big-skill", "limit": 1000}, task_id="paging-wiring-2"))
        assert page["success"] is True
        assert page["content_total_lines"] == len(big_skill["body"].split("\n"))
        assert page["content"].split("\n") == big_skill["body"].split("\n")[:1000]

