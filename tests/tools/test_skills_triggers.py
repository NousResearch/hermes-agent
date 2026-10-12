"""Tests for declared skill triggers — the phrases the composer may offer a skill on.

Why the field exists: the desktop composer's skill pill offers a skill only when the draft NAMES it,
so a skill whose name the user never types — because they write in another language, or because the
name is a common word — could never be offered. A declared phrase list, written in the user's own
words, is the only signal that reaches it.

These tests pin the contract its one consumer depends on, and nothing more: the parser lives in the
metadata layer (``agent.skill_utils``) beside its ``extract_skill_*`` siblings, both declaration
shapes parse, the list is bounded, garbage degrades to an EMPTY LIST rather than to a stringified
scalar, the phrases stay OUT of the model-facing listing unless asked for, and every row the desktop
receives when it does ask carries a list (it maps over it).
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from agent.skill_utils import MAX_TRIGGERS, extract_skill_triggers
from tools.skills_tool import _find_all_skills, _parse_triggers


def _make_skill(skills_dir, name, frontmatter_extra="", body="Step 1: Do the thing.", category=None):
    """A minimal skill directory — mirrors the helper in tests/tools/test_skills_tool.py."""
    skill_dir = skills_dir / category / name if category else skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Description for {name}.\n{frontmatter_extra}---\n\n# {name}\n\n{body}\n",
        encoding="utf-8",
    )
    return skill_dir


class TestExtractSkillTriggers:
    def test_top_level_list(self):
        assert extract_skill_triggers({"triggers": ["دمج ملفين", "ملف pdf"]}) == ["دمج ملفين", "ملف pdf"]

    def test_metadata_hermes_list(self):
        assert extract_skill_triggers({"metadata": {"hermes": {"triggers": ["a phrase"]}}}) == ["a phrase"]

    def test_metadata_hermes_takes_precedence_when_both_are_declared(self):
        # House precedence, the same one skill_view uses for tags/related_skills: the
        # agentskills.io location (metadata.hermes.*) is read first, then the top level.
        frontmatter = {"triggers": ["top"], "metadata": {"hermes": {"triggers": ["nested"]}}}
        assert extract_skill_triggers(frontmatter) == ["nested"]

    def test_comma_separated_string(self):
        assert extract_skill_triggers({"triggers": "one, two , three"}) == ["one", "two", "three"]

    def test_quoted_string_entries_are_unquoted(self):
        assert extract_skill_triggers({"triggers": '"a phrase", \'another\''}) == ["a phrase", "another"]

    def test_absent_yields_empty(self):
        assert extract_skill_triggers({}) == []
        assert extract_skill_triggers({"description": "no triggers here"}) == []

    def test_blank_entries_are_dropped(self):
        assert extract_skill_triggers({"triggers": ["ok", "", "   ", "\t"]}) == ["ok"]

    def test_entries_are_trimmed(self):
        assert extract_skill_triggers({"triggers": ["  spaced  "]}) == ["spaced"]

    def test_list_is_capped(self):
        assert len(extract_skill_triggers({"triggers": [f"phrase {i}" for i in range(40)]})) == MAX_TRIGGERS

    @pytest.mark.parametrize("bad", [7, None, True, {"a": 1}, [["nested"], {"k": "v"}]])
    def test_non_phrase_values_yield_empty(self, bad):
        # A scalar must NOT become its stringification ("7" is a phrase nobody wrote), and a nested
        # list is not a phrase list.
        assert extract_skill_triggers({"triggers": bad}) == []

    def test_numbers_inside_a_list_are_dropped_not_stringified(self):
        assert extract_skill_triggers({"triggers": ["kept", 5, None, {"x": 1}]}) == ["kept"]

    def test_nested_garbage_yields_empty(self):
        assert extract_skill_triggers({"metadata": {"hermes": {"triggers": 5}}}) == []
        assert extract_skill_triggers({"metadata": "not a mapping"}) == []

    def test_the_tool_layer_delegates_to_the_metadata_layer(self):
        # The seam, the same one parse_frontmatter and get_disabled_skill_names use: a lazy call-time
        # delegate, so patching agent.skill_utils reaches the tool layer too.
        assert _parse_triggers({"triggers": ["via the delegate"]}) == ["via the delegate"]
        with patch("agent.skill_utils.extract_skill_triggers", return_value=["patched"]):
            assert _parse_triggers({"triggers": ["ignored"]}) == ["patched"]


class TestMirroredCap:
    """The cap exists in two languages and drifts silently: a backend that keeps twelve while the
    desktop clamps to eight drops a phrase the author believes is live. Pin the two literals."""

    def test_the_desktop_cap_matches_the_backend_cap(self):
        ts = Path(__file__).resolve().parents[2] / "apps/desktop/src/store/suggestion-providers/skill.ts"
        if not ts.is_file():
            pytest.skip("desktop sources are not part of this checkout")
        declared = [line for line in ts.read_text(encoding="utf-8").splitlines() if "const MAX_TRIGGERS" in line]
        assert declared, "skill.ts no longer declares MAX_TRIGGERS — move this cap with it"
        assert f"= {MAX_TRIGGERS}" in declared[0].rstrip(), f"the two caps disagree: {declared[0].strip()!r}"


class TestFindAllSkillsCarriesTriggers:
    """The API contract: the desktop asks for the phrases, and every row it gets carries a list."""

    def test_the_phrases_stay_out_of_the_default_listing(self, tmp_path):
        # This listing also feeds the model-facing skills_list tool. The phrases are a composer
        # affordance — shipping them to the model on every call is attention spent for nothing.
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "with-triggers", 'triggers: ["phrase one"]\n')
            rows = _find_all_skills()
        assert rows and all("triggers" not in row for row in rows)

    def test_declared_triggers_reach_the_listing_when_asked_for(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "with-triggers", 'triggers: ["phrase one", "عبارة عربية"]\n')
            _make_skill(tmp_path, "without-triggers")
            rows = {row["name"]: row for row in _find_all_skills(include_triggers=True)}
        assert rows["with-triggers"]["triggers"] == ["phrase one", "عبارة عربية"]
        assert rows["without-triggers"]["triggers"] == []

    def test_block_sequence_form_parses(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "blocky", 'triggers:\n  - "first phrase"\n  - "ثانية"\n')
            rows = {row["name"]: row for row in _find_all_skills(include_triggers=True)}
        assert rows["blocky"]["triggers"] == ["first phrase", "ثانية"]

    def test_metadata_nested_form_reaches_the_listing(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "nested", "metadata:\n  hermes:\n    triggers:\n      - \"from metadata\"\n")
            rows = {row["name"]: row for row in _find_all_skills(include_triggers=True)}
        assert rows["nested"]["triggers"] == ["from metadata"]

    def test_every_row_carries_a_list_even_when_the_declaration_is_junk(self, tmp_path):
        # The desktop maps over this value. A string or a number here is the shape that breaks it,
        # so the guarantee is asserted for EVERY row, not only the well-formed declarations.
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "junk", "triggers: 7\n")
            _make_skill(tmp_path, "good", 'triggers: ["x"]\n')
            _make_skill(tmp_path, "none")
            rows = _find_all_skills(include_triggers=True)
        assert rows, "expected a non-empty listing"
        assert all(isinstance(row.get("triggers"), list) for row in rows)

    def test_the_returned_list_is_not_the_cached_one(self, tmp_path):
        # _skill_catalog hands out shallow copies and its own comment warns that callers mutate rows.
        # A caller appending to `triggers` must not poison the cache for the next reader.
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "mutable", 'triggers: ["orig"]\n')
            first = next(r for r in _find_all_skills(include_triggers=True) if r["name"] == "mutable")
            first["triggers"].append("appended by a caller")
            second = next(r for r in _find_all_skills(include_triggers=True) if r["name"] == "mutable")
        assert second["triggers"] == ["orig"]

    def test_skills_without_the_field_are_unchanged(self, tmp_path):
        # Backwards compatibility: nothing else about a skill row changes when the field is absent.
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "plain", category="devops")
            (row,) = [r for r in _find_all_skills(include_triggers=True) if r["name"] == "plain"]
        assert row["category"] == "devops"
        assert row["description"] == "Description for plain."
        assert row["triggers"] == []
