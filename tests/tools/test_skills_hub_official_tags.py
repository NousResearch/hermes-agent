"""Skills hub: the optional-skills source honours top-level ``tags:`` like the GitHub source.

``skills/AGENTS.md`` accepts top-level ``tags:`` beside ``metadata.hermes.tags`` and the
authoring-standards gate blesses either spelling, but ``OptionalSkillSource._scan_all`` read only
the nested one, so skills using the top-level form shipped with no tags and were unfindable by them.
"""

from tools.skills_hub_models import _hermes_tags, _parse_frontmatter
from tools.skills_hub_official import OptionalSkillSource


class TestOptionalSkillSourceTopLevelTags:
    @staticmethod
    def _write_skill(optional_root, frontmatter: str):
        skill_dir = optional_root / "finance" / "tagged-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: tagged-skill\ndescription: test\n{frontmatter}---\n\nBody\n",
            encoding="utf-8",
        )
        src = OptionalSkillSource()
        src._optional_dir = optional_root
        src._list_remote_skill_dirs = dict
        return src

    def test_scan_all_honors_top_level_tags(self, tmp_path):
        """Top-level ``tags:`` is an accepted spelling (skills/AGENTS.md) and
        ``tests/skills/test_authoring_standards.py`` blesses either one, but this source only
        read ``metadata.hermes.tags`` — so a skill using the top-level form shipped with no tags and
        was unfindable by them. ``GitHubSource.inspect`` already accepts both."""
        src = self._write_skill(tmp_path / "optional-skills",
                                "tags: [polymarket, prediction-markets, market-data]\n")

        assert src.inspect("official/finance/tagged-skill").tags == [
            "polymarket", "prediction-markets", "market-data"]
        # ...and tags are a search field (`_matches_query` over name/description/tags).
        assert [m.name for m in src.search("prediction-markets")] == ["tagged-skill"]

    def test_scan_all_prefers_hermes_tags_when_both_spellings_are_present(self, tmp_path):
        """``metadata.hermes.tags`` stays authoritative — same precedence as the GitHub source."""
        src = self._write_skill(
            tmp_path / "optional-skills",
            "tags: [top-level]\nmetadata:\n  hermes:\n    tags: [canonical]\n")

        assert src.inspect("official/finance/tagged-skill").tags == ["canonical"]

    def test_scan_all_leaves_a_malformed_tags_value_empty(self, tmp_path):
        """A non-list ``tags:`` must not leak a string into the list-typed field."""
        src = self._write_skill(tmp_path / "optional-skills", "tags: not-a-list\n")

        assert src.inspect("official/finance/tagged-skill").tags == []

    def test_shipped_optional_skills_keep_every_declared_tag(self):
        """Regression guard against the real catalog: four shipped skills use the top-level spelling
        (optional-skills/finance/polymarket, gaming/minecraft-modpack-server, gaming/pokemon-player,
        mlops/models/huggingface-hub) and lost their tags in this source."""
        src = OptionalSkillSource()
        by_path = {m.path: m.tags for m in src._scan_all()}
        assert by_path, "no optional skills found in the local checkout"

        dropped = []
        for skill_md in src._local_skill_mds():
            fm = _parse_frontmatter(skill_md.read_text(encoding="utf-8"))
            declared = _hermes_tags(fm) or fm.get("tags")
            if not isinstance(declared, list) or not declared:
                continue
            rel = skill_md.parent.relative_to(src._optional_dir).as_posix()
            if not by_path.get(f"optional-skills/{rel}"):
                dropped.append(rel)
        assert dropped == [], f"optional skills whose declared tags were dropped: {dropped}"
