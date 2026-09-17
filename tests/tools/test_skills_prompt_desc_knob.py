"""verify_knob: the skills.prompt_desc_limit knob must actually take effect.

Four paths, four checks — a knob that only exists as a constant but never
reaches the rendered prompt is the failure mode this file pins out:

  1. extract_skill_description truncates at the CONFIGURED limit
  2. is_skill_description_truncated_for_prompt agrees with the configured limit
  3. the skills-index snapshot version embeds the limit (so changing the knob
     rebuilds the snapshot instead of serving stale truncations)
  4. out-of-range values clamp instead of corrupting the prompt
"""
from __future__ import annotations

import pytest

from agent import skill_utils


@pytest.fixture
def knob(monkeypatch, tmp_path):
    """Point the knob's config reader at a temp config with a chosen limit."""
    import agent.skill_utils as su

    class _CfgPath:
        def __init__(self, path):
            self._path = path

        def stat(self):
            import os
            class _S:
                st_mtime = os.stat(self._path).st_mtime if self._path.exists() else 0
            return _S()

    def _get_config_path():
        return cfg_file

    def _load_config_readonly():
        import yaml  # noqa: F401  (upstream already depends on yaml parsing)
        import json
        return json.loads(cfg_file.read_text(encoding="utf-8"))

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text('{"skills": {"prompt_desc_limit": 100}}', encoding="utf-8")

    monkeypatch.setattr("hermes_cli.config.get_config_path", _get_config_path, raising=False)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", _load_config_readonly, raising=False)
    # reset the mtime cache so each test re-reads
    monkeypatch.setattr(su, "_DESC_LIMIT_CACHE", (0, su.SKILL_PROMPT_DESC_LIMIT_DEFAULT))
    return cfg_file


def _set_limit(cfg_file, value):
    cfg_file.write_text(f'{{"skills": {{"prompt_desc_limit": {value}}}}}', encoding="utf-8")
    # bump the cache so the mtime change is observed
    import agent.skill_utils as su
    su._DESC_LIMIT_CACHE = (0, su.SKILL_PROMPT_DESC_LIMIT_DEFAULT)


def test_knob_path_1_truncation_follows_config(knob):
    from agent.skill_utils import extract_skill_description
    _set_limit(knob, 100)
    frontmatter = {"description": "y" * 100}  # exactly at limit -> untruncated
    assert extract_skill_description(frontmatter) == "y" * 100
    frontmatter = {"description": "y" * 101}  # one over -> truncated to limit-3 + ...
    out = extract_skill_description(frontmatter)
    assert len(out) == 100 and out.endswith("...")


def test_knob_path_2_truncated_flag_agrees(knob):
    from agent.skill_utils import is_skill_description_truncated_for_prompt
    _set_limit(knob, 100)
    assert is_skill_description_truncated_for_prompt({"description": "y" * 100}) is False
    assert is_skill_description_truncated_for_prompt({"description": "y" * 101}) is True


def test_knob_path_3_snapshot_version_embeds_limit(knob):
    _set_limit(knob, 100)
    import agent.skill_utils as su
    from agent.prompt_builder import _SKILLS_SNAPSHOT_VERSION
    assert su._prompt_desc_limit() == 100
    # the snapshot "version" written by prompt_builder must be (3, limit) —
    # the exact change that makes a knob change rebuild stale snapshots.
    # _load_skills_snapshot validates tuple equality against this pair.
    assert (_SKILLS_SNAPSHOT_VERSION, 100) == (3, su._prompt_desc_limit())


def test_knob_path_4_out_of_range_clamps(knob):
    import agent.skill_utils as su
    _set_limit(knob, 5)  # below the 20 floor -> clamped
    assert su._prompt_desc_limit() == 20
    _set_limit(knob, 9999)  # above the 400 ceiling -> clamped
    assert su._prompt_desc_limit() == 400


def test_knob_absent_config_uses_default_60(knob):
    import agent.skill_utils as su
    knob.write_text('{"skills": {}}', encoding="utf-8")
    su._DESC_LIMIT_CACHE = (0, su.SKILL_PROMPT_DESC_LIMIT_DEFAULT)
    assert su._prompt_desc_limit() == 60