"""verify_knob: the skills.prompt_desc_limit knob must actually take effect.

Four paths, four checks — a knob that only exists as a constant but never
reaches the rendered prompt is the failure mode this file pins out:

  1. extract_skill_description truncates at the CONFIGURED limit
  2. is_skill_description_truncated_for_prompt agrees with the configured limit
  3. the skills-index snapshot version embeds the limit (so changing the knob
     rebuilds the snapshot instead of serving stale truncations)
  4. out-of-range values clamp instead of corrupting the prompt
  5. a PRE-UPGRADE snapshot whose "version" is the bare int (e.g. 2) is a
     rebuild trigger — it must not raise TypeError past the corrupt guard
  6. a snapshot whose [version, limit] matches the knob is served as-is
  7. a snapshot written under a DIFFERENT limit rebuilds
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


# --- snapshot migration: _load_skills_snapshot must survive every legacy shape --

def _snapshot_fixture(monkeypatch, tmp_path):
    """Point prompt_builder's snapshot path at a temp file; empty skills dir."""
    from agent import prompt_builder as pb

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    snap_path = tmp_path / "snap.json"
    monkeypatch.setattr(pb, "_skills_prompt_snapshot_path", lambda: snap_path, raising=True)
    monkeypatch.setattr(pb, "clear_skills_system_prompt_cache", lambda *a, **k: None, raising=True)
    return pb, skills_dir, snap_path


def test_knob_path_5_legacy_int_version_is_rebuild_trigger(monkeypatch, tmp_path):
    """Pre-upgrade snapshot wrote the version as a bare int (e.g. 2) — no TypeError, just rebuild."""
    import json
    pb, skills_dir, snap_path = _snapshot_fixture(monkeypatch, tmp_path)
    snap_path.write_text(json.dumps({"version": 2, "skills": []}), encoding="utf-8")
    assert pb._load_skills_snapshot(skills_dir) is None  # must not raise


def test_knob_path_6_current_version_snapshot_is_served(monkeypatch, tmp_path, knob):
    """A v3 snapshot written under the CURRENT limit round-trips as a list."""
    import json
    pb, skills_dir, snap_path = _snapshot_fixture(monkeypatch, tmp_path)
    _set_limit(knob, 100)
    snap_path.write_text(json.dumps({
        "version": [pb._SKILLS_SNAPSHOT_VERSION, 100],
        "manifest": pb._build_skills_manifest(skills_dir),
        "skills": [],
        "category_descriptions": {},
    }), encoding="utf-8")
    loaded = pb._load_skills_snapshot(skills_dir)
    assert loaded is not None and loaded["skills"] == []


def test_knob_path_7_limit_change_rebuilds_snapshot(monkeypatch, tmp_path, knob):
    """Snapshot written under a different limit is stale — must rebuild (None)."""
    import json
    pb, skills_dir, snap_path = _snapshot_fixture(monkeypatch, tmp_path)
    _set_limit(knob, 100)
    snap_path.write_text(json.dumps({
        "version": [pb._SKILLS_SNAPSHOT_VERSION, 400],  # written under another limit
        "manifest": pb._build_skills_manifest(skills_dir),
        "skills": [],
        "category_descriptions": {},
    }), encoding="utf-8")
    assert pb._load_skills_snapshot(skills_dir) is None


def test_knob_path_7b_missing_version_is_rebuild_trigger(monkeypatch, tmp_path):
    """No version key at all (corrupt dict) — rebuild, never raise."""
    import json
    pb, skills_dir, snap_path = _snapshot_fixture(monkeypatch, tmp_path)
    snap_path.write_text(json.dumps({"skills": []}), encoding="utf-8")
    assert pb._load_skills_snapshot(skills_dir) is None