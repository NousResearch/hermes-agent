"""Tests for tools/skills_hub.py — source adapters, lock file, taps, dedup logic."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from tools.skills_hub import (
    GitHubAuth,
    GitHubSource,
    LobeHubSource,
    SkillMeta,
    SkillBundle,
    HubLockFile,
    TapsManager,
    unified_search,
    append_audit_log,
    _skill_meta_to_dict,
)


# ---------------------------------------------------------------------------
# GitHubSource._parse_frontmatter_quick
# ---------------------------------------------------------------------------


class TestParseFrontmatterQuick:
    def test_valid_frontmatter(self):
        content = "---\nname: test-skill\ndescription: A test.\n---\n\n# Body\n"
        fm = GitHubSource._parse_frontmatter_quick(content)
        assert fm["name"] == "test-skill"
        assert fm["description"] == "A test."

    def test_no_frontmatter(self):
        content = "# Just a heading\nSome body text.\n"
        fm = GitHubSource._parse_frontmatter_quick(content)
        assert fm == {}

    def test_no_closing_delimiter(self):
        content = "---\nname: test\ndescription: desc\nno closing here\n"
        fm = GitHubSource._parse_frontmatter_quick(content)
        assert fm == {}

    def test_empty_content(self):
        fm = GitHubSource._parse_frontmatter_quick("")
        assert fm == {}

    def test_nested_yaml(self):
        content = "---\nname: test\nmetadata:\n  hermes:\n    tags: [a, b]\n---\n\nBody.\n"
        fm = GitHubSource._parse_frontmatter_quick(content)
        assert fm["metadata"]["hermes"]["tags"] == ["a", "b"]

    def test_invalid_yaml_returns_empty(self):
        content = "---\n: : : invalid{{\n---\n\nBody.\n"
        fm = GitHubSource._parse_frontmatter_quick(content)
        assert fm == {}

    def test_non_dict_yaml_returns_empty(self):
        content = "---\n- just a list\n- of items\n---\n\nBody.\n"
        fm = GitHubSource._parse_frontmatter_quick(content)
        assert fm == {}


# ---------------------------------------------------------------------------
# GitHubSource.trust_level_for
# ---------------------------------------------------------------------------


class TestTrustLevelFor:
    def _source(self):
        auth = MagicMock(spec=GitHubAuth)
        return GitHubSource(auth=auth)

    def test_trusted_repo(self):
        src = self._source()
        # TRUSTED_REPOS is imported from skills_guard, test with known trusted repo
        from tools.skills_guard import TRUSTED_REPOS
        if TRUSTED_REPOS:
            repo = next(iter(TRUSTED_REPOS))
            assert src.trust_level_for(f"{repo}/some-skill") == "trusted"

    def test_community_repo(self):
        src = self._source()
        assert src.trust_level_for("random-user/random-repo/skill") == "community"

    def test_short_identifier(self):
        src = self._source()
        assert src.trust_level_for("no-slash") == "community"

    def test_two_part_identifier(self):
        src = self._source()
        result = src.trust_level_for("owner/repo")
        # No path part — still resolves repo correctly
        assert result in ("trusted", "community")


# ---------------------------------------------------------------------------
# HubLockFile
# ---------------------------------------------------------------------------


class TestHubLockFile:
    def test_load_missing_file(self, tmp_path):
        lock = HubLockFile(path=tmp_path / "lock.json")
        data = lock.load()
        assert data == {"version": 1, "installed": {}}

    def test_load_valid_file(self, tmp_path):
        lock_file = tmp_path / "lock.json"
        lock_file.write_text(json.dumps({
            "version": 1,
            "installed": {"my-skill": {"source": "github"}}
        }))
        lock = HubLockFile(path=lock_file)
        data = lock.load()
        assert "my-skill" in data["installed"]

    def test_load_corrupt_json(self, tmp_path):
        lock_file = tmp_path / "lock.json"
        lock_file.write_text("not json{{{")
        lock = HubLockFile(path=lock_file)
        data = lock.load()
        assert data == {"version": 1, "installed": {}}

    def test_save_creates_parent_dir(self, tmp_path):
        lock_file = tmp_path / "subdir" / "lock.json"
        lock = HubLockFile(path=lock_file)
        lock.save({"version": 1, "installed": {}})
        assert lock_file.exists()

    def test_record_install(self, tmp_path):
        lock = HubLockFile(path=tmp_path / "lock.json")
        lock.record_install(
            name="test-skill",
            source="github",
            identifier="owner/repo/test-skill",
            trust_level="trusted",
            scan_verdict="pass",
            skill_hash="abc123",
            install_path="test-skill",
            files=["SKILL.md", "references/api.md"],
        )
        data = lock.load()
        assert "test-skill" in data["installed"]
        entry = data["installed"]["test-skill"]
        assert entry["source"] == "github"
        assert entry["trust_level"] == "trusted"
        assert entry["content_hash"] == "abc123"
        assert "installed_at" in entry

    def test_record_uninstall(self, tmp_path):
        lock = HubLockFile(path=tmp_path / "lock.json")
        lock.record_install(
            name="test-skill", source="github", identifier="x",
            trust_level="community", scan_verdict="pass",
            skill_hash="h", install_path="test-skill", files=["SKILL.md"],
        )
        lock.record_uninstall("test-skill")
        data = lock.load()
        assert "test-skill" not in data["installed"]

    def test_record_uninstall_nonexistent(self, tmp_path):
        lock = HubLockFile(path=tmp_path / "lock.json")
        lock.save({"version": 1, "installed": {}})
        # Should not raise
        lock.record_uninstall("nonexistent")

    def test_get_installed(self, tmp_path):
        lock = HubLockFile(path=tmp_path / "lock.json")
        lock.record_install(
            name="skill-a", source="github", identifier="x",
            trust_level="trusted", scan_verdict="pass",
            skill_hash="h", install_path="skill-a", files=["SKILL.md"],
        )
        assert lock.get_installed("skill-a") is not None
        assert lock.get_installed("nonexistent") is None

    def test_list_installed(self, tmp_path):
        lock = HubLockFile(path=tmp_path / "lock.json")
        lock.record_install(
            name="s1", source="github", identifier="x",
            trust_level="trusted", scan_verdict="pass",
            skill_hash="h1", install_path="s1", files=["SKILL.md"],
        )
        lock.record_install(
            name="s2", source="clawhub", identifier="y",
            trust_level="community", scan_verdict="pass",
            skill_hash="h2", install_path="s2", files=["SKILL.md"],
        )
        installed = lock.list_installed()
        assert len(installed) == 2
        names = {e["name"] for e in installed}
        assert names == {"s1", "s2"}

    def test_is_hub_installed(self, tmp_path):
        lock = HubLockFile(path=tmp_path / "lock.json")
        lock.record_install(
            name="my-skill", source="github", identifier="x",
            trust_level="trusted", scan_verdict="pass",
            skill_hash="h", install_path="my-skill", files=["SKILL.md"],
        )
        assert lock.is_hub_installed("my-skill") is True
        assert lock.is_hub_installed("other") is False


# ---------------------------------------------------------------------------
# TapsManager
# ---------------------------------------------------------------------------


class TestTapsManager:
    def test_load_missing_file(self, tmp_path):
        mgr = TapsManager(path=tmp_path / "taps.json")
        assert mgr.load() == []

    def test_load_valid_file(self, tmp_path):
        taps_file = tmp_path / "taps.json"
        taps_file.write_text(json.dumps({"taps": [{"repo": "owner/repo", "path": "skills/"}]}))
        mgr = TapsManager(path=taps_file)
        taps = mgr.load()
        assert len(taps) == 1
        assert taps[0]["repo"] == "owner/repo"

    def test_load_corrupt_json(self, tmp_path):
        taps_file = tmp_path / "taps.json"
        taps_file.write_text("bad json")
        mgr = TapsManager(path=taps_file)
        assert mgr.load() == []

    def test_add_new_tap(self, tmp_path):
        mgr = TapsManager(path=tmp_path / "taps.json")
        assert mgr.add("owner/repo", "skills/") is True
        taps = mgr.load()
        assert len(taps) == 1
        assert taps[0]["repo"] == "owner/repo"

    def test_add_duplicate_tap(self, tmp_path):
        mgr = TapsManager(path=tmp_path / "taps.json")
        mgr.add("owner/repo")
        assert mgr.add("owner/repo") is False
        assert len(mgr.load()) == 1

    def test_remove_existing_tap(self, tmp_path):
        mgr = TapsManager(path=tmp_path / "taps.json")
        mgr.add("owner/repo")
        assert mgr.remove("owner/repo") is True
        assert mgr.load() == []

    def test_remove_nonexistent_tap(self, tmp_path):
        mgr = TapsManager(path=tmp_path / "taps.json")
        assert mgr.remove("nonexistent") is False

    def test_list_taps(self, tmp_path):
        mgr = TapsManager(path=tmp_path / "taps.json")
        mgr.add("repo-a/skills")
        mgr.add("repo-b/tools")
        taps = mgr.list_taps()
        assert len(taps) == 2


# ---------------------------------------------------------------------------
# LobeHubSource._convert_to_skill_md
# ---------------------------------------------------------------------------


class TestConvertToSkillMd:
    def test_basic_conversion(self):
        agent_data = {
            "identifier": "test-agent",
            "meta": {
                "title": "Test Agent",
                "description": "A test agent.",
                "tags": ["testing", "demo"],
            },
            "config": {
                "systemRole": "You are a helpful test agent.",
            },
        }
        result = LobeHubSource._convert_to_skill_md(agent_data)
        assert "---" in result
        assert "name: test-agent" in result
        assert "description: A test agent." in result
        assert "tags: [testing, demo]" in result
        assert "# Test Agent" in result
        assert "You are a helpful test agent." in result

    def test_missing_system_role(self):
        agent_data = {
            "identifier": "no-role",
            "meta": {"title": "No Role", "description": "Desc."},
        }
        result = LobeHubSource._convert_to_skill_md(agent_data)
        assert "(No system role defined)" in result

    def test_missing_meta(self):
        agent_data = {"identifier": "bare-agent"}
        result = LobeHubSource._convert_to_skill_md(agent_data)
        assert "name: bare-agent" in result


# ---------------------------------------------------------------------------
# unified_search — dedup logic
# ---------------------------------------------------------------------------


class TestUnifiedSearchDedup:
    def _make_source(self, source_id, results):
        """Create a mock SkillSource that returns fixed results."""
        src = MagicMock()
        src.source_id.return_value = source_id
        src.search.return_value = results
        return src

    def test_dedup_keeps_first_seen(self):
        s1 = SkillMeta(name="skill", description="from A", source="a",
                        identifier="a/skill", trust_level="community")
        s2 = SkillMeta(name="skill", description="from B", source="b",
                        identifier="b/skill", trust_level="community")
        src_a = self._make_source("a", [s1])
        src_b = self._make_source("b", [s2])
        results = unified_search("skill", [src_a, src_b])
        assert len(results) == 1
        assert results[0].description == "from A"

    def test_dedup_prefers_trusted_over_community(self):
        community = SkillMeta(name="skill", description="community", source="a",
                               identifier="a/skill", trust_level="community")
        trusted = SkillMeta(name="skill", description="trusted", source="b",
                             identifier="b/skill", trust_level="trusted")
        src_a = self._make_source("a", [community])
        src_b = self._make_source("b", [trusted])
        results = unified_search("skill", [src_a, src_b])
        assert len(results) == 1
        assert results[0].trust_level == "trusted"

    def test_dedup_prefers_builtin_over_trusted(self):
        """Regression: builtin must not be overwritten by trusted."""
        builtin = SkillMeta(name="skill", description="builtin", source="a",
                             identifier="a/skill", trust_level="builtin")
        trusted = SkillMeta(name="skill", description="trusted", source="b",
                             identifier="b/skill", trust_level="trusted")
        src_a = self._make_source("a", [builtin])
        src_b = self._make_source("b", [trusted])
        results = unified_search("skill", [src_a, src_b])
        assert len(results) == 1
        assert results[0].trust_level == "builtin"

    def test_dedup_trusted_not_overwritten_by_community(self):
        trusted = SkillMeta(name="skill", description="trusted", source="a",
                             identifier="a/skill", trust_level="trusted")
        community = SkillMeta(name="skill", description="community", source="b",
                               identifier="b/skill", trust_level="community")
        src_a = self._make_source("a", [trusted])
        src_b = self._make_source("b", [community])
        results = unified_search("skill", [src_a, src_b])
        assert results[0].trust_level == "trusted"

    def test_source_filter(self):
        s1 = SkillMeta(name="s1", description="d", source="a",
                        identifier="x", trust_level="community")
        s2 = SkillMeta(name="s2", description="d", source="b",
                        identifier="y", trust_level="community")
        src_a = self._make_source("a", [s1])
        src_b = self._make_source("b", [s2])
        results = unified_search("query", [src_a, src_b], source_filter="a")
        assert len(results) == 1
        assert results[0].name == "s1"

    def test_limit_respected(self):
        skills = [
            SkillMeta(name=f"s{i}", description="d", source="a",
                       identifier=f"a/s{i}", trust_level="community")
            for i in range(20)
        ]
        src = self._make_source("a", skills)
        results = unified_search("query", [src], limit=5)
        assert len(results) == 5

    def test_source_error_handled(self):
        failing = MagicMock()
        failing.source_id.return_value = "fail"
        failing.search.side_effect = RuntimeError("boom")
        ok = self._make_source("ok", [
            SkillMeta(name="s1", description="d", source="ok",
                       identifier="x", trust_level="community")
        ])
        results = unified_search("query", [failing, ok])
        assert len(results) == 1


# ---------------------------------------------------------------------------
# append_audit_log
# ---------------------------------------------------------------------------


class TestAppendAuditLog:
    def test_creates_log_entry(self, tmp_path):
        log_file = tmp_path / "audit.log"
        with patch("tools.skills_hub.AUDIT_LOG", log_file):
            append_audit_log("INSTALL", "test-skill", "github", "trusted", "pass")
        content = log_file.read_text()
        assert "INSTALL" in content
        assert "test-skill" in content
        assert "github:trusted" in content
        assert "pass" in content

    def test_appends_multiple_entries(self, tmp_path):
        log_file = tmp_path / "audit.log"
        with patch("tools.skills_hub.AUDIT_LOG", log_file):
            append_audit_log("INSTALL", "s1", "github", "trusted", "pass")
            append_audit_log("UNINSTALL", "s1", "github", "trusted", "n/a")
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_extra_field_included(self, tmp_path):
        log_file = tmp_path / "audit.log"
        with patch("tools.skills_hub.AUDIT_LOG", log_file):
            append_audit_log("INSTALL", "s1", "github", "trusted", "pass", extra="hash123")
        content = log_file.read_text()
        assert "hash123" in content


# ---------------------------------------------------------------------------
# _skill_meta_to_dict
# ---------------------------------------------------------------------------


class TestSkillMetaToDict:
    def test_roundtrip(self):
        meta = SkillMeta(
            name="test", description="desc", source="github",
            identifier="owner/repo/test", trust_level="trusted",
            repo="owner/repo", path="skills/test", tags=["a", "b"],
        )
        d = _skill_meta_to_dict(meta)
        assert d["name"] == "test"
        assert d["tags"] == ["a", "b"]
        # Can reconstruct from dict
        restored = SkillMeta(**d)
        assert restored.name == meta.name
        assert restored.trust_level == meta.trust_level
