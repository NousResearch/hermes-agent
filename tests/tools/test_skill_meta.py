"""Tests for tools/skill_meta.py — local skill confidence metadata management."""

import json
import os
import tempfile
import shutil
from datetime import datetime, timezone

import pytest

# We test the module in isolation without needing the full hermes environment.


@pytest.fixture()
def meta_db():
    """Create a SkillMetaDB backed by a temporary directory."""
    from tools.skill_meta import SkillMetaDB

    tmpdir = tempfile.mkdtemp()
    try:
        # Point to a fake HERMES_HOME so the meta file lands inside tmpdir
        hermes_home = os.path.dirname(tmpdir)
        os.environ["HERMES_HOME"] = hermes_home
        os.makedirs(os.path.join(hermes_home, "skills"), exist_ok=True)
        db = SkillMetaDB()
        yield db
    finally:
        shutil.rmtree(hermes_home, ignore_errors=True)


# ── Registration ──────────────────────────────────────────────────


class TestRegistration:
    def test_empty_db(self, meta_db):
        assert meta_db.list_all() == []

    def test_register_creates_untested(self, meta_db):
        meta = meta_db.register("test-skill", reason="new")
        assert meta.confidence == "untested"
        assert len(meta.grade_history) == 1
        assert meta.grade_history[0]["grader"] == "system"
        assert meta.grade_history[0]["to"] == "untested"

    def test_get_nonexistent_returns_none(self, meta_db):
        assert meta_db.get("does-not-exist") is None

    def test_register_twice_is_idempotent(self, meta_db):
        meta_db.register("dup-test", reason="first")
        meta_db.register("dup-test", reason="second")  # should not add history
        meta = meta_db.get("dup-test")
        # Only the first registration added "initial → untested"
        assert len(meta.grade_history) == 1

    def test_name_normalization(self, meta_db):
        meta_db.register("TEST-SKILL")
        assert meta_db.get("test-skill") is not None
        assert meta_db.get("Test_Skill") is not None


# ── Confidence Grading ────────────────────────────────────────────


class TestGrading:
    def test_upgrades_in_order(self, meta_db):
        meta = meta_db.grade("test-skill", "trial", reason="t1")
        assert meta.confidence == "trial"

        meta = meta_db.grade("test-skill", "verified", reason="t2")
        assert meta.confidence == "verified"

        meta = meta_db.grade("test-skill", "default", reason="t3")
        assert meta.confidence == "default"
        assert meta.default_skill_pool is True  # auto-add to pool

    def test_same_level_blocked(self, meta_db):
        meta_db.register("s")
        meta_db.grade("s", "trial")
        with pytest.raises(ValueError, match="Cannot downgrade"):
            meta_db.grade("s", "trial")

    def test_downgrade_blocked(self, meta_db):
        meta_db.register("d")
        meta_db.grade("d", "verified")
        with pytest.raises(ValueError, match="Cannot downgrade"):
            meta_db.grade("d", "trial")

    def test_invalid_level_raises(self, meta_db):
        meta_db.register("i")
        with pytest.raises(ValueError, match="Invalid confidence level"):
            meta_db.grade("i", "superior")

    def test_grade_history_tracked(self, meta_db):
        meta_db.register("h")
        meta_db.grade("h", "trial", reason="t1", grader="user")
        meta_db.grade("h", "verified", reason="t2", grader="auto")
        meta = meta_db.get("h")
        assert len(meta.grade_history) == 3  # initial + 2 grades
        assert meta.grade_history[1]["reason"] == "t1"
        assert meta.grade_history[2]["grader"] == "auto"


# ── Usage Tracking ────────────────────────────────────────────────


class TestUsageTracking:
    def test_usage_count_increments(self, meta_db):
        meta_db.register("u")
        meta_db.record_usage("u")
        meta_db.record_usage("u")
        meta = meta_db.get("u")
        assert meta.usage_count == 2
        assert meta.last_used is not None

    def test_auto_promote_after_3_uses(self, meta_db):
        meta_db.register("a")
        meta_db.record_usage("a")
        meta_db.record_usage("a")
        meta_db.record_usage("a")
        meta = meta_db.get("a")
        assert meta.confidence == "trial"
        # Should have auto-promotion entry in history
        auto_promotions = [
            e for e in meta.grade_history if e["grader"] == "auto"
        ]
        assert len(auto_promotions) == 1

    def test_nonexistent_skill_automatically_registered(self, meta_db):
        meta_db.record_usage("ghost")
        meta = meta_db.get("ghost")
        assert meta is not None
        assert meta.usage_count == 1


# ── Default Pool Management ───────────────────────────────────────


class TestDefaultPool:
    def test_add_to_default_pool(self, meta_db):
        meta_db.register("dp")
        meta_db.add_to_default_pool("dp")
        assert "dp" in meta_db.list_default_pool()
        meta = meta_db.get("dp")
        assert meta.default_skill_pool is True

    def test_remove_from_default_pool(self, meta_db):
        meta_db.register("dr")
        meta_db.add_to_default_pool("dr")
        meta_db.remove_from_default_pool("dr")
        assert "dr" not in meta_db.list_default_pool()
        meta = meta_db.get("dr")
        assert meta.default_skill_pool is False

    def test_add_to_pool_promotes_untested(self, meta_db):
        meta_db.register("pu")
        meta_db.add_to_default_pool("pu")
        meta = meta_db.get("pu")
        # untested → trial when added to pool
        assert meta.confidence == "trial"

    def test_remove_nonexistent_raises(self, meta_db):
        with pytest.raises(KeyError):
            meta_db.remove_from_default_pool("nonexistent")

    def test_add_nonexistent_raises(self, meta_db):
        with pytest.raises(KeyError):
            meta_db.add_to_default_pool("nonexistent")

    def test_default_pool_in_report(self, meta_db):
        meta_db.register("rpt")
        meta_db.grade("rpt", "verified")
        meta_db.add_to_default_pool("rpt")
        report = meta_db.report()
        assert "rpt" in report
        assert "[DEFAULT]" in report
        assert "auto-loaded" in report


# ── Filtering & Listing ──────────────────────────────────────────


class TestFiltering:
    def test_list_by_confidence(self, meta_db):
        meta_db.register("f-untested")
        meta_db.register("f-verified")
        meta_db.grade("f-verified", "verified")
        untested = meta_db.list_by_confidence("untested")
        verified = meta_db.list_by_confidence("verified")
        assert len(untested) == 1
        assert len(verified) == 1
        assert untested[0][0] == "f-untested"
        assert verified[0][0] == "f-verified"

    def test_list_empty_filter(self, meta_db):
        assert meta_db.list_by_confidence("default") == []

    def test_list_all_sorted(self, meta_db):
        meta_db.register("z-skill")
        meta_db.register("a-skill")
        names = [name for name, _ in meta_db.list_all()]
        assert names == ["a-skill", "z-skill"]


# ── Removal ──────────────────────────────────────────────────────


class TestRemoval:
    def test_remove_existing(self, meta_db):
        meta_db.register("rm")
        assert meta_db.remove("rm") is True
        assert meta_db.get("rm") is None

    def test_remove_nonexistent(self, meta_db):
        assert meta_db.remove("nonexistent") is False


# ── Report ───────────────────────────────────────────────────────


class TestReport:
    def test_report_contains_levels(self, meta_db):
        meta_db.register("rep")
        meta_db.grade("rep", "verified")
        report = meta_db.report()
        assert "rep" in report
        assert "VERIFIED" in report

    def test_report_empty(self, meta_db):
        report = meta_db.report()
        assert "Local Skill Confidence Database (0 skills)" in report


# ── Persistence ──────────────────────────────────────────────────


class TestPersistence:
    def test_atomic_write_valid_json(self, meta_db):
        meta_db.register("p")
        meta_path = meta_db.db_path
        assert meta_path.exists()
        with open(meta_path, "r") as f:
            data = json.load(f)
        assert "skills" in data
        assert "p" in data["skills"]

    def test_reopen_loads_data(self, meta_db):
        from tools.skill_meta import SkillMetaDB

        meta_db.register("rl")
        meta_db.grade("rl", "verified")

        # Re-open a new instance
        db2 = SkillMetaDB()
        meta = db2.get("rl")
        assert meta is not None
        assert meta.confidence == "verified"

    def test_grade_history_survives_reopen(self, meta_db):
        meta_db.register("rh")
        meta_db.grade("rh", "trial", reason="t1")
        meta_db.grade("rh", "verified", reason="t2")

        db2 = SkillMetaDB()
        meta = db2.get("rh")
        assert len(meta.grade_history) == 3
