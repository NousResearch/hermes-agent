"""Tests for the features.json data-dir activation ledger.

The ledger persists which lazy features the user has activated so that
feature state survives venv replacement (e.g. ``hermes update`` swapping
in a fresh slot).  Functions tested here live in ``tools.lazy_deps``:

- ``record_feature(name, via)`` — atomic write to ``$HERMES_HOME/state/features.json``
- ``ledger_features()`` — list feature names from the ledger
- ``remove_feature(name)`` — remove a feature from the ledger
- ``apply_ledger(venv_python)`` — re-run ``ensure()`` per ledger feature

These tests do NOT read source code (AGENTS.md rule).  They exercise the
public API with temp ``HERMES_HOME`` (isolated by the conftest autouse
fixture).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.lazy_deps as ld


# ---------------------------------------------------------------------------
# record_feature
# ---------------------------------------------------------------------------


class TestRecordFeature:
    def test_creates_state_file_with_correct_schema(self, monkeypatch):
        monkeypatch.setattr(ld, "active_features", lambda: [])
        ld.record_feature("memory.honcho", via="ensure")
        state_file = ld._ledger_path()
        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert data["schema"] == 1
        assert "memory.honcho" in data["features"]
        entry = data["features"]["memory.honcho"]
        assert entry["via"] == "ensure"
        assert "activated_at" in entry

    def test_updates_existing_entry(self, monkeypatch):
        monkeypatch.setattr(ld, "active_features", lambda: [])
        ld.record_feature("memory.honcho", via="ensure")
        first_data = json.loads(ld._ledger_path().read_text())
        first_ts = first_data["features"]["memory.honcho"]["activated_at"]

        ld.record_feature("memory.honcho", via="manual")
        second_data = json.loads(ld._ledger_path().read_text())
        assert second_data["features"]["memory.honcho"]["via"] == "manual"
        # activated_at should be updated
        assert second_data["features"]["memory.honcho"]["activated_at"] != first_ts

    def test_preserves_other_features(self, monkeypatch):
        monkeypatch.setattr(ld, "active_features", lambda: [])
        ld.record_feature("memory.honcho", via="ensure")
        ld.record_feature("tts.mistral", via="ensure")
        ld.record_feature("memory.honcho", via="manual")
        data = json.loads(ld._ledger_path().read_text())
        assert "tts.mistral" in data["features"]
        assert data["features"]["tts.mistral"]["via"] == "ensure"

    def test_atomic_write_no_partial_file(self, monkeypatch):
        monkeypatch.setattr(ld, "active_features", lambda: [])
        ld.record_feature("memory.honcho", via="ensure")
        state_file = ld._ledger_path()
        # File should be valid JSON (atomic replace means no partial writes)
        data = json.loads(state_file.read_text())
        assert data["schema"] == 1


# ---------------------------------------------------------------------------
# ledger_features
# ---------------------------------------------------------------------------


class TestLedgerFeatures:
    def test_empty_when_no_state_file(self, monkeypatch):
        monkeypatch.setattr(ld, "active_features", lambda: [])
        assert ld.ledger_features() == []

    def test_returns_recorded_features(self, monkeypatch):
        monkeypatch.setattr(ld, "active_features", lambda: [])
        ld.record_feature("memory.honcho", via="ensure")
        ld.record_feature("tts.mistral", via="ensure")
        features = ld.ledger_features()
        assert "memory.honcho" in features
        assert "tts.mistral" in features

    def test_one_time_seed_from_venv_probe(self, monkeypatch):
        """When state file is absent and venv probe finds features, first
        ledger_features() call writes them with via='venv-probe-migration'."""
        monkeypatch.setattr(
            ld, "active_features", lambda: ["memory.honcho", "tts.mistral"]
        )
        features = ld.ledger_features()
        assert "memory.honcho" in features
        assert "tts.mistral" in features
        # The seed should have written the state file
        data = json.loads(ld._ledger_path().read_text())
        for feat in ("memory.honcho", "tts.mistral"):
            assert data["features"][feat]["via"] == "venv-probe-migration"

    def test_seed_merges_pending_file(self, monkeypatch, tmp_path):
        """features.pending.json (written by phase-2 adopt) is consumed on
        first seed."""
        # Ensure state/features.json is absent
        assert not ld._ledger_path().exists()

        # Write a pending file with one feature
        pending = ld._ledger_path().parent / "features.pending.json"
        pending.parent.mkdir(parents=True, exist_ok=True)
        pending.write_text(json.dumps({
            "schema": 1,
            "features": {"platform.slack": {"activated_at": "2025-01-01T00:00:00", "via": "adopt"}},
        }))

        # Venv probe finds a different feature
        monkeypatch.setattr(ld, "active_features", lambda: ["memory.honcho"])
        features = ld.ledger_features()
        # Both the probe result and the pending file should be seeded
        assert "memory.honcho" in features
        assert "platform.slack" in features

        # pending file consumed (removed)
        assert not pending.exists()

    def test_no_reseed_after_state_exists(self, monkeypatch):
        """Once the state file exists, venv probe is not run again."""
        monkeypatch.setattr(ld, "active_features", lambda: [])
        ld.record_feature("memory.honcho", via="manual")
        # Now if active_features would return something, it shouldn't be seeded
        monkeypatch.setattr(ld, "active_features", lambda: ["tts.mistral"])
        features = ld.ledger_features()
        assert "memory.honcho" in features
        assert "tts.mistral" not in features


# ---------------------------------------------------------------------------
# remove_feature
# ---------------------------------------------------------------------------


class TestRemoveFeature:
    def test_removes_existing_feature(self, monkeypatch):
        monkeypatch.setattr(ld, "active_features", lambda: [])
        ld.record_feature("memory.honcho", via="ensure")
        ld.record_feature("tts.mistral", via="ensure")
        ld.remove_feature("memory.honcho")
        features = ld.ledger_features()
        assert "memory.honcho" not in features
        assert "tts.mistral" in features

    def test_remove_nonexistent_is_noop(self, monkeypatch):
        monkeypatch.setattr(ld, "active_features", lambda: [])
        # No state file at all — should not raise
        ld.remove_feature("nonexistent.feature")
        # Still no state file (or if created, empty features)
        if ld._ledger_path().exists():
            data = json.loads(ld._ledger_path().read_text())
            assert "nonexistent.feature" not in data["features"]

    def test_remove_preserves_schema(self, monkeypatch):
        monkeypatch.setattr(ld, "active_features", lambda: [])
        ld.record_feature("memory.honcho", via="ensure")
        ld.remove_feature("memory.honcho")
        data = json.loads(ld._ledger_path().read_text())
        assert data["schema"] == 1


# ---------------------------------------------------------------------------
# apply_ledger
# ---------------------------------------------------------------------------


class TestApplyLedger:
    def test_current_feature_returns_current(self, monkeypatch):
        monkeypatch.setattr(ld, "active_features", lambda: [])
        ld.record_feature("test.feat", via="ensure")
        monkeypatch.setitem(ld.LAZY_DEPS, "test.feat", ("zzzfake==1.0.0",))
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: True)
        result = ld.apply_ledger("/fake/venv/bin/python")
        assert result["test.feat"] == "current"

    def test_stale_pin_triggers_refresh(self, monkeypatch):
        monkeypatch.setattr(ld, "active_features", lambda: [])
        ld.record_feature("test.feat", via="ensure")
        monkeypatch.setitem(ld.LAZY_DEPS, "test.feat", ("zzzfake==2.0.0",))
        states = iter([False, True])
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: next(states))
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: True)
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda specs, **kw: ld._InstallResult(True, "ok", ""),
        )
        result = ld.apply_ledger("/fake/venv/bin/python")
        assert result["test.feat"] == "refreshed"

    def test_install_failure_returns_failed(self, monkeypatch):
        monkeypatch.setattr(ld, "active_features", lambda: [])
        ld.record_feature("test.feat", via="ensure")
        monkeypatch.setitem(ld.LAZY_DEPS, "test.feat", ("zzzfake==2.0.0",))
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: False)
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: True)
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda specs, **kw: ld._InstallResult(False, "", "PyPI 404"),
        )
        result = ld.apply_ledger("/fake/venv/bin/python")
        assert result["test.feat"].startswith("failed:")

    def test_lazy_installs_disabled_returns_skipped(self, monkeypatch):
        monkeypatch.setattr(ld, "active_features", lambda: [])
        ld.record_feature("test.feat", via="ensure")
        monkeypatch.setitem(ld.LAZY_DEPS, "test.feat", ("zzzfake==2.0.0",))
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: False)
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: False)
        result = ld.apply_ledger("/fake/venv/bin/python")
        assert result["test.feat"].startswith("skipped:")

    def test_mixed_results_returns_per_feature_status(self, monkeypatch):
        monkeypatch.setattr(ld, "active_features", lambda: [])
        ld.record_feature("a.ok", via="ensure")
        ld.record_feature("b.fail", via="ensure")
        monkeypatch.setitem(ld.LAZY_DEPS, "a.ok", ("pkga==1.0",))
        monkeypatch.setitem(ld.LAZY_DEPS, "b.fail", ("pkgb==1.0",))

        def fake_satisfied(spec):
            return ld._pkg_name_from_spec(spec) == "pkga"

        monkeypatch.setattr(ld, "_is_satisfied", fake_satisfied)
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: True)
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda specs, **kw: ld._InstallResult(False, "", "nope"),
        )
        result = ld.apply_ledger("/fake/venv/bin/python")
        assert result["a.ok"] == "current"
        assert result["b.fail"].startswith("failed:")

    def test_empty_ledger_returns_empty(self, monkeypatch):
        monkeypatch.setattr(ld, "active_features", lambda: [])
        result = ld.apply_ledger("/fake/venv/bin/python")
        assert result == {}

    def test_unsupported_feature_returns_skipped(self, monkeypatch):
        monkeypatch.setattr(ld, "active_features", lambda: [])
        ld.record_feature("platform.matrix", via="ensure")
        monkeypatch.setattr(ld.sys, "platform", "win32")
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: False)
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: True)
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda *a, **kw: pytest.fail("pip should not be called"),
        )
        result = ld.apply_ledger("/fake/venv/bin/python")
        assert result["platform.matrix"].startswith("skipped:")


# ---------------------------------------------------------------------------
# ensure() records feature on first install
# ---------------------------------------------------------------------------


class TestEnsureRecordsFeature:
    def test_ensure_records_on_successful_install(self, monkeypatch):
        monkeypatch.setattr(ld, "active_features", lambda: [])
        monkeypatch.setitem(ld.LAZY_DEPS, "test.install", ("zzzfake>=1",))
        call_count = {"n": 0}

        def fake_satisfied(spec):
            call_count["n"] += 1
            return call_count["n"] > 1  # missing first, installed after

        monkeypatch.setattr(ld, "_is_satisfied", fake_satisfied)
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: True)
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda specs, **kw: ld._InstallResult(True, "ok", ""),
        )
        ld.ensure("test.install", prompt=False)
        # Feature should now be in the ledger
        assert "test.install" in ld.ledger_features()
        data = json.loads(ld._ledger_path().read_text())
        assert data["features"]["test.install"]["via"] == "ensure"

    def test_ensure_does_not_record_when_already_satisfied(self, monkeypatch):
        monkeypatch.setattr(ld, "active_features", lambda: [])
        monkeypatch.setitem(ld.LAZY_DEPS, "test.satisfied", ("zzzfake>=1",))
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: True)
        ld.ensure("test.satisfied", prompt=False)
        # Should NOT be recorded (already satisfied = no install)
        assert "test.satisfied" not in ld.ledger_features()

    def test_ensure_does_not_record_on_failure(self, monkeypatch):
        monkeypatch.setattr(ld, "active_features", lambda: [])
        monkeypatch.setitem(ld.LAZY_DEPS, "test.fail", ("zzzfake>=1",))
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: False)
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: True)
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda specs, **kw: ld._InstallResult(False, "", "error"),
        )
        with pytest.raises(ld.FeatureUnavailable):
            ld.ensure("test.fail", prompt=False)
        assert "test.fail" not in ld.ledger_features()
