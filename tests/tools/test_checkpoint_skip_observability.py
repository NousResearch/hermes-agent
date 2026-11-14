"""A workdir refused a checkpoint has to say why (#154876).

``_take`` writes the project record *before* the file-count gate, so a refused
workdir shows up in the store with an advancing ``last_touch``, zero commits and
no ref — and the refusal itself was only ``logger.debug``. These tests pin the
four surfaces that now carry the reason: the project record, the (throttled)
log line, ``hermes checkpoints``, and ``hermes doctor``.

Symbols the fix adds are reached through the module (``getattr``) so reverting
the fix makes these tests fail on an assertion about behaviour instead of dying
at collection with an ImportError.
"""

import argparse
import json
import logging

import pytest

from tools import checkpoint_maintenance as maint
from tools import checkpoint_manager as cm

# Key the reason is persisted under on ``projects/<hash>.json``.
SKIP_REASON_KEY = getattr(maint, "SKIP_REASON_KEY", "skip_reason")


@pytest.fixture()
def work_dir(tmp_path):
    d = tmp_path / "project"
    d.mkdir()
    (d / "main.py").write_text("print('hello')\n")
    return d


@pytest.fixture()
def checkpoint_base(tmp_path):
    """Isolated checkpoint base — never writes to ~/.hermes/."""
    return tmp_path / "checkpoints"


@pytest.fixture()
def mgr(checkpoint_base, monkeypatch):
    monkeypatch.setattr(cm, "CHECKPOINT_BASE", checkpoint_base)
    # The warning throttle is per-process state; start each test from empty.
    monkeypatch.setattr(maint, "_skip_warned", {}, raising=False)
    return cm.CheckpointManager(enabled=True)


def _project_record(checkpoint_base, work_dir) -> dict:
    meta = checkpoint_base / "store" / "projects" / f"{cm._project_hash(str(work_dir))}.json"
    return json.loads(meta.read_text(encoding="utf-8-sig"))


def _skip_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("Checkpoint skipped for")]


def test_file_cap_records_the_reason_and_warns(mgr, work_dir, checkpoint_base, monkeypatch, caplog):
    monkeypatch.setattr(cm, "_MAX_FILES", 0)

    with caplog.at_level(logging.WARNING, logger="tools.checkpoint_maintenance"):
        assert mgr.ensure_checkpoint(str(work_dir), "auto") is False

    assert _project_record(checkpoint_base, work_dir)[SKIP_REASON_KEY] == "file_count>0"
    warnings = _skip_warnings(caplog)
    assert len(warnings) == 1
    assert "file_count>0" in warnings[0]
    assert str(work_dir) in warnings[0]


def test_skip_warning_is_throttled_per_workdir(mgr, work_dir, monkeypatch, caplog):
    monkeypatch.setattr(cm, "_MAX_FILES", 0)

    with caplog.at_level(logging.WARNING, logger="tools.checkpoint_maintenance"):
        for _ in range(3):
            mgr.new_turn()
            assert mgr.ensure_checkpoint(str(work_dir), "auto") is False
        assert len(_skip_warnings(caplog)) == 1

        # A throttle window, not a once-ever flag: a later window warns again.
        monkeypatch.setattr(maint, "_SKIP_WARN_INTERVAL", 0.0)
        mgr.new_turn()
        mgr.ensure_checkpoint(str(work_dir), "auto")
        assert len(_skip_warnings(caplog)) == 2


def test_a_successful_snapshot_clears_the_recorded_reason(mgr, work_dir, checkpoint_base, monkeypatch):
    monkeypatch.setattr(cm, "_MAX_FILES", 0)
    assert mgr.ensure_checkpoint(str(work_dir), "auto") is False
    assert _project_record(checkpoint_base, work_dir)[SKIP_REASON_KEY] == "file_count>0"

    monkeypatch.setattr(cm, "_MAX_FILES", 50_000)
    mgr.new_turn()
    assert mgr.ensure_checkpoint(str(work_dir), "auto") is True

    record = _project_record(checkpoint_base, work_dir)
    assert SKIP_REASON_KEY not in record
    assert "last_touch" in record  # the record itself survives


def test_store_status_carries_the_reason(mgr, work_dir, checkpoint_base, monkeypatch):
    monkeypatch.setattr(cm, "_MAX_FILES", 0)
    assert mgr.ensure_checkpoint(str(work_dir), "auto") is False

    rows = maint.store_status(checkpoint_base)["projects"]
    assert [row.get("skip_reason") for row in rows] == ["file_count>0"]
    assert rows[0]["commits"] == 0


def test_hermes_checkpoints_names_the_reason(mgr, work_dir, checkpoint_base, monkeypatch, capsys):
    monkeypatch.setattr(cm, "_MAX_FILES", 0)
    assert mgr.ensure_checkpoint(str(work_dir), "auto") is False

    from hermes_cli.checkpoints import cmd_status

    assert cmd_status(argparse.Namespace(limit=None)) == 0
    out = capsys.readouterr().out
    assert "no snapshots: file_count>0" in out


def test_doctor_warns_about_skipped_projects(mgr, work_dir, checkpoint_base, monkeypatch, capsys):
    monkeypatch.setattr(cm, "_MAX_FILES", 0)
    assert mgr.ensure_checkpoint(str(work_dir), "auto") is False

    monkeypatch.setattr("hermes_cli.config.load_config", dict)
    from hermes_cli.doctor_state import _check_checkpoint_store

    # ``doctor_check`` wraps it to ``(should_fix) -> Finding``.
    _check_checkpoint_store(False)  # pyright: ignore[reportCallIssue]
    out = capsys.readouterr().out
    assert "Checkpoints are disabled for 1 working directory" in out
    assert "file_count>0" in out


def test_notice_is_silent_without_skipped_projects(mgr, work_dir, checkpoint_base, monkeypatch):
    assert mgr.ensure_checkpoint(str(work_dir), "auto") is True
    monkeypatch.setattr("hermes_cli.config.load_config", dict)

    notice = getattr(maint, "skipped_projects_notice", None)
    assert notice is not None, "skipped_projects_notice is missing"
    assert notice() is None
