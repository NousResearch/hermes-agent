"""Importing the tool registry must not touch state.db; the durable completion replay runs when a
consumer asks for it (#123265)."""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_model_tools_import_creates_no_state_db(tmp_path):
    typo_home = tmp_path / "profiles" / "typo"  # a missing named profile: a typo'd HERMES_HOME
    env = {**os.environ, "HERMES_HOME": str(typo_home), "PYTHONPATH": str(REPO)}
    env.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)
    proc = subprocess.run([sys.executable, "-c", "import model_tools"], cwd=REPO, env=env,
                          text=True, capture_output=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert not (typo_home / "state.db").exists(), sorted(os.listdir(typo_home)) if typo_home.exists() else None


def test_restore_runs_once_on_first_drain(monkeypatch):
    from tools import async_delegation, process_registry as pr_mod
    calls = []
    monkeypatch.setattr(async_delegation, "restore_undelivered_completions", lambda q: calls.append(q) or 1)
    registry = pr_mod.ProcessRegistry()
    assert calls == []  # construction (module import) does not replay the ledger
    registry.drain_notifications("sess")
    registry.drain_notifications("sess")
    assert registry.restore_completions() == 0
    assert calls == [registry.completion_queue]  # first consumer restores, exactly once


def test_first_drain_under_secondary_scope_replays_the_launch_ledger(monkeypatch, tmp_path):
    """Under multi-profile ``hermes serve`` the first consumer is a session bound to a secondary
    profile (TUI poller / prompt_turn drain); the once-per-process replay must still read the LAUNCH
    ledger, or it is never replayed for the life of the process."""
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
    from tools import async_delegation, process_registry as pr_mod
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    ledgers = []
    monkeypatch.setattr(async_delegation, "restore_undelivered_completions",
                        lambda q: ledgers.append(async_delegation._db_path()) or 0)
    registry = pr_mod.ProcessRegistry()
    token = set_hermes_home_override(tmp_path / "launch" / "profiles" / "b")
    try:
        registry.drain_notifications("sess")
        assert get_hermes_home() == tmp_path / "launch" / "profiles" / "b"  # caller scope untouched
    finally:
        reset_hermes_home_override(token)
    assert ledgers == [tmp_path / "launch" / "state.db"]
