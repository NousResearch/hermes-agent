"""``session.undo``'s file half on the TUI sidecar reads the SESSION's profile checkpoint store.

One sidecar process serves several profiles: a session in a secondary profile wrote its checkpoints
under that profile's home, so the revert must bind it (the launch profile's store has none of them).
"""
from pathlib import Path


def test_sidecar_undo_reverts_files_from_the_sessions_own_profile_store(tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.checkpoint_manager import CheckpointManager
    from tui_gateway.checkpoints import revert_undone_files

    launch, other = tmp_path / "launch", tmp_path / "profiles" / "work"
    for home in (launch, other):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("{}\n")
    work = tmp_path / "proj"
    work.mkdir()
    target = work / "app.py"
    target.write_text("print('v1')\n")

    mgr = CheckpointManager(enabled=True)
    token = set_hermes_home_override(str(other))  # the turn ran in the session's profile
    try:
        mgr.new_turn()
        mgr.begin_turn(100.0, "s1")
        mgr.ensure_checkpoint(str(work), "before write_file")
        mgr.note_pending_write(str(target))
        target.write_text("print('v2')\n")
        mgr.record_agent_write(str(target))
    finally:
        reset_hermes_home_override(token)

    class Agent:
        _checkpoint_mgr = mgr
        session_id = "s1"

    monkeypatch.setenv("TERMINAL_CWD", str(work))
    token = set_hermes_home_override(str(launch))  # /undo arrives on the launch profile's thread
    try:
        lines = revert_undone_files({"agent": Agent(), "profile_home": str(other), "cwd": str(work)},
                                    {"timestamp": 100.0})
    finally:
        reset_hermes_home_override(token)

    assert target.read_text() == "print('v1')\n", lines
    assert any("app.py" in line for line in lines)
    assert not (Path(launch) / "checkpoints" / "store").exists()
