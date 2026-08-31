"""Transaction root identity, reentrancy and profile isolation (PR #99780)."""

import multiprocessing
from pathlib import Path

import pytest

from tools import checkpoint_manager as cpm
from tools import checkpoint_manager_lock as leases


def _probe_lease(base, results):
    try:
        with leases._CheckpointTransaction(Path(base), 0.15, "child probe"):
            results.put("acquired")
    except leases.CheckpointStoreBusy:
        results.put("busy")


def _hold_lease(base, entered, release):
    with leases._CheckpointTransaction(Path(base), 5, "test holder"):
        entered.set()
        if not release.wait(30):
            raise RuntimeError("lease holder was not released")


def _probe(context, base):
    results = context.Queue()
    child = context.Process(target=_probe_lease, args=(str(base), results))
    child.start()
    try:
        result = results.get(timeout=15)
        child.join(5)
        assert child.exitcode == 0
        return result
    finally:
        child.join(5)
        if child.is_alive():
            child.terminate()
            child.join(5)
        if child.is_alive():
            child.kill()
            child.join(5)
        results.close()
        results.join_thread()


@pytest.mark.parametrize("child_method", [
    pytest.param("fork", marks=pytest.mark.macos_only, id="macos-fork"),
    pytest.param("fork", marks=pytest.mark.linux_only, id="linux-fork"),
    pytest.param("spawn", marks=pytest.mark.windows_only, id="windows-spawn"),
])
def test_external_lease_survives_nested_clear_and_child_reset(tmp_path, child_method):
    """Clearing cannot replace the lock inode; a fork child must not unlock its parent."""
    base = tmp_path / "checkpoints"
    base.mkdir()
    (base / "history").write_text("keep until clear", encoding="utf-8")
    spawn = multiprocessing.get_context("spawn")
    with leases._CheckpointTransaction(base, 1, "outer"):
        with leases._CheckpointTransaction(base / ".." / base.name, 1, "nested alias"):
            assert cpm.clear_all(base)["deleted"] is True
            assert leases._transaction_lock_path(base).exists()
        assert _probe(multiprocessing.get_context(child_method), base) == "busy"
        # An independent open proves fork reset did not LOCK_UN the parent's descriptor.
        assert _probe(spawn, base) == "busy"
    assert _probe(spawn, base) == "acquired"


@pytest.mark.parametrize("operation", ["snapshot", "ledger", "plan", "diff", "restore", "auto-prune"])
def test_transaction_captures_one_live_profile_root(tmp_path, monkeypatch, operation):
    """A→B→A calls stay isolated; nested operations never re-resolve the leased root."""
    homes = [tmp_path / "profile-a", tmp_path / "profile-b"]
    for home in homes:
        home.mkdir()
        (home / "config.yaml").write_text("plugins:\n  enabled: []\n", encoding="utf-8")
    monkeypatch.setattr(cpm, "_INTERACTIVE_LOCK_TIMEOUT", 0.15)
    monkeypatch.setattr(cpm, "_MAINTENANCE_LOCK_TIMEOUT", 0.15)
    work = tmp_path / "project"
    work.mkdir()
    (work / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    content = work / "main.py"
    manager = cpm.CheckpointManager(enabled=True, max_snapshots=20)
    tips = {}
    for home in [*homes, homes[0]]:
        monkeypatch.setenv("HERMES_HOME", str(home))
        content.write_text(str(home), encoding="utf-8")
        manager.new_turn()
        taken = manager.ensure_checkpoint(str(work), home.name)
        if home not in tips:
            assert taken
            tips[home] = manager.list_checkpoints(str(work))[0]["hash"]
        else:
            assert not taken  # A is unchanged; it must not use B's ref/index.
        manager.record_agent_write(str(content))
        assert cpm._ledger_path(home / "checkpoints" / "store", manager._ledger_key(str(content))).exists()
    assert tips[homes[0]] != tips[homes[1]]

    # A changing resolver catches locked A / mutated B without racing a global env change.
    calls = []

    def live_base():
        calls.append(True)
        return homes[0 if len(calls) == 1 else 1] / "checkpoints" / ".." / "checkpoints"

    monkeypatch.setattr(cpm, "_resolve_checkpoint_base", live_base)
    content.write_text("agent edit", encoding="utf-8")
    a_store, b_store = (home / "checkpoints" / "store" for home in homes)
    before_b = {p.relative_to(b_store): p.read_bytes() for p in b_store.rglob("*") if p.is_file()}
    actions = {
        "snapshot": lambda: manager._take(str(work), "captured root"),
        "ledger": lambda: manager.record_agent_write(str(content)),
        "plan": lambda: manager.safe_restore_plan(str(work), tips[homes[0]]),
        "diff": lambda: manager.diff(str(work), tips[homes[0]]),
        "restore": lambda: manager.restore(str(work), tips[homes[0]], safe=True),
        "auto-prune": lambda: cpm.maybe_auto_prune_checkpoints(retention_days=0),
    }
    # Auto-prune must claim its interval before doing work, even under a nested lease.
    prune_calls = []
    if operation == "auto-prune":
        real_prune = cpm.prune_checkpoints

        def observe_prune(**kwargs):
            assert kwargs["checkpoint_base"] == (homes[0] / "checkpoints").resolve()
            assert (kwargs["checkpoint_base"] / ".last_prune").exists()
            prune_calls.append(True)
            return real_prune(**kwargs)

        monkeypatch.setattr(cpm, "prune_checkpoints", observe_prune)
    if operation == "ledger":
        real_hash = cpm._hash_file

        def hash_outside_lease(path):
            state = leases._transaction_lock_state(leases._transaction_lock_path(homes[0] / "checkpoints"))
            assert state.depth == 0
            return real_hash(path)

        monkeypatch.setattr(cpm, "_hash_file", hash_outside_lease)
    # Exercise nested planning AND the pre-rollback snapshot, not the early no-op branch.
    if operation == "restore":
        key = manager._ledger_key(str(content))
        cpm._save_ledger(a_store, key, {str(content): {"sha256": cpm._hash_file(content), "ts": 1}})
    ctx = multiprocessing.get_context("spawn")
    entered, release = ctx.Event(), ctx.Event()
    holder = ctx.Process(target=_hold_lease, args=(str(homes[0] / "checkpoints"), entered, release))
    holder.start()
    before_a = {p.relative_to(a_store): p.read_bytes() for p in a_store.rglob("*") if p.is_file()}
    try:
        assert entered.wait(15)
        blocked = actions[operation]()
        if operation == "snapshot":
            assert blocked is False
        elif operation != "ledger":
            assert "checkpoint store busy" in blocked["error"]
        assert len(calls) == 1
        assert {p.relative_to(a_store): p.read_bytes() for p in a_store.rglob("*") if p.is_file()} == before_a
        assert content.read_text(encoding="utf-8") == "agent edit"
        assert not (homes[0] / "checkpoints" / ".last_prune").exists()
    finally:
        release.set()
        holder.join(5)
        if holder.is_alive():
            holder.terminate()
            holder.join(5)
        if holder.is_alive():
            holder.kill()
            holder.join(5)
    assert holder.exitcode == 0
    calls.clear()
    result = actions[operation]()
    assert len(calls) == 1
    if isinstance(result, dict):
        assert not result.get("error"), result
    assert {p.relative_to(b_store): p.read_bytes() for p in b_store.rglob("*") if p.is_file()} == before_b
    assert leases._transaction_lock_path(homes[0] / "checkpoints").exists()
    if operation == "restore":
        assert content.read_text(encoding="utf-8") == str(homes[0])
    if operation == "auto-prune":
        calls.clear()
        assert actions[operation]()["skipped"] is True
        assert len(prune_calls) == 1
