from pathlib import Path
import pytest

from hermes_state import SessionDB
from workstation.browser_readiness import BrowserReadinessContract


def test_semantic_readiness_differentiates_loaded_vs_ready():
    contract = BrowserReadinessContract("/card", ["description-editor"], require_hydration=True, entity_identity="card-1")
    observation = {"url": "https://example.com/card", "readyState": "complete", "buttons": 20}
    assert not contract.evaluate(observation)["ready"]
    observation.update(hydrated=True, entity_identity="card-1", semantic_targets=["description-editor"])
    assert contract.evaluate(observation)["ready"]
    observation["auth_required"] = True
    assert contract.evaluate(observation)["code"] == "auth_required"


def test_trusted_ingress_creates_and_reuses_work_but_internal_text_never_creates():
    from types import SimpleNamespace
    from workstation.contracts import MessageEnvelope, MessageOrigin, IntentAuthority
    from workstation.work_intent import prepare_turn_work
    from hermes_cli import kanban_db
    agent = SimpleNamespace(session_id="session")
    text = "First extract reports then verify each result"
    assert not prepare_turn_work(agent, text).requires_task
    assert not hasattr(agent, "_canonical_work_task_id")
    envelope = MessageEnvelope(MessageOrigin.HUMAN, IntentAuthority.CREATE_WORK, "session", text)
    prepare_turn_work(agent, text, envelope)
    task_id = agent._canonical_work_task_id
    assert task_id
    prepare_turn_work(agent, text, envelope)
    assert agent._canonical_work_task_id == task_id
    conn = kanban_db.connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
        task = kanban_db.get_task(conn, task_id)
        assert task.session_id == "session" and task.body == text
    finally:
        conn.close()
    with pytest.raises(ValueError, match="owning conversation"):
        prepare_turn_work(agent, text, MessageEnvelope(MessageOrigin.HUMAN, IntentAuthority.CREATE_WORK, "other", text))


@pytest.mark.parametrize("stopped,tampered", [(False, False), (True, False), (False, True)])
def test_turn_candidate_reverifies_actual_durable_outputs_before_acceptance(stopped, tampered):
    from types import SimpleNamespace
    from workstation.contracts import MessageEnvelope, MessageOrigin, IntentAuthority
    from workstation.work_intent import prepare_turn_work
    from workstation.task_compiler import TaskCompiler
    from workstation.kanban import WorkstationKanbanBridge
    from hermes_cli import kanban_db
    agent = SimpleNamespace(session_id="session")
    text = "First extract reports then verify each result"
    prepare_turn_work(agent, text, MessageEnvelope(MessageOrigin.HUMAN, IntentAuthority.CREATE_WORK, "session", text))
    compiler = TaskCompiler()
    try:
        run = compiler.execute({"operation_key": "verify-records", "items": [{"id": 1}],
            "steps": [{"tool": "read_file", "args": {"path": "record"}, "expect": {"ok": True}}]},
            task_id="technical-browser-id", session_id="session", canonical_task_id=agent._canonical_work_task_id,
            dispatch=lambda *a: {"ok": True})
        if tampered:
            item = compiler.store.get_work_items(run["plan_id"])[0]
            compiler.artifacts.resolve_ref(item.normalized_output_ref).write_text("corrupted", encoding="utf-8")
        candidate = WorkstationKanbanBridge().finalize_turn_candidate(agent._canonical_work_task_id, "session",
            {"completed": not stopped, "interrupted": stopped, "final_response": "Record verified"})
        conn = kanban_db.connect()
        try:
            task = kanban_db.get_task(conn, agent._canonical_work_task_id)
            assert (task.status == "done") is (not stopped and not tampered)
            assert candidate["acceptance_approved"] is (not stopped and not tampered)
        finally:
            conn.close()
    finally:
        compiler.store.close()


def test_repeated_compacted_snapshots_do_not_bias_search_projection(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("s", "cli")
        for _ in range(12):
            db.append_message("s", "assistant", "unique snapshot marker")
        with db._lock:
            db._conn.execute("UPDATE messages SET active=0, compacted=1 WHERE session_id='s'")
            db._conn.commit()
        assert len(db.search_messages("snapshot", limit=100)) == 1
        assert db.compacted_duplicate_metrics()["duplicate_compacted_rows"] == 11
        assert db.get_messages("s") == []
        assert len(db.get_messages("s", include_inactive=True)) == 12
    finally:
        db.close()


def test_cron_drift_transitions_to_needs_migration(tmp_path):
    from cron import jobs
    with jobs.use_cron_store(tmp_path):
        jobs.save_jobs([{"id": "job", "name": "job", "prompt": "work", "enabled": True,
                         "state": "scheduled", "schedule": {"kind": "interval", "minutes": 5},
                         "provider_snapshot": "old", "model_snapshot": "old-model"}])
        assert jobs.mark_needs_migration("job", current_provider="new", current_model="new-model", reason="provider drift")
        job = jobs.load_jobs()[0]
        assert job["migration_state"] == "NEEDS_MIGRATION"
        assert job["enabled"] is False
        resolved = jobs.resolve_model_migration("job", "adopt_global")
        assert resolved["model_policy"] == "follow_global_model"
        assert resolved["enabled"] is True


def test_cron_creation_policy_pins_or_explicitly_follows_global(tmp_path, monkeypatch):
    from cron import jobs
    monkeypatch.setattr(jobs, "_compute_provider_model_snapshots", lambda **kw: (None if kw.get("provider") else "provider-a",
        None if kw.get("model") else "model-a"))
    with jobs.use_cron_store(tmp_path):
        pinned = jobs.create_job("Read reports", "every 5m", model_policy="pin_current_model")
        followed = jobs.create_job("Read reports", "every 5m", model_policy="follow_global_model")
        legacy = jobs.create_job("Read reports", "every 5m")
        assert pinned["provider"] == "provider-a" and pinned["model"] == "model-a"
        assert followed["provider"] is None and followed["model"] is None
        assert "model_policy" not in legacy
        assert jobs.get_job(pinned["id"])["model"] == "model-a"
        with pytest.raises(ValueError, match="inference overrides"):
            jobs.create_job("Read reports", "every 5m", model="other", model_policy="follow_global_model")
        with pytest.raises(ValueError, match="model_policy"):
            jobs.update_job(followed["id"], {"model_policy": "typo"})
        assert jobs.get_job(followed["id"])["model_policy"] == "follow_global_model"


def test_canonical_lineage_resolves_workplan(tmp_path):
    from hermes_cli import kanban_db
    from workstation.cockpit import task_cockpit
    from workstation.durable_tasks import DurableTaskStore
    conn = kanban_db.connect(db_path=tmp_path / "kanban.db")
    try:
        task_id = kanban_db.create_task(conn, title="Real work", session_id="session", initial_status="running")
        store = DurableTaskStore(conn=conn)
        plan = store.create_plan("technical-work", "Batch", [{"id": 1}], session_id="session",
                                 metadata={"canonical_task_id": task_id, "browser_task_id": "browser-page"})
        cockpit = task_cockpit(conn, task_id)
        assert cockpit["workplans"][0]["id"] == plan.id
        assert cockpit["lineage"]["agent_task_id"] == task_id
        assert cockpit["browser_tasks"] == ["browser-page"]
    finally:
        conn.close()


def test_trello_legacy_migration_preview_is_idempotent(tmp_path):
    from hermes_cli import kanban_db
    from workstation.trello_migration import reconcile_trello_legacy
    conn = kanban_db.connect(db_path=tmp_path / "kanban.db")
    try:
        task_id = kanban_db.create_task(conn, title="Legacy human card", body="Description", created_by="trello-sync", initial_status="running")
        original_status = kanban_db.get_task(conn, task_id).status
        records = [{"agent_task_id": task_id, "board_id": "remote-board", "list_id": "remote-list", "card_id": "remote-card",
                    "board_name": "Board", "list_name": "A fazer", "url": "https://trello.com/c/example"}]
        assert reconcile_trello_legacy(conn, records)[0]["action"] == "project"
        assert conn.execute("SELECT COUNT(*) FROM hybrid_cards").fetchone()[0] == 0
        first = reconcile_trello_legacy(conn, records, dry_run=False)[0]
        second = reconcile_trello_legacy(conn, records, dry_run=False)[0]
        assert first["human_card_id"] == second["human_card_id"]
        assert conn.execute("SELECT COUNT(*) FROM hybrid_card_delegations").fetchone()[0] == 0
        assert kanban_db.get_task(conn, task_id).status == original_status
    finally:
        conn.close()


def test_precommit_outcome_candidate_is_not_presented_as_accepted():
    from hermes_cli import kanban_db
    from workstation.cockpit import task_cockpit
    from workstation.contracts import ExecutionEventKind
    from workstation.journal import ExecutionJournal
    conn = kanban_db.connect()
    try:
        task_id = kanban_db.create_task(conn, title="Work", session_id="session", created_by="workstation")
        ExecutionJournal(task_id, "session").record(ExecutionEventKind.PROGRESS, "candidate persisted before commit",
            metadata={"boundary": "acceptance", "outcome": {"status": "verified_completed"}})
        cockpit = task_cockpit(conn, task_id)
        assert cockpit["outcome_status"] == "uncertain"
        assert cockpit["acceptance_approved"] is False
        assert cockpit["next_action"] == "reconcile_or_verify"
    finally:
        conn.close()


def test_trello_migration_same_name_lists_keep_distinct_remote_identity_and_rollback(tmp_path):
    from hermes_cli import kanban_db
    from workstation.trello_migration import reconcile_trello_legacy
    conn = kanban_db.connect(db_path=tmp_path / "kanban.db")
    try:
        tasks = [kanban_db.create_task(conn, title="Card", created_by="trello-sync") for _ in range(2)]
        records = [{"agent_task_id": task_id, "board_id": "board", "list_id": f"list-{i}", "card_id": f"card-{i}",
                    "board_name": "Board", "list_name": "Same name"} for i, task_id in enumerate(tasks)]
        with pytest.raises(ValueError, match="provenance"):
            reconcile_trello_legacy(conn, records + [{"agent_task_id": "missing"}], dry_run=False)
        assert conn.execute("SELECT COUNT(*) FROM hybrid_boards").fetchone()[0] == 0
        reconcile_trello_legacy(conn, records, dry_run=False)
        assert conn.execute("SELECT COUNT(*) FROM hybrid_columns").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(DISTINCT column_id) FROM hybrid_cards").fetchone()[0] == 2
    finally:
        conn.close()


def test_two_processes_migrate_same_trello_manifest_once(tmp_path):
    import json
    import subprocess
    import sys
    from hermes_cli import kanban_db
    db_path = tmp_path / "kanban.db"
    conn = kanban_db.connect(db_path=db_path)
    try:
        task_id = kanban_db.create_task(conn, title="Card", created_by="trello-sync")
        records = [{"agent_task_id": task_id, "board_id": "board", "list_id": "list", "card_id": "card",
                    "board_name": "Board", "list_name": "List"}]
        source = "from hermes_cli import kanban_db; from workstation.trello_migration import reconcile_trello_legacy; import sys,json; " \
                 "conn=kanban_db.connect(db_path=sys.argv[1]); reconcile_trello_legacy(conn,json.loads(sys.argv[2]),dry_run=False); conn.close()"
        processes = [subprocess.Popen([sys.executable, "-c", source, str(db_path), json.dumps(records)],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
        for process in processes:
            stdout, stderr = process.communicate(timeout=30)
            assert process.returncode == 0, stdout + stderr
        assert conn.execute("SELECT COUNT(*) FROM hybrid_boards").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM hybrid_columns").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM hybrid_cards").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM hybrid_card_delegations").fetchone()[0] == 0
    finally:
        conn.close()


def test_kanban_db_connect_handles_str_and_path(tmp_path):
    from hermes_cli import kanban_db
    p_path = tmp_path / "as_path" / "kanban.db"
    conn1 = kanban_db.connect(db_path=p_path)
    try:
        assert p_path.is_file()
    finally:
        conn1.close()

    p_str = str(tmp_path / "as_str" / "kanban.db")
    conn2 = kanban_db.connect(db_path=p_str)
    try:
        assert Path(p_str).is_file()
    finally:
        conn2.close()
