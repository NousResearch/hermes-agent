"""Private contract probe for real profile-resume retirement retries."""

import pytest

from hermes_state import SessionDB
from tests.tui_gateway import test_stranded_session_adoption as adoption_cases
from tests.hermes_state.test_adoption_transfer_records import _retire_fails_for, _seed_compression_lineage


gateway = adoption_cases.gateway
_seed_stranded = adoption_cases._seed_stranded


def _resume(mod, sid, request_id):
    return mod.handle_request({"id": request_id, "method": "session.resume",
                               "params": {"session_id": sid, "profile": "developer", "lazy": True}})


@pytest.mark.parametrize("fail_root", [True, False], ids=["first-retirement", "partial-retirement"])
def test_profile_resume_finishes_the_saved_transfer(gateway, monkeypatch, fail_root):
    mod, donor, profile_home = gateway
    root, tip = "resume-transfer-root", "resume-transfer-tip"
    _seed_compression_lineage(donor, root, tip)
    with monkeypatch.context() as patch:
        patch.setattr(SessionDB, "_retire_donor_segment", _retire_fails_for(root if fail_root else tip))
        _resume(mod, tip, "first")
    assert donor.get_session(tip)["end_reason"] is None
    response = _resume(mod, tip if fail_root else root, "retry")
    assert not response.get("error"), response.get("error")
    assert donor.get_session(root)["end_reason"] == "adopted_by_profile"
    assert donor.get_session(tip)["end_reason"] == "adopted_by_profile"
    target = SessionDB(profile_home / "state.db")
    try:
        target.append_message(tip, "user", "after profile resume")
        assert len(target.get_messages(tip)) == 3
    finally:
        target.close()


def test_profile_resume_does_not_retire_a_copy_only_donor(gateway):
    mod, donor, profile_home = gateway
    sid = "copy-only-resume"
    _seed_stranded(donor, sid)
    target = SessionDB(profile_home / "state.db")
    try:
        assert target.adopt_session_lineage_from(donor, sid, retire_donor=False)["adopted"]
    finally:
        target.close()
    _resume(mod, sid, "view-copy")
    assert donor.get_session(sid)["end_reason"] is None
    assert not donor.get_session(sid)["archived"]

"""Additional controls for a stale resume hint and completed transfers."""


def test_stale_resume_hint_cannot_override_copy_only_intent(gateway, monkeypatch):
    import hermes_state_adoption_records as records
    mod, donor, profile_home = gateway
    sid = "stale-copy-only-resume"
    _seed_stranded(donor, sid)
    target = SessionDB(profile_home / "state.db")
    try:
        assert target.adopt_session_lineage_from(donor, sid, retire_donor=False)["adopted"]
    finally:
        target.close()
    monkeypatch.setattr(records, "retirement_pending_source", lambda db, member: str(donor.db_path.resolve()))
    _resume(mod, sid, "stale-hint")
    assert donor.get_session(sid)["end_reason"] is None
    assert not donor.get_session(sid)["archived"]


def test_completed_resume_does_not_repeat_adoption(gateway, monkeypatch):
    mod, donor, profile_home = gateway
    sid = "completed-transfer-resume"
    _seed_stranded(donor, sid)
    target = SessionDB(profile_home / "state.db")
    try:
        assert target.adopt_session_lineage_from(donor, sid)["donor_retired"]
    finally:
        target.close()

    def no_adoption(*args, **kwargs):
        raise AssertionError("completed transfers must not revisit the donor")

    monkeypatch.setattr(SessionDB, "adopt_session_lineage_from", no_adoption)
    response = _resume(mod, sid, "completed")
    assert not response.get("error"), response.get("error")


def test_default_resume_does_not_take_over_an_explicit_named_donor(gateway, monkeypatch):
    from hermes_state_adoption_records import retirement_pending_source

    mod, default, profile_home = gateway
    other_home = profile_home.parent / "science"
    other_home.mkdir()
    sid = "explicit-named-donor-resume"
    other, target = SessionDB(other_home / "state.db"), SessionDB(profile_home / "state.db")
    try:
        _seed_stranded(other, sid)
        with monkeypatch.context() as patch:
            patch.setattr(SessionDB, "_retire_donor_segment", _retire_fails_for(sid))
            outcome = target.adopt_session_lineage_from(other, sid)
        assert outcome["adopted"] and not outcome["donor_retired"]
        source = str(other.db_path.resolve())
        assert retirement_pending_source(target, sid) == source
        before_other = [tuple(row) for row in other._conn.execute("SELECT * FROM sessions ORDER BY id")]
        before_default = [tuple(row) for row in default._conn.execute("SELECT * FROM sessions ORDER BY id")]
        response = _resume(mod, sid, "named-donor")
        assert response["error"]["code"] == 5000
        assert "multiple live owners" in response["error"]["message"]
        assert [tuple(row) for row in other._conn.execute("SELECT * FROM sessions ORDER BY id")] == before_other
        assert [tuple(row) for row in default._conn.execute("SELECT * FROM sessions ORDER BY id")] == before_default
        assert retirement_pending_source(target, sid) == source
        completed = target.adopt_session_lineage_from(other, sid, _pending_only=True)
        assert completed["adopted"] and completed["donor_retired"]
        response = _resume(mod, sid, "named-donor-completed")
        assert not response.get("error"), response.get("error")
        assert response["result"]["resumed"] == sid
        target.append_message(sid, "user", "Explicit transfer remains usable")
    finally:
        target.close()
        other.close()


def test_profile_resume_retries_partial_retirement_with_queued_ancestor(gateway, monkeypatch):
    mod, donor, profile_home = gateway
    root, tip = "queued-resume-root", "queued-resume-tip"
    _seed_compression_lineage(donor, root, tip)
    donor._conn.execute(
        "UPDATE messages SET display_metadata=? WHERE session_id=? AND role='user'",
        ('{"_queued_prompt":true}', root))
    donor._conn.commit()
    with monkeypatch.context() as patch:
        patch.setattr(SessionDB, "_retire_donor_segment", _retire_fails_for(tip))
        _resume(mod, tip, "queued-first")
    assert donor.get_session(root)["end_reason"] == "adopted_by_profile"
    assert donor.get_session(tip)["end_reason"] is None
    response = _resume(mod, root, "queued-retry")
    assert not response.get("error"), response.get("error")
    assert donor.get_session(tip)["end_reason"] == "adopted_by_profile"
    target = SessionDB(profile_home / "state.db")
    try:
        assert len(target.get_messages(root, include_inactive=True)) == 2
        assert len(target.get_messages(tip, include_inactive=True)) == 2
    finally:
        target.close()


def test_resume_reopens_valid_half_retired_queued_donor(gateway, monkeypatch):
    mod, donor, profile_home = gateway
    root, tip = "half-retired-root", "half-retired-tip"
    _seed_compression_lineage(donor, root, tip)
    donor._conn.execute("UPDATE messages SET display_metadata=? WHERE session_id=? AND role='user'", ('{"_queued_prompt":true}', root))
    donor._conn.commit()
    target = SessionDB(profile_home / "state.db")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(SessionDB, "_retire_donor_segment", _retire_fails_for(root))
            first = target.adopt_session_lineage_from(donor, tip)
        assert first["adopted"] and first["pending_ids"] == [root, tip]
        assert target.get_messages(root, include_inactive=True)[0]["active"] == 1
        assert donor.end_session_as_adopted(root)
        assert donor.get_session(root)["end_reason"] == "adopted_by_profile"
        assert not donor.get_session(root)["archived"]
        assert donor.get_messages(root, include_inactive=True)[0]["active"] == 0
    finally:
        target.close()
    donor_path = donor.db_path
    donor.close()
    reopened = SessionDB(donor_path)
    mod._db = reopened
    try:
        response = _resume(mod, root, "half-retired-retry")
        assert not response.get("error"), response.get("error")
        assert all(reopened.get_session(mid)["archived"] for mid in (root, tip))
        with SessionDB(profile_home / "state.db") as verified:
            assert [len(verified.get_messages(mid, include_inactive=True)) for mid in (root, tip)] == [2, 2]
            # Resuming the root retires its own undrained queued prompt normally.
            assert verified.get_messages(root, include_inactive=True)[0]["active"] == 0
    finally:
        reopened.close()
        mod._db = donor


@pytest.mark.parametrize("damage", ["missing_root", "missing_manifest", "malformed_manifest", "inconsistent_member", "malformed_donor"])
def test_resume_refuses_corrupt_pending_component_without_mutation(gateway, monkeypatch, damage):
    mod, donor, profile_home = gateway
    root, tip = "corrupt-resume-root", "corrupt-resume-tip"
    _seed_compression_lineage(donor, root, tip)
    donor._conn.execute("UPDATE messages SET display_metadata=? WHERE session_id=? AND role='user'", ('{"_queued_prompt":true}', root))
    donor._conn.commit()
    target = SessionDB(profile_home / "state.db")
    try:
        target.create_session("unrelated", "tui")
        with monkeypatch.context() as patch:
            patch.setattr(SessionDB, "_retire_donor_segment", _retire_fails_for(tip))
            assert target.adopt_session_lineage_from(donor, tip)["pending_ids"] == [tip]
        if damage == "malformed_donor":
            import json
            manifest = json.loads(target._conn.execute(
                "SELECT manifest_json FROM adoption_transfers WHERE member_id=?", (root,)).fetchone()[0])
            manifest["donor"][1] = "invalid"
            statement = ("UPDATE adoption_transfers SET manifest_json=? WHERE member_id=?", (json.dumps(manifest), root))
        else:
            statement = {
                "missing_root": ("DELETE FROM adoption_transfers WHERE member_id=?", (root,)),
                "missing_manifest": ("UPDATE adoption_transfers SET manifest_json=NULL WHERE member_id=?", (root,)),
                "malformed_manifest": ("UPDATE adoption_transfers SET manifest_json='{' WHERE member_id=?", (root,)),
                "inconsistent_member": ("UPDATE adoption_transfers SET member_order=7 WHERE member_id=?", (tip,)),
            }[damage]
        target._conn.execute(*statement)
        target._conn.commit()
        assert not _resume(mod, "unrelated", "unrelated-control").get("error")
        def snapshot():
            return [[tuple(row) for row in db._conn.execute("SELECT * FROM " + table + " ORDER BY rowid")]
                    for db in (donor, target) for table in ("sessions", "messages")]
        before = snapshot()
        response = _resume(mod, root, "corrupt-retry")
        assert response.get("error"), response
        assert "adoption" in response["error"]["message"].lower()
        assert snapshot() == before
        assert not donor.get_session(tip)["archived"]
    finally:
        target.close()


def test_branch_stored_collision_preserves_existing_local_conversation(gateway, monkeypatch):
    from tests.tui_gateway.test_protocol import _stub_session_create_dependencies
    mod, db, _ = gateway
    db.create_session("branch-parent", "cli")
    db.append_message("branch-parent", "user", "parent history")
    db.create_session("branch-existing", "cli")
    db.append_message("branch-existing", "user", "unrelated history")
    before = [[tuple(row) for row in db._conn.execute("SELECT * FROM " + table + " ORDER BY rowid")]
              for table in ("sessions", "messages")]
    real_seed = mod._seed_branch_row
    _stub_session_create_dependencies(mod, monkeypatch)
    monkeypatch.setattr(mod, "_seed_branch_row", real_seed)
    monkeypatch.setattr(mod, "_new_session_key", lambda: "branch-existing")
    response = mod.handle_request({"id": "branch-collision", "method": "session.branch_stored",
                                   "params": {"parent_session_id": "branch-parent", "source": "desktop"}})
    after = [[tuple(row) for row in db._conn.execute("SELECT * FROM " + table + " ORDER BY rowid")]
             for table in ("sessions", "messages")]
    assert after == before
    assert response.get("error"), response


def test_branch_stored_fresh_id_and_cached_replay(gateway, monkeypatch):
    from tests.tui_gateway.test_protocol import _stub_session_create_dependencies
    mod, db, _ = gateway
    db.create_session('fresh-parent', 'cli')
    db.append_message('fresh-parent', 'user', 'parent history')
    real_seed = mod._seed_branch_row
    _stub_session_create_dependencies(mod, monkeypatch)
    monkeypatch.setattr(mod, '_seed_branch_row', real_seed)
    monkeypatch.setattr(mod, '_new_session_key', lambda: 'fresh-branch')
    params = {'parent_session_id': 'fresh-parent', 'source': 'desktop', 'idempotency_key': 'branch-fresh-replay'}
    first = mod.handle_request({'id': 'branch-first', 'method': 'session.branch_stored', 'params': params})
    assert not first.get('error'), first
    assert first['result']['stored_session_id'] == 'fresh-branch'
    assert [row['content'] for row in db.get_messages('fresh-branch')] == ['parent history']
    before = [[tuple(row) for row in db._conn.execute('SELECT * FROM ' + table + ' ORDER BY rowid')]
              for table in ('sessions', 'messages')]
    retry = mod.handle_request({'id': 'branch-retry', 'method': 'session.branch_stored', 'params': params})
    assert retry['result']['session_id'] == first['result']['session_id']
    assert [[tuple(row) for row in db._conn.execute('SELECT * FROM ' + table + ' ORDER BY rowid')]
            for table in ('sessions', 'messages')] == before


def test_branch_copy_failure_compensates_only_new_child(gateway, monkeypatch):
    mod, db, _ = gateway
    db.create_session('comp-parent', 'cli')
    db.create_session('unrelated', 'cli')
    db.append_message('unrelated', 'user', 'untouched')
    before = [tuple(row) for row in db._conn.execute("SELECT * FROM messages WHERE session_id='unrelated'")]
    def fail_copy(*args, **kwargs):
        raise RuntimeError('injected branch copy failure')
    monkeypatch.setattr(db, 'append_messages_batch', fail_copy)
    with pytest.raises(RuntimeError, match='injected branch copy failure'):
        mod._persist_branch(db, 'new-child', 'comp-parent', 'Branch', [{'role': 'user', 'content': 'seed'}],
                            source='desktop', cwd=None, profile_name='default', model='m', compensate=True)
    assert db.get_session('new-child') is None
    assert [tuple(row) for row in db._conn.execute("SELECT * FROM messages WHERE session_id='unrelated'")] == before


@pytest.mark.parametrize('seeded', [False, True])
def test_session_create_generated_collision_refuses_existing(gateway, monkeypatch, seeded):
    from tests.tui_gateway.test_protocol import _stub_session_create_dependencies
    mod, db, _ = gateway
    db.create_session('existing-create', 'cli')
    db.append_message('existing-create', 'user', 'unrelated history')
    before = [[tuple(row) for row in db._conn.execute('SELECT * FROM ' + table + ' ORDER BY rowid')]
              for table in ('sessions', 'messages')]
    seed, ensure = mod._seed_row, mod._ensure_session_db_row
    _stub_session_create_dependencies(mod, monkeypatch)
    monkeypatch.setattr(mod, '_seed_row', seed)
    monkeypatch.setattr(mod, '_ensure_session_db_row', ensure)
    monkeypatch.setattr(mod, '_new_session_key', lambda: 'existing-create')
    params = {'source': 'desktop', 'idempotency_key': 'collision-create'}
    if seeded:
        params['messages'] = [{'role': 'user', 'content': 'new seed'}]
    response = mod.handle_request({'id': 'create-collision', 'method': 'session.create', 'params': params})
    after = [[tuple(row) for row in db._conn.execute('SELECT * FROM ' + table + ' ORDER BY rowid')]
             for table in ('sessions', 'messages')]
    assert after == before
    assert response.get('error'), response


@pytest.mark.parametrize('late_collision', [False, True])
def test_fresh_draft_first_persist_refuses_late_collision_or_reuses_own_row(gateway, monkeypatch, late_collision):
    from hermes_state_lineage import FreshSessionCollision
    from tests.tui_gateway.test_protocol import _stub_session_create_dependencies

    mod, db, _ = gateway
    ensure = mod._ensure_session_db_row
    _stub_session_create_dependencies(mod, monkeypatch)
    monkeypatch.setattr(mod, '_ensure_session_db_row', ensure)
    monkeypatch.setattr(mod, '_new_session_key', lambda: 'fresh-draft')
    first = mod.handle_request({'id': 'draft-first', 'method': 'session.create',
                                'params': {'source': 'desktop', 'idempotency_key': 'fresh-draft-replay'}})
    assert not first.get('error'), first
    record = mod._sessions[first['result']['session_id']]
    assert record['_fresh_generated_key'] is True
    if late_collision:
        db.create_session('fresh-draft', 'cli')
        db.append_message('fresh-draft', 'user', 'unrelated')
        before = [[tuple(row) for row in db._conn.execute('SELECT * FROM ' + table + ' ORDER BY rowid')]
                  for table in ('sessions', 'messages')]
        with pytest.raises(FreshSessionCollision):
            mod._ensure_session_db_row(record)
        assert [[tuple(row) for row in db._conn.execute('SELECT * FROM ' + table + ' ORDER BY rowid')]
                for table in ('sessions', 'messages')] == before
        assert record['_fresh_generated_key'] is True
    else:
        assert mod._ensure_session_db_row(record) is True
        assert record['_fresh_generated_key'] is False
        assert mod._ensure_session_db_row(record) is True
        retry = mod.handle_request({'id': 'draft-retry', 'method': 'session.create',
                                    'params': {'source': 'desktop', 'idempotency_key': 'fresh-draft-replay'}})
        assert retry['result']['session_id'] == first['result']['session_id']


def test_fresh_draft_failed_birth_does_not_continue_as_ensured(gateway, monkeypatch):
    from tests.tui_gateway.test_protocol import _stub_session_create_dependencies

    mod, db, _ = gateway
    ensure = mod._ensure_session_db_row
    _stub_session_create_dependencies(mod, monkeypatch)
    monkeypatch.setattr(mod, '_ensure_session_db_row', ensure)
    monkeypatch.setattr(mod, '_new_session_key', lambda: 'failure-draft')
    first = mod.handle_request({'id': 'draft', 'method': 'session.create', 'params': {'source': 'desktop'}})
    assert not first.get('error'), first
    record = mod._sessions[first['result']['session_id']]
    def fail_birth(*args, **kwargs):
        raise RuntimeError('injected birth failure')
    monkeypatch.setattr(db, '_create_fresh_session', fail_birth)
    assert mod._ensure_session_db_row(record) is False
    assert record['_fresh_generated_key'] is True
    assert db.get_session('failure-draft') is None


@pytest.mark.parametrize('mode', ['retry', 'local_collision', 'foreign_collision', 'discard_false', 'discard_unknown'])
def test_seed_compensation_preserves_fresh_retry_intent(gateway, monkeypatch, mode):
    from tests.tui_gateway.test_protocol import _stub_session_create_dependencies
    mod, db, _ = gateway
    ensure, seed, persist = mod._ensure_session_db_row, mod._seed_row, mod._persist_branch_seed
    _stub_session_create_dependencies(mod, monkeypatch)
    monkeypatch.setattr(mod, '_ensure_session_db_row', ensure)
    monkeypatch.setattr(mod, '_seed_row', seed)
    monkeypatch.setattr(mod, '_persist_branch_seed', persist)
    monkeypatch.setattr(mod, '_new_session_key', lambda: 'seed-compensation-id')
    append = db.append_messages_batch
    discard = db.discard_unadmitted_session
    def fail_seed(*args, **kwargs):
        raise RuntimeError('seed fault')
    monkeypatch.setattr(db, 'append_messages_batch', fail_seed)
    if mode == 'discard_false':
        monkeypatch.setattr(db, 'discard_unadmitted_session', lambda key: False)
    if mode == 'discard_unknown':
        def unknown(key):
            discard(key)
            raise RuntimeError('discard acknowledgment lost')
        monkeypatch.setattr(db, 'discard_unadmitted_session', unknown)
    created = mod.handle_request({'id': 'seed-create', 'method': 'session.create', 'params': {
        'source': 'desktop', 'messages': [{'role': 'user', 'content': 'seed'}]}})
    assert not created.get('error'), created
    record = mod._sessions[created['result']['session_id']]
    monkeypatch.setattr(db, 'append_messages_batch', append)
    sibling = None
    try:
        if mode == 'local_collision':
            db.create_session('seed-compensation-id', 'cli')
            db.append_message('seed-compensation-id', 'user', 'unrelated')
        elif mode == 'foreign_collision':
            profile = db.db_path.parent / 'profiles' / 'science'
            profile.mkdir(parents=True, exist_ok=True)
            sibling = SessionDB(profile / 'state.db')
            sibling.create_session('seed-compensation-id', 'cli')
            sibling.append_message('seed-compensation-id', 'user', 'foreign')
        stores = (db,) if sibling is None else (db, sibling)
        def snapshot():
            return [[tuple(row) for row in store._conn.execute('SELECT * FROM ' + table + ' ORDER BY rowid')]
                    for store in stores for table in ('sessions', 'messages', 'state_meta')]
        before = snapshot()
        reply = mod._persist_session_row_for_submit('submit', record, 'followup')
        if mode == 'retry':
            assert reply is None
            assert [m['content'] for m in db.get_messages('seed-compensation-id')] == ['seed', 'followup']
            assert record['_branch_seed_persisted']
        else:
            assert reply and reply.get('error'), reply
            assert snapshot() == before
    finally:
        if sibling is not None:
            sibling.close()


@pytest.mark.parametrize('mode', ['success', 'collision', 'compensate'])
def test_named_profile_real_registry_branch_fresh_contract(gateway, monkeypatch, mode):
    from tests.tui_gateway.test_protocol import _stub_session_create_dependencies

    mod, launch_db, profile_home = gateway
    profile_path = profile_home / 'state.db'
    with SessionDB(profile_path) as setup:
        setup.create_session('named-parent', 'cli')
        setup.append_message('named-parent', 'user', 'profile history')
        if mode == 'collision':
            setup.create_session('named-child', 'cli')
            setup.append_message('named-child', 'user', 'unrelated')
    launch_before = [tuple(row) for row in launch_db._conn.execute('SELECT * FROM sessions ORDER BY id')]
    with SessionDB(profile_path) as before_db:
        before = [[tuple(row) for row in before_db._conn.execute('SELECT * FROM ' + table + ' ORDER BY rowid')]
                  for table in ('sessions', 'messages')]
    seed = mod._seed_branch_row
    _stub_session_create_dependencies(mod, monkeypatch)
    monkeypatch.setattr(mod, '_seed_branch_row', seed)
    monkeypatch.setattr(mod, '_profile_home', lambda profile: profile_home if profile == 'developer' else None)
    monkeypatch.setattr(mod, '_new_session_key', lambda: 'named-child')
    if mode == 'compensate':
        original = SessionDB.append_messages_batch
        def fail_named(self, session_id, messages, **kwargs):
            if self.db_path == profile_path and session_id == 'named-child':
                raise RuntimeError('named seed failure')
            return original(self, session_id, messages, **kwargs)
        monkeypatch.setattr(SessionDB, 'append_messages_batch', fail_named)
    response = mod.handle_request({'id': 'named-branch', 'method': 'session.branch_stored',
        'params': {'parent_session_id': 'named-parent', 'profile': 'developer', 'source': 'desktop'}})
    with SessionDB(profile_path) as verified:
        rows = [[tuple(row) for row in verified._conn.execute('SELECT * FROM ' + table + ' ORDER BY rowid')]
                for table in ('sessions', 'messages')]
        if mode == 'success':
            assert not response.get('error'), response
            assert [msg['content'] for msg in verified.get_messages('named-child')] == ['profile history']
        elif mode == 'collision':
            assert response.get('error'), response
            assert rows == before
        else:
            assert not response.get('error'), response
            assert verified.get_session('named-child') is None
            assert rows == before
    assert [tuple(row) for row in launch_db._conn.execute('SELECT * FROM sessions ORDER BY id')] == launch_before
