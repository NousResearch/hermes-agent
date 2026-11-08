"""Preserve and soft-archive one exact live legacy conversation through its primary turn."""
from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped
_CLOSEOUT_WAIT_SECONDS = 30.0


def _recover_closeout_receipt(rid, params):
    from hermes_cli.conversation_closeout import CloseoutStore, public_receipt
    if current_transport() is None:
        return _err(rid, 4001, 'permission_denied')
    with _profile_db(params, writer=True) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        store = CloseoutStore(db)
        record = store.for_request(params['request_id'])
        if (record is None or params['session_id'] not in {
                record['old'].get('runtime_id'), record['old']['stored_session_id']}):
            return _err(rid, 4001, 'session not found')
        row = db.get_session(record['old']['stored_session_id'])
        if row is None or (row.get('user_id') and row['user_id'] != _transport_auth_user_id(current_transport())):
            return _err(rid, 4001, 'permission_denied')
        if record['status'] == 'reviewing':
            record = store.recover_uncertain(record['closeout_id'])
        if record['status'] == 'preserved' and not record.get('_fence_released'):
            record = store.archive(record['closeout_id'], settled=False)
        return _ok(rid, public_receipt(record))


@method('session.closeout')
@_profile_scoped
def _session_closeout(rid, params):
    from agent.memory_provider import spawn_context_thread
    from hermes_cli.conversation_closeout_jobs import begin_closeout, old_identity
    from hermes_constants import profile_name_for_home
    import asyncio

    sid = params['session_id']
    session = _sessions.get(sid)
    if session is None:
        return _recover_closeout_receipt(rid, params)
    transport = current_transport()
    if transport is None or not _session_transport_contains(session, transport):
        return _err(rid, 4001, 'permission_denied')
    if params.get('profile') is not None and not _live_profile_matches(session, _profile_home(params['profile'])):
        return _err(rid, 4001, 'profile_mismatch')
    if _ensure_session_db_row(session) is False:
        return _db_unavailable_error(rid, code=5007)

    def submit(prompt, operation):
        return handle_request({'id': 'closeout-review-' + operation, 'method': 'prompt.submit',
            'params': {'session_id': sid, 'text': prompt, 'queued': True}})

    def launch(coroutine):
        spawn_context_thread(lambda: asyncio.run(coroutine), name='hermes-closeout-' + sid[:24], daemon=True).start()

    def settled():
        return _sessions.get(sid) is session and _viewer_detach_is_settled(sid, session)

    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        try:
            old = old_identity(db, _submit_row_target_key(session) or session['session_key'],
                               runtime_id=sid, profile=profile_name_for_home(db.db_path.parent) or 'default')
            receipt = begin_closeout(db, old, params['request_id'], params['intent'], submit=submit,
                settled=settled, launch=launch, wait_seconds=_CLOSEOUT_WAIT_SECONDS,
                publish=lambda result: _emit('session.closeout.updated', sid, result))
        except ValueError as exc:
            return _err(rid, 5007, str(exc))
    return _ok(rid, receipt)


def register(server):
    bind_module(globals(), server)
