"""Preflight a TUI-minted key without changing existing ensure semantics."""


def fresh_key_collision(profile_db, params, key):
    from hermes_state_lineage import FreshSessionCollision
    try:
        with profile_db(params) as db:
            check = getattr(db, "_assert_fresh_session_id", None)
            if callable(check):
                check(key)
    except FreshSessionCollision:
        return True
    return False


def forget_failed_birth(sessions, lock, keys, idem_key, sid):
    with lock:
        sessions.pop(sid, None)
        if idem_key is not None and keys.get(idem_key, (None,))[0] == sid:
            keys.pop(idem_key, None)
