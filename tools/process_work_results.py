"""Finite, opt-in work results from a native process's output, not readiness matches.

The trusted spawn binds the owner; output can supply only a terminal status and
opaque evidence text for that binding. SQLite commits precede native admission.
The gateway's existing drain is the only consumer. No external replay transport.
"""
import json
import hashlib
import uuid
import sqlite3
import time
import logging
from contextlib import closing
from pathlib import Path

from hermes_constants import get_hermes_home
from hermes_state import SessionDB

PREFIX = 'HERMES_WORK_RESULT_V1 '
MAX_FRAME = 8192
MAX_WORK = 16
MAX_RECORDS = 4096
SUPERVISION_ERRORS = {'never_started': 'os_spawn_failed', 'supervision_failed': 'missing_terminal_result'}
_TABLE = '''CREATE TABLE IF NOT EXISTS process_work_results (
    result_id TEXT PRIMARY KEY, process_id TEXT NOT NULL,
    work_id TEXT NOT NULL, incarnation TEXT NOT NULL,
    route TEXT NOT NULL, parent TEXT NOT NULL,
    payload TEXT, digest TEXT, disposition TEXT NOT NULL DEFAULT 'enrolled', error TEXT,
    UNIQUE(process_id, work_id, incarnation))'''
_FENCES = '''CREATE TABLE IF NOT EXISTS process_work_fences (
    route TEXT PRIMARY KEY, generation INTEGER NOT NULL DEFAULT 0)'''
_OWNER_FENCES = '''CREATE TABLE IF NOT EXISTS process_work_owner_fences (
    parent TEXT PRIMARY KEY, generation INTEGER NOT NULL DEFAULT 0)'''


def _exists(conn):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='process_work_results'").fetchone() is not None


def enroll(session, work):
    if not isinstance(work, list) or not 1 <= len(work) <= MAX_WORK:
        raise ValueError('completion_work requires 1..16 work bindings')
    keys = []
    for item in work:
        if (not isinstance(item, dict) or set(item) != {'work_id', 'incarnation'}
                or any(not isinstance(v, str) or not v or len(v) > 128 for v in item.values())):
            raise ValueError('work bindings require bounded work_id and incarnation strings')
        keys.append((item['work_id'], item['incarnation']))
    if len(set(keys)) != len(keys) or not session.parent_session_id or not session.session_key.startswith('agent:'):
        raise ValueError('completion_work requires unique bindings and a native parent route')
    home = get_hermes_home()
    with closing(SessionDB(home / 'state.db')) as db:
        # Reserve a boundary snapshot before slow parent validation. Stop and
        # enrollment serialize through this same writer, including first use.
        def prepare(conn):
            conn.execute(_TABLE)
            conn.execute(_FENCES)
            conn.execute(_OWNER_FENCES)
            conn.execute('INSERT OR IGNORE INTO process_work_fences(route) VALUES(?)', (session.session_key,))
            conn.execute('INSERT OR IGNORE INTO process_work_owner_fences(parent) VALUES(?)',
                         (session.parent_session_id,))
            return (
                conn.execute('SELECT generation FROM process_work_fences WHERE route=?',
                             (session.session_key,)).fetchone()[0],
                conn.execute('SELECT generation FROM process_work_owner_fences WHERE parent=?',
                             (session.parent_session_id,)).fetchone()[0],
            )
        generation = db._execute_write(prepare)
        if db.get_session(session.parent_session_id) is None:
            raise ValueError('completion_work parent is not persisted')
        def write(conn):
            current = (
                conn.execute('SELECT generation FROM process_work_fences WHERE route=?',
                             (session.session_key,)).fetchone()[0],
                conn.execute('SELECT generation FROM process_work_owner_fences WHERE parent=?',
                             (session.parent_session_id,)).fetchone()[0],
            )
            if current != generation:
                raise ValueError('completion_work enrollment crossed a user boundary')
            if conn.execute('SELECT COUNT(*) FROM process_work_results').fetchone()[0] + len(keys) > MAX_RECORDS:
                raise ValueError('completion_work ledger full; no enrollment was created')
            for work_id, incarnation in keys:
                if conn.execute('SELECT 1 FROM process_work_results WHERE parent=? AND work_id=? AND incarnation=?',
                                (session.parent_session_id, work_id, incarnation)).fetchone():
                    raise ValueError('work incarnation already enrolled for this parent')
                conn.execute('INSERT INTO process_work_results(result_id,process_id,work_id,incarnation,route,parent) '
                             'VALUES(?,?,?,?,?,?)', (uuid.uuid4().hex, session.id, work_id, incarnation,
                                                   session.session_key, session.parent_session_id))
        db._execute_write(write)
    session.completion_work_home = str(home)


def mark_never_started(session):
    """A definite pre-spawn failure is lifecycle evidence, not a task result."""
    try:
        with closing(SessionDB(Path(session.completion_work_home) / 'state.db')) as db:
            db._execute_write(lambda conn: conn.execute(
                "UPDATE process_work_results SET disposition='never_started',error='os_spawn_failed' "
                "WHERE process_id=? AND disposition='enrolled'", (session.id,)))
    except (sqlite3.Error, OSError):
        session.work_result_error = 'spawn_lifecycle_storage_unresolved'
        logging.getLogger(__name__).error('Could not persist spawn rejection for %s', session.id, exc_info=True)


def finish_without_result(session):
    """Only untouched enrollments lack evidence. Exit code is not task status."""
    if not session.completion_work_home or session.work_result_error:
        return
    try:
        with closing(SessionDB(Path(session.completion_work_home) / 'state.db')) as db:
            db._execute_write(lambda conn: conn.execute(
                "UPDATE process_work_results SET disposition='supervision_failed',error='missing_terminal_result' "
                "WHERE process_id=? AND disposition='enrolled'", (session.id,)))
    except (sqlite3.Error, OSError):
        session.work_result_error = 'lifecycle_storage_unresolved'
        logging.getLogger(__name__).error('Could not finalize work supervision for %s', session.id, exc_info=True)


def records(db, process_id=None, pending_only=False):
    with db._read_ctx() as conn:
        if not _exists(conn):
            return []
        clauses, args = [], []
        if process_id is not None:
            clauses.append('process_id=?')
            args.append(process_id)
        if pending_only:
            clauses.append("(disposition='pending' OR (disposition IN ('never_started','supervision_failed') "
                           "AND payload IS NULL AND digest IS NULL))")
        sql = 'SELECT * FROM process_work_results' + (' WHERE ' + ' AND '.join(clauses) if clauses else '')
        return [dict(row) for row in conn.execute(sql, args)]


def ingest(session, text):
    home = session.completion_work_home
    if not home:
        return text
    if session.work_result_error:
        # The durable output log is the evidence. Do not skip a failed frame and
        # accept a later one, or turn a transport outage into a task result.
        return ''
    ordinary = []
    # Newlines delimit frames. An overlong line is discarded through its delimiter,
    # not reinterpreted as a fresh record when a chunk happens to begin with PREFIX.
    for part in text.splitlines(keepends=True):
        end = part.endswith('\n')
        if not session._work_discard:
            session._work_buffer += part
            if len(session._work_buffer) > MAX_FRAME:
                session._work_buffer = ''
                session._work_discard = True
        if end:
            line, session._work_buffer = session._work_buffer, ''
            discard, session._work_discard = session._work_discard, False
            if not discard and line.startswith(PREFIX):
                # Keep this exact frame (including its delimiter) until commit.
                # No decoder/chunk replay: earlier frames may already be durable.
                for attempt in range(3):
                    try:
                        _record(Path(home), session.id, line[len(PREFIX):])
                        break
                    except (sqlite3.Error, OSError):
                        if attempt == 2:
                            session.work_result_error = 'storage_retry_exhausted'
                            logging.getLogger(__name__).error(
                                'Work result storage unresolved for %s; evidence retained in %s',
                                session.id, session.output_log, exc_info=True)
                            return ''
                        time.sleep(.05)
            elif not discard:
                ordinary.append(line)
    return ''.join(ordinary)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate result field')
        result[key] = value
    return result


def _record(home, process_id, raw):
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
        if (not isinstance(value, dict) or set(value) != {'work_id', 'incarnation', 'status', 'evidence'}
                or not all(isinstance(v, str) for v in value.values())
                or value['status'] not in {'completed', 'failed', 'cancelled'}
                or len(value['evidence']) > 4096):
            return
        # JSON permits escaped lone surrogates; SQLite text bindings do not.
        # Reject invalid identifiers before the untrusted record reaches SQLite.
        value['work_id'].encode('utf-8')
        value['incarnation'].encode('utf-8')
    except (ValueError, RecursionError):
        return
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'))
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    from agent.redact import redact_sensitive_text
    value = {**value, 'evidence': redact_sensitive_text(value['evidence'], force=True)}
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'))
    with closing(SessionDB(home / 'state.db')) as db:
        def write(conn):
            row = conn.execute('SELECT * FROM process_work_results WHERE process_id=? AND work_id=? AND incarnation=?',
                               (process_id, value['work_id'], value['incarnation'])).fetchone()
            if row is None:
                return
            if row['payload'] is not None and row['digest'] != digest:
                conn.execute("UPDATE process_work_results SET error='conflicting_terminal_result', "
                             "disposition=CASE WHEN disposition='pending' THEN 'conflict' ELSE disposition END "
                             "WHERE result_id=?", (row['result_id'],))
                return
            if row['disposition'] != 'enrolled':
                return
            conn.execute("UPDATE process_work_results SET payload=?,digest=?,disposition='pending' WHERE result_id=?",
                         (encoded, digest, row['result_id']))
        db._execute_write(write)


def supervision_kind(row):
    """Trusted lifecycle observation, separate from producer status and notice delivery."""
    if row['payload'] is None and row['digest'] is None:
        for kind, error in SUPERVISION_ERRORS.items():
            if row['error'] == error and row['disposition'] in (kind, kind + '_admitted', 'dropped'):
                return kind
    return None


def admission_identity(row):
    producer = 'process_work_supervision' if supervision_kind(row) else 'process_work_result'
    return json.dumps([producer, row['process_id'], row['result_id']], separators=(',', ':'))


def admission_guard(result_id, *, disposition='pending', error=None):
    """Runs in the native admission transaction: no ACK gap, no separate outbox ACK."""
    def guard(conn):
        from hermes_state_runtime import RuntimeStoreError
        if disposition in SUPERVISION_ERRORS and error == SUPERVISION_ERRORS[disposition]:
            changed = conn.execute(
                "UPDATE process_work_results SET disposition=? WHERE result_id=? AND disposition=? "
                "AND error=? AND payload IS NULL AND digest IS NULL",
                (disposition + '_admitted', result_id, disposition, error)).rowcount
        elif disposition == 'pending':
            changed = conn.execute("UPDATE process_work_results SET disposition='admitted' "
                                   "WHERE result_id=? AND disposition='pending'", (result_id,)).rowcount
        else:
            changed = 0
        if changed != 1:
            raise RuntimeStoreError('stale_generation')
    return guard


def drop(db, *, route=None, result_id=None, reason='user_boundary', parents=()):
    """Same SQLite writer as admission: Stop wins before admission, never revives on reopen."""
    def write(conn):
        if not _exists(conn):
            return
        if route is not None:
            conn.execute(_FENCES)
            conn.execute('INSERT INTO process_work_fences(route,generation) VALUES(?,1) '
                         'ON CONFLICT(route) DO UPDATE SET generation=generation+1', (route,))
        column, value = ('route', route) if route is not None else ('result_id', result_id)
        predicate, values = f'{column}=?', [value]
        if parents:
            # Fence in-flight enrollment too: the owner may now be on another
            # route, and its obligation row need not exist at Stop time yet.
            conn.execute(_OWNER_FENCES)
            for parent in parents:
                conn.execute('INSERT INTO process_work_owner_fences(parent,generation) VALUES(?,1) '
                             'ON CONFLICT(parent) DO UPDATE SET generation=generation+1', (parent,))
            # A resumed owner can have work enrolled on its former detached route.
            predicate += ' OR parent IN (' + ','.join('?' for _ in parents) + ')'
            values.extend(parents)
        conn.execute("UPDATE process_work_results SET disposition='dropped', "
                     "error=CASE WHEN disposition IN ('never_started','supervision_failed') THEN error ELSE ? END "
                     f"WHERE ({predicate}) AND disposition IN "
                     "('enrolled','pending','conflict','never_started','supervision_failed')", (reason, *values))
    db._execute_write(write)


def receipts(session):
    """Read-only process status: model execution is not evidence acceptance."""
    with closing(SessionDB(Path(session.completion_work_home) / 'state.db', read_only=True)) as db:
        result = []
        for row in records(db, session.id):
            item = {key: row[key] for key in ('result_id', 'work_id', 'incarnation', 'disposition', 'error')}
            item['acceptance'] = 'not_verified'
            if row['payload']:
                item.update(json.loads(row['payload']))
            kind = supervision_kind(row)
            if kind:
                item['supervision'] = kind
            identity = admission_identity(row)
            with db._read_ctx() as conn:
                admission = conn.execute('SELECT status,outcome FROM session_admissions WHERE request_id=?',
                                         (identity,)).fetchone()
            item['admission_status'] = admission['status'] if admission else None
            item['admission_outcome'] = admission['outcome'] if admission else None
            result.append(item)
        return result
