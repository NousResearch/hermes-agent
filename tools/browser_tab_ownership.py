"""Profile-local creation ledger and durable admission fence.

Unknown task lifetimes are retained. This prerequisite does not close tabs or
infer task completion; only synchronous Harness requests are captured.
"""

import contextlib
import sqlite3
import uuid
from pathlib import Path


class OwnershipBusy(RuntimeError):
    pass


class OwnershipRegistry:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS calls (
                    token TEXT PRIMARY KEY, owner TEXT, generation TEXT, browser TEXT,
                    daemon TEXT, state TEXT NOT NULL,
                    daemon_pid INTEGER, daemon_start TEXT, inflight INTEGER NOT NULL DEFAULT 0);
                CREATE UNIQUE INDEX IF NOT EXISTS one_call_per_daemon
                    ON calls(daemon) WHERE state IN ('active', 'quarantined');
                CREATE TABLE IF NOT EXISTS targets (
                    browser TEXT, target TEXT, owner TEXT, generation TEXT,
                    call_token TEXT, PRIMARY KEY(browser, target));
            ''')

    @contextlib.contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def admit(self, owner, generation, browser, daemon):
        """Commit before spawning exec; a lost caller remains fenced, never expired."""
        token = uuid.uuid4().hex
        with self._db() as db:
            try:
                db.execute('INSERT INTO calls(token,owner,generation,browser,daemon,state) VALUES (?, ?, ?, ?, ?, ?)',
                           (token, owner, generation, browser, daemon, 'active'))
            except sqlite3.IntegrityError:
                raise OwnershipBusy('daemon has an active or quarantined call') from None
        return token

    def call(self, token):
        with self._db() as db:
            row = db.execute('SELECT * FROM calls WHERE token=?', (token,)).fetchone()
            if row is None:
                raise OwnershipBusy('unknown call')
            return dict(row)

    def bind_daemon(self, token, pid, start):
        with self._db() as db:
            row = db.execute('SELECT * FROM calls WHERE token=?', (token,)).fetchone()
            if row is None or row['state'] != 'active':
                raise OwnershipBusy('call is not active')
            if row['daemon_pid'] is not None and (row['daemon_pid'], row['daemon_start']) != (pid, start):
                raise OwnershipBusy('daemon changed during call')
            db.execute('UPDATE calls SET daemon_pid=?, daemon_start=? WHERE token=?',
                       (pid, start, token))

    def record_created(self, token, target):
        if not isinstance(target, str) or not target:
            raise ValueError('missing exact creation target')
        with self._db() as db:
            call = db.execute("SELECT * FROM calls WHERE token=? AND state IN ('active', 'quarantined')",
                              (token,)).fetchone()
            if call is None:
                raise OwnershipBusy('creation without admitted call')
            previous = db.execute('SELECT call_token FROM targets WHERE browser=? AND target=?',
                                  (call['browser'], target)).fetchone()
            if previous:
                if previous[0] != token:
                    raise OwnershipBusy('target already attributed')
                return
            db.execute('INSERT INTO targets(browser,target,owner,generation,call_token) VALUES (?,?,?,?,?)',
                       (call['browser'], target, call['owner'], call['generation'], token))

    def request_started(self, token):
        with self._db() as db:
            if not db.execute("UPDATE calls SET inflight=inflight+1 WHERE token=? AND state='active'",
                              (token,)).rowcount:
                raise OwnershipBusy('call is not active')

    def request_finished(self, token):
        with self._db() as db:
            db.execute('UPDATE calls SET inflight=inflight-1 WHERE token=? AND inflight>0', (token,))

    def finish(self, token):
        """Only after synchronous helper replies; cannot release quarantine."""
        with self._db() as db:
            db.execute("UPDATE calls SET state=CASE WHEN inflight=0 THEN 'drained' ELSE 'quarantined' END WHERE token=? AND state='active'", (token,))

    def quarantine(self, token):
        with self._db() as db:
            db.execute("UPDATE calls SET state='quarantined' WHERE token=? AND state='active'", (token,))

    def owned(self, token):
        call = self.call(token)
        with self._db() as db:
            return [r[0] for r in db.execute('SELECT target FROM targets WHERE browser=? AND owner=? '
                                            'AND generation=? ORDER BY rowid',
                                            (call['browser'], call['owner'], call['generation']))]

    def owned_for_daemon(self, token):
        """Entry reuse is scoped to the admitted daemon, not just its owner."""
        call = self.call(token)
        if call['state'] != 'active':
            raise OwnershipBusy('call is not active')
        with self._db() as db:
            return [r[0] for r in db.execute(
                'SELECT t.target FROM targets t JOIN calls c ON c.token=t.call_token '
                'WHERE t.browser=? AND t.owner=? AND t.generation=? '
                'AND c.daemon=? ORDER BY t.rowid',
                (call['browser'], call['owner'], call['generation'], call['daemon']))]
