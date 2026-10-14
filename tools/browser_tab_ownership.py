"""Profile-local exact creation ledger. No tool schema and no inferred task ends.

Only the capture seam may record creations. Lifecycle adapters must prove
retirement before calling retire; UNKNOWN is deliberately not a terminal state.
This module uses only stdlib so the Harness interpreter can import it too.
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
                CREATE TABLE IF NOT EXISTS owners (
                    owner TEXT, generation TEXT, terminal INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(owner, generation));
                CREATE TABLE IF NOT EXISTS calls (
                    token TEXT PRIMARY KEY, owner TEXT, generation TEXT, browser TEXT,
                    daemon TEXT, state TEXT NOT NULL,
                    daemon_pid INTEGER, daemon_start TEXT, inflight INTEGER NOT NULL DEFAULT 0);
                CREATE UNIQUE INDEX IF NOT EXISTS one_call_per_daemon
                    ON calls(daemon) WHERE state IN ('active', 'quarantined');
                CREATE TABLE IF NOT EXISTS targets (
                    browser TEXT, target TEXT, owner TEXT, generation TEXT,
                    call_token TEXT, pending_close INTEGER NOT NULL DEFAULT 0,
                    diagnostic TEXT, PRIMARY KEY(browser, target));
                CREATE TABLE IF NOT EXISTS retirements (
                    owner TEXT, generation TEXT, evidence TEXT NOT NULL,
                    PRIMARY KEY(owner, generation));
                CREATE TABLE IF NOT EXISTS native_completions (
                    owner TEXT, generation TEXT, PRIMARY KEY(owner,generation));
                CREATE TABLE IF NOT EXISTS holds (
                    browser TEXT, target TEXT, actor TEXT,
                    PRIMARY KEY(browser, target, actor));
                CREATE TABLE IF NOT EXISTS cleanup_cursors (
                    browser TEXT, scope TEXT, position INTEGER NOT NULL,
                    PRIMARY KEY(browser, scope));
            ''')
            if 'inflight' not in {r[1] for r in db.execute('PRAGMA table_info(calls)')}:
                db.execute('ALTER TABLE calls ADD COLUMN inflight INTEGER NOT NULL DEFAULT 0')

            for column in ('client_pid INTEGER', 'client_start TEXT'):
                if column.split()[0] not in {r[1] for r in db.execute('PRAGMA table_info(calls)')}:
                    db.execute('ALTER TABLE calls ADD COLUMN ' + column)
            if 'current_target' not in {r[1] for r in db.execute('PRAGMA table_info(calls)')}:
                db.execute('ALTER TABLE calls ADD COLUMN current_target TEXT')

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
            db.execute('INSERT OR IGNORE INTO owners VALUES (?, ?, 0)', (owner, generation))
            if db.execute('SELECT terminal FROM owners WHERE owner=? AND generation=?',
                          (owner, generation)).fetchone()[0]:
                raise OwnershipBusy('owner retired')
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

    def publish_current(self, token, target):
        """Publish only a daemon-reported, exact mission-owned current page."""
        if target not in self.owned_for_daemon(token):
            raise OwnershipBusy('current target is not mission-owned')
        with self._db() as db:
            if not db.execute("UPDATE calls SET current_target=? WHERE token=? AND state='active' AND inflight=0",
                              (target, token)).rowcount:
                raise OwnershipBusy('current target handoff is not drained')

    def current_for_vault(self, token):
        """A completed call, never an active/uncertain or retired mission."""
        with self._db() as db:
            row = db.execute('''SELECT c.browser, c.current_target FROM calls c
                JOIN targets t ON t.browser=c.browser AND t.target=c.current_target
                JOIN calls created ON created.token=t.call_token
                JOIN owners o ON o.owner=c.owner AND o.generation=c.generation
                WHERE c.token=? AND c.state='drained' AND c.inflight=0
                AND o.terminal=0 AND t.pending_close=0
                AND c.owner NOT LIKE 'unknown:%' AND c.generation!='unresolved'
                AND NOT EXISTS (SELECT 1 FROM holds h WHERE h.browser=t.browser AND h.target=t.target)
                AND t.owner=c.owner AND t.generation=c.generation AND created.daemon=c.daemon
                AND NOT EXISTS (SELECT 1 FROM calls busy WHERE busy.browser=c.browser
                    AND busy.daemon=c.daemon AND busy.state IN ('active','quarantined'))
                AND NOT EXISTS (SELECT 1 FROM calls newer WHERE newer.daemon=c.daemon
                    AND newer.rowid>c.rowid)''', (token,)).fetchone()
            if row is None:
                raise OwnershipBusy('owned current target unavailable')
            return tuple(row)

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

    def retire(self, owner, generation, *, evidence=None):
        """Atomic admission fence. The caller must supply authoritative terminal proof."""
        if not evidence:
            return False
        with self._db() as db:
            if db.execute("SELECT 1 FROM calls WHERE owner=? AND generation=? "
                          "AND state IN ('active','quarantined') LIMIT 1", (owner, generation)).fetchone():
                return False
            if not db.execute('SELECT 1 FROM owners WHERE owner=? AND generation=?', (owner, generation)).fetchone():
                return False
            db.execute('INSERT OR IGNORE INTO retirements VALUES (?,?,?)', (owner, generation, evidence))
            return bool(db.execute('UPDATE owners SET terminal=1 WHERE owner=? AND generation=?',
                                   (owner, generation)).rowcount)

    def hold(self, browser, target, actor):
        """An exact durable actor key, never a blanket parent/process hold."""
        with self._db() as db:
            row = db.execute('SELECT t.pending_close, o.terminal FROM targets t JOIN owners o '
                             'USING(owner,generation) WHERE browser=? AND target=?',
                             (browser, target)).fetchone()
            if row is None or row['pending_close'] or row['terminal']:
                raise OwnershipBusy('target unavailable for new holds')
            db.execute('INSERT OR IGNORE INTO holds VALUES (?,?,?)', (browser, target, actor))

    def release(self, browser, target, actor):
        with self._db() as db:
            db.execute('DELETE FROM holds WHERE browser=? AND target=? AND actor=?', (browser, target, actor))

    def owned(self, token):
        call = self.call(token)
        with self._db() as db:
            return [r[0] for r in db.execute('SELECT target FROM targets WHERE browser=? AND owner=? '
                                            'AND generation=? AND pending_close=0 ORDER BY rowid',
                                            (call['browser'], call['owner'], call['generation']))]

    def owned_for_daemon(self, token):
        """Entry reuse is mission-scoped; owned() remains owner-wide for cleanup."""
        call = self.call(token)
        if call['state'] != 'active':
            raise OwnershipBusy('call is not active')
        with self._db() as db:
            return [r[0] for r in db.execute(
                'SELECT t.target FROM targets t JOIN calls c ON c.token=t.call_token '
                'WHERE t.browser=? AND t.owner=? AND t.generation=? '
                'AND c.daemon=? AND t.pending_close=0 ORDER BY t.rowid',
                (call['browser'], call['owner'], call['generation'], call['daemon']))]

    def request_owner_close(self, owner, generation):
        """Explicit resource release, not owner/session retirement.

        Mark only exact creations; fences and holds remain authoritative. New
        browser work can continue, but cannot reattach to pending-close targets.
        """
        if owner.startswith('unknown:') or generation == 'unresolved':
            raise OwnershipBusy('owner identity unresolved')
        with self._db() as db:
            if db.execute("SELECT 1 FROM calls WHERE owner=? AND generation=? "
                          "AND state IN ('active','quarantined') LIMIT 1", (owner, generation)).fetchone():
                raise OwnershipBusy('owner has active or quarantined work')
            db.execute('UPDATE targets SET pending_close=1 WHERE owner=? AND generation=?',
                       (owner, generation))
            return [dict(r) for r in db.execute(
                'SELECT * FROM targets WHERE owner=? AND generation=? ORDER BY rowid',
                (owner, generation))]

    def pending(self, browser, limit=32, owner_scope=None):
        with self._db() as db:
            scope = repr(owner_scope)
            previous = db.execute('SELECT position FROM cleanup_cursors WHERE browser=? AND scope=?',
                                  (browser, scope)).fetchone()
            position = previous[0] if previous else 0
            query = '''SELECT t.rowid AS position,t.* FROM targets t JOIN owners o USING(owner,generation)
                WHERE t.browser=? AND (o.terminal=1 OR t.pending_close=1)
                AND (? IS NULL OR (t.owner=? AND t.generation=?))
                AND NOT EXISTS (SELECT 1 FROM holds h WHERE h.browser=t.browser AND h.target=t.target)
                AND NOT EXISTS (SELECT 1 FROM calls c WHERE c.owner=t.owner AND c.generation=t.generation
                                AND c.state IN ('active','quarantined'))
                AND t.rowid>? ORDER BY t.rowid LIMIT ?'''
            params = (browser, owner_scope[0] if owner_scope else None,
                      *(owner_scope or (None, None)))
            rows = db.execute(query, (*params, position, limit)).fetchall()
            if not rows:
                rows = db.execute(query, (*params, 0, limit)).fetchall()
            db.execute('INSERT OR REPLACE INTO cleanup_cursors VALUES (?,?,?)',
                       (browser, scope, rows[-1]['position'] if rows else 0))
            for row in rows:
                db.execute('UPDATE targets SET pending_close=1 WHERE browser=? AND target=?',
                           (browser, row['target']))
            return [dict(r) for r in rows]

    def cleanup(self, browser, list_targets, close_target, limit=32, owner_scope=None):
        """Callbacks MUST be pinned to this immutable browser WS, not a rediscovered port.

        A successful close reply is not absence proof. Failed/ambiguous replies keep
        the pending record for the next bounded attempt. Never persist exception text.
        """
        for row in self.pending(browser, limit, owner_scope=owner_scope):
            try:
                if row['target'] in list_targets():
                    close_target(row['target'])
                absent = row['target'] not in list_targets()
                with self._db() as db:
                    if absent:
                        db.execute('DELETE FROM targets WHERE browser=? AND target=? AND pending_close=1',
                                   (browser, row['target']))
                    else:
                        db.execute("UPDATE targets SET diagnostic='still_present' WHERE browser=? AND target=?",
                                   (browser, row['target']))
            except Exception as exc:
                with self._db() as db:
                    db.execute('UPDATE targets SET diagnostic=? WHERE browser=? AND target=?',
                               (type(exc).__name__, browser, row['target']))
