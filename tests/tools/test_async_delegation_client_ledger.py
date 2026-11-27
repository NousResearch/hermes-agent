"""A process that holds no store writer (the classic CLI beside a running gateway) restores delegation completions at
startup without writing ``state.db`` or opening a store writer when the ledger holds nothing for it."""
import json
import os
from pathlib import Path
import subprocess
import sys

from persistence.sqlite._holder import StoreHolder

_WITNESS = r'''
import json, queue, sqlite3, sys
writes, writers = [], []
def trace(frame, event, arg):
    if event == 'call' and frame.f_code.co_name == '__init__':
        if type(frame.f_locals.get('self')).__name__ == 'StoreHolder' and not frame.f_locals.get('read_only'):
            writers.append('StoreHolder')
sys.setprofile(trace)
_connect = sqlite3.connect
def connect(database, *a, **k):
    conn = _connect(database, *a, **k)
    if 'state.db' in str(database):
        def deny(action, table, *rest):
            if action in (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE) and table != 'sqlite_master':
                writes.append('write:' + str(table))
            return sqlite3.SQLITE_OK
        conn.set_authorizer(deny)
    return conn
sqlite3.connect = connect
from tools.async_delegation import restore_undelivered_completions
replayed = restore_undelivered_completions(queue.Queue())
sys.setprofile(None)
print(json.dumps({"replayed": replayed, "writes": writes, "writers": writers}))
'''


def test_startup_restore_with_an_empty_ledger_neither_writes_nor_opens_a_writer():
    home = Path(os.environ["HERMES_HOME"])
    StoreHolder(home / "state.db").close()
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(root))
    result = subprocess.run([sys.executable, "-c", _WITNESS], cwd=root, env=env, capture_output=True, text=True,
                            timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {"replayed": 0, "writes": [], "writers": []}
