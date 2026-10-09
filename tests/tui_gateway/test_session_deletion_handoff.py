"""Actual backend RPC -> separate gateway consumer, using only public receipt reads."""

import json
import os
from pathlib import Path
import subprocess
import sys

from hermes_state import SessionDB


_BACKEND = """
import os, sys
from pathlib import Path
from hermes_state import SessionDB
from tui_gateway import server
with SessionDB(Path(sys.argv[1])) as db:
    server._get_db = lambda: db
    result = server.handle_request({"id": "receipt-test", "method": "session.delete",
        "params": {"session_id": "paired-session"}})
    assert "result" in result, result
    # Simulate a backend exit after commit, before a process-local observer can hand off.
    os._exit(0)
"""

_CONSUMER = """
import json, sqlite3, sys
from pathlib import Path
from hermes_state import SessionDB
# Plugin-owned ledger/cursor. No core private table access; no external deletion in this test.
with sqlite3.connect(sys.argv[2]) as ledger:
    ledger.execute("CREATE TABLE IF NOT EXISTS cursor (store_id TEXT PRIMARY KEY, sequence INTEGER)")
    ledger.execute("CREATE TABLE IF NOT EXISTS pending (operation_id TEXT PRIMARY KEY, payload TEXT)")
    current = ledger.execute("SELECT sequence FROM cursor WHERE store_id = ?", (sys.argv[1],)).fetchone()
    with SessionDB(Path(sys.argv[1]), read_only=True) as db:
        rows = db.list_session_deletion_receipts(after_sequence=current[0] if current else 0)
    for row in rows:
        deletion = row["deletion"]
        assert deletion["reason"] == "explicit_user"
        ledger.execute("INSERT OR IGNORE INTO pending VALUES (?, ?)",
                       (deletion["operation_id"], json.dumps(deletion)))
        ledger.execute("INSERT OR REPLACE INTO cursor VALUES (?, ?)", (sys.argv[1], row["sequence"]))
    print(json.dumps({"new": len(rows), "pending": [json.loads(row[0]) for row in
        ledger.execute("SELECT payload FROM pending ORDER BY operation_id")]}))
"""


def test_rpc_commit_survives_backend_exit_and_gateway_owns_durable_handoff(tmp_path):
    profile = tmp_path / "profile"
    path = profile / "state.db"
    with SessionDB(path) as db:
        db.create_session("paired-session", source="example", chat_id="chat-a", thread_id="thread-a")
    env = dict(os.environ, HERMES_HOME=str(profile))
    cwd = Path(__file__).resolve().parents[2]
    backend = subprocess.run([sys.executable, "-c", _BACKEND, str(path)], env=env, cwd=cwd,
                             capture_output=True, text=True, timeout=90)
    assert backend.returncode == 0, backend.stderr
    ledger = tmp_path / "plugin-owned.sqlite"
    def consume():
        result = subprocess.run([sys.executable, "-c", _CONSUMER, str(path), str(ledger)],
                                env=env, cwd=cwd, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    first = consume()
    assert first["new"] == 1
    assert len(first["pending"]) == 1
    deletion = first["pending"][0]
    assert deletion["surface"] == "session_rpc"
    assert deletion["profile_home"] == str(profile.resolve())
    assert deletion["identities"][0]["thread_id"] == "thread-a"
    assert consume() == {"new": 0, "pending": first["pending"]}
    with SessionDB(tmp_path / "other-profile" / "state.db") as other:
        assert other.list_session_deletion_receipts() == []
        assert other.get_session_deletion_receipt(deletion["operation_id"]) is None
