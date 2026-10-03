"""Child self-scope write paths on a FILE-BACKED board (amended directive §6).

The amended composition (CTO ruling t_e645df7c): the PATH FENCE decides WHERE,
the in-process context decides the blanket deny, and the parent-grant is the
sole exception to both — exactly its one task id, on the only two sanctioned
write paths (``add_comment`` / ``add_attachment``).

Every case runs against a real file-backed temp board (``tmp_path / *.db``) —
never ``:memory:`` — because the production gate fences on the connection's
backing path and ``connect()`` resolves real paths. Exec'd-descendant cases run
in sidecar processes with a scrubbed env (``HERMES_DELEGATED_CHILD_CONTEXT`` is
popped before any hermes import); descendant markers are path-VALUED, never
``"1"``, when scoping is intended.

Spec mapping (§6 of the amended directive; §10 of the original):
  (e) granted in-process child CAN add_comment + add_attachment on exactly its
      granted id (the case that failed on the pre-amendment tree).
  (f) same child DENIED on a different task id and on status/claim/complete/
      link/board-metadata mutators.
  (a) non-granted in-process child mutator => PermissionError; its connect()
      handle is read-only (a write attempt raises).
  (b) exec'd descendant with a path-valued env marker cannot write the lineage
      board; (b2) nor the pinned ``HERMES_KANBAN_DB`` (sidecar subprocess).
  (c) legacy ``"1"`` marker fences everything (sidecar subprocess).
  (j') descendant's temp-``HERMES_HOME`` scratch board (outside the fenced
      root) init + ``create_task`` stays WRITABLE — stage2 scratch-root
      restoration (sidecar subprocess).
  (d) grandchild holds no grant (the ContextVar dies at the process boundary).
  (g) ``is_self_scoped_write`` False when ``HERMES_KANBAN_TASK`` is set in env
      but no grant exists (env-spoofing resistance).
  (h) a nested grant cannot widen scope past the outer id.
  (i) a fenced connect() never initializes/migrates; missing schema =>
      PermissionError.
  (k) the granted child's handle carries busy_timeout (waits on a lock).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent.delegation_context import (
    delegated_child_context,
    delegated_self_scope_grant,
    is_delegated_child_in_process_context,
    is_self_scoped_write,
    self_scope_grant,
)
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect

_LIVE_TREE = str(Path(__file__).resolve().parents[2])


def _file_board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A dispatcher-owned worker board backed by a real file, with own+foreign tasks."""
    db = tmp_path / "self-scope.db"
    assert not str(db).startswith(":memory:")
    conn = connect(db)
    assert Path(str(db)).exists()
    own = kb.create_task(conn, title="own")
    foreign = kb.create_task(conn, title="foreign")
    for tid in (own, foreign):
        kb.claim_task(conn, tid)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    monkeypatch.setenv("HERMES_KANBAN_TASK", own)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    return conn, db, own, foreign


def _denied(fn) -> str:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - the denial type is the assertion
        return type(exc).__name__
    return "ok"


def _run_sidecar(script: str, env_extra: dict, tmp_path: Path, *, marker: str | None) -> str:
    """Run a child sidecar with a scrubbed env; return its last ``OUT`` line.

    The inherited ``HERMES_DELEGATED_CHILD_CONTEXT`` is popped inside the child
    before any hermes import (this host's shell exports it), then re-set to the
    case's intended *marker* value — path-valued for scoped descendants, ``"1"``
    for the legacy case, never left at the ambient agent's value.
    """
    env = dict(os.environ)
    env.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)
    env.update(env_extra)
    script_path = tmp_path / "_sidecar_child.py"
    script_path.write_text(script, encoding="utf-8")
    r = subprocess.run(
        [sys.executable, str(script_path), marker or ""],
        capture_output=True, text=True, env=env, timeout=120,
    )
    lines = [l for l in r.stdout.splitlines() if l.startswith("OUT")]
    return lines[-1] if lines else f"OUT-ERROR {r.stderr[-400:]}"


# ---------------------------------------------------------------------------
# In-process child cases — the gate + connect() branch run in THIS process.
# ---------------------------------------------------------------------------

def test_e_granted_child_can_self_report_on_granted_id(tmp_path, monkeypatch):
    """(e) granted in-process child CAN comment + attach on exactly its granted id."""
    conn, _db, own, _foreign = _file_board(tmp_path, monkeypatch)
    with delegated_child_context("child"), delegated_self_scope_grant(own, "lock"):
        assert is_delegated_child_in_process_context()
        cid = kb.add_comment(conn, own, author="sub", body="child report")
        assert cid > 0
        aid = kb.store_attachment_bytes(conn, own, "finding.txt", b"payload", board="default")
        assert aid > 0
    assert [c.body for c in kb.list_comments(conn, own)] == ["child report"]
    assert [a.filename for a in kb.list_attachments(conn, own)] == ["finding.txt"]


def test_f_granted_child_denied_elsewhere(tmp_path, monkeypatch):
    """(f) same child DENIED on a different task id and on lifecycle mutators."""
    conn, _db, own, foreign = _file_board(tmp_path, monkeypatch)
    with delegated_child_context("child"), delegated_self_scope_grant(own, "lock"):
        assert _denied(lambda: kb.add_comment(conn, foreign, author="sub", body="x")) == "PermissionError"
        assert _denied(lambda: kb.store_attachment_bytes(
            conn, foreign, "x.txt", b"x", board="default")) == "PermissionError"
        assert _denied(lambda: kb.create_task(conn, title="child-created")) == "PermissionError"
        assert _denied(lambda: kb.claim_task(conn, foreign)) == "PermissionError"
        assert _denied(lambda: kb.complete_task(conn, own, summary="not yours")) == "PermissionError"
        assert _denied(lambda: kb.link_tasks(conn, own, foreign)) == "PermissionError"
        assert _denied(lambda: kb.set_reasoning_effort(conn, own, effort="high")) == "PermissionError"
        assert _denied(lambda: kb.write_board_metadata(board="default", name="x")) == "PermissionError"
    assert kb.list_comments(conn, foreign) == []
    own_task = kb.get_task(conn, own)
    assert own_task is not None and own_task.status == "running"


def test_g_env_identity_without_grant_is_not_self_scoped(tmp_path, monkeypatch):
    """(g) HERMES_KANBAN_TASK in env with NO grant => is_self_scoped_write False + denied."""
    conn, _db, own, _foreign = _file_board(tmp_path, monkeypatch)
    assert os.environ["HERMES_KANBAN_TASK"] == own
    with delegated_child_context("child"):
        assert is_self_scoped_write(own) is False
        assert _denied(lambda: kb.add_comment(conn, own, author="sub", body="x")) == "PermissionError"
    assert kb.list_comments(conn, own) == []


def test_h_nested_grant_cannot_widen_scope(tmp_path, monkeypatch):
    """(h) an inner grant for another id does not widen the outer grant's scope."""
    conn, _db, own, foreign = _file_board(tmp_path, monkeypatch)
    with delegated_child_context("child"), delegated_self_scope_grant(own, "lock"):
        assert is_self_scoped_write(own) is True
        with delegated_self_scope_grant(foreign, "lock"):
            assert is_self_scoped_write(foreign) is True
            assert is_self_scoped_write(own) is False
            assert _denied(lambda: kb.add_comment(conn, own, author="sub", body="x")) == "PermissionError"
        assert is_self_scoped_write(own) is True
        assert _denied(lambda: kb.add_comment(conn, foreign, author="sub", body="x")) == "PermissionError"


def test_a_non_granted_child_mutator_denied_and_handle_readonly(tmp_path, monkeypatch):
    """(a) non-granted in-process child mutator => PermissionError; handle read-only."""
    conn, db, own, _foreign = _file_board(tmp_path, monkeypatch)
    with delegated_child_context("child"):
        assert self_scope_grant() is None
        assert _denied(lambda: kb.add_comment(conn, own, author="sub", body="x")) == "PermissionError"
        assert _denied(lambda: kb.create_task(conn, title="x")) == "PermissionError"
        # The connect() handle itself must be read-only.
        ro = connect(db)
        assert _denied(lambda: ro.execute("CREATE TABLE IF NOT EXISTS zz(x)")) in (
            "OperationalError", "PermissionError",
        )
        ro.close()


def test_k_granted_child_handle_carries_busy_timeout(tmp_path, monkeypatch):
    """(k) the granted child's writable handle keeps the generous busy_timeout."""
    conn, db, own, _foreign = _file_board(tmp_path, monkeypatch)
    with delegated_child_context("child"), delegated_self_scope_grant(own, "lock"):
        granted_conn = connect(db)
        timeout_ms = granted_conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert int(timeout_ms) >= 1000
        granted_conn.close()


def test_i_fenced_connect_never_initializes_missing_board(tmp_path, monkeypatch):
    """(i) a fenced connect() never initializes/migrates; missing schema fails closed."""
    conn, _db, own, _foreign = _file_board(tmp_path, monkeypatch)
    # An EXISTING but schema-less board file: a fenced connect must neither
    # create a missing file nor migrate this one — it fails closed instead.
    schemaless = tmp_path / "schemaless-board.db"
    schemaless.write_bytes(b"")
    with delegated_child_context("child"):
        assert _denied(lambda: connect(schemaless)) == "PermissionError"
    assert schemaless.read_bytes() == b""


# ---------------------------------------------------------------------------
# Exec'd descendant cases — sidecar processes with a scrubbed env.
# ---------------------------------------------------------------------------

_DESCENDANT_WRITE = """\
import os, sys
L = {live!r}
sys.path.insert(0, L)
os.environ.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)
if sys.argv[1]:
    os.environ["HERMES_DELEGATED_CHILD_CONTEXT"] = sys.argv[1]
from hermes_cli import kanban_db as kb
import hermes_cli.kanban_db_connect as kbc
try:
    conn = kbc.connect(kbc.Path({db!r}))
    kb.add_comment(conn, {task!r}, author="desc", body="should not land")
    print("OUT WROTE (BAD)")
except PermissionError:
    print("OUT PermissionError")
except Exception as e:
    print("OUT", type(e).__name__, str(e)[:120])
"""

_SCRATCH_INIT = """\
import os, sys
L = {live!r}
sys.path.insert(0, L)
os.environ.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)
if sys.argv[1]:
    os.environ["HERMES_DELEGATED_CHILD_CONTEXT"] = sys.argv[1]
import hermes_cli.kanban_db_connect as kbc
from hermes_cli import kanban_db as kb
try:
    kbc.init_db(kbc.Path({db!r}))
    conn = kbc.connect(kbc.Path({db!r}))
    tid = kb.create_task(conn, title="scratch", created_by="desc")
    print("OUT SCRATCH-OK", tid)
except Exception as e:
    print("OUT SCRATCH-FAIL", type(e).__name__, str(e)[:120])
"""

_GRANDCHILD = """\
import os, sys
L = {live!r}
sys.path.insert(0, L)
os.environ.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)
if sys.argv[1]:
    os.environ["HERMES_DELEGATED_CHILD_CONTEXT"] = sys.argv[1]
from agent.delegation_context import self_scope_grant
g = self_scope_grant()
print("OUT GRANT", "none" if g is None else g[0])
"""


def test_b_exec_descendant_cannot_write_lineage_board(tmp_path, monkeypatch):
    """(b) exec'd descendant (path-valued marker) cannot write the lineage board."""
    conn, db, own, _foreign = _file_board(tmp_path, monkeypatch)
    out = _run_sidecar(
        _DESCENDANT_WRITE.format(live=_LIVE_TREE, db=str(db), task=own),
        {}, tmp_path,
        marker=str(tmp_path),
    )
    assert "PermissionError" in out, out
    assert kb.list_comments(conn, own) == []


def test_b2_exec_descendant_cannot_write_pinned_db(tmp_path, monkeypatch):
    """(b2) ...nor the board it pins via HERMES_KANBAN_DB."""
    conn, db, own, _foreign = _file_board(tmp_path, monkeypatch)
    out = _run_sidecar(
        _DESCENDANT_WRITE.format(live=_LIVE_TREE, db=str(db), task=own),
        {"HERMES_KANBAN_DB": str(db)},
        tmp_path,
        marker=str(tmp_path),
    )
    assert "PermissionError" in out, out


def test_c_legacy_marker_fences_everything(tmp_path, monkeypatch):
    """(c) a legacy \"1\" marker fences everything, including unrelated boards."""
    conn, db, own, _foreign = _file_board(tmp_path, monkeypatch)
    out = _run_sidecar(
        _DESCENDANT_WRITE.format(live=_LIVE_TREE, db=str(db), task=own),
        {}, tmp_path,
        marker="1",
    )
    assert "PermissionError" in out, out


def test_jp_scratch_root_board_stays_writable(tmp_path, monkeypatch):
    """(j') descendant's temp-HERMES_HOME scratch board (outside the fenced root)
    stays writable — init + create_task succeed (stage2 scratch-root restoration)."""
    scratch_home = tmp_path / "scratch-home"
    scratch_home.mkdir()
    scratch_db = scratch_home / "kanban.db"
    out = _run_sidecar(
        _SCRATCH_INIT.format(live=_LIVE_TREE, db=str(scratch_db)),
        {"HERMES_KANBAN_HOME": str(scratch_home)},
        tmp_path,
        marker=str(tmp_path / "real-home"),
    )
    assert "SCRATCH-OK" in out, out
    assert scratch_db.exists()


def test_d_grandchild_holds_no_grant(tmp_path, monkeypatch):
    """(d) a grandchild process holds no grant — the ContextVar dies at exec."""
    conn, db, own, _foreign = _file_board(tmp_path, monkeypatch)
    out = _run_sidecar(
        _GRANDCHILD.format(live=_LIVE_TREE),
        {"HERMES_KANBAN_TASK": own},
        tmp_path,
        marker=str(tmp_path),
    )
    assert "OUT GRANT none" in out, out
