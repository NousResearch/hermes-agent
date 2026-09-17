"""kanban_create tool-layer body guard (title-only cards are undispatchable junk).

DB-layer ``create_task`` stays permissive (CLI/dashboard title-only drafts are
legitimate); this guards the *worker tool* only. Regression for the 83-stub /
275-crashed-runs defect class (t_da696979 forensics, 2026-09-17): every leaked
fixture card landed body-less through ``_handle_create`` with no guard to stop it.
"""
import json

from tests.tools.test_kanban_tools import worker_env  # noqa: F401 — shared fixture


def _set_body(task_id: str, body: str) -> None:
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET body = ? WHERE id = ?", (body, task_id))


def _get(task_id: str):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        return kb.get_task(conn, task_id)


def test_refuses_bodyless_create_from_plain_session(tmp_path, monkeypatch):
    """A caller with no dispatcher-owned task gets a clear tool error, and
    nothing is written to the board."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from tools import kanban_tools as kt

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    out = json.loads(kt._handle_create({"title": "stub", "assignee": "default"}))
    assert out.get("error") and "body" in out["error"]
    with kbc.connect_closing() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_refuses_bodyless_create_from_bodyless_worker(worker_env):  # noqa: F811
    """A dispatcher-owned worker whose OWN task has no body cannot inherit one —
    the guard refuses rather than emitting another body-less card."""
    from tools import kanban_tools as kt

    out = json.loads(kt._handle_create({"title": "stub child", "assignee": "peer"}))
    assert out.get("error") and "body" in out["error"]


def test_triage_flag_parks_bodyless_card(worker_env):  # noqa: F811
    """triage=true is the explicit specifier path: body stays empty, card lands
    in triage, and a specifier fills it before promotion."""
    from tools import kanban_tools as kt

    out = json.loads(kt._handle_create({
        "title": "needs spec", "assignee": "default", "triage": True}))
    assert out["ok"], out
    assert out["body_inherited"] is False
    task = _get(out["task_id"])
    assert task is not None and task.status == "triage"
    assert (task.body or "").strip() == ""


def test_worker_create_inherits_own_task_body_with_provenance(worker_env):  # noqa: F811
    """A dispatcher-owned worker omitting body inherits its own task's body with
    a provenance note — the child is never body-less."""
    from tools import kanban_tools as kt

    _set_body(worker_env, "SPEC: do the thing.\nAcceptance: it is done.")
    out = json.loads(kt._handle_create({
        "title": "fan-out child", "assignee": "peer", "parents": [worker_env]}))
    assert out["ok"], out
    assert out["body_inherited"] is True
    task = _get(out["task_id"])
    assert task is not None
    assert task.body and worker_env in task.body
    assert task.body.startswith("> Inherited from")
    assert "SPEC: do the thing." in task.body


def test_explicit_body_beats_inheritance(worker_env):  # noqa: F811
    """An explicit body is used verbatim; nothing is inherited or prepended."""
    from tools import kanban_tools as kt

    _set_body(worker_env, "PARENT SPEC")
    out = json.loads(kt._handle_create({
        "title": "explicit body child", "assignee": "peer", "body": "CHILD SPEC"}))
    assert out["ok"], out
    assert out["body_inherited"] is False
    task = _get(out["task_id"])
    assert task is not None and task.body == "CHILD SPEC"


def test_blank_body_is_treated_as_missing(worker_env):  # noqa: F811
    """A whitespace-only body must not slip past the guard via .strip() truthiness."""
    from tools import kanban_tools as kt

    out = json.loads(kt._handle_create({
        "title": "blank body child", "assignee": "peer", "body": "   \n\t "}))
    assert out.get("error") and "body" in out["error"]


def test_db_layer_still_permits_title_only(worker_env):  # noqa: F811
    """Acceptance: dashboard/CLI title-only creation still works — the guard is a
    worker-tool guard, not a DB constraint."""
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="cli draft", assignee="default")
        task = kb.get_task(conn, tid)
    assert task is not None and task.body is None


def test_inherited_card_is_dispatch_ready(worker_env):  # noqa: F811
    """The inherited body flows through create_task's event path intact; the
    created card carries a spec the next worker can actually read."""
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from tools import kanban_tools as kt

    _set_body(worker_env, "SPEC: inherit me.")
    out = json.loads(kt._handle_create({
        "title": "event probe", "assignee": "peer", "parents": [worker_env]}))
    assert out["ok"], out
    with kbc.connect_closing() as conn:
        assert any(e.kind == "created" for e in kb.list_events(conn, out["task_id"]))
        task = kb.get_task(conn, out["task_id"])
    assert task is not None and "SPEC: inherit me." in (task.body or "")
