"""Human todo Mark done/Reopen JSON-RPC handlers (Desktop phase).

Authoritative human terminal actions over the live session agent's
TodoStore. The desktop composer status stack reads ``todo.snapshot`` when a
session binds and calls ``todo.update_status`` from the shared
ConfirmDialog; both return and emit full snapshots so the renderer replaces
its local view instead of patching it.

The model's own ``todo`` tool calls are untouched — this module only adds
the human-written path, keyed off the same agent._todo_store the tool
writes, so user terminal status stays authoritative against stale model
merges (TodoStore._user_status_overrides semantics).

Handlers are rebound onto server.py's globals at install time (see
method_ctx.py) — helper functions must stay NESTED inside the handler
bodies, because the rebind swaps handler __globals__ to server.py's
namespace where this module's names do not exist. See methods_images.py
for the same constraint.
"""

from .method_ctx import HandlerRegistry

_registry = HandlerRegistry()
method = _registry.method

# Machine-distinguishable gateway error codes for the todo contract. The
# renderer branches on these codes (not messages): STALE_REVISION carries the
# current snapshot in error.data so the client can roll forward instead of
# rolling back; ITEM_MISSING means the plan changed under the user. Mirrored
# as literals inside the handlers (globals are swapped at install time — the
# constants here are the readable contract, the literals are the live path).
STALE_REVISION = 4096
ITEM_MISSING = 4044


@method("todo.snapshot")
def _(rid, params: dict) -> dict:
    # Nested so install()'s __globals__ rebind can't lose them.
    def snapshot_payload(store, session_id):
        state = store.snapshot_state()
        return {
            "session_id": session_id,
            "revision": state["revision"],
            "generation": state["generation"],
            "todos": state["todos"],
        }

    sid = str(params.get("session_id") or "").strip()
    if not sid:
        return _err(rid, 4004, "session_id required")

    # _sess (not _sess_nowait): a session whose agent is still building must
    # wait for the built agent's _todo_store — a snapshot taken before the
    # build would report an empty plan for a session that has one. _sess
    # returns (session, err) with err a 4001 envelope for a missing session.
    session, err = _sess(params, rid)
    if err is not None:
        return err

    store = getattr(session.get("agent"), "_todo_store", None)
    if store is None:
        return _err(rid, 4004, "session has no todo store")

    return _ok(rid, snapshot_payload(store, sid))


@method("todo.update_status")
def _(rid, params: dict) -> dict:
    # Nested so install()'s __globals__ rebind can't lose them.
    def snapshot_payload(store, session_id):
        state = store.snapshot_state()
        return {
            "session_id": session_id,
            "revision": state["revision"],
            "generation": state["generation"],
            "todos": state["todos"],
        }

    # Exactly the two human terminal actions the composer exposes. The model
    # keeps its full four-state vocabulary through the todo tool; Mark done
    # and Reopen are the only human verbs and must not widen.
    ALLOWED_HUMAN_STATUSES = {"pending", "completed"}
    ALLOWED_ACTORS = {"user"}
    INVALID_PARAMS = 4004
    STALE = 4096
    MISSING = 4044

    sid = str(params.get("session_id") or "").strip()
    if not sid:
        return _err(rid, INVALID_PARAMS, "session_id required")

    session, err = _sess(params, rid)
    if err is not None:
        return err

    store = getattr(session.get("agent"), "_todo_store", None)
    if store is None:
        return _err(rid, INVALID_PARAMS, "session has no todo store")

    item_id = str(params.get("item_id") or "").strip()
    status = str(params.get("status") or "").strip().lower()
    actor = str(params.get("actor") or "").strip().lower()
    expected_revision = params.get("expected_revision")

    if not item_id or status not in ALLOWED_HUMAN_STATUSES or actor not in ALLOWED_ACTORS:
        return _err(
            rid,
            INVALID_PARAMS,
            "todo.update_status requires item_id, status in {pending, completed}, actor='user'",
        )

    if not isinstance(expected_revision, int) or isinstance(expected_revision, bool):
        return _err(rid, INVALID_PARAMS, "todo.update_status requires integer expected_revision")

    # CAS: recheck the expected revision atomically inside the store. A stale
    # request must fail LOUDLY (a silent success would complete a task from a
    # plan the user was no longer looking at), and it must carry the current
    # snapshot so the renderer can adopt it instead of guessing.
    if not store.update_status(item_id, status, actor=actor, expected_revision=expected_revision):
        current = snapshot_payload(store, sid)
        item_exists = any(t["id"] == item_id for t in current["todos"])
        if not item_exists:
            return _err(rid, MISSING, "todo item not found", data=current)
        return _err(rid, STALE, "stale revision: task list changed", data=current)

    snapshot = snapshot_payload(store, sid)
    # Dedicated full-snapshot event: every other surface (and late responses
    # racing this one) reconciles from this, keyed on generation.
    _emit("todo.updated", sid, snapshot)
    return _ok(rid, snapshot)


def register(server) -> None:
    _registry.install(server)
