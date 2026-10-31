"""The sidecar's whole-tree ``rollback.restore`` keeps the user's hand-edits like the owner's.

``CheckpointManager.restore(safe=True)`` unless the client sends ``all: true`` (main's ``--all``); a
file-scoped restore is never ``safe``. Omitted, null and false ``all`` all mean the safe restore."""
import threading
import types

from tui_gateway import server


def test_rollback_restore_keeps_hand_edits_unless_all():
    calls = []

    class _Mgr:
        enabled = True
        unsupported_backend_reason = staticmethod(lambda task_id="default": None)

        def list_checkpoints(self, cwd):
            return [{"hash": "aaa111"}]

        def restore(self, cwd, target, file_path=None, safe=False):
            calls.append((file_path, safe))
            return {"success": True, "message": "done"}

    try:
        for extra in ({}, {"all": None}, {"all": False}, {"all": True}, {"file_path": "a.py"}):
            server._sessions["sid"] = _session(types.SimpleNamespace(_checkpoint_mgr=_Mgr()))
            resp = server.handle_request({"id": "1", "method": "rollback.restore",
                                          "params": {"session_id": "sid", "hash": "1", **extra}})
            assert resp.get("result", {}).get("success") is True, resp
    finally:
        server._sessions.pop("sid", None)
    assert calls == [(None, True), (None, True), (None, True), (None, False), ("a.py", False)]


def test_rollback_restore_refuses_a_non_boolean_all_before_touching_anything():
    """``all: "false"`` or ``all: 1`` is not ``true``: refused by name, like the owner, never read as
    an override. Nothing is restored and the history is not rewound."""
    calls = []

    class _Mgr:
        enabled = True
        unsupported_backend_reason = staticmethod(lambda task_id="default": None)

        def list_checkpoints(self, cwd):
            return [{"hash": "aaa111"}]

        def restore(self, cwd, target, file_path=None, safe=False):
            calls.append((file_path, safe))
            return {"success": True, "message": "done"}

    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
    try:
        for value in ("false", "true", 1, 0, [], {}):
            for extra in ({}, {"file_path": "a.py"}):
                session = _session(types.SimpleNamespace(_checkpoint_mgr=_Mgr()), history=list(history))
                server._sessions["sid"] = session
                resp = server.handle_request({"id": "1", "method": "rollback.restore",
                                              "params": {"session_id": "sid", "hash": "1", "all": value, **extra}})
                assert resp.get("error", {}).get("data") == {
                    "reason": "invalid_params", "field": "all", "detail": "must be a boolean"}, (value, resp)
                assert session["history"] == history
    finally:
        server._sessions.pop("sid", None)
    assert calls == []


def _session(agent, **extra):
    return {
        "agent": agent,
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        **extra,
    }
