"""``rollback.diff`` refuses a container-backed session like every other ``/rollback diff`` surface.

The host tree vs a host checkpoint is not the sandbox session's diff, so the RPC must not run
``mgr.diff`` once the manager classifies the task's backend as a container."""
import threading
import types

from tui_gateway import server


def test_rollback_diff_refuses_container_backed_session():
    seen = {}

    class _Mgr:
        enabled = True

        def unsupported_backend_reason(self, task_id="default"):
            seen["task_id"] = task_id
            return "Checkpoints are not taken for terminal.backend=docker: file paths belong to the container."

        def list_checkpoints(self, cwd):
            return [{"hash": "abc123"}]

        def diff(self, cwd, target):
            raise AssertionError("mgr.diff must not run for a container-backed session")

    server._sessions["sid"] = _session(
        agent=types.SimpleNamespace(_checkpoint_mgr=_Mgr()), session_key="container-session"
    )
    try:
        resp = server.handle_request(
            {"id": "1", "method": "rollback.diff", "params": {"session_id": "sid", "hash": "abc123"}}
        )
    finally:
        server._sessions.pop("sid", None)

    assert "result" not in resp
    assert "terminal.backend=docker" in resp["error"]["message"]
    assert seen["task_id"] == "container-session"


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
