"""RED-first behavioral tests for the ask_user_questions.respond RPC.

Contract (mirrors the clarify.respond late-answer family):
  ask_user_questions.respond {request_id, answers} -> {status: ok|expired}

The TUI (ui-tui/src/app/useMainApp.ts) sends ``answers`` as the params key
and blocks the agent thread in agent_callbacks via ``_block`` keyed by the
request_id emitted on ``ask_user_questions.request``.  Before the fix the
method was never registered → JSON-RPC -32601 and the agent hung until the
clarify timeout.
"""

import threading

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture()
def server():
    """Import the gateway server with sys.modules mocks (test_protocol.py pattern)."""
    with patch.dict(
        "sys.modules",
        {
            "hermes_constants": MagicMock(get_hermes_home=MagicMock(return_value="/tmp/hermes_test_auq_rpc")),
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
            "hermes_state": MagicMock(),
        },
    ):
        import importlib

        mod = importlib.import_module("tui_gateway.server")

    methods = dict(mod._methods)
    real_stdout = mod._real_stdout
    yield mod
    mod._methods.clear()
    mod._methods.update(methods)
    mod._real_stdout = real_stdout
    for sid in list(mod._sessions):
        mod._close_session_by_id(sid, end_reason="test_cleanup")
    mod._pending.clear()
    mod._answers.clear()


@pytest.fixture()
def auq_server(server):
    """Server with one live session, like a real ask_user_questions turn."""
    server._sessions["auq-sid"] = {
        "agent": MagicMock(),
        "agent_error": None,
        "agent_ready": None,
    }
    return server


def call(server, method, params, rid="r1"):
    handler = server._methods[method]
    return handler(rid, params)


def block_in_thread(server, sid, questions):
    """Create a pending entry exactly the way agent_callbacks._block does."""
    result = {}

    def run():
        result["answer"] = server._block(
            "ask_user_questions.request", sid, {"questions": questions}, timeout=5
        )

    t = threading.Thread(target=run, daemon=True)
    t.start()
    # _block publishes the rid into the event payload via _emit; recover it
    # from the pending registry instead (there is exactly one).  Poll like
    # _drain_batch_block does — thread.start() does not synchronously register.
    import time

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        with server._prompt_lock:
            if server._pending:
                return t, next(iter(server._pending)), result
        time.sleep(0.01)
    raise AssertionError("ask_user_questions request never registered")


_QUESTIONS = [
    {"question": "Scope?", "options": [{"label": "Full"}, {"label": "Partial"}], "header": "SCOPE"},
    {"question": "Model?", "options": [{"label": "A"}, {"label": "B"}], "header": "MODEL"},
]


class TestAuqRespondRpc:
    def test_method_is_registered(self, auq_server):
        assert "ask_user_questions.respond" in auq_server._methods

    def test_resolves_pending_block_with_answers(self, auq_server):
        t, rid, result = block_in_thread(auq_server, "auq-sid", _QUESTIONS)
        answers = {0: "Full", 1: "B"}
        resp = call(auq_server, "ask_user_questions.respond", {"answers": answers, "request_id": rid})
        assert resp["result"]["status"] == "ok"
        t.join(timeout=5)
        assert not t.is_alive()
        # _block returns the _answers entry — whatever params["answers"] carried.
        assert result["answer"] == answers

    def test_late_answer_after_expiry_is_tolerated(self, auq_server):
        # allow_expired=True: a tool's bounded wait expired while the card was visible.
        resp = call(auq_server, "ask_user_questions.respond", {"answers": {}, "request_id": "gone"})
        assert resp["result"]["status"] == "expired"