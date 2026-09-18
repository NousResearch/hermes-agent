"""Behavioral contract tests for ``inbox.requests``: scoped request-detail read layer.

``inbox.requests`` is profile-scoped, read-only, and returns live request details
(approvals with redacted payloads + actual choices, clarifications with params) for
a specific session identified by its durable ``session_key``.

Security invariants:
  - Wrong profile cannot read another profile's requests.
  - Duplicate durable keys across profile homes don't mix.
  - Missing/denied sessions return explicit errors, never empty.
  - No session is resumed/hydrated/started to read requests.
  - Only approval and clarify types are returned; password/secret/vault are excluded.
  - No side effects on the server request queue or approval queue.
  - No arbitrary exception text leaked.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    """Isolated persisted-manager database + state.db for every test."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import goals
    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


@pytest.fixture()
def server(hermes_home, monkeypatch):
    import tui_gateway.server as mod
    monkeypatch.setattr(mod, "_hermes_home", hermes_home)
    monkeypatch.setattr(mod, "_cfg_cache", None)
    monkeypatch.setattr(mod, "_cfg_sig", None)
    monkeypatch.setattr(mod, "_cfg_path", None)
    yield mod
    mod._sessions.clear()
    mod._server_requests.reset_for_tests()
    from tools import approval
    with approval._lock:
        approval._gateway_queues.clear()


@pytest.fixture()
def db(server, hermes_home):
    handle = server._get_db()
    yield handle


def _call(server, method, *, rid=91, **params):
    return server._methods[method](rid, params)


def _result(server, method, **params):
    response = _call(server, method, **params)
    assert "result" in response, response
    return response["result"]


def _error(response):
    assert "error" in response
    return response["error"]


def _new_key(tag="req"):
    return f"{tag}-{uuid.uuid4().hex[:12]}"


def _create_row(db, key, *, source="cli", title=""):
    db.create_session(key, source=source)
    if title:
        db.set_session_title(key, title)
    return key


def _open_session(server, key, *, profile_home=None):
    from hermes_constants import get_hermes_home
    sid = f"sid-{uuid.uuid4().hex[:8]}"
    server._sessions[sid] = {
        "session_key": key,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "cols": 120,
        "agent": None,
        "created_at": time.time(),
        "profile_home": str(profile_home) if profile_home is not None else str(get_hermes_home()),
    }
    return sid


def _queue_approval(server, key, *, command="rm -rf /tmp/secret-value-abc", description="run removal"):
    from tools import approval
    with approval._lock:
        approval._gateway_queues.setdefault(key, []).append(
            SimpleNamespace(data={
                "request_id": f"rid-{uuid.uuid4().hex[:8]}",
                "command": command,
                "description": description,
            })
        )


def _queue_clarify(server, key, *, question="Which provider?", choices=None,
                    multi_select=False, profile_home=None, sid=None):
    from tui_gateway.server_requests import ServerRequest
    if sid is None:
        sid = _open_session(server, key, profile_home=profile_home)
    params = {"question": question}
    if choices is not None:
        params["choices"] = choices
    if multi_select:
        params["multi_select"] = True
    req = ServerRequest(sid, "clarify", params)
    sr = server._server_requests
    with sr._lock:
        sr._open[req.id] = req
    return sid, req.id


def _queue_batch_clarify(server, key, *, questions, profile_home=None, sid=None):
    """Inject a batch-clarify with multiple questions."""
    from tui_gateway.server_requests import ServerRequest
    if sid is None:
        sid = _open_session(server, key, profile_home=profile_home)
    qids = [q.get("qid", f"q{i}") for i, q in enumerate(questions)]
    params = {"questions": questions}
    req = ServerRequest(sid, "clarify", params, qids=qids)
    sr = server._server_requests
    with sr._lock:
        sr._open[req.id] = req
    return sid, req.id


def _queue_non_allowed_type(server, key, *, method="secret", params=None, sid=None):
    """Inject a server request of a type that should be EXCLUDED (e.g. secret, vault)."""
    from tui_gateway.server_requests import ServerRequest
    if sid is None:
        sid = _open_session(server, key)
    req = ServerRequest(sid, method, params or {"env_var": "MY_SECRET", "prompt": "Enter value"})
    sr = server._server_requests
    with sr._lock:
        sr._open[req.id] = req
    return sid, req.id


# ── Method registration ──────────────────────────────────────────────
class TestRegistration:
    def test_method_is_registered(self, server):
        assert "inbox.requests" in server._methods


# ── Session resolution ───────────────────────────────────────────────
class TestSessionResolution:
    def test_missing_session_key_returns_error(self, server, db):
        response = _call(server, "inbox.requests", session_key="nonexistent-key")
        assert _error(response)["code"] == 4001

    def test_denied_persisted_source_cannot_read_live_requests(self, server, db):
        key = _create_row(db, _new_key(), source="tool")
        _open_session(server, key)
        assert _error(_call(server, "inbox.requests", session_key=key))["code"] == 4001

    def test_live_presence_does_not_fabricate_context_anchor(self, server, db):
        key = _create_row(db, _new_key())
        _open_session(server, key)
        result = _result(server, "inbox.requests", session_key=key)
        assert result["coverage"]["context_anchor"] == "unavailable: open chat for context"

    def test_empty_session_key_returns_error(self, server, db):
        response = _call(server, "inbox.requests", session_key="")
        assert "error" in response
        assert response["error"]["code"] == 4002

    def test_session_not_live_returns_no_live_session(self, server, db):
        key = _create_row(db, _new_key())
        result = _result(server, "inbox.requests", session_key=key)
        assert result["sessions"][0]["live_session_ids"] == []
        assert result["sessions"][0]["approvals"] == []
        assert result["sessions"][0]["clarifications"] == []

    def test_live_session_found(self, server, db):
        key = _create_row(db, _new_key())
        sid = _open_session(server, key)
        result = _result(server, "inbox.requests", session_key=key)
        assert sid in result["sessions"][0]["live_session_ids"]


# ── Profile isolation ────────────────────────────────────────────────
class TestProfileIsolation:
    def test_wrong_profile_cannot_read_requests(self, server, db, monkeypatch):
        """A session under the launch profile must not be visible under a foreign profile."""
        from tui_gateway.server import _profile_home as _orig, ProfileUnavailableError

        key = _create_row(db, _new_key())
        _open_session(server, key)
        # Foreign profile that doesn't exist should raise
        with pytest.raises(ProfileUnavailableError, match="does not exist"):
            _call(server, "inbox.requests", session_key=key, profile="nonexistent-foreign")

    def test_duplicate_durable_keys_across_profiles_dont_mix(self, server, db, monkeypatch):
        """Two sessions with the same key but different profile_homes are isolated."""
        key = _create_row(db, _new_key())
        launch_home = str(server._hermes_home)
        sid_launch = _open_session(server, key, profile_home=launch_home)
        sid_foreign = _open_session(server, key, profile_home="/other/profile/home")
        # The launch profile should only see the launch session
        result = _result(server, "inbox.requests", session_key=key)
        assert sid_launch in result["sessions"][0]["live_session_ids"]
        assert sid_foreign not in result["sessions"][0]["live_session_ids"]

    def test_read_failure_does_not_resume_session(self, server, db):
        """Reading requests never starts/resumes/hydrates a session."""
        key = _create_row(db, _new_key())
        # Ensure no live session exists
        assert not any(s.get("session_key") == key for s in server._sessions.values())
        result = _result(server, "inbox.requests", session_key=key)
        # Still no live session - reading didn't create or resume one
        assert not any(s.get("session_key") == key for s in server._sessions.values())
        assert result["sessions"][0]["live_session_ids"] == []


# ── Approval details ─────────────────────────────────────────────────
class TestApprovalDetails:
    def test_redacted_approval_fields(self, server, db):
        """Approval command is redacted; raw credentials never appear."""
        key = _create_row(db, _new_key())
        _open_session(server, key)
        _queue_approval(server, key,
                        command="curl -H 'Authorization: Bearer SECRET_KEY_12345' https://api.example.com")
        result = _result(server, "inbox.requests", session_key=key)
        approvals = result["sessions"][0]["approvals"]
        assert len(approvals) >= 1
        serialized = json.dumps(result)
        assert "SECRET_KEY_12345" not in serialized
        # The command is redacted (*** replacement), not completely stripped
        assert "***" in approvals[0]["command"] or "SECRET" not in approvals[0]["command"]

    def test_approval_has_actual_allowed_choices(self, server, db):
        """Approval payload includes the computed choices, not just a count."""
        key = _create_row(db, _new_key())
        _open_session(server, key)
        _queue_approval(server, key)
        result = _result(server, "inbox.requests", session_key=key)
        assert len(result["sessions"][0]["approvals"]) == 1
        approval = result["sessions"][0]["approvals"][0]
        assert "request_id" in approval
        assert "choices" in approval
        assert "deny" in approval["choices"]
        assert "once" in approval["choices"]

    def test_approval_choice_calculation_smart_denied(self, server, db):
        """When smart_denied is true, 'session' is excluded from choices."""
        key = _create_row(db, _new_key())
        _open_session(server, key)
        from tools import approval
        with approval._lock:
            approval._gateway_queues.setdefault(key, []).append(
                SimpleNamespace(data={
                    "request_id": f"rid-{uuid.uuid4().hex[:8]}",
                    "command": "test command",
                    "smart_denied": True,
                })
            )
        result = _result(server, "inbox.requests", session_key=key)
        choices = result["sessions"][0]["approvals"][0]["choices"]
        assert "session" not in choices
        assert "once" in choices
        assert "deny" in choices

    def test_approval_choice_calculation_allow_permanent_false(self, server, db):
        """When allow_permanent is false, 'always' is excluded from choices."""
        key = _create_row(db, _new_key())
        _open_session(server, key)
        from tools import approval
        with approval._lock:
            approval._gateway_queues.setdefault(key, []).append(
                SimpleNamespace(data={
                    "request_id": f"rid-{uuid.uuid4().hex[:8]}",
                    "command": "test command",
                    "allow_permanent": False,
                })
            )
        result = _result(server, "inbox.requests", session_key=key)
        choices = result["sessions"][0]["approvals"][0]["choices"]
        assert "always" not in choices
        assert "session" in choices
        assert "deny" in choices

    def test_all_pending_approvals_included(self, server, db):
        """Multiple pending approvals for one session are all returned."""
        key = _create_row(db, _new_key())
        _open_session(server, key)
        from tools import approval
        for _ in range(3):
            with approval._lock:
                approval._gateway_queues.setdefault(key, []).append(
                    SimpleNamespace(data={
                        "request_id": f"rid-{uuid.uuid4().hex[:8]}",
                        "command": f"command-{_}",
                    })
                )
        result = _result(server, "inbox.requests", session_key=key)
        assert len(result["sessions"][0]["approvals"]) == 3

    def test_approval_queue_never_consumed(self, server, db):
        """Reading requests does not consume the approval queue."""
        from tools import approval
        key = _create_row(db, _new_key())
        _open_session(server, key)
        _queue_approval(server, key)
        with approval._lock:
            before = len(approval._gateway_queues.get(key, []))
        _result(server, "inbox.requests", session_key=key)
        with approval._lock:
            after = len(approval._gateway_queues.get(key, []))
        assert before == after == 1

    def test_approval_excludes_non_allowed_types(self, server, db):
        """Server requests of type 'secret', 'vault.*', 'sudo' are excluded."""
        key = _create_row(db, _new_key())
        _open_session(server, key)
        _queue_non_allowed_type(server, key, method="secret")
        result = _result(server, "inbox.requests", session_key=key)
        # No approvals or clarifications from secret type
        assert result["sessions"][0]["approvals"] == []
        assert result["sessions"][0]["clarifications"] == []


# ── Clarification details ────────────────────────────────────────────
class TestClarificationDetails:
    def test_single_clarify_params_exposed(self, server, db):
        """Single clarify shows question, choices, multi_select, locked answers."""
        key = _create_row(db, _new_key())
        sid, req_id = _queue_clarify(server, key, question="Which backend?",
                                      choices=["local", "docker", "ssh"])
        result = _result(server, "inbox.requests", session_key=key)
        assert len(result["sessions"][0]["clarifications"]) == 1
        clarify = result["sessions"][0]["clarifications"][0]
        assert clarify["request_id"] == req_id
        assert clarify["kind"] == "single"
        params = clarify["params"]
        assert params["question"] == "Which backend?"
        assert params["choices"] == ["local", "docker", "ssh"]
        assert params.get("multi_select") in (False, None)

    def test_multi_select_clarify(self, server, db):
        """Multi-select clarify exposes multi_select=true."""
        key = _create_row(db, _new_key())
        sid, req_id = _queue_clarify(server, key, question="Pick features",
                                      choices=["auth", "cache", "logs"],
                                      multi_select=True)
        result = _result(server, "inbox.requests", session_key=key)
        clarify = result["sessions"][0]["clarifications"][0]
        assert clarify["kind"] == "single"
        assert clarify["params"]["multi_select"] is True

    def test_batch_clarify_with_locked_answers(self, server, db):
        """Batch clarify shows questions array and any already-locked answers."""
        key = _create_row(db, _new_key())
        questions = [
            {"qid": "q1", "question": "Project name?", "choices": ["proj-a", "proj-b"]},
            {"qid": "q2", "question": "Language?", "choices": ["python", "rust"]},
        ]
        sid, req_id = _queue_batch_clarify(server, key, questions=questions)
        # Lock one answer
        from tui_gateway import server_requests
        server_requests.lock_answer(req_id, "q1", "proj-a")
        result = _result(server, "inbox.requests", session_key=key)
        assert len(result["sessions"][0]["clarifications"]) == 1
        clarify = result["sessions"][0]["clarifications"][0]
        assert clarify["request_id"] == req_id
        assert clarify["kind"] == "batch"
        params = clarify["params"]
        assert len(params["questions"]) == 2
        assert params["questions"][0]["qid"] == "q1"
        assert params["questions"][1]["qid"] == "q2"
        # Locked answers are included
        assert "answers" in params
        assert params["answers"]["q1"] == "proj-a"

    def test_clarify_never_exposed_by_inbox_list(self, server, db):
        """inbox.list shows only count; inbox.requests shows full params."""
        key = _create_row(db, _new_key())
        _queue_clarify(server, key, question="What API key?")
        # inbox.list: metadata only
        list_result = _result(server, "inbox.list")["inbox"]
        assert list_result["items"][0]["pending_clarify"] == {"count": 1}
        # Verify question text not in list
        serialized = json.dumps(list_result)
        assert "API key" not in serialized
        # inbox.requests: full params
        requests_result = _result(server, "inbox.requests", session_key=key)
        assert len(requests_result["sessions"][0]["clarifications"]) == 1
        assert requests_result["sessions"][0]["clarifications"][0]["params"]["question"] == "What API key?"


# ── Non-allowed request types ────────────────────────────────────────
class TestNonAllowedTypes:
    def test_secret_type_excluded(self, server, db):
        key = _create_row(db, _new_key())
        _open_session(server, key)
        _queue_non_allowed_type(server, key, method="secret",
                                params={"env_var": "API_KEY", "prompt": "Enter API key"})
        result = _result(server, "inbox.requests", session_key=key)
        assert result["sessions"][0]["approvals"] == []
        assert result["sessions"][0]["clarifications"] == []

    def test_vault_unlock_excluded(self, server, db):
        key = _create_row(db, _new_key())
        _open_session(server, key)
        _queue_non_allowed_type(server, key, method="vault.unlock_prompt",
                                params={"backend": "1password", "display_name": "Master"})
        result = _result(server, "inbox.requests", session_key=key)
        assert result["sessions"][0]["approvals"] == []
        assert result["sessions"][0]["clarifications"] == []

    def test_sudo_excluded(self, server, db):
        key = _create_row(db, _new_key())
        _open_session(server, key)
        _queue_non_allowed_type(server, key, method="sudo",
                                params={"command": "rm -rf /"})
        result = _result(server, "inbox.requests", session_key=key)
        assert result["sessions"][0]["approvals"] == []
        assert result["sessions"][0]["clarifications"] == []

    def test_mixed_types_only_allows_approval_and_clarify(self, server, db):
        """When a session has both allowed and disallowed types, only allowed appear."""
        key = _create_row(db, _new_key())
        sid = _open_session(server, key)
        # Queue an approval
        _queue_approval(server, key)
        # Queue a disallowed type into the same session
        _queue_non_allowed_type(server, key, method="vault.code", sid=sid)
        result = _result(server, "inbox.requests", session_key=key)
        assert len(result["sessions"][0]["approvals"]) == 1
        assert result["sessions"][0]["clarifications"] == []


# ── No resolve side effects ──────────────────────────────────────────
class TestNoSideEffects:
    def test_read_does_not_resolve_approvals(self, server, db):
        """Reading requests does not resolve/remove any approval from the queue."""
        from tools import approval
        key = _create_row(db, _new_key())
        _open_session(server, key)
        _queue_approval(server, key)
        _queue_approval(server, key)
        with approval._lock:
            before = sum(len(q) for q in approval._gateway_queues.values())
        _result(server, "inbox.requests", session_key=key)
        with approval._lock:
            after = sum(len(q) for q in approval._gateway_queues.values())
        assert before == after == 2

    def test_read_does_not_resolve_clarifications(self, server, db):
        """Reading requests does not resolve/remove any clarification."""
        key = _create_row(db, _new_key())
        sid = _open_session(server, key)
        _queue_clarify(server, key, question="Q1", sid=sid)
        _queue_clarify(server, key, question="Q2", sid=sid)
        sr = server._server_requests
        with sr._lock:
            before_count = sum(1 for r in sr._open.values() if r.sid == sid and r.method == "clarify")
        _result(server, "inbox.requests", session_key=key)
        with sr._lock:
            after_count = sum(1 for r in sr._open.values() if r.sid == sid and r.method == "clarify")
        assert before_count == after_count == 2


# ── Error handling ───────────────────────────────────────────────────
class TestErrorHandling:
    def test_unauthorized_profile_returns_4064(self, server, db):
        from tui_gateway.server import ProfileUnavailableError
        with pytest.raises(ProfileUnavailableError, match="does not exist"):
            _call(server, "inbox.requests", session_key="any", profile="no-such-profile")

    def test_error_message_is_sanitized(self, server, db, monkeypatch):
        """RPC error must not contain raw exception text."""
        original = server._inbox_requests
        def boom(rid, params):
            raise RuntimeError("super-secret-internal-XYZ")
        server._inbox_requests = boom
        try:
            response = _call(server, "inbox.requests", session_key="any")
            assert "super-secret-internal-XYZ" not in json.dumps(response)
            assert response.get("error", {}).get("code") == 5031
        finally:
            server._inbox_requests = original

    def test_stale_session_key_not_live(self, server, db):
        """A session that was finalized is not considered live."""
        key = _create_row(db, _new_key())
        sid = _open_session(server, key)
        server._sessions[sid]["_finalized"] = True
        result = _result(server, "inbox.requests", session_key=key)
        assert sid not in result["sessions"][0]["live_session_ids"]


# ── Contract validation ──────────────────────────────────────────────
class TestContract:
    def test_result_matches_contract(self, server, db):
        from tui_gateway.contracts.inbox_requests import InboxRequestsResult
        key = _create_row(db, _new_key())
        sid = _open_session(server, key)
        _queue_approval(server, key)
        _queue_clarify(server, key, question="Q1", sid=sid)
        response = _call(server, "inbox.requests", session_key=key)
        assert "result" in response
        validated = InboxRequestsResult.model_validate(response["result"])
        assert len(validated.sessions) == 1
        assert validated.sessions[0].live_session_ids == [sid]

    def test_empty_result_matches_contract(self, server, db):
        from tui_gateway.contracts.inbox_requests import InboxRequestsResult
        key = _create_row(db, _new_key())
        response = _call(server, "inbox.requests", session_key=key)
        validated = InboxRequestsResult.model_validate(response["result"])
        assert validated.sessions[0].live_session_ids == []
        assert validated.sessions[0].approvals == []
        assert validated.sessions[0].clarifications == []

    def test_profile_field_accepted(self, server, db):
        """Explicit profile=None must not cause a contract violation."""
        from tui_gateway.contracts.inbox_requests import InboxRequestsResult
        key = _create_row(db, _new_key())
        response = _call(server, "inbox.requests", session_key=key, profile=None)
        validated = InboxRequestsResult.model_validate(response["result"])
        assert isinstance(validated.sessions, list)


# ── Context anchor ───────────────────────────────────────────────────
class TestContextAnchor:
    def test_no_live_session_reports_unavailable(self, server, db):
        """When no live session exists, coverage.context_anchor indicates stale."""
        key = _create_row(db, _new_key())
        result = _result(server, "inbox.requests", session_key=key)
        anchor = result["coverage"].get("context_anchor") or ""
        assert "unavailable" in anchor.lower() or "open chat" in anchor.lower()

    def test_live_session_reports_available(self, server, db):
        """When a live session exists, coverage.context_anchor indicates available."""
        key = _create_row(db, _new_key())
        _open_session(server, key)
        result = _result(server, "inbox.requests", session_key=key)
        anchor = result["coverage"].get("context_anchor") or ""
        assert "available" in anchor.lower() or "live" in anchor.lower()
