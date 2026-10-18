"""Live request evidence survives only its admitted task and reserved children."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from types import SimpleNamespace
from unittest.mock import patch

from tools.approval_context import (approval_task_context, background_approval_task,
                                    current_approval_task_context, delegated_approval_task,
                                    reserve_background_approval_task)
from tools.approval_smart import _smart_approve


def _reply(answer="APPROVE"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer))])


def test_reviewer_receives_quoted_request_and_child_ancestry():
    with patch("agent.auxiliary_client.call_llm", return_value=_reply()) as call_llm, \
         patch("tools.approval_context._get_approval_config", return_value={}):
        with approval_task_context("Edit only src/cli.py; no tests.", identity="user-7",
                                   profile="p", session="s", task_id="a", run_id="3",
                                   is_current=lambda: True):
            with delegated_approval_task("child-1"):
                assert _smart_approve("python -c 'print(1)'", "script execution") == "approve"
    sent = call_llm.call_args.kwargs["messages"]
    assert "Edit only src/cli.py; no tests." in sent[1]["content"]
    assert '"ancestry": ["child-1"]' in sent[1]["content"]
    assert "operator rules" in sent[0]["content"]


def test_copied_context_is_revoked_on_stop_and_late_approve_is_unusable():
    owned = []
    with patch("agent.auxiliary_client.call_llm", side_effect=lambda **_: (owned[0].revoke(), _reply())[1]), \
         patch("tools.approval_context._get_approval_config", return_value={}):
        with approval_task_context("Edit src/cli.py", identity="u", profile="p", session="s",
                                   is_current=lambda: True, owner=owned.append):
            copied = copy_context()
            assert _smart_approve("python -c 'print(1)'", "script execution") == "deny"
            with ThreadPoolExecutor(max_workers=1) as pool:
                assert pool.submit(copied.run, current_approval_task_context).result() is None


def test_detached_child_ends_with_its_reservation():
    owned = []
    with approval_task_context("Edit src/cli.py", identity="u", profile="p", session="s",
                               is_current=lambda: True, owner=owned.append):
        reservation = reserve_background_approval_task()
    with background_approval_task(reservation, "unit-1"):
        with delegated_approval_task("child-1"):
            assert current_approval_task_context().ancestry == ("unit-1", "child-1")
    assert not owned[0].live()


def test_real_child_worker_receives_narrowed_request(monkeypatch):
    from tools.delegate_tool_child_run import _ChildRun

    observations = []

    def conversation(**_kwargs):
        evidence = current_approval_task_context()
        observations.append((evidence.text, evidence.ancestry))
        return {"final_response": "done"}

    child = SimpleNamespace(session_id="child-session", run_conversation=conversation)
    run = _ChildRun(child=child, parent_agent=SimpleNamespace(), task_index=0, goal="Edit src/cli.py",
                    subagent_id=None, child_progress_cb=None, child_task_id="child-real")
    monkeypatch.setattr("tools.delegate_tool_config._get_child_timeout", lambda: None)
    with approval_task_context("Edit src/cli.py", identity="u", profile="p", session="s",
                               is_current=lambda: True):
        result, error, deferred = run.await_child()
    assert result["final_response"] == "done" and error is None and not deferred
    assert observations == [("Edit src/cli.py", ("child-real",))]


def test_background_unit_retains_request_after_parent_returns(monkeypatch):
    from tools.delegate_tool_dispatch import _dispatch_unit

    accepted = {}
    observed = []

    def dispatch(**kwargs):
        accepted.update(kwargs)
        return {"status": "dispatched", "delegation_id": "unit-1"}

    monkeypatch.setattr("tools.async_delegation.dispatch_async_delegation_batch", dispatch)
    monkeypatch.setattr("tools.delegate_tool_dispatch._execute_and_aggregate",
                        lambda *_a, **_k: observed.append(current_approval_task_context().ancestry) or {})
    unit = SimpleNamespace(children=[(0, {"goal": "work"}, SimpleNamespace())],
                           task_list=[{"goal": "work"}], context=None, top_role="worker",
                           creds={"model": "fixture"}, live_writers=[], live_home=None)
    routing = dict(session_key="s", origin_ui_session_id="", origin_session_id="s",
                   parent_session_id="s", max_async_children=1)
    with approval_task_context("Edit src/cli.py", identity="u", profile="p", session="s",
                               is_current=lambda: True):
        assert _dispatch_unit(unit, "unit-1", None, routing)["status"] == "dispatched"
    accepted["runner"]()
    assert observed == [("unit-1",)]


def test_stop_before_detached_child_starts_removes_request():
    leases = []
    with approval_task_context("Edit src/cli.py", identity="u", profile="p", session="s",
                               is_current=lambda: True, owner=leases.append):
        reservation = reserve_background_approval_task()
    leases[0].revoke()
    with background_approval_task(reservation, "late-unit"):
        with delegated_approval_task("late-child"):
            assert current_approval_task_context() is None


def test_revoked_parent_child_keeps_hard_denial(monkeypatch):
    from tools import approval as A, approval_context
    monkeypatch.setattr(approval_context, '_get_approval_mode', lambda: 'off')
    leases = []
    with approval_task_context('Edit src/cli.py', identity='u', profile='p', session='s',
                               is_current=lambda: True, owner=leases.append):
        leases[0].revoke()
        with delegated_approval_task('late-child'):
            assert approval_context.approval_task_revoked()
            assert A.check_all_command_guards('python -c "print(1)"', 'local')['approved'] is False


def test_background_reservation_release_is_idempotent_and_copied_child_stales():
    from tools.approval_context import release_background_approval_task, approval_task_revoked
    with approval_task_context('Edit src/cli.py', identity='u', profile='p', session='s',
                               is_current=lambda: True):
        first = reserve_background_approval_task()
        second = reserve_background_approval_task()
    with background_approval_task(first, 'first'):
        copied = copy_context()
    release_background_approval_task(first)
    assert copied.run(approval_task_revoked) is True
    assert second.lease.live()
    release_background_approval_task(second)
    release_background_approval_task(second)
    assert not second.lease.live()


def test_copied_parent_binding_cannot_borrow_live_detached_child(monkeypatch):
    from tools import approval as A, approval_context
    monkeypatch.setattr(approval_context, '_get_approval_mode', lambda: 'off')
    with approval_task_context('Edit only alpha.py', identity='u', profile='p', session='s',
                               is_current=lambda: True):
        copied_parent = copy_context()
        child = reserve_background_approval_task()
    try:
        assert child.lease.live()
        assert copied_parent.run(current_approval_task_context) is None
        assert copied_parent.run(approval_context.approval_task_revoked)
        assert copied_parent.run(lambda: A.check_all_command_guards(
            'python -c "print(1)"', 'local')['approved']) is False
        with background_approval_task(child, 'child'):
            assert current_approval_task_context().text == 'Edit only alpha.py'
    finally:
        child.release()


def test_coalesced_completions_with_different_origins_cannot_borrow_scope():
    from tools.approval_context import completion_approval_task, approval_task_revoked
    with approval_task_context('Edit only first.py', identity='alice', profile='p', session='s',
                               is_current=lambda: True):
        first = reserve_background_approval_task()
    with approval_task_context('Edit only second.py', identity='bob', profile='p', session='s',
                               is_current=lambda: True):
        second = reserve_background_approval_task()
    with completion_approval_task([first, second], 'batch'):
        assert current_approval_task_context() is None
        assert approval_task_revoked()
    first.release()
    second.release()


def test_concurrent_request_profiles_do_not_mix():
    def review(identity, profile):
        with approval_task_context(f"Task for {profile}", identity=identity, profile=profile,
                                   session=profile, is_current=lambda: True):
            return current_approval_task_context()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(review, "alice", "profile-a")
        second = pool.submit(review, "bob", "profile-b")
        assert (first.result().identity, first.result().profile) == ("alice", "profile-a")
        assert (second.result().identity, second.result().profile) == ("bob", "profile-b")
    assert current_approval_task_context() is None


def test_stale_task_cannot_reuse_off_mode_or_session_approval(monkeypatch):
    from tools import approval as A, approval_context

    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "off")
    leases = []
    with approval_task_context("Edit src/cli.py", identity="u", profile="p", session="s",
                               is_current=lambda: True, owner=leases.append):
        leases[0].revoke()
        assert A.check_all_command_guards("python -c 'print(1)'", "local")["approved"] is False
        assert A.check_execute_code_guard("print(1)", "local")["approved"] is False
        assert A.request_tool_approval("write", "change file")["approved"] is False
