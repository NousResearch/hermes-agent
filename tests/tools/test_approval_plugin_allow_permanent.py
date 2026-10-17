"""Plugin approve directives can opt out of the permanent allowlist.

A plugin escalation whose ``allow_permanent`` is false must be decided per call: an
"always" choice downgrades to session. Default plugins (no field) keep the documented
"always" behavior. Guardrail keys stay session-max regardless.
"""

from unittest import mock

from tools import approval as A


def _persist(choice, key, **kw):
    with mock.patch.object(A, "approve_session") as sess, \
            mock.patch.object(A, "approve_permanent") as perm, \
            mock.patch.object(A, "save_permanent_allowlist") as save:
        A._persist_choice("s-test", choice, [key], **kw)
    return sess.called, perm.called, save.called


def test_opted_out_always_is_session_only():
    sess, perm, save = _persist("always", "plugin_rule:jev:escalate:rm", allow_permanent=False)
    assert sess and not perm and not save


def test_default_always_still_persists_for_plugins():
    sess, perm, save = _persist("always", "plugin_rule:deploy:rule-a")
    assert sess and perm and save


def test_guardrail_always_stays_session_only():
    sess, perm, save = _persist("always", "guardrail: replace or remove crontab")
    assert sess and not perm and not save


def test_session_choice_unaffected_by_opt_out():
    sess, perm, save = _persist("session", "plugin_rule:jev:escalate:rm", allow_permanent=False)
    assert sess and not perm and not save


def test_request_tool_approval_forwards_opt_out_to_gate():
    with mock.patch.object(A, "_run_approval_gate", return_value={"approved": True}) as gate:
        A.request_tool_approval("terminal", "why", rule_key="jev:escalate:x", allow_permanent=False)
    assert gate.call_args.kwargs["allow_permanent"] is False


def test_request_tool_approval_default_is_permissive():
    with mock.patch.object(A, "_run_approval_gate", return_value={"approved": True}) as gate:
        A.request_tool_approval("terminal", "why", rule_key="deploy:rule-a")
    assert gate.call_args.kwargs["allow_permanent"] is True
