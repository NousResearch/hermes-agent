"""LOCAL PATCH tests: guardrail patterns + approval_prefilter hook (see fix/local-patches).

Guardrail findings (commands that disable the approval system itself) must skip the smart
guardian and never persist to the permanent allowlist. The approval_prefilter hook may approve
or escalate ahead of the guardian LLM, never deny, and any failure falls through to the LLM.
"""

from unittest.mock import patch

import pytest

from tools import approval as A
from tools import approval_smart as S
from tools.approval_detection import detect_dangerous_command, is_guardrail_key

GUARDRAIL = [
    'echo "" > ~/.hermes/.env',
    "truncate -s0 ~/.hermes/.env",
    "rm ~/.hermes/.env",
    "cp /tmp/x ~/.hermes/.env",
    "hermes config set security.tirith.enabled false",
    "hermes config set approvals.mode off",
    "hermes config set plugins.hermes-jev.settings.approval_prefilter on",
    "hermes config set plugins.entries.hermes-jev.settings.approval_prefilter off",
    'sed -i "s/mode: smart/mode: off/" ~/.hermes/config.yaml',
    "python3 -c \"import yaml;c=yaml.safe_load(open('/x'));c['approvals']['mode']='off';"
    "yaml.safe_dump(c,open('$HOME/.hermes/config.yaml','w'))\"",
    "systemctl --user disable --now hermes-gateway.service",
    "systemctl --user mask hermes-gateway",
    'pkill -9 -f "hermes gateway"',
    "killall hermes",
    "hermes gateway stop",
    "crontab -l | grep -v backup | crontab -",
    "crontab -r",
    "sudo crontab -r",
    "crontab /tmp/new-cron",
    "chmod 644 ~/.ssh/id_ed25519",
    "chmod -R 777 ~/.ssh",
    "chmod go+r ~/.ssh/id_rsa",
]

NOT_GUARDRAIL = [
    'echo "FOO=1" >> ~/.hermes/.env',
    "cp ~/.hermes/.env /tmp/envbak",
    "cp ~/.hermes/.env ~/.hermes/.env.bak",
    "cat ~/.hermes/.env.example",
    "hermes config set model.default foo",
    "grep -n approvals ~/.hermes/config.yaml",
    "systemctl --user restart hermes-gateway",
    "systemctl --user stop hermes-webui",
    "systemctl --user disable --now hermes-bridge.service",
    "pkill -f hermes-runner-service",
    "crontab -l",
    "crontab -e",
    "which crontab",
    "chmod 600 ~/.ssh/id_ed25519",
    "chmod 700 ~/.ssh",
    "chmod -R go-rwx ~/.ssh",
]


@pytest.mark.parametrize("command", GUARDRAIL)
def test_guardrail_detected(command):
    _, key, _ = detect_dangerous_command(command)
    assert is_guardrail_key(key), (command, key)


@pytest.mark.parametrize("command", NOT_GUARDRAIL)
def test_routine_not_guardrail(command):
    _, key, _ = detect_dangerous_command(command)
    assert not is_guardrail_key(key), (command, key)


def test_config_lookahead_is_linear_on_huge_commands():
    import time
    big = "echo " + "x " * 30000 + " ~/.hermes/config.yaml"
    t0 = time.monotonic()
    detect_dangerous_command(big)
    assert time.monotonic() - t0 < 2.0


def _decide(command, key, *, smart=True):
    return A._human_decision(
        A._COMMAND_GATE, command=command, description=key, pattern_key=key, pattern_keys=[key],
        warnings=[(key, key, False)], session_key="s-guardrail", approval_callback=None,
        is_cli=False, is_gateway=False, is_ask=False, smart=smart)


def test_guardrail_skips_smart_guardian():
    key = detect_dangerous_command("crontab -r")[1]
    with patch.object(A, "_smart_gate") as smart_gate:
        result = _decide("crontab -r", key)
    smart_gate.assert_not_called()
    assert not result.get("approved")


def test_non_guardrail_still_uses_guardian():
    key = "recursive delete"
    with patch.object(A, "_smart_gate", return_value=({"approved": True, "smart_approved": True}, False)) as sg:
        result = _decide("rm -rf ./build", key)
    sg.assert_called_once()
    assert result["approved"]


def test_guardrail_always_downgrades_to_session():
    key = detect_dangerous_command("crontab -r")[1]
    with patch.object(A, "approve_session") as sess, patch.object(A, "approve_permanent") as perm, \
            patch.object(A, "save_permanent_allowlist") as save:
        A._persist_choice("s1", "always", [(key, key, False), ("recursive delete", "rd", False)])
    assert sess.call_count == 2
    perm.assert_called_once_with("recursive delete")
    save.assert_called_once()


# --- approval_prefilter hook -----------------------------------------------------------------

def _run_verdict(hook_results, llm="escalate", has_hook=True):
    with patch("hermes_cli.lifecycle.has_hook", return_value=has_hook), \
            patch("hermes_cli.lifecycle.invoke_hook", side_effect=lambda name, **kw:
                  hook_results if name == "approval_prefilter" else []), \
            patch.object(S, "_smart_approve", return_value=llm) as llm_call:
        verdict = S._smart_verdict("rm -rf ./build", "recursive delete", "recursive delete",
                                   ["recursive delete"], "s1")
    return verdict, llm_call


def test_prefilter_approve_skips_llm():
    verdict, llm = _run_verdict([{"verdict": "approve", "decided_by": "jev"}], llm="deny")
    assert verdict == "approve"
    llm.assert_not_called()


def test_prefilter_escalate_skips_llm():
    verdict, llm = _run_verdict([{"verdict": "escalate"}], llm="approve")
    assert verdict == "escalate"
    llm.assert_not_called()


def test_prefilter_cannot_deny():
    verdict, llm = _run_verdict([{"verdict": "deny"}], llm="approve")
    assert verdict == "approve"
    llm.assert_called_once()


def test_prefilter_defer_and_absent_use_llm():
    for results, has in (([None], True), ([], False)):
        verdict, llm = _run_verdict(results, llm="deny", has_hook=has)
        assert verdict == "deny"
        llm.assert_called_once()


def test_prefilter_error_falls_through_to_llm():
    with patch("hermes_cli.lifecycle.has_hook", return_value=True), \
            patch("hermes_cli.lifecycle.invoke_hook", side_effect=RuntimeError("boom")), \
            patch.object(S, "_smart_approve", return_value="approve") as llm:
        assert S._prefilter_verdict("ls", "d", "k", "command") is None
        assert S._smart_verdict("ls", "d", "k", ["k"], "s1") == "approve"
    llm.assert_called_once()


def test_approval_prefilter_is_a_valid_hook():
    from hermes_cli.plugins import VALID_HOOKS
    assert "approval_prefilter" in VALID_HOOKS
