"""Synthetic DOM security contracts for saved Messenger restore PIN fills."""

import json
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import shutil

import pytest


from tools import browser_vault_tool as vault
from agent.vault_login_classifier import (
    build_fill_js,
    classify_login_control,
    LoginControl,
)


@pytest.fixture(autouse=True, scope="module")
def require_dom_runtime():
    if not shutil.which("node"):
        pytest.skip("Messenger DOM contracts require Node.js and npm ci")
    probe = subprocess.run(
        ["node", "-e", "require('jsdom')"],
        capture_output=True,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        timeout=15,
    )
    if probe.returncode:
        pytest.skip("Messenger DOM contracts require jsdom from npm ci")


ORIGIN = "https://www.facebook.com"
LABEL = "facebook messenger pin"


def run_dom(
    expression,
    *,
    origin=ORIGIN,
    heading="Enter your PIN to restore your chats",
    field='type="text" aria-label="PIN"',
    mutate="",
):
    program = """
const {JSDOM} = require('jsdom');
const p = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const dom = new JSDOM('<div role="dialog"><h2>'+p.heading+'</h2><input '+p.field+'></div>', {url:p.origin, runScripts:'outside-only'});
const w = dom.window;
w.HTMLElement.prototype.getClientRects = function(){return [{width:10,height:10}]};
const result = w.eval(p.expression);
w.eval(p.mutate);
console.log(JSON.stringify({result, value:w.document.querySelector('input').value}));
"""
    result = subprocess.run(
        ["node", "-e", program],
        input=json.dumps(
            dict(
                expression=expression,
                origin=origin,
                heading=heading,
                field=field,
                mutate=mutate,
            )
        ),
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    "origin,heading,field,expected",
    [
        (
            ORIGIN,
            "Enter your PIN to restore your chats",
            'type="text" aria-label="PIN"',
            True,
        ),
        (
            "https://facebook.com.evil.test",
            "Enter your PIN to restore your chats",
            'aria-label="PIN"',
            False,
        ),
        (ORIGIN, "Enter your PIN", 'aria-label="PIN"', False),
        (ORIGIN, "Enter your PIN to restore your chats", 'type="password"', False),
        (
            ORIGIN,
            "Enter your PIN to restore your chats",
            'aria-label="PIN" disabled',
            False,
        ),
        (
            ORIGIN,
            "Enter your PIN to restore your chats",
            'aria-label="PIN" readonly',
            False,
        ),
        (
            ORIGIN,
            "Enter your PIN to restore your chats",
            'aria-label="PIN" style="display:none"',
            False,
        ),
        (
            ORIGIN,
            "Enter your PIN to restore your chats",
            'aria-label="PIN"><input aria-label="PIN"',
            False,
        ),
        (
            "http://www.facebook.com",
            "Enter your PIN to restore your chats",
            'aria-label="PIN"',
            False,
        ),
        (
            "https://www.facebook.com:8443",
            "Enter your PIN to restore your chats",
            'aria-label="PIN"',
            False,
        ),
    ],
)
def test_restore_tab_probe(origin, heading, field, expected):
    from agent.vault_messenger_restore import RESTORE_TAB_PROBE

    assert (
        run_dom(RESTORE_TAB_PROBE, origin=origin, heading=heading, field=field)[
            "result"
        ]
        is expected
    )


def test_restore_fill_rechecks_dialog_and_origin_at_write():
    from agent.vault_messenger_restore import (
        build_restore_inspection_js,
        build_restore_fill_js,
    )

    inspect = build_restore_inspection_js("testnonce")
    fill = build_restore_fill_js(
        [{"index": 0, "token": "messenger-restore-pin", "value": "synthetic-pin"}],
        ORIGIN,
        "testnonce",
    )
    for mutation, expected in [
        ("", "synthetic-pin"),
        ("document.querySelector('h2').textContent='Log in'", ""),
        ("document.querySelector('input').setAttribute('aria-label','Password')", ""),
        ("document.querySelector('input').type='password'", ""),
        ("document.querySelector('input').disabled=true", ""),
        ("document.querySelector('input').readOnly=true", ""),
        ("document.querySelector('input').style.display='none'", ""),
        ("document.querySelector('[role=dialog]').removeAttribute('role')", ""),
        (
            "document.querySelector('input').setAttribute('data-hermes-vault-slot','other:0')",
            "",
        ),
        (
            "document.querySelector('input').after(document.querySelector('input').cloneNode())",
            "",
        ),
    ]:
        expr = f"(() => {{ {inspect}; {mutation}; return {fill}; }})()"
        assert run_dom(expr)["value"] == expected
    assert (
        run_dom(
            f"(() => {{ {inspect}; return {fill}; }})()", origin="https://evil.test"
        )["value"]
        == ""
    )


@pytest.mark.parametrize(
    "label,origin,heading,field,allowed,expected",
    [
        (
            LABEL,
            ORIGIN,
            "Enter your PIN to restore your chats",
            'aria-label="PIN"',
            (ORIGIN,),
            True,
        ),
        (
            "Facebook login",
            ORIGIN,
            "Enter your PIN to restore your chats",
            'aria-label="PIN"',
            (ORIGIN,),
            False,
        ),
        (
            LABEL,
            "https://evil.test",
            "Enter your PIN to restore your chats",
            'aria-label="PIN"',
            ("https://evil.test",),
            False,
        ),
        (LABEL, ORIGIN, "Log in", 'type="password"', (ORIGIN,), False),
        (
            LABEL,
            ORIGIN,
            "Enter your PIN to restore your chats",
            'aria-label="PIN"',
            ("https://facebook.com",),
            False,
        ),
        (
            LABEL,
            ORIGIN,
            "Enter your PIN to restore your chats",
            'aria-label="PIN"',
            ("https://facebook.com", ORIGIN),
            True,
        ),
    ],
)
def test_saved_pin_handler_is_scoped(
    monkeypatch, label, origin, heading, field, allowed, expected
):
    class Backend(SimpleNamespace):
        def fill_context(self):
            return nullcontext()

    backend = Backend(
        name="local",
        needs_unlock=False,
        get_meta=lambda _: SimpleNamespace(
            id="synthetic",
            kind="login",
            label=label,
            origin=allowed[0],
            allowed_origins=allowed,
            has_otp=False,
        ),
        resolve_password=Mock(return_value="synthetic-pin"),
    )
    monkeypatch.setattr("agent.vault_backends.backend_for_handle", lambda _: backend)
    probes = []

    class Supervisor:
        def focus_page(self, target, accept=None):
            probes.append(accept)
            return {
                "ok": bool(
                    run_dom(accept, origin=origin, heading=heading, field=field)[
                        "result"
                    ]
                )
            }

    monkeypatch.setattr(vault, "_ensure_supervisor", lambda _: Supervisor())
    monkeypatch.setattr(vault, "_current_page_origin", lambda _: origin)
    inspection = []

    def evaluate(_, expression):
        inspection.append(expression)
        return {
            "success": True,
            "result": run_dom(expression, origin=origin, heading=heading, field=field)[
                "result"
            ],
        }

    monkeypatch.setattr(vault, "_eval_js", evaluate)

    def fill(_, expression):
        combined = f"(() => {{ {inspection[-1]}; return {expression}; }})()"
        result = run_dom(combined, origin=origin, heading=heading, field=field)
        assert (result["value"] == "synthetic-pin") is expected
        return {"success": True, "result": result["result"]}

    monkeypatch.setattr(vault, "_eval_js_secret", fill)
    out = vault._handle_vault({"action": "fill", "handle": "synthetic"}, task_id="test")
    assert json.loads(out)["success"] is expected
    assert "synthetic-pin" not in out
    if not expected and label == LABEL:
        backend.resolve_password.assert_not_called()


def test_general_login_classifier_does_not_accept_pin():
    assert classify_login_control(LoginControl("", None, 0, "PIN", "", "text")) is None
    assert (
        run_dom(build_fill_js([{"index": 0, "value": "synthetic-pin"}], ORIGIN))[
            "value"
        ]
        == ""
    )
