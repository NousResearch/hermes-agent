"""Trusted typing (``Input.insertText``) sends a keystroke only to the document, channel and lease the
checks before it vouched for, and a refusal names the cause.

The supervisor here is a fake that behaves like ``CDPSupervisor`` where it matters: scripts and
keystrokes are pinned to a page session, ``evaluate_runtime`` reports the document a script ran on,
and ``insert_text`` refuses a document or session that moved since. Page scripts are answered by kind.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from typing import Any, Callable
from unittest.mock import patch

import pytest

from agent.vault_store import VaultStore
from tests.bot_desktop_lease import reset_lease
from tools.bot_desktop import lease, runtime

PASSWORD = "s3cret-pw"
_PW = {"autocomplete": "current-password", "formIndex": 0, "index": 1, "label": "", "name": "pw", "type": "password"}


class _Supervisor:
    """One tab on session S1. ``on_focus`` runs after each focus check is answered (the page or
    another tool acting between the check and the keystroke)."""

    def __init__(self, controls):
        self.controls, self.sid, self.document = controls, "S1", 1
        self.typed: list[str] = []
        self.on_focus: Callable[[], object] = lambda: None

    def current_page_session(self):
        return self.sid

    def focus_page(self, origin, accept=None):
        return {"ok": True, "url": origin + "/login", "session_id": self.sid}

    def evaluate_runtime(self, expression, *, session_id=None, **kwargs):
        if session_id is not None and session_id != self.sid:
            return {"ok": False, "error": "the page session changed since it was validated", "code": "session_changed"}
        document = self.document
        result = self._answer(expression)
        if 'refused: "focus_lost"' in expression:
            self.on_focus()
        return {"ok": True, "result": result, "document": document}

    def _answer(self, expression):
        if "location.href" in expression:
            return "https://example.com/login"
        if "ok: true" in expression:
            return {"ok": True}
        if 'refused: "focus_lost"' in expression:
            return {"focused": int(expression.split('"index": ', 1)[1].split(",", 1)[0])}
        if "reverted" in expression:
            return {"landed": expression.count('"index"'), "reverted": 0, "missing": []}
        if "unstamped" in expression:
            return {"unstamped": True}
        return self.controls

    def insert_text(self, text, *, session_id, document=None, **kwargs):
        if session_id != self.sid:
            return {"ok": False, "error": "the page session changed since it was validated", "code": "session_changed"}
        if document is not None and document != self.document:
            return {"ok": False, "error": "the page navigated or moved since it was checked", "code": "document_changed"}
        self.typed.append(text)
        return {"ok": True}


@pytest.fixture(autouse=True)
def _fresh_lease(monkeypatch):
    reset_lease()
    monkeypatch.setattr(runtime, "published_env", lambda: {"DISPLAY": ":37"})
    yield
    reset_lease()


def _fill(sup, tmp_path, *, bot_desktop=False, page=lambda t: None, with_identifier=False):
    from tools import browser_vault_tool

    store = VaultStore(base_dir=tmp_path / "vault")
    meta = store.add_item(kind="login", label="Example", origin="https://example.com",
                          secret={"identifier_type": "phone", "identifier": "+48600100200",
                                  "password": PASSWORD, "origin": "https://example.com"})
    with ExitStack() as stack:
        stack.enter_context(patch("agent.vault_store.get_vault_store", return_value=store))
        stack.enter_context(patch.object(browser_vault_tool, "_ensure_supervisor", return_value=sup))
        stack.enter_context(patch.object(browser_vault_tool, "_page_backend", side_effect=page))
        stack.enter_context(patch.object(browser_vault_tool, "_bot_desktop_browser_session", return_value=bot_desktop))
        stack.enter_context(patch.object(browser_vault_tool, "_FILL_VERIFY_STABLE_S", 0))
        return json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id="t", with_identifier=with_identifier))


def test_a_redirect_between_the_focus_check_and_the_keystroke_types_nothing(tmp_path):
    """The session id survives a navigation; the document does not. The keystroke carries the
    document the focus check ran on, so the redirected page's focused input never gets it."""
    sup = _Supervisor([_PW])

    def redirect():
        sup.document += 1  # e.g. a login page that redirects on focus, same tab, same session

    sup.on_focus = redirect
    out = _fill(sup, tmp_path)
    assert out["success"] is False and out["error_type"] == "page_navigated"
    assert sup.typed == []
    assert PASSWORD not in json.dumps(out)


def test_without_a_redirect_the_password_is_typed(tmp_path):
    sup = _Supervisor([_PW])
    out = _fill(sup, tmp_path)
    assert out["success"] is True and sup.typed == [PASSWORD]
