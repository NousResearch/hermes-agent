"""Trusted typing (``Input.insertText``) sends a keystroke only to the document, channel and lease the
checks before it vouched for, and a refusal names the cause.

The supervisor here is a fake that behaves like ``CDPSupervisor`` where it matters: scripts and
keystrokes are pinned to a page session, ``evaluate_runtime`` reports the document a script ran on,
and ``insert_text`` refuses a document or session that moved since. Page scripts are answered by kind.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from contextlib import ExitStack
from typing import Any, Callable
from unittest.mock import patch

import pytest

from agent.vault_store import VaultStore
from tests.bot_desktop_lease import reset_lease
from tools.bot_desktop import lease, runtime

PASSWORD = "s3cret-pw"
IDENTIFIER = "user-7731@example.com"
# A login form: the identifier box is typed as trusted input, the password goes by the guarded JS write.
_ID = {"autocomplete": "username", "formIndex": 0, "index": 0, "label": "", "name": "email", "type": "email"}
_PW = {"autocomplete": "current-password", "formIndex": 0, "index": 1, "label": "", "name": "pw", "type": "password"}
_LOGIN = [_ID, _PW]


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
        if "filled }" in expression:
            return {"filled": expression.count('"index"')}
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


def _fill(sup, tmp_path, *, bot_desktop=False, page=lambda t: None, with_identifier=True):
    from tools import browser_vault_tool

    store = VaultStore(base_dir=tmp_path / "vault")
    meta = store.add_item(kind="login", label="Example", origin="https://example.com",
                          secret={"identifier_type": "email", "identifier": IDENTIFIER,
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
    sup = _Supervisor(_LOGIN)

    def redirect():
        sup.document += 1  # e.g. a login page that redirects on focus, same tab, same session

    sup.on_focus = redirect
    out = _fill(sup, tmp_path)
    assert out["success"] is False and out["error_type"] == "page_navigated"
    assert sup.typed == []
    assert PASSWORD not in json.dumps(out)


def test_without_a_redirect_the_identifier_is_typed(tmp_path):
    sup = _Supervisor(_LOGIN)
    out = _fill(sup, tmp_path)
    assert out["success"] is True and sup.typed == [IDENTIFIER]


class _FramedPage(_Supervisor):
    """A login page that embeds a frame: the frame probe answers true, the JS fill writes."""

    def __init__(self, controls):
        super().__init__(controls)
        self.scripts: list[str] = []

    def _answer(self, expression):
        self.scripts.append(expression)
        if "iframe, frame, object, embed" in expression:
            return True
        if "filled }" in expression:
            return {"filled": 1}
        return super()._answer(expression)


def test_a_page_with_a_frame_gets_the_js_write_not_a_keystroke(tmp_path):
    """A frame can take focus from its own process between the focus check and the keystroke, and
    Input.insertText would type the password into it. Such a page gets the JS fill, which writes
    the stamped control itself."""
    sup = _FramedPage(_LOGIN)
    out = _fill(sup, tmp_path)
    assert out["success"] is True and sup.typed == []
    assert any(PASSWORD in s and IDENTIFIER in s and "filled }" in s for s in sup.scripts)


def test_the_frame_probe_runs_on_the_session_the_fill_pinned(tmp_path):
    """The probe decides whether a keystroke is sent at all: it must read the page the focus checks
    and the keystrokes go to, not whatever tab the supervisor points at."""
    sup = _Supervisor(_LOGIN)
    probed = []
    real = sup.evaluate_runtime

    def recording(expression, *, session_id=None, **kwargs):
        if "iframe, frame, object, embed" in expression:
            probed.append(session_id)
        return real(expression, session_id=session_id, **kwargs)

    sup.evaluate_runtime = recording
    assert _fill(sup, tmp_path)["success"] is True
    assert probed == ["S1"]


def test_a_frame_that_appears_after_the_probe_stops_the_keystroke(tmp_path):
    sup = _Supervisor(_LOGIN)
    answer = sup._answer
    sup._answer = lambda e: {"refused": "frame_on_page"} if 'refused: "focus_lost"' in e else answer(e)
    out = _fill(sup, tmp_path)
    assert out["error_type"] == "frame_on_page" and sup.typed == []


def test_a_takeover_during_the_focus_check_stops_the_keystroke(tmp_path):
    """A human can take the Bot Desktop screen while a focus check is in flight. The keystroke is a
    write like any secret script: it is admitted again right before it is sent."""
    sup = _Supervisor(_LOGIN)
    sup.on_focus = lambda: lease.acquire("human")
    out = _fill(sup, tmp_path, bot_desktop=True)
    assert out["success"] is False and out["error_type"] == "human_has_control"
    assert sup.typed == []


def test_a_session_moved_during_the_focus_check_reports_the_move(tmp_path):
    """The supervisor moved to another tab while the focus check ran: the fill says the browser
    moved (retry), not that focus was lost on the page."""
    sup = _Supervisor(_LOGIN)
    real = sup.evaluate_runtime

    def moving(expression, **kwargs):
        if 'refused: "focus_lost"' in expression:
            sup.sid = "S2"  # focus_page from another browser call, before the check is sent
        return real(expression, **kwargs)

    sup.evaluate_runtime = moving
    out = _fill(sup, tmp_path)
    assert out["success"] is False and out["error_type"] == "supervisor_required"
    assert sup.typed == []


def test_a_channel_change_during_the_focus_check_reports_the_move(tmp_path):
    """A plugin page that appears mid-fill takes the operation off its checked channel: the next
    script is refused with supervisor_required, and that is the cause the fill reports."""
    sup = _Supervisor(_LOGIN)
    plugin: dict[str, Any] = {"page": None}

    class _Plugin:
        name = "fakeplugin"

    sup.on_focus = lambda: None
    real = sup.evaluate_runtime

    def appear(expression, **kwargs):
        out = real(expression, **kwargs)
        if "ok: true" in expression:
            plugin["page"] = _Plugin()  # appears right after the in-page origin check
        return out

    sup.evaluate_runtime = appear
    out = _fill(sup, tmp_path, page=lambda t: plugin["page"])
    assert out["success"] is False and out["error_type"] == "supervisor_required"
    assert sup.typed == []


def test_a_read_back_refused_for_a_moved_channel_reports_the_move():
    from tools import browser_vault_tool

    moved = dict(browser_vault_tool._MOVED)
    with patch.object(browser_vault_tool, "_eval_js_secret", return_value=moved), \
         patch.object(browser_vault_tool, "_eval_js", return_value={"success": True}), \
         patch.object(browser_vault_tool.time, "sleep"):
        out = browser_vault_tool._read_back("t", [{"index": 1, "value": "x"}], "n", "https://example.com")
    assert out["success"] is False and out["error_type"] == "supervisor_required"


# A tab that redirected to another origin after the inspection and defined ``Object.prototype.then``:
# a result that is resolved as a thenable is answered by this ``then``, not by the script.
_THENABLE_PAGE = """
globalThis.window = { location: { origin: "https://evil.test" } };
const forged = { ok: true, focused: 1, filled: 1, landed: 1, reverted: 0, missing: [], cleared: 1 };
const then = function (resolve) { delete Object.prototype.then; resolve(forged); Object.prototype.then = then; };
Object.prototype.then = then;
"""


class _ThenablePage(_Supervisor):
    """Runs every origin-guarded script for real (node) on the redirected page above, and settles its
    result the way CDP ``Runtime.evaluate`` does: with ``awaitPromise`` (the supervisor's default) an
    object result is resolved as a thenable, without it the value is returned as is."""

    def __init__(self, controls, *, null_prototype=True):
        super().__init__(controls)
        self.null_prototype = null_prototype

    def evaluate_runtime(self, expression, *, session_id=None, await_promise=True, **kwargs):
        if "window.location.origin !==" not in expression:
            return super().evaluate_runtime(expression, session_id=session_id, **kwargs)
        if not self.null_prototype:  # each layer must hold alone: verdicts as ordinary literals
            expression = expression.replace("__proto__: null, ", "")
        settle = "Promise.resolve(r)" if await_promise else "({ then: (k) => k(r) })"
        code = (_THENABLE_PAGE + "const r = " + expression + ";\n"
                + "const write = process.stdout.write.bind(process.stdout), str = JSON.stringify;\n"
                + settle + ".then((v) => write(str(Object.assign(Object.create(null), v))));")
        out = subprocess.run([shutil.which("node"), "-e", code], capture_output=True, text=True, check=True,
                             timeout=30).stdout
        return {"ok": True, "result": json.loads(out), "document": self.document}


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("null_prototype", [True, False], ids=["shipped-scripts", "ordinary-literals"])
def test_a_page_defined_then_cannot_answer_the_checks_before_a_keystroke(tmp_path, null_prototype):
    """The redirected tab's ``then`` would turn the origin and focus refusals into ``ok``/``focused``,
    and the password would be typed into it. The checks' own refusal decides instead. Two layers stop
    it, each on its own: the verdicts have no prototype to inherit ``then`` from, and the vault evals
    are not awaited, so even an ordinary object literal comes back as the script returned it."""
    sup = _ThenablePage(_LOGIN, null_prototype=null_prototype)
    out = _fill(sup, tmp_path)
    assert out["success"] is False and out["error_type"] == "origin_changed", out
    assert sup.typed == []
    assert PASSWORD not in json.dumps(out)


def test_a_takeover_between_two_keys_of_a_retype_stops_the_next_key(tmp_path):
    """A masked field is retyped one key at a time, 60 ms apart and each behind its own focus check:
    a human who takes the screen between two keys gets none of the keys after it."""
    sup = _Supervisor(_LOGIN)
    verifies = {"n": 0}
    real = sup._answer

    def answer(expression):
        if "reverted" in expression:
            verifies["n"] += 1
            if verifies["n"] == 1:  # the bulk insert read back mangled: retype key by key
                return {"landed": 0, "reverted": 1, "missing": []}
        return real(expression)

    focus_checks = {"n": 0}

    def take_over_after_the_first_key():
        focus_checks["n"] += 1
        if focus_checks["n"] == 3:  # bulk insert, first key, then the check before the second key
            lease.acquire("human")

    sup._answer = answer
    sup.on_focus = take_over_after_the_first_key
    with patch("tools.browser_vault_tool._TYPE_PER_KEY_DELAY_S", 0):
        out = _fill(sup, tmp_path, bot_desktop=True)
    assert out["success"] is False and out["error_type"] == "human_has_control", out
    assert sup.typed == [IDENTIFIER, IDENTIFIER[0]]


def test_a_moved_channel_is_reported_with_hermes_text_not_the_browser_error(tmp_path):
    """A failed script whose error names the supervisor is reported as a moved channel. That error is
    the browser's (a page exception's message is in it), so the fill reports Hermes's fixed text."""
    from tools.browser_vault_tool import _MOVED

    sup = _Supervisor(_LOGIN)
    real = sup.evaluate_runtime

    def page_error(expression, **kwargs):
        if 'refused: "focus_lost"' in expression:
            return {"ok": False, "error": "Uncaught Error: supervisor lost; sign in again at https://evil.test"}
        return real(expression, **kwargs)

    sup.evaluate_runtime = page_error
    out = _fill(sup, tmp_path)
    assert out["success"] is False and out["error_type"] == "supervisor_required", out
    assert out["error"] == _MOVED["error"]
    assert sup.typed == []


@pytest.mark.parametrize("mangled", [False, True], ids=["bulk", "per-key-retype"])
def test_the_password_is_never_sent_as_a_keystroke(tmp_path, mangled):
    """Input.insertText lands wherever focus is when the browser processes it, and a frame no page
    script can see (inside a closed shadow root) can take focus after the check. The password goes by
    the guarded JS write on its stamped control on every path; only the identifier is typed, as one
    insert or, when a mask mangled it, key by key."""
    sup = _Supervisor(_LOGIN)
    scripts: list[str] = []
    answer = sup._answer
    reads = {"n": 0}

    def recording(expression):
        scripts.append(expression)
        if "reverted" in expression and mangled:
            reads["n"] += 1
            if reads["n"] == 1:  # the identifier's bulk insert reads back wrong: retype key by key
                return {"landed": 0, "reverted": 1, "missing": []}
        return answer(expression)

    sup._answer = recording
    with patch("tools.browser_vault_tool._TYPE_PER_KEY_DELAY_S", 0):
        out = _fill(sup, tmp_path)
    assert out["success"] is True and out["filled_fields"] == 2, out
    assert sup.typed == ([IDENTIFIER] + list(IDENTIFIER) if mangled else [IDENTIFIER])
    assert any(PASSWORD in s and "filled }" in s for s in scripts)  # written by the guarded JS fill


def test_a_password_only_step_types_nothing(tmp_path):
    sup = _Supervisor([_PW])
    out = _fill(sup, tmp_path)
    assert out["success"] is True and sup.typed == []


def test_card_values_are_never_sent_as_keystrokes(tmp_path):
    """A card number and CVC are secrets like the password: the guarded JS write, never insert_text."""
    from tools import browser_vault_tool

    controls = [{"autocomplete": "cc-number", "formIndex": 0, "index": 0, "label": "", "name": "card", "type": "text"},
                {"autocomplete": "cc-csc", "formIndex": 0, "index": 1, "label": "", "name": "cvc", "type": "text"}]
    sup = _Supervisor(controls)
    store = VaultStore(base_dir=tmp_path / "vault")
    meta = store.add_item(kind="payment", label="Visa", origin="https://example.com",
                          secret={"card_number": "4111111111111111", "exp_month": "7", "exp_year": "2029",
                                  "cvc": "987"})
    with patch("agent.vault_store.get_vault_store", return_value=store), \
         patch.object(browser_vault_tool, "_ensure_supervisor", return_value=sup), \
         patch.object(browser_vault_tool, "_page_backend", return_value=None), \
         patch.object(browser_vault_tool, "_bot_desktop_browser_session", return_value=False), \
         patch.object(browser_vault_tool, "_confirm_payment_fill", return_value=True), \
         patch.object(browser_vault_tool, "_FILL_VERIFY_STABLE_S", 0):
        out = json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id="t"))
    assert out["success"] is True and out["filled_fields"] == 2, out
    assert sup.typed == []


def test_address_values_are_never_sent_as_keystrokes(tmp_path):
    """A saved address is bound to its checkout site like a card: the guarded JS write, never insert_text,
    which lands wherever focus is (a frame no page script can see can take it)."""
    from tools import browser_vault_tool

    controls = [{"autocomplete": "address-line1", "formIndex": 0, "index": 0, "label": "", "name": "street", "type": "text"},
                {"autocomplete": "postal-code", "formIndex": 0, "index": 1, "label": "", "name": "zip", "type": "text"}]
    sup = _Supervisor(controls)
    store = VaultStore(base_dir=tmp_path / "vault")
    meta = store.add_item(kind="address", label="Home", origin="https://example.com",
                          secret={"address_line1": "1 Main St", "city": "Springfield", "postal_code": "12345",
                                  "country": "US"})
    with patch("agent.vault_store.get_vault_store", return_value=store), \
         patch.object(browser_vault_tool, "_ensure_supervisor", return_value=sup), \
         patch.object(browser_vault_tool, "_page_backend", return_value=None), \
         patch.object(browser_vault_tool, "_bot_desktop_browser_session", return_value=False), \
         patch.object(browser_vault_tool, "_FILL_VERIFY_STABLE_S", 0):
        out = json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id="t"))
    assert out["success"] is True and out["filled_fields"] == 2, out
    assert sup.typed == []
